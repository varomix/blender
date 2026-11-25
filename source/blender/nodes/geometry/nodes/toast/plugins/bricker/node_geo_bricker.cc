/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Bricker: divides faces into bricks by cutting them with a lattice of planes.
 *
 * Every selected face is cut by three families of parallel planes (a "lattice"). Plane family
 * `i` has normal `N_i` and the planes sit at `dot(P, N_i) = bbox_min[i] + offset[i] + k * size[i]`
 * for `k = 0 .. ceil(bbox_extent[i] / size[i]) - 1`, where the bounding box is the one of the
 * whole input mesh and the offset is wrapped into `[0, size)`. Each family is tilted by its own
 * angle:
 *
 *   N_x = (cos(ax), sin(ax), 0)
 *   N_y = (sin(ay), cos(ay), 0)
 *   N_z = (0, sin(az), cos(az))
 *
 * Topology guarantees:
 * - Points created on an input edge are created once per edge and shared by every face using
 *   that edge, so the result is watertight where the input was.
 * - With "Fix Shared Edges", unselected faces receive the points created on their edges, which
 *   avoids T-junctions at the selection border.
 * - Concave faces are supported (a plane can split a face into more than two pieces).
 *
 * Attributes: point and corner attributes are interpolated linearly, face attributes are copied
 * from the source face, edge attributes are copied to sub-edges of the original edges.
 */

#include <algorithm>
#include <atomic>
#include <limits>

#include "BLI_array.hh"
#include "BLI_bounds.hh"
#include "BLI_map.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_offset_indices.hh"
#include "BLI_ordered_edge.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "DNA_mesh_types.h"

#include "BKE_attribute.hh"
#include "BKE_attribute_math.hh"
#include "BKE_mesh.h"
#include "BKE_mesh.hh"

#include "BLT_translation.hh"

#include "GEO_foreach_geometry.hh"

#include "NOD_toast_plugins.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_bricker_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();

  b.add_input<decl::Geometry>("Mesh"_ustr)
      .supported_type(GeometryComponent::Type::Mesh)
      .is_default_link_socket()
      .description("Mesh to divide into bricks");
  b.add_output<decl::Geometry>("Mesh"_ustr).propagate_all_geometry().align_with_previous();
  b.add_input<decl::Bool>("Selection"_ustr)
      .default_value(true)
      .evaluated_geometry_field()
      .hide_value()
      .description("Faces to divide");
  b.add_input<decl::Vector>("Size"_ustr)
      .default_value({1.0f, 1.0f, 1.0f})
      .min(0.0f)
      .subtype(PROP_XYZ)
      .description("Distance between cutting planes along each axis. Zero disables an axis");
  b.add_input<decl::Vector>("Offset"_ustr)
      .subtype(PROP_TRANSLATION)
      .description("Shift of the cutting planes, wrapped into the range [0, Size)");
  b.add_input<decl::Vector>("Angle"_ustr)
      .subtype(PROP_EULER)
      .description("Tilt of each family of cutting planes");
  b.add_input<decl::Bool>("Fix Shared Edges"_ustr)
      .default_value(true)
      .description(
          "Insert the new points into unselected faces that share an edge with selected faces, "
          "avoiding T-junctions");
}

/* -------------------------------------------------------------------- */
/** \name Lattice
 * \{ */

struct Lattice {
  double3 normal[3];
  double base[3];
  double size[3];
  double eps[3];
  int count[3];
  /** Global index of the first plane of each family. */
  int start[3];
  int total = 0;

  int axis_of(const int plane) const
  {
    return plane < start[1] ? 0 : (plane < start[2] ? 1 : 2);
  }

  double value(const int plane) const
  {
    const int axis = this->axis_of(plane);
    return base[axis] + double(plane - start[axis]) * size[axis];
  }

  double distance(const double3 &p, const int plane) const
  {
    const int axis = this->axis_of(plane);
    return math::dot(p, normal[axis]) - (base[axis] + double(plane - start[axis]) * size[axis]);
  }
};

static constexpr int max_planes_per_axis = 1000000;

static std::optional<Lattice> build_lattice(const Bounds<float3> &bounds,
                                            const float3 &size,
                                            const float3 &offset,
                                            const float3 &angle,
                                            bool &r_too_many)
{
  Lattice lattice;
  lattice.normal[0] = double3(std::cos(double(angle.x)), std::sin(double(angle.x)), 0.0);
  lattice.normal[1] = double3(std::sin(double(angle.y)), std::cos(double(angle.y)), 0.0);
  lattice.normal[2] = double3(0.0, std::sin(double(angle.z)), std::cos(double(angle.z)));

  int start = 0;
  for (const int axis : IndexRange(3)) {
    const double s = size[axis];
    const double extent = double(bounds.max[axis]) - double(bounds.min[axis]);
    int count = 0;
    double o = 0.0;
    if (s > 0.0) {
      const double divisions = std::ceil(extent / s - 1e-5);
      if (divisions > max_planes_per_axis) {
        r_too_many = true;
        return std::nullopt;
      }
      count = std::max(0, int(divisions));
      o = std::fmod(double(offset[axis]), s);
      if (o < 0.0) {
        o += s;
      }
    }
    lattice.size[axis] = s;
    lattice.eps[axis] = 1e-5 * s;
    lattice.base[axis] = double(bounds.min[axis]) + o;
    lattice.count[axis] = count;
    lattice.start[axis] = start;
    start += count;
  }
  lattice.total = start;
  if (lattice.total == 0) {
    return std::nullopt;
  }
  return lattice;
}

static int side_of(const double d, const double eps)
{
  return d > eps ? 1 : (d < -eps ? -1 : 0);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Edge Points
 *
 * Points created where lattice planes strictly cross an input edge. They are computed once per
 * edge from the full edge (never from a sub-segment), so every face using the edge agrees on
 * them exactly.
 * \{ */

struct EdgePoints {
  /** CSR offsets per edge. Edge point `i` gets the output vertex index `verts_num + i`. */
  Array<int> offsets_data;
  /** Parameter along the edge, from `edge[0]` to `edge[1]`. */
  Array<float> t;
  Array<int> plane;
  Array<int> edge;
  Array<float3> position;

  OffsetIndices<int> offsets() const
  {
    return offsets_data.as_span();
  }
  int size() const
  {
    return t.size();
  }
};

template<typename Fn>
static void foreach_edge_crossing(const Lattice &lattice,
                                  const double3 &a,
                                  const double3 &b,
                                  const Fn &fn)
{
  for (const int axis : IndexRange(3)) {
    const int count = lattice.count[axis];
    if (count == 0) {
      continue;
    }
    const double s = lattice.size[axis];
    const double pa = math::dot(a, lattice.normal[axis]);
    const double pb = math::dot(b, lattice.normal[axis]);
    const double lo = std::min(pa, pb);
    const double hi = std::max(pa, pb);
    const int k0 = std::max(0, int(std::floor((lo - lattice.base[axis]) / s)));
    const int k1 = std::min(count - 1, int(std::ceil((hi - lattice.base[axis]) / s)));
    for (int k = k0; k <= k1; k++) {
      const double c = lattice.base[axis] + double(k) * s;
      const double da = pa - c;
      const double db = pb - c;
      const int sa = side_of(da, lattice.eps[axis]);
      const int sb = side_of(db, lattice.eps[axis]);
      if (sa * sb < 0) {
        fn(lattice.start[axis] + k, da / (da - db));
      }
    }
  }
}

static EdgePoints compute_edge_points(const Lattice &lattice,
                                      const Span<float3> positions,
                                      const Span<int2> edges,
                                      const Span<bool> edge_mask)
{
  EdgePoints result;
  result.offsets_data.reinitialize(edges.size() + 1);
  MutableSpan<int> counts = result.offsets_data.as_mutable_span().drop_back(1);
  threading::parallel_for(edges.index_range(), 2048, [&](const IndexRange range) {
    for (const int e : range) {
      int count = 0;
      if (edge_mask[e]) {
        foreach_edge_crossing(lattice,
                              double3(positions[edges[e][0]]),
                              double3(positions[edges[e][1]]),
                              [&](int /*plane*/, double /*t*/) { count++; });
      }
      counts[e] = count;
    }
  });
  const OffsetIndices<int> offsets = offset_indices::accumulate_counts_to_offsets(
      result.offsets_data);
  const int total = offsets.total_size();
  result.t.reinitialize(total);
  result.plane.reinitialize(total);
  result.edge.reinitialize(total);
  result.position.reinitialize(total);

  threading::parallel_for(edges.index_range(), 1024, [&](const IndexRange range) {
    Vector<std::pair<double, int>, 16> crossings;
    for (const int e : range) {
      const IndexRange points = offsets[e];
      if (points.is_empty()) {
        continue;
      }
      const double3 a(positions[edges[e][0]]);
      const double3 b(positions[edges[e][1]]);
      crossings.clear();
      foreach_edge_crossing(
          lattice, a, b, [&](const int plane, const double t) { crossings.append({t, plane}); });
      std::sort(crossings.begin(), crossings.end());
      for (const int i : crossings.index_range()) {
        const int dst = points[i];
        const double t = crossings[i].first;
        result.t[dst] = float(t);
        result.plane[dst] = crossings[i].second;
        result.edge[dst] = e;
        result.position[dst] = float3(a * (1.0 - t) + b * t);
      }
    }
  });
  return result;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Per Face Splitting
 * \{ */

using Weights = Vector<std::pair<int, float>, 4>;

/** A point known to the face currently being processed. */
struct LocalPoint {
  double3 position;
  /** Original vertex index, or -1. */
  int vert = -1;
  /** Edge point index, or -1. */
  int edge_point = -1;
  /** Lattice planes the point lies on exactly (by construction). */
  int plane_a = -1;
  int plane_b = -1;
  /** Interpolation weights over face-local corner indices. */
  Weights weights;
  /** Index in the chunk interior point table once emitted, or -1. */
  int chunk_interior = -1;
};

/** Output of a contiguous range of input faces, assembled into the final mesh afterwards. */
struct ChunkResult {
  Vector<int> face_src;
  Vector<int> face_sizes;
  /** >= 0: final vertex index. < 0: `-(chunk interior index) - 1`. */
  Vector<int> corner_refs;
  Vector<int> corner_weight_counts;
  Vector<int> corner_weight_src;
  Vector<float> corner_weights;
  Vector<float3> interior_positions;
  Vector<int> interior_weight_counts;
  /** Source corner indices (mapped to vertices for point attributes). */
  Vector<int> interior_weight_src;
  Vector<float> interior_weights;
};

static void add_weighted(Weights &dst, const Weights &src, const float factor)
{
  if (factor == 0.0f) {
    return;
  }
  for (const std::pair<int, float> &item : src) {
    bool found = false;
    for (std::pair<int, float> &existing : dst) {
      if (existing.first == item.first) {
        existing.second += item.second * factor;
        found = true;
        break;
      }
    }
    if (!found) {
      dst.append({item.first, item.second * factor});
    }
  }
}

struct CrossNode {
  int point;
  bool is_cross;
  int side;
  /** Sort keys along the cut line (primary position, secondary symbolic perturbation). */
  double key1 = 0.0;
  double key2 = 0.0;
};

class FaceSplitter {
 public:
  const Lattice &lattice;
  const Span<float3> positions;
  const Span<int2> edges;
  const EdgePoints &edge_points;

  Vector<LocalPoint> points;
  Map<int64_t, int> interior_by_key;
  double3 face_normal;
  int unique_key_counter = 0;

  /* Scratch buffers, reused between faces and splits. */
  Vector<int> sides_;
  Vector<CrossNode> nodes_;
  Vector<CrossNode> ring_;
  Vector<int> cross_order_;
  Vector<int> partner_;
  Vector<bool> visited_;

  FaceSplitter(const Lattice &lattice,
               const Span<float3> positions,
               const Span<int2> edges,
               const EdgePoints &edge_points)
      : lattice(lattice), positions(positions), edges(edges), edge_points(edge_points)
  {
  }

  void reset()
  {
    points.clear();
    interior_by_key.clear();
    unique_key_counter = 0;
  }

  int edge_point_side(const int ep, const int plane) const
  {
    if (edge_points.plane[ep] == plane) {
      return 0;
    }
    const int axis = lattice.axis_of(plane);
    const int2 edge = edges[edge_points.edge[ep]];
    const double da = lattice.distance(double3(positions[edge[0]]), plane);
    const double db = lattice.distance(double3(positions[edge[1]]), plane);
    const int sa = side_of(da, lattice.eps[axis]);
    const int sb = side_of(db, lattice.eps[axis]);
    if (sa * sb < 0) {
      /* Same formula as when the edge points were created, so the ordering is consistent. */
      const float tq = float(da / (da - db));
      const float t = edge_points.t[ep];
      return t < tq ? sa : (t > tq ? sb : 0);
    }
    return sa != 0 ? sa : sb;
  }

  int side(const int point_i, const int plane) const
  {
    const LocalPoint &point = points[point_i];
    const int axis = lattice.axis_of(plane);
    if (point.vert >= 0) {
      return side_of(lattice.distance(double3(positions[point.vert]), plane), lattice.eps[axis]);
    }
    if (point.edge_point >= 0) {
      return this->edge_point_side(point.edge_point, plane);
    }
    if (point.plane_a == plane || point.plane_b == plane) {
      return 0;
    }
    return side_of(lattice.distance(point.position, plane), lattice.eps[axis]);
  }

  double distance(const int point_i, const int plane) const
  {
    return lattice.distance(points[point_i].position, plane);
  }

  /** Point where the segment between two local points crosses #plane, shared within the face. */
  int interior_point(const int v, const int w, const int plane)
  {
    const LocalPoint &pv = points[v];
    const LocalPoint &pw = points[w];
    int common = -1;
    for (const int a : {pv.plane_a, pv.plane_b}) {
      if (a >= 0 && (a == pw.plane_a || a == pw.plane_b)) {
        common = (common < 0) ? a : std::min(common, a);
      }
    }
    int64_t key;
    if (common >= 0) {
      const int64_t lo = std::min(common, plane);
      const int64_t hi = std::max(common, plane);
      key = lo * int64_t(lattice.total) + hi;
    }
    else {
      /* Should not happen with consistent sides; keep going with an unshared point. */
      key = -1 - int64_t(unique_key_counter++);
    }
    if (const int *existing = interior_by_key.lookup_ptr(key)) {
      return *existing;
    }
    const double dv = this->distance(v, plane);
    const double dw = this->distance(w, plane);
    const double t = dv / (dv - dw);
    LocalPoint point;
    point.position = pv.position * (1.0 - t) + pw.position * t;
    point.plane_a = common;
    point.plane_b = plane;
    add_weighted(point.weights, pv.weights, float(1.0 - t));
    add_weighted(point.weights, points[w].weights, float(t));
    const int index = points.append_and_get_index(std::move(point));
    interior_by_key.add_new(key, index);
    return index;
  }

  /**
   * Split #poly by #plane. Appends the resulting polygons to #r_pieces.
   * Returns false if the plane does not cut the polygon (nothing is appended then).
   */
  bool split(const Span<int> poly, const int plane, Vector<Vector<int>> &r_pieces)
  {
    const int size = poly.size();
    sides_.resize(size);
    bool has_neg = false;
    bool has_pos = false;
    for (const int i : poly.index_range()) {
      sides_[i] = this->side(poly[i], plane);
      has_neg |= sides_[i] < 0;
      has_pos |= sides_[i] > 0;
    }
    if (!(has_neg && has_pos)) {
      return false;
    }
    const double3 &plane_normal = lattice.normal[lattice.axis_of(plane)];
    const double3 line_dir = math::cross(face_normal, plane_normal);

    /* Insert strict crossing points. */
    nodes_.clear();
    for (const int i : poly.index_range()) {
      const int next = (i + 1) % size;
      nodes_.append({poly[i], false, sides_[i]});
      if (sides_[i] * sides_[next] < 0) {
        const int x = this->interior_point(poly[i], poly[next], plane);
        CrossNode node{x, true, 0};
        node.key1 = math::dot(points[x].position, line_dir);
        nodes_.append(node);
      }
    }

    /* Vertices on the plane count as being on the positive side (symbolic perturbation). They
     * become crossings when a neighbor is on the negative side. */
    ring_.clear();
    const int nodes_num = nodes_.size();
    for (const int i : IndexRange(nodes_num)) {
      const CrossNode &node = nodes_[i];
      if (node.is_cross || node.side != 0) {
        ring_.append(node);
        continue;
      }
      const CrossNode &prev = nodes_[(i + nodes_num - 1) % nodes_num];
      const CrossNode &next = nodes_[(i + 1) % nodes_num];
      const double3 &p = points[node.point].position;
      const double key1 = math::dot(p, line_dir);
      bool any = false;
      for (const CrossNode *neighbor : {&prev, &next}) {
        if (neighbor->side < 0) {
          const double3 &q = points[neighbor->point].position;
          const double dq = std::abs(this->distance(neighbor->point, plane));
          CrossNode cross{node.point, true, 0, key1, 0.0};
          cross.key2 = dq > 0.0 ? math::dot(q - p, line_dir) / dq : 0.0;
          ring_.append(cross);
          any = true;
        }
      }
      if (!any) {
        ring_.append(node);
      }
    }

    /* Pair crossings along the cut line. */
    const int ring_num = ring_.size();
    cross_order_.clear();
    for (const int i : IndexRange(ring_num)) {
      if (ring_[i].is_cross) {
        cross_order_.append(i);
      }
    }
    if (cross_order_.size() % 2 != 0 || cross_order_.is_empty()) {
      /* Degenerate configuration, leave the polygon untouched. */
      return false;
    }
    std::sort(cross_order_.begin(), cross_order_.end(), [&](const int a, const int b) {
      if (ring_[a].key1 != ring_[b].key1) {
        return ring_[a].key1 < ring_[b].key1;
      }
      if (ring_[a].key2 != ring_[b].key2) {
        return ring_[a].key2 < ring_[b].key2;
      }
      return a < b;
    });
    partner_.resize(ring_num);
    for (int i = 0; i < cross_order_.size(); i += 2) {
      partner_[cross_order_[i]] = cross_order_[i + 1];
      partner_[cross_order_[i + 1]] = cross_order_[i];
    }

    /* Trace the pieces. Each crossing is reached once along the ring (then jumps along the
     * chord) and once as the landing of a chord. */
    visited_.resize(ring_num);
    visited_.fill(false);
    for (const int start : IndexRange(ring_num)) {
      if (visited_[start]) {
        continue;
      }
      Vector<int> piece;
      int i = start;
      for (int guard = 0; guard < 4 * ring_num + 8; guard++) {
        visited_[i] = true;
        piece.append(ring_[i].point);
        if (ring_[i].is_cross) {
          const int j = partner_[i];
          piece.append(ring_[j].point);
          i = (j + 1) % ring_num;
        }
        else {
          i = (i + 1) % ring_num;
        }
        if (i == start) {
          break;
        }
      }
      r_pieces.append(std::move(piece));
    }
    return true;
  }

  /** Cut #poly by all planes of #axis with index in [lo, hi], divide and conquer. */
  void split_axis(Vector<int> poly, const int axis, int lo, int hi, Vector<Vector<int>> &r_out)
  {
    const int start = lattice.start[axis];
    const double s = lattice.size[axis];
    if (lo <= hi) {
      /* Narrow the plane range to the projection of the polygon. */
      double pmin = std::numeric_limits<double>::max();
      double pmax = std::numeric_limits<double>::lowest();
      for (const int p : poly) {
        const double proj = math::dot(points[p].position, lattice.normal[axis]);
        pmin = std::min(pmin, proj);
        pmax = std::max(pmax, proj);
      }
      lo = std::max(lo, int(std::floor((pmin - lattice.base[axis]) / s)));
      hi = std::min(hi, int(std::ceil((pmax - lattice.base[axis]) / s)));
    }
    if (lo > hi) {
      r_out.append(std::move(poly));
      return;
    }
    const int mid = (lo + hi) / 2;
    const int plane = start + mid;
    Vector<Vector<int>> pieces;
    if (!this->split(poly, plane, pieces)) {
      bool has_neg = false;
      for (const int p : poly) {
        if (this->side(p, plane) < 0) {
          has_neg = true;
          break;
        }
      }
      if (has_neg) {
        this->split_axis(std::move(poly), axis, lo, mid - 1, r_out);
      }
      else {
        this->split_axis(std::move(poly), axis, mid + 1, hi, r_out);
      }
      return;
    }
    for (Vector<int> &piece : pieces) {
      bool has_neg = false;
      for (const int p : piece) {
        if (this->side(p, plane) < 0) {
          has_neg = true;
          break;
        }
      }
      if (has_neg) {
        this->split_axis(std::move(piece), axis, lo, mid - 1, r_out);
      }
      else {
        this->split_axis(std::move(piece), axis, mid + 1, hi, r_out);
      }
    }
  }
};

static double3 newell_normal(const Span<LocalPoint> points, const Span<int> ring)
{
  double3 n(0.0);
  const int size = ring.size();
  for (const int i : ring.index_range()) {
    const double3 &a = points[ring[i]].position;
    const double3 &b = points[ring[(i + 1) % size]].position;
    n += math::cross(a, b);
  }
  return n;
}

static void emit_piece(FaceSplitter &splitter,
                       const Span<int> piece,
                       const int src_face,
                       const IndexRange face_corners,
                       const int verts_num,
                       ChunkResult &chunk,
                       Vector<int> &scratch)
{
  /* Remove consecutive duplicates (including wrap-around). */
  scratch.clear();
  for (const int p : piece) {
    if (scratch.is_empty() || scratch.last() != p) {
      scratch.append(p);
    }
  }
  while (scratch.size() > 1 && scratch.first() == scratch.last()) {
    scratch.remove_last();
  }
  if (scratch.size() < 3) {
    return;
  }
  {
    /* Require at least three distinct points. */
    int distinct = 0;
    for (const int i : scratch.index_range()) {
      bool seen = false;
      for (const int j : IndexRange(i)) {
        if (scratch[j] == scratch[i]) {
          seen = true;
          break;
        }
      }
      distinct += seen ? 0 : 1;
      if (distinct >= 3) {
        break;
      }
    }
    if (distinct < 3) {
      return;
    }
  }

  chunk.face_src.append(src_face);
  chunk.face_sizes.append(scratch.size());
  for (const int p : scratch) {
    LocalPoint &point = splitter.points[p];
    if (point.vert >= 0) {
      chunk.corner_refs.append(point.vert);
    }
    else if (point.edge_point >= 0) {
      chunk.corner_refs.append(verts_num + point.edge_point);
    }
    else {
      if (point.chunk_interior < 0) {
        point.chunk_interior = chunk.interior_positions.size();
        chunk.interior_positions.append(float3(point.position));
        chunk.interior_weight_counts.append(point.weights.size());
        for (const std::pair<int, float> &w : point.weights) {
          chunk.interior_weight_src.append(face_corners[w.first]);
          chunk.interior_weights.append(w.second);
        }
      }
      chunk.corner_refs.append(-point.chunk_interior - 1);
    }
    chunk.corner_weight_counts.append(point.weights.size());
    for (const std::pair<int, float> &w : point.weights) {
      chunk.corner_weight_src.append(face_corners[w.first]);
      chunk.corner_weights.append(w.second);
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Attribute Interpolation
 * \{ */

/** Mix source values into `dst[i]` using the weights in `groups[i]`. */
static void interpolate_attribute(const GSpan src,
                                  const OffsetIndices<int> groups,
                                  const Span<int> indices,
                                  const Span<float> weights,
                                  GMutableSpan dst)
{
  bke::attribute_math::to_static_type(src.type(), [&]<typename T>() {
    const Span<T> src_typed = src.typed<T>();
    MutableSpan<T> dst_typed = dst.typed<T>();
    if constexpr (!std::is_void_v<bke::attribute_math::DefaultMixer<T>>) {
      bke::attribute_math::DefaultMixer<T> mixer(dst_typed);
      threading::parallel_for(dst_typed.index_range(), 1024, [&](const IndexRange range) {
        for (const int i : range) {
          for (const int j : groups[i]) {
            mixer.mix_in(i, src_typed[indices[j]], weights[j]);
          }
        }
      });
      mixer.finalize();
    }
    else {
      /* Types that can't be mixed: take the value with the largest weight. */
      threading::parallel_for(dst_typed.index_range(), 1024, [&](const IndexRange range) {
        for (const int i : range) {
          const IndexRange group = groups[i];
          if (group.is_empty()) {
            dst_typed[i] = T();
            continue;
          }
          int best = group.first();
          for (const int j : group) {
            if (weights[j] > weights[best]) {
              best = j;
            }
          }
          dst_typed[i] = src_typed[indices[best]];
        }
      });
    }
  });
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Main Algorithm
 * \{ */

struct BrickerSettings {
  float3 size;
  float3 offset;
  float3 angle;
  bool fix_shared_edges;
};

static Mesh *brick_mesh(const Mesh &src_mesh,
                        const IndexMask &selection,
                        const BrickerSettings &settings,
                        const bke::AttributeFilter &attribute_filter,
                        bool &r_too_many_planes)
{
  const Span<float3> positions = src_mesh.vert_positions();
  const Span<int2> edges = src_mesh.edges();
  const OffsetIndices<int> faces = src_mesh.faces();
  const Span<int> corner_verts = src_mesh.corner_verts();
  const Span<int> corner_edges = src_mesh.corner_edges();
  const int verts_num = src_mesh.verts_num;

  const std::optional<Bounds<float3>> bounds = bounds::min_max(positions);
  if (!bounds) {
    return nullptr;
  }
  const std::optional<Lattice> lattice_opt = build_lattice(
      *bounds, settings.size, settings.offset, settings.angle, r_too_many_planes);
  if (!lattice_opt) {
    return nullptr;
  }
  const Lattice &lattice = *lattice_opt;

  Array<bool> face_selected(faces.size(), false);
  selection.to_bools(face_selected);

  Array<bool> edge_mask(edges.size(), false);
  selection.foreach_index([&](const int face) {
    for (const int edge : corner_edges.slice(faces[face])) {
      edge_mask[edge] = true;
    }
  });

  const EdgePoints edge_points = compute_edge_points(lattice, positions, edges, edge_mask);
  const OffsetIndices<int> edge_point_offsets = edge_points.offsets();

  /* Process faces in fixed chunks so the output order is deterministic. */
  constexpr int chunk_size = 64;
  const int chunks_num = (faces.size() + chunk_size - 1) / chunk_size;
  Array<ChunkResult> chunks(chunks_num);

  threading::parallel_for(IndexRange(chunks_num), 1, [&](const IndexRange chunk_range) {
    FaceSplitter splitter(lattice, positions, edges, edge_points);
    Vector<int> ring;
    Vector<int> scratch;
    Vector<Vector<int>> pieces;
    Vector<Vector<int>> next_pieces;
    for (const int chunk_i : chunk_range) {
      ChunkResult &chunk = chunks[chunk_i];
      const IndexRange chunk_faces = IndexRange(chunk_i * chunk_size, chunk_size)
                                         .intersect(faces.index_range());
      for (const int face_i : chunk_faces) {
        const IndexRange face = faces[face_i];
        const bool selected = face_selected[face_i];
        const bool insert_edge_points = selected || settings.fix_shared_edges;
        splitter.reset();
        ring.clear();
        const int size = face.size();
        for (const int i : IndexRange(size)) {
          const int corner = face[i];
          const int vert = corner_verts[corner];
          LocalPoint point;
          point.position = double3(positions[vert]);
          point.vert = vert;
          point.weights.append({i, 1.0f});
          ring.append(splitter.points.append_and_get_index(std::move(point)));
          if (!insert_edge_points) {
            continue;
          }
          const int edge = corner_edges[corner];
          const IndexRange edge_range = edge_point_offsets[edge];
          if (edge_range.is_empty()) {
            continue;
          }
          const bool forward = edges[edge][0] == vert;
          const int next_local = (i + 1) % size;
          for (const int j : edge_range.index_range()) {
            const int ep = forward ? edge_range[j] : edge_range[edge_range.size() - 1 - j];
            const float t = forward ? edge_points.t[ep] : 1.0f - edge_points.t[ep];
            LocalPoint ep_point;
            ep_point.position = double3(edge_points.position[ep]);
            ep_point.edge_point = ep;
            ep_point.plane_a = edge_points.plane[ep];
            ep_point.weights.append({i, 1.0f - t});
            ep_point.weights.append({next_local, t});
            ring.append(splitter.points.append_and_get_index(std::move(ep_point)));
          }
        }

        if (!selected) {
          emit_piece(splitter, ring, face_i, face, verts_num, chunk, scratch);
          continue;
        }

        splitter.face_normal = newell_normal(splitter.points, ring);
        if (math::length_squared(splitter.face_normal) == 0.0) {
          emit_piece(splitter, ring, face_i, face, verts_num, chunk, scratch);
          continue;
        }

        pieces.clear();
        pieces.append(ring);
        for (const int axis : IndexRange(3)) {
          if (lattice.count[axis] == 0) {
            continue;
          }
          next_pieces.clear();
          for (Vector<int> &piece : pieces) {
            splitter.split_axis(std::move(piece), axis, 0, lattice.count[axis] - 1, next_pieces);
          }
          std::swap(pieces, next_pieces);
        }
        for (const Vector<int> &piece : pieces) {
          emit_piece(splitter, piece, face_i, face, verts_num, chunk, scratch);
        }
      }
    }
  });

  /* Accumulate chunk sizes. */
  Array<int> face_start(chunks_num + 1);
  Array<int> corner_start(chunks_num + 1);
  Array<int> corner_weight_start(chunks_num + 1);
  Array<int> interior_start(chunks_num + 1);
  Array<int> interior_weight_start(chunks_num + 1);
  face_start[0] = corner_start[0] = corner_weight_start[0] = interior_start[0] =
      interior_weight_start[0] = 0;
  for (const int i : IndexRange(chunks_num)) {
    face_start[i + 1] = face_start[i] + chunks[i].face_src.size();
    corner_start[i + 1] = corner_start[i] + chunks[i].corner_refs.size();
    corner_weight_start[i + 1] = corner_weight_start[i] + chunks[i].corner_weights.size();
    interior_start[i + 1] = interior_start[i] + chunks[i].interior_positions.size();
    interior_weight_start[i + 1] = interior_weight_start[i] + chunks[i].interior_weights.size();
  }
  const int dst_faces_num = face_start.last();
  const int dst_corners_num = corner_start.last();
  const int edge_points_num = edge_points.size();
  const int interior_num = interior_start.last();
  const int new_points_num = edge_points_num + interior_num;
  const int dst_verts_num = verts_num + new_points_num;

  Mesh *dst_mesh = BKE_mesh_new_nomain(dst_verts_num, 0, dst_faces_num, dst_corners_num);
  BKE_mesh_copy_parameters_for_eval(dst_mesh, &src_mesh);

  MutableSpan<float3> dst_positions = dst_mesh->vert_positions_for_write();
  MutableSpan<int> dst_face_offsets = dst_mesh->face_offsets_for_write();
  MutableSpan<int> dst_corner_verts = dst_mesh->corner_verts_for_write();

  Array<int> dst_face_src(dst_faces_num);
  /* Interpolation tables. New points: edge points first, then interior points. */
  Array<int> new_point_weight_offsets(new_points_num + 1);
  Array<int> corner_weight_offsets(dst_corners_num + 1);
  const int edge_point_weights_num = edge_points_num * 2;
  Array<int> new_point_weight_src(edge_point_weights_num + interior_weight_start.last());
  Array<float> new_point_weights(new_point_weight_src.size());
  Array<int> corner_weight_src(corner_weight_start.last());
  Array<float> corner_weights(corner_weight_start.last());

  dst_positions.take_front(verts_num).copy_from(positions);
  dst_positions.slice(verts_num, edge_points_num).copy_from(edge_points.position);
  threading::parallel_for(IndexRange(edge_points_num), 4096, [&](const IndexRange range) {
    for (const int ep : range) {
      new_point_weight_offsets[ep] = ep * 2;
      new_point_weight_src[ep * 2] = edges[edge_points.edge[ep]][0];
      new_point_weight_src[ep * 2 + 1] = edges[edge_points.edge[ep]][1];
      new_point_weights[ep * 2] = 1.0f - edge_points.t[ep];
      new_point_weights[ep * 2 + 1] = edge_points.t[ep];
    }
  });

  threading::parallel_for(IndexRange(chunks_num), 1, [&](const IndexRange chunk_range) {
    for (const int chunk_i : chunk_range) {
      const ChunkResult &chunk = chunks[chunk_i];
      const int interior_base = verts_num + edge_points_num + interior_start[chunk_i];
      /* Faces. */
      int corner = corner_start[chunk_i];
      for (const int i : chunk.face_src.index_range()) {
        dst_face_offsets[face_start[chunk_i] + i] = corner;
        dst_face_src[face_start[chunk_i] + i] = chunk.face_src[i];
        corner += chunk.face_sizes[i];
      }
      /* Corners. */
      int weight = corner_weight_start[chunk_i];
      for (const int i : chunk.corner_refs.index_range()) {
        const int ref = chunk.corner_refs[i];
        const int dst_corner = corner_start[chunk_i] + i;
        dst_corner_verts[dst_corner] = ref >= 0 ? ref : interior_base + (-ref - 1);
        corner_weight_offsets[dst_corner] = weight;
        weight += chunk.corner_weight_counts[i];
      }
      corner_weight_src.as_mutable_span()
          .slice(corner_weight_start[chunk_i], chunk.corner_weight_src.size())
          .copy_from(chunk.corner_weight_src);
      corner_weights.as_mutable_span()
          .slice(corner_weight_start[chunk_i], chunk.corner_weights.size())
          .copy_from(chunk.corner_weights);
      /* Interior points. */
      dst_positions.slice(interior_base, chunk.interior_positions.size())
          .copy_from(chunk.interior_positions);
      int iw = edge_point_weights_num + interior_weight_start[chunk_i];
      for (const int i : chunk.interior_positions.index_range()) {
        const int new_point = edge_points_num + interior_start[chunk_i] + i;
        new_point_weight_offsets[new_point] = iw;
        iw += chunk.interior_weight_counts[i];
      }
      const int iw_start = edge_point_weights_num + interior_weight_start[chunk_i];
      for (const int i : chunk.interior_weight_src.index_range()) {
        /* Point attributes interpolate over the vertices of the source corners. */
        new_point_weight_src[iw_start + i] = corner_verts[chunk.interior_weight_src[i]];
        new_point_weights[iw_start + i] = chunk.interior_weights[i];
      }
    }
  });
  dst_face_offsets.last() = dst_corners_num;
  new_point_weight_offsets.last() = new_point_weight_src.size();
  corner_weight_offsets.last() = corner_weight_src.size();

  /* Attributes. */
  const bke::AttributeAccessor src_attributes = src_mesh.attributes();
  bke::MutableAttributeAccessor dst_attributes = dst_mesh->attributes_for_write();

  for (bke::AttributeTransferData &attribute : bke::retrieve_attributes_for_transfer(
           src_attributes,
           dst_attributes,
           {bke::AttrDomain::Point},
           bke::attribute_filter_with_skip_ref(attribute_filter, {"position"})))
  {
    const GSpan src = attribute.src;
    attribute.dst.span.take_front(verts_num).copy_from(src);
    interpolate_attribute(src,
                          new_point_weight_offsets.as_span(),
                          new_point_weight_src,
                          new_point_weights,
                          attribute.dst.span.drop_front(verts_num));
    attribute.dst.finish();
  }

  for (bke::AttributeTransferData &attribute : bke::retrieve_attributes_for_transfer(
           src_attributes,
           dst_attributes,
           {bke::AttrDomain::Corner},
           bke::attribute_filter_with_skip_ref(attribute_filter, {".corner_vert", ".corner_edge"})))
  {
    interpolate_attribute(attribute.src,
                          corner_weight_offsets.as_span(),
                          corner_weight_src,
                          corner_weights,
                          attribute.dst.span);
    attribute.dst.finish();
  }

  bke::gather_attributes(src_attributes,
                         bke::AttrDomain::Face,
                         bke::AttrDomain::Face,
                         attribute_filter,
                         dst_face_src.as_span(),
                         dst_attributes);

  bke::mesh_calc_edges(*dst_mesh, false, false, attribute_filter);

  /* Edge attributes: sub-edges of original edges keep their values. */
  Vector<bke::AttributeTransferData> edge_attributes = bke::retrieve_attributes_for_transfer(
      src_attributes,
      dst_attributes,
      {bke::AttrDomain::Edge},
      bke::attribute_filter_with_skip_ref(attribute_filter, {".edge_verts"}));
  if (!edge_attributes.is_empty()) {
    Map<OrderedEdge, int> src_edge_map;
    src_edge_map.reserve(edges.size());
    for (const int e : edges.index_range()) {
      src_edge_map.add(OrderedEdge(edges[e][0], edges[e][1]), e);
    }
    const Span<int2> dst_edges = dst_mesh->edges();
    Array<int> dst_edge_src(dst_edges.size());
    threading::parallel_for(dst_edges.index_range(), 2048, [&](const IndexRange range) {
      for (const int i : range) {
        const int a = dst_edges[i][0];
        const int b = dst_edges[i][1];
        auto edge_of = [&](const int v) {
          return (v >= verts_num && v < verts_num + edge_points_num) ?
                     edge_points.edge[v - verts_num] :
                     -1;
        };
        const bool a_orig = a < verts_num;
        const bool b_orig = b < verts_num;
        int src = -1;
        if (a_orig && b_orig) {
          src = src_edge_map.lookup_default(OrderedEdge(a, b), -1);
        }
        else if (a_orig || b_orig) {
          const int v = a_orig ? a : b;
          const int e = edge_of(a_orig ? b : a);
          if (e >= 0 && (edges[e][0] == v || edges[e][1] == v)) {
            src = e;
          }
        }
        else {
          const int ea = edge_of(a);
          if (ea >= 0 && ea == edge_of(b)) {
            src = ea;
          }
        }
        dst_edge_src[i] = src;
      }
    });
    for (bke::AttributeTransferData &attribute : edge_attributes) {
      bke::attribute_math::to_static_type(attribute.src.type(), [&]<typename T>() {
        const Span<T> src = attribute.src.typed<T>();
        MutableSpan<T> dst = attribute.dst.span.typed<T>();
        threading::parallel_for(dst.index_range(), 4096, [&](const IndexRange range) {
          for (const int i : range) {
            dst[i] = dst_edge_src[i] >= 0 ? src[dst_edge_src[i]] : T();
          }
        });
      });
      attribute.dst.finish();
    }
  }

  return dst_mesh;
}

/** \} */

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh"_ustr);
  const Field<bool> selection_field = params.extract_input<Field<bool>>("Selection"_ustr);
  BrickerSettings settings;
  settings.size = params.extract_input<float3>("Size"_ustr);
  settings.offset = params.extract_input<float3>("Offset"_ustr);
  settings.angle = params.extract_input<float3>("Angle"_ustr);
  settings.fix_shared_edges = params.extract_input<bool>("Fix Shared Edges"_ustr);
  const AttributeFilter &attribute_filter = params.get_attribute_filter("Mesh"_ustr);

  std::atomic<bool> too_many_planes = false;
  geometry::foreach_real_geometry(geometry_set, [&](GeometrySet &geometry_set) {
    const Mesh *src_mesh = geometry_set.get_mesh();
    if (!src_mesh || src_mesh->faces_num == 0) {
      return;
    }
    const bke::MeshFieldContext context(*src_mesh, AttrDomain::Face);
    FieldEvaluator evaluator{context, src_mesh->faces_num};
    evaluator.add(selection_field);
    evaluator.evaluate();
    const IndexMask selection = evaluator.get_evaluated_as_mask(0);
    if (selection.is_empty()) {
      return;
    }
    bool too_many = false;
    Mesh *mesh = brick_mesh(*src_mesh, selection, settings, attribute_filter, too_many);
    if (too_many) {
      too_many_planes = true;
    }
    if (mesh) {
      geometry_set.replace_mesh(mesh);
    }
  });

  if (too_many_planes) {
    params.error_message_add(NodeWarningType::Error,
                             TIP_("Size is too small for the mesh bounds, too many cuts"));
  }
  params.set_output("Mesh"_ustr, std::move(geometry_set));
}

/* Registered as a Toast plugin (see `NOD_toast_plugins.hh`), not compiled into Blender. */

static bke::bNodeType ntype;
static bool registered = false;

static bool register_nodes()
{
  if (!toast_plugins::geo_node_type_base(ntype, "GeometryNodeBricker")) {
    return false;
  }
  ntype.ui_name = "Bricker";
  ntype.ui_description =
      "Cut faces into bricks with a lattice of planes";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  toast_plugins::register_node_type(ntype);
  registered = true;
  return true;
}

static void unregister_nodes()
{
  if (registered) {
    toast_plugins::unregister_node_type(ntype);
    registered = false;
  }
}

}  // namespace blender::nodes::node_geo_bricker_cc

TOAST_PLUGIN_EXPORT const ToastPluginInfo *toast_plugin_info()
{
  namespace ns = blender::nodes::node_geo_bricker_cc;
  static const ToastPluginInfo info = {
      TOAST_PLUGIN_API_VERSION, "Bricker", ns::register_nodes, ns::unregister_nodes};
  return &info;
}
