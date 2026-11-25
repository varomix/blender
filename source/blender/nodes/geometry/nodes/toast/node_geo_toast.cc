/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Toast node: runs a native kernel that the Toast add-on compiled from Python-syntax code.
 *
 * The add-on owns the language. It compiles the code in the node's Text data-block and writes
 * the result into the node storage: the kernel library path and its signature (JSON). This node
 * never calls Python. During evaluation it:
 * 1. loads the kernel with dlopen (cached per path for the session),
 * 2. builds the param() block from the sockets declared from the signature,
 * 3. binds attribute arrays in signature order,
 * 4. runs the kernel over fixed-size chunks of points with `parallel_for`,
 * 5. merges the per-chunk command buffers, sorted by source point, and applies them.
 *
 * Fixed chunk sizes and sorted merging make results independent of the thread count. The
 * contract matches `harness.py` in the add-on, which is the reference implementation.
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#ifndef _WIN32
#  include <dlfcn.h>
#endif

#include "BLI_array.hh"
#include "BLI_array_utils.hh"
#include "BLI_generic_array.hh"
#include "BLI_kdtree_new.hh"
#include "BLI_map.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_serialize.hh"
#include "BLI_string.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "DNA_mesh_types.h"
#include "DNA_pointcloud_types.h"
#include "DNA_scene_types.h"
#include "DNA_text_types.h"

#include "BKE_attribute.hh"
#include "BKE_bvh.hh"
#include "BKE_geometry_fields.hh"
#include "BKE_global.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLO_read_write.hh"

#include "BLT_translation.hh"

#include "DEG_depsgraph_query.hh"

#include "GEO_foreach_geometry.hh"
#include "GEO_mesh_copy_selection.hh"

#include "NOD_rna_define.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"
#include "RNA_enum_types.hh"
#include "RNA_prototypes.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "node_geometry_util.hh"
#include "toast_abi.h"

namespace blender::nodes::node_geo_toast_cc {

NODE_STORAGE_FUNCS(NodeGeometryToast)

using bke::AttrDomain;
using bke::AttrType;

enum class RunOver : int8_t {
  Points = 0,
  Faces = 1,
  Edges = 2,
  Corners = 3,
  /* Once for the whole geometry (stored value 4, files saved with "Detail" keep working). */
  Geometry = 4,
};

static const EnumPropertyItem run_over_items[] = {
    {int(RunOver::Points), "POINTS", ICON_NONE, "Points", "Run the code once for every point"},
    {int(RunOver::Faces), "FACES", ICON_NONE, "Faces", "Run the code once for every face"},
    {int(RunOver::Edges), "EDGES", ICON_NONE, "Edges", "Run the code once for every edge"},
    {int(RunOver::Corners),
     "CORNERS",
     ICON_NONE,
     "Face Corners",
     "Run the code once for every face corner"},
    {int(RunOver::Geometry),
     "GEOMETRY",
     ICON_NONE,
     "Geometry",
     "Run the code once for the whole geometry, for setup or to build geometry from scratch"},
    {0, nullptr, 0, nullptr, nullptr},
};

/** Domain of `@` attributes for a run-over mode, from the kernel signature. */
static std::optional<AttrDomain> run_domain(const StringRef run_over)
{
  if (run_over == "points") {
    return AttrDomain::Point;
  }
  if (run_over == "faces") {
    return AttrDomain::Face;
  }
  if (run_over == "edges") {
    return AttrDomain::Edge;
  }
  if (run_over == "corners") {
    return AttrDomain::Corner;
  }
  return std::nullopt;
}

static int32_t abi_domain(const AttrDomain domain)
{
  switch (domain) {
    case AttrDomain::Face:
      return TOAST_DOMAIN_FACE;
    case AttrDomain::Edge:
      return TOAST_DOMAIN_EDGE;
    case AttrDomain::Corner:
      return TOAST_DOMAIN_CORNER;
    default:
      return TOAST_DOMAIN_POINT;
  }
}

static std::optional<AttrDomain> domain_from_string(const StringRef domain)
{
  if (domain == "point") {
    return AttrDomain::Point;
  }
  if (domain == "face") {
    return AttrDomain::Face;
  }
  if (domain == "edge") {
    return AttrDomain::Edge;
  }
  if (domain == "corner") {
    return AttrDomain::Corner;
  }
  return std::nullopt;
}

/** Chunk size. Fixed, so chunk boundaries and command order don't depend on thread count. */
static constexpr int64_t GRAIN_SIZE = 1024;
static constexpr int MAX_INPUTS = 2;

/* -------------------------------------------------------------------- */
/** \name Signature
 * \{ */

struct SigAttr {
  std::string name;
  int type_code = TOAST_TYPE_FLOAT;
  int input = 0;
  bool read = false;
  bool write = false;
  bool create = false;
  bool snapshot = false;
  AttrDomain domain = AttrDomain::Point;
};

struct SigParam {
  std::string name;
  std::string type;
  int offset = 0;
  Vector<double, 4> default_value;
  std::optional<double> min;
  std::optional<double> max;
};

struct Signature {
  int abi = 0;
  std::string run_over;
  Vector<SigAttr> attrs;
  Vector<SigParam> params;
  int params_size = 0;
  bool uses_cmd = false;
  bool uses_time = false;
  bool uses_neighbors = false;
  Vector<int> nearpoints_inputs;
  Vector<int> surface_inputs;
};

static std::optional<double> as_number(const io::serialize::Value *value)
{
  if (value == nullptr) {
    return std::nullopt;
  }
  if (const io::serialize::DoubleValue *v = value->as_double_value()) {
    return v->value();
  }
  if (const io::serialize::IntValue *v = value->as_int_value()) {
    return double(v->value());
  }
  if (const io::serialize::BooleanValue *v = value->as_boolean_value()) {
    return v->value() ? 1.0 : 0.0;
  }
  return std::nullopt;
}

static const io::serialize::Value *dict_value(const io::serialize::DictionaryValue &dict,
                                              const StringRef key)
{
  const std::shared_ptr<io::serialize::Value> *value = dict.lookup(key);
  return value ? value->get() : nullptr;
}

static std::optional<Signature> parse_signature(const StringRefNull json_str)
{
  if (json_str.is_empty()) {
    return std::nullopt;
  }
  io::serialize::JsonFormatter json;
  std::istringstream stream{std::string(json_str)};
  std::unique_ptr<io::serialize::Value> root;
  try {
    root = json.deserialize(stream);
  }
  catch (...) {
    return std::nullopt;
  }
  const io::serialize::DictionaryValue *dict = root ? root->as_dictionary_value() : nullptr;
  if (dict == nullptr) {
    return std::nullopt;
  }

  Signature sig;
  sig.abi = int(dict->lookup_int("abi").value_or(0));
  sig.run_over = std::string(dict->lookup_str("run_over").value_or(""));
  sig.params_size = int(dict->lookup_int("params_size").value_or(0));

  if (const io::serialize::ArrayValue *attrs = dict->lookup_array("attrs")) {
    for (const std::shared_ptr<io::serialize::Value> &item : attrs->elements()) {
      const io::serialize::DictionaryValue *a = item->as_dictionary_value();
      if (a == nullptr) {
        return std::nullopt;
      }
      SigAttr attr;
      attr.name = std::string(a->lookup_str("name").value_or(""));
      attr.type_code = int(a->lookup_int("type_code").value_or(-1));
      attr.input = int(a->lookup_int("input").value_or(0));
      const StringRef access = a->lookup_str("access").value_or("r");
      attr.read = access.find('r') != StringRef::not_found;
      attr.write = access.find('w') != StringRef::not_found;
      attr.create = a->lookup_bool("create").value_or(false);
      attr.snapshot = a->lookup_bool("snapshot").value_or(false);
      const std::optional<AttrDomain> domain = domain_from_string(
          a->lookup_str("domain").value_or("point"));
      if (!domain) {
        return std::nullopt;
      }
      attr.domain = *domain;
      sig.attrs.append(std::move(attr));
    }
  }

  if (const io::serialize::ArrayValue *params = dict->lookup_array("params")) {
    for (const std::shared_ptr<io::serialize::Value> &item : params->elements()) {
      const io::serialize::DictionaryValue *p = item->as_dictionary_value();
      if (p == nullptr) {
        return std::nullopt;
      }
      SigParam param;
      param.name = std::string(p->lookup_str("name").value_or(""));
      param.type = std::string(p->lookup_str("type").value_or(""));
      param.offset = int(p->lookup_int("offset").value_or(0));
      param.min = as_number(dict_value(*p, "min"));
      param.max = as_number(dict_value(*p, "max"));
      const io::serialize::Value *def = dict_value(*p, "default");
      if (const io::serialize::ArrayValue *arr = def ? def->as_array_value() : nullptr) {
        for (const std::shared_ptr<io::serialize::Value> &c : arr->elements()) {
          param.default_value.append(as_number(c.get()).value_or(0.0));
        }
      }
      else {
        param.default_value.append(as_number(def).value_or(0.0));
      }
      sig.params.append(std::move(param));
    }
  }

  if (const io::serialize::DictionaryValue *uses = dict->lookup_dict("uses")) {
    sig.uses_cmd = uses->lookup_bool("cmd").value_or(false);
    sig.uses_time = uses->lookup_bool("time").value_or(false);
    if (const io::serialize::ArrayValue *host = uses->lookup_array("host")) {
      for (const std::shared_ptr<io::serialize::Value> &c : host->elements()) {
        if (const io::serialize::StringValue *v = c->as_string_value()) {
          sig.uses_neighbors |= v->value() == "point_neighbors";
        }
      }
    }
    if (const io::serialize::ArrayValue *inputs = uses->lookup_array("surface_inputs")) {
      for (const std::shared_ptr<io::serialize::Value> &c : inputs->elements()) {
        sig.surface_inputs.append(int(as_number(c.get()).value_or(0.0)));
      }
    }
    if (const io::serialize::ArrayValue *inputs = uses->lookup_array("nearpoints_inputs")) {
      for (const std::shared_ptr<io::serialize::Value> &c : inputs->elements()) {
        sig.nearpoints_inputs.append(int(as_number(c.get()).value_or(0.0)));
      }
    }
  }
  return sig;
}

static std::string param_identifier(const StringRef name)
{
  return "p_" + std::string(name);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Kernel Loading
 * \{ */

struct LoadedKernel {
  void *handle = nullptr;
  ToastKernelFn fn = nullptr;
  Signature signature;
};

/**
 * Kernels stay loaded for the whole session: another evaluation may still be running one, and
 * a new compile always gets a new path (the cache key), so nothing is ever replaced in place.
 */
static const LoadedKernel *load_kernel(const StringRefNull path, std::string &r_error)
{
#ifdef _WIN32
  UNUSED_VARS(path);
  r_error = "Toast kernels are not supported on Windows yet";
  return nullptr;
#else
  static std::mutex mutex;
  static Map<std::string, std::unique_ptr<LoadedKernel>> kernels;
  static Map<std::string, std::string> failures;

  std::lock_guard lock{mutex};
  if (const std::unique_ptr<LoadedKernel> *kernel = kernels.lookup_ptr(path)) {
    return kernel->get();
  }
  if (const std::string *error = failures.lookup_ptr(path)) {
    r_error = *error;
    return nullptr;
  }

  auto fail = [&](std::string message) -> const LoadedKernel * {
    failures.add(path, message);
    r_error = std::move(message);
    return nullptr;
  };

  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char *err = dlerror();
    return fail(std::string("Can't load kernel: ") + (err ? err : path.c_str()));
  }
  auto abi_fn = reinterpret_cast<ToastAbiVersionFn>(dlsym(handle, "toast_abi_version"));
  auto sig_fn = reinterpret_cast<ToastSignatureFn>(dlsym(handle, "toast_signature"));
  auto kernel_fn = reinterpret_cast<ToastKernelFn>(dlsym(handle, "toast_kernel"));
  if (!abi_fn || !sig_fn || !kernel_fn) {
    dlclose(handle);
    return fail("Not a Toast kernel: " + std::string(path));
  }
  /* Read before any dlclose: the functions are gone once the library is closed. */
  const int32_t abi = abi_fn();
  if (abi != TOAST_ABI_VERSION) {
    dlclose(handle);
    return fail("Kernel built for Toast ABI " + std::to_string(abi) + ", this Blender uses " +
                std::to_string(TOAST_ABI_VERSION) + ". It is recompiled automatically when the "
                "Toast add-on is enabled");
  }
  std::optional<Signature> sig = parse_signature(sig_fn());
  if (!sig) {
    dlclose(handle);
    return fail("Kernel has an invalid signature: " + std::string(path));
  }

  auto kernel = std::make_unique<LoadedKernel>();
  kernel->handle = handle;
  kernel->fn = kernel_fn;
  kernel->signature = std::move(*sig);
  const LoadedKernel *result = kernel.get();
  kernels.add(path, std::move(kernel));
  return result;
#endif
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Host Services
 * \{ */

/** Keeps the best `max` points by (distance, index), within the radius. Ties go to lower index,
 * as in the harness, so results don't depend on the tree layout. */
class NearPointsQuery {
 public:
  static constexpr bool inclusive = true;

 private:
  float radius_sq_;
  int max_;
  int count_ = 0;
  int64_t *indices_;
  float *dists_;

 public:
  NearPointsQuery(const float radius_sq, const int max, int64_t *indices, float *dists)
      : radius_sq_(radius_sq), max_(max), indices_(indices), dists_(dists)
  {
  }

  int count() const
  {
    return count_;
  }

  bool add_point(const float distance_sq, const int index)
  {
    if (distance_sq > radius_sq_) {
      return true;
    }
    if (count_ == max_) {
      const float worst = dists_[count_ - 1];
      if (distance_sq > worst || (distance_sq == worst && index > indices_[count_ - 1])) {
        return true;
      }
      count_--;
    }
    int k = count_;
    while (k > 0 &&
           (dists_[k - 1] > distance_sq || (dists_[k - 1] == distance_sq && indices_[k - 1] > index)))
    {
      dists_[k] = dists_[k - 1];
      indices_[k] = indices_[k - 1];
      k--;
    }
    dists_[k] = distance_sq;
    indices_[k] = index;
    count_++;
    return true;
  }

  float max_distance_sq() const
  {
    return count_ == max_ ? dists_[count_ - 1] : radius_sq_;
  }
};

enum class CmdKind : int8_t { AddPoint, AddPrim, RemovePoint, RemovePrim, SetAttr };

struct Cmd {
  CmdKind kind;
  int64_t src;
  /* AddPoint: position. */
  float3 position;
  /* RemovePoint: point. RemovePrim: face. SetAttr: element. AddPrim: range in
   * #CmdBuffer::prim_points. */
  int64_t point = 0;
  IndexRange prim_points;
  /* SetAttr: binding index and the value's bytes in #CmdBuffer::values. */
  int32_t attr = 0;
  IndexRange value;
};

/** Behind the opaque `ToastCmdBuffer *` handed to the kernel. One per chunk. */
struct CmdBuffer {
  int64_t chunk_id = 0;
  int64_t added = 0;
  Vector<Cmd> cmds;
  Vector<int64_t> prim_points;
  Vector<uint8_t> values;
};

struct HostData {
  std::array<const KDTreeNew<float3> *, MAX_INPUTS> trees = {};
  std::array<int64_t, MAX_INPUTS> npoints = {};
  /* For face queries; null for inputs without a mesh. */
  std::array<const Mesh *, MAX_INPUTS> meshes = {};
  /* Points sharing an edge with each point, ascending (built when the code uses them). */
  std::array<Array<int>, MAX_INPUTS> neighbor_offsets;
  std::array<Array<int>, MAX_INPUTS> neighbors;
  /* Bytes per element of each bound attribute, for cmd_set_attr. */
  Span<ToastAttr> bindings;
  /* Meshes for ray_cast/closest_point, as they were before the run. */
  std::array<const Mesh *, MAX_INPUTS> surface_meshes = {};
};

static int32_t fill_hit(const Mesh &mesh,
                        const int tri_index,
                        const float3 &position,
                        const float3 &bary,
                        const float distance,
                        ToastHit *out)
{
  const int3 tri = mesh.corner_tris()[tri_index];
  const Span<int> corner_verts = mesh.corner_verts();
  const Span<float3> positions = mesh.vert_positions();
  const int3 points(corner_verts[tri[0]], corner_verts[tri[1]], corner_verts[tri[2]]);
  const float3 normal = math::normalize(math::cross(positions[points[1]] - positions[points[0]],
                                                    positions[points[2]] - positions[points[0]]));
  for (const int k : IndexRange(3)) {
    out->position[k] = position[k];
    out->normal[k] = normal[k];
    out->bary[k] = bary[k];
  }
  out->distance = distance;
  out->face = mesh.corner_tri_faces()[tri_index];
  out->tri_points[0] = points[0];
  out->tri_points[1] = points[1];
  out->tri_points[2] = points[2];
  out->hit = 1;
  return 1;
}

static int32_t host_ray_cast(void *self,
                             const int32_t input,
                             const float *origin,
                             const float *dir,
                             const float max_dist,
                             ToastHit *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.surface_meshes[input] == nullptr) {
    return 0;
  }
  const Mesh &mesh = *host.surface_meshes[input];
  float3 direction(dir[0], dir[1], dir[2]);
  const float length = math::length(direction);
  if (mesh.faces_num == 0 || !(length > 0.0f)) {
    return 0;
  }
  direction /= length;
  const bke::bvh::Ray ray(float3(origin[0], origin[1], origin[2]), direction, max_dist);
  const std::optional<bke::bvh::RayHit> hit = mesh.bvh_tris().ray_intersect(ray);
  if (!hit) {
    return 0;
  }
  return fill_hit(mesh, hit->index, hit->position(ray), hit->bary_coord, hit->distance, out);
}

static int32_t host_closest_point(
    void *self, const int32_t input, const float *pos, const float max_dist, ToastHit *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.surface_meshes[input] == nullptr) {
    return 0;
  }
  const Mesh &mesh = *host.surface_meshes[input];
  if (mesh.faces_num == 0) {
    return 0;
  }
  const float3 position(pos[0], pos[1], pos[2]);
  const std::optional<bke::bvh::ClosestPointResult> result = mesh.bvh_tris().closest_point(
      position, max_dist);
  if (!result) {
    return 0;
  }
  return fill_hit(mesh,
                  int(result->index),
                  result->position,
                  result->bary_coord,
                  math::distance(position, result->position),
                  out);
}

static int64_t type_size(const int32_t type_code)
{
  switch (type_code) {
    case TOAST_TYPE_FLOAT:
    case TOAST_TYPE_INT:
      return 4;
    case TOAST_TYPE_BOOL:
      return 1;
    case TOAST_TYPE_FLOAT2:
      return 8;
    case TOAST_TYPE_FLOAT3:
      return 12;
    case TOAST_TYPE_FLOAT4:
      return 16;
  }
  return 0;
}

static void build_neighbors(const Mesh &mesh, Array<int> &r_offsets, Array<int> &r_neighbors)
{
  const Span<int2> edges = mesh.edges();
  r_offsets = Array<int>(mesh.verts_num + 1, 0);
  for (const int2 e : edges) {
    r_offsets[e[0]]++;
    r_offsets[e[1]]++;
  }
  offset_indices::accumulate_counts_to_offsets(r_offsets);
  r_neighbors = Array<int>(r_offsets.last());
  Array<int> fill(mesh.verts_num, 0);
  for (const int2 e : edges) {
    r_neighbors[r_offsets[e[0]] + fill[e[0]]++] = e[1];
    r_neighbors[r_offsets[e[1]] + fill[e[1]]++] = e[0];
  }
  threading::parallel_for(IndexRange(mesh.verts_num), 4096, [&](const IndexRange range) {
    for (const int v : range) {
      std::sort(&r_neighbors[r_offsets[v]], &r_neighbors[r_offsets[v + 1]]);
    }
  });
}

static int32_t write_indices(const Span<int> values, const int32_t max, int64_t *out)
{
  const int64_t n = std::min<int64_t>(values.size(), std::max(max, 0));
  for (const int64_t k : IndexRange(n)) {
    out[k] = values[k];
  }
  return int32_t(values.size());
}

static int32_t host_point_faces(
    void *self, const int32_t input, const int64_t point, const int32_t max, int64_t *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.meshes[input] == nullptr) {
    return 0;
  }
  const Mesh &mesh = *host.meshes[input];
  if (point < 0 || point >= mesh.verts_num) {
    return 0;
  }
  return write_indices(mesh.vert_to_face_map()[point], max, out);
}

static int32_t host_point_neighbors(
    void *self, const int32_t input, const int64_t point, const int32_t max, int64_t *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.neighbor_offsets[input].is_empty()) {
    return 0;
  }
  const Span<int> offsets = host.neighbor_offsets[input];
  if (point < 0 || point >= offsets.size() - 1) {
    return 0;
  }
  return write_indices(
      host.neighbors[input].as_span().slice(offsets[point], offsets[point + 1] - offsets[point]),
      max,
      out);
}

static int64_t host_corner_point(void *self, const int32_t input, const int64_t corner)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.meshes[input] == nullptr) {
    return -1;
  }
  const Mesh &mesh = *host.meshes[input];
  return (corner >= 0 && corner < mesh.corners_num) ? mesh.corner_verts()[corner] : -1;
}

static int64_t host_corner_face(void *self, const int32_t input, const int64_t corner)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.meshes[input] == nullptr) {
    return -1;
  }
  const Mesh &mesh = *host.meshes[input];
  return (corner >= 0 && corner < mesh.corners_num) ? mesh.corner_to_face_map()[corner] : -1;
}

static void host_set_attr(void *self,
                          ToastCmdBuffer *cmd,
                          const int64_t src,
                          const int32_t attr,
                          const int64_t elem,
                          const void *value)
{
  const HostData &host = *static_cast<const HostData *>(self);
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  if (attr < 0 || attr >= host.bindings.size()) {
    return;
  }
  const int64_t size = type_size(host.bindings[attr].type);
  Cmd c{CmdKind::SetAttr, src, float3(0.0f)};
  c.point = elem;
  c.attr = attr;
  c.value = IndexRange(buffer.values.size(), size);
  buffer.values.extend(Span<uint8_t>(static_cast<const uint8_t *>(value), size));
  buffer.cmds.append(c);
}

/**
 * Apply set_point_attr()/set_face_attr() into the bound arrays, in (source index, emission order):
 * chunks cover increasing ranges, so that is chunk order. Done before topology changes.
 */
static bool apply_sets(Span<CmdBuffer> buffers, Span<ToastAttr> bindings, std::string &r_error)
{
  for (const CmdBuffer &buffer : buffers) {
    for (const Cmd &cmd : buffer.cmds) {
      if (cmd.kind != CmdKind::SetAttr) {
        continue;
      }
      const ToastAttr &attr = bindings[cmd.attr];
      if (cmd.point < 0 || cmd.point >= attr.size) {
        r_error = "Element " + std::to_string(cmd.point) + " does not exist (set_*_attr)";
        return false;
      }
      memcpy(static_cast<uint8_t *>(attr.data) + cmd.point * cmd.value.size(),
             &buffer.values[cmd.value.start()],
             cmd.value.size());
    }
  }
  return true;
}

static int32_t host_nearpoints(void *self,
                               const int32_t input,
                               const float *pos3,
                               const float radius,
                               const int32_t max,
                               int64_t *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.trees[input] == nullptr || max <= 0 ||
      !(radius >= 0.0f))
  {
    return 0;
  }
  Array<float, 64> dists(max);
  NearPointsQuery query(radius * radius, max, out, dists.data());
  host.trees[input]->search(query, float3(pos3[0], pos3[1], pos3[2]));
  return query.count();
}

static int64_t host_npoints(void *self, const int32_t input)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS) {
    return 0;
  }
  return host.npoints[input];
}

static int64_t host_nprims(void *self, const int32_t input)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.meshes[input] == nullptr) {
    return 0;
  }
  return host.meshes[input]->faces_num;
}

static int32_t host_primpoints(
    void *self, const int32_t input, const int64_t prim, const int32_t max, int64_t *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.meshes[input] == nullptr) {
    return 0;
  }
  const Mesh &mesh = *host.meshes[input];
  if (prim < 0 || prim >= mesh.faces_num) {
    return 0;
  }
  const IndexRange face = mesh.faces()[prim];
  const Span<int> corner_verts = mesh.corner_verts();
  const int64_t n = std::min<int64_t>(face.size(), std::max(max, 0));
  for (const int64_t k : IndexRange(n)) {
    out[k] = corner_verts[face[k]];
  }
  return int32_t(face.size());
}

static void host_remove_prim(void * /*self*/,
                             ToastCmdBuffer *cmd,
                             const int64_t src,
                             const int64_t prim)
{
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  Cmd c{CmdKind::RemovePrim, src, float3(0.0f)};
  c.point = prim;
  buffer.cmds.append(c);
}

static int64_t host_add_point(void * /*self*/,
                              ToastCmdBuffer *cmd,
                              const int64_t src,
                              const float *pos3)
{
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  /* Same encoding as the harness: negative, unique per (chunk, local index). */
  const int64_t handle = -((buffer.chunk_id << 32) | buffer.added) - 1;
  buffer.added++;
  Cmd c{CmdKind::AddPoint, src, float3(pos3[0], pos3[1], pos3[2])};
  buffer.cmds.append(c);
  return handle;
}

static void host_add_prim(
    void * /*self*/, ToastCmdBuffer *cmd, const int64_t src, const int64_t *pts, const int32_t n)
{
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  const IndexRange range(buffer.prim_points.size(), std::max(n, 0));
  buffer.prim_points.extend(Span<int64_t>(pts, range.size()));
  Cmd c{CmdKind::AddPrim, src, float3(0.0f)};
  c.prim_points = range;
  buffer.cmds.append(c);
}

static void host_remove_point(void * /*self*/,
                              ToastCmdBuffer *cmd,
                              const int64_t src,
                              const int64_t pt)
{
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  Cmd c{CmdKind::RemovePoint, src, float3(0.0f)};
  c.point = pt;
  buffer.cmds.append(c);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Attribute Binding
 * \{ */

static std::optional<AttrType> attr_type_from_code(const int code)
{
  switch (code) {
    case TOAST_TYPE_FLOAT:
      return AttrType::Float;
    case TOAST_TYPE_INT:
      return AttrType::Int32;
    case TOAST_TYPE_BOOL:
      return AttrType::Bool;
    case TOAST_TYPE_FLOAT2:
      return AttrType::Float2;
    case TOAST_TYPE_FLOAT3:
      return AttrType::Float3;
    case TOAST_TYPE_FLOAT4:
      return AttrType::Float4;
  }
  return std::nullopt;
}

/** Attributes with the same layout as a Toast type, which can be bound without conversion. */
static bool layout_compatible(const AttrType existing, const AttrType wanted)
{
  if (existing == wanted) {
    return true;
  }
  if (wanted == AttrType::Float4) {
    return ELEM(existing, AttrType::ColorFloat, AttrType::Quaternion);
  }
  return false;
}

/** Name of the color attribute `@color` maps to: the mesh's active color, otherwise "Color". */
static std::string color_attribute_name(const Mesh *mesh)
{
  if (mesh && mesh->active_color_attribute && mesh->attributes().contains(
                                                   mesh->active_color_attribute))
  {
    return mesh->active_color_attribute;
  }
  return "Color";
}

/**
 * Blender name of a Toast attribute. Toast uses Blender's names, except `color`, which is the
 * active color attribute. `mesh` is the mesh being read or written, if any.
 */
static std::string blender_name(const StringRef name, const Mesh *mesh)
{
  if (name == "color") {
    return color_attribute_name(mesh);
  }
  return name;
}

/** The points of one geometry: the mesh if it has vertices, otherwise the point cloud. */
struct PointsSource {
  const Mesh *mesh = nullptr;
  const PointCloud *pointcloud = nullptr;

  int64_t size() const
  {
    return mesh ? mesh->verts_num : (pointcloud ? pointcloud->totpoint : 0);
  }
  /** Element count of a domain; point clouds only have points. */
  int64_t size(const AttrDomain domain) const
  {
    if (mesh) {
      return mesh->attributes().domain_size(domain);
    }
    return domain == AttrDomain::Point ? this->size() : 0;
  }
  std::optional<bke::AttributeAccessor> attributes() const
  {
    if (mesh) {
      return mesh->attributes();
    }
    if (pointcloud) {
      return pointcloud->attributes();
    }
    return std::nullopt;
  }
};

static PointsSource points_of(const GeometrySet &geometry)
{
  PointsSource source;
  const Mesh *mesh = geometry.get_mesh();
  if (mesh && mesh->verts_num > 0) {
    source.mesh = mesh;
    return source;
  }
  source.pointcloud = geometry.get_pointcloud();
  if (source.pointcloud == nullptr) {
    source.mesh = mesh;
  }
  return source;
}

/** A read-only attribute bound for the kernel: a view of Blender's array when possible. */
struct ReadBinding {
  /** Keeps the data of a zero-copy view alive. */
  bke::GAttributeReader reader;
  GArray<> copy;
  const void *data = nullptr;
};

/**
 * Bind an attribute for reading, converted to the Toast type and interpolated to points.
 * With `allow_view`, an attribute already stored as a span of the right type is bound without a
 * copy; otherwise it is copied in parallel (snapshots must be copies, since the kernel writes the
 * same attribute in place). `N` on a mesh without an `N` attribute reads the vertex normals.
 */
static bool read_attribute(const PointsSource &source,
                           const SigAttr &attr,
                           const AttrType type,
                           const bool allow_view,
                           ReadBinding &r_binding,
                           std::string &r_error)
{
  const CPPType &cpp_type = bke::attribute_type_to_cpp_type(type);
  const AttrDomain domain = attr.domain;
  const int64_t size = source.size(domain);
  std::optional<bke::AttributeAccessor> attributes = source.attributes();
  const std::string name = blender_name(attr.name, source.mesh);
  if (attributes && attributes->contains(name)) {
    bke::GAttributeReader reader = attributes->lookup(name, domain, type);
    if (!reader) {
      r_error = "Attribute \"" + std::string(attr.name) + "\" can't be read as " +
                std::string(cpp_type.name());
      return false;
    }
    if (allow_view && reader.varray.is_span()) {
      r_binding.data = reader.varray.get_internal_span().data();
      r_binding.reader = std::move(reader);
      return true;
    }
    r_binding.copy = GArray<>(cpp_type, size);
    array_utils::copy(reader.varray, r_binding.copy.as_mutable_span());
    r_binding.data = r_binding.copy.data();
    return true;
  }
  if (attr.name == "normal" && source.mesh && type == AttrType::Float3 && domain != AttrDomain::Edge) {
    const Span<float3> normals = domain == AttrDomain::Point ? source.mesh->vert_normals() :
                                 domain == AttrDomain::Face  ? source.mesh->face_normals() :
                                                               source.mesh->corner_normals();
    if (allow_view) {
      r_binding.data = normals.data();
      return true;
    }
    r_binding.copy = GArray<>(cpp_type, size);
    array_utils::copy(normals, r_binding.copy.as_mutable_span().typed<float3>());
    r_binding.data = r_binding.copy.data();
    return true;
  }
  if (attr.create) {
    r_binding.copy = GArray<>(cpp_type, size);
    cpp_type.value_initialize_n(r_binding.copy.data(), size);
    r_binding.data = r_binding.copy.data();
    return true;
  }
  r_error = "Attribute \"" + std::string(attr.name) + "\" not found on input " +
            std::to_string(attr.input);
  return false;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Applying Commands
 * \{ */

struct MergedCommands {
  Vector<float3> new_positions;
  Vector<Vector<int>> new_prims;
  Vector<int> removed;
  Vector<int> removed_prims;
  bool has_changes() const
  {
    return !new_positions.is_empty() || !new_prims.is_empty() || !removed.is_empty() ||
           !removed_prims.is_empty();
  }
};

/**
 * Merge per-chunk buffers. Chunks cover increasing ranges of the mask and commands inside a
 * chunk are in element order, so chunk order is already (source index, emission order).
 */
static bool merge_commands(Span<CmdBuffer> buffers,
                           const int64_t points_num,
                           const int64_t prims_num,
                           MergedCommands &r_merged,
                           std::string &r_error)
{
  Array<int64_t> added_offset(buffers.size() + 1, 0);
  for (const int64_t i : buffers.index_range()) {
    added_offset[i + 1] = added_offset[i] + buffers[i].added;
  }
  const int64_t total_points = points_num + added_offset.last();
  if (total_points > std::numeric_limits<int>::max()) {
    r_error = "Too many points added";
    return false;
  }

  auto resolve = [&](const int64_t pt, int &r_index) -> bool {
    if (pt >= 0) {
      /* From points_num on, indices are points added in this run, in merged order. */
      if (pt >= total_points) {
        r_error = "Point " + std::to_string(pt) + " does not exist";
        return false;
      }
      r_index = int(pt);
      return true;
    }
    const int64_t code = -(pt + 1);
    const int64_t chunk = code >> 32;
    const int64_t local = code & 0xFFFFFFFF;
    if (chunk < 0 || chunk >= buffers.size() || local >= buffers[chunk].added) {
      r_error = "Invalid point handle";
      return false;
    }
    r_index = int(points_num + added_offset[chunk] + local);
    return true;
  };

  for (const CmdBuffer &buffer : buffers) {
    for (const Cmd &cmd : buffer.cmds) {
      switch (cmd.kind) {
        case CmdKind::AddPoint:
          r_merged.new_positions.append(cmd.position);
          break;
        case CmdKind::AddPrim: {
          Vector<int> prim;
          for (const int64_t pt : buffer.prim_points.as_span().slice(cmd.prim_points)) {
            int index;
            if (!resolve(pt, index)) {
              return false;
            }
            prim.append(index);
          }
          r_merged.new_prims.append(std::move(prim));
          break;
        }
        case CmdKind::SetAttr:
          break; /* Applied before the merge, see apply_sets. */
        case CmdKind::RemovePrim: {
          if (cmd.point < 0 || cmd.point >= prims_num) {
            r_error = "Face " + std::to_string(cmd.point) + " does not exist";
            return false;
          }
          r_merged.removed_prims.append(int(cmd.point));
          break;
        }
        case CmdKind::RemovePoint: {
          int index;
          if (!resolve(cmd.point, index)) {
            return false;
          }
          r_merged.removed.append(index);
          break;
        }
      }
    }
  }
  return true;
}

/**
 * Grow a domain from `old_num` to `old_num + added`: existing values are kept, new elements get
 * default (zero) values, as the harness does.
 */
static void grow_domain(const bke::AttributeAccessor src_attributes,
                        bke::MutableAttributeAccessor dst_attributes,
                        const AttrDomain domain,
                        const int old_num,
                        const int added,
                        const bke::AttributeFilter &attribute_filter)
{
  if (old_num > 0) {
    Array<int> indices(old_num + added, 0);
    array_utils::fill_index_range<int>(indices.as_mutable_span().take_front(old_num));
    bke::gather_attributes(
        src_attributes, domain, domain, attribute_filter, indices.as_span(), dst_attributes);
  }
  if (added > 0) {
    bke::fill_attribute_range_default(
        dst_attributes, domain, attribute_filter, IndexRange(old_num, added));
  }
}

static Mesh *mesh_add_elements(const Mesh &src,
                               const MergedCommands &merged,
                               const bke::AttributeFilter &attribute_filter)
{
  int new_edges = 0;
  int new_faces = 0;
  int new_corners = 0;
  for (const Span<int> prim : merged.new_prims) {
    if (prim.size() == 2) {
      new_edges++;
    }
    else {
      new_faces++;
      new_corners += prim.size();
    }
  }
  const int new_verts = merged.new_positions.size();

  Mesh *dst = bke::mesh_new_no_attributes(src.verts_num + new_verts,
                                          src.edges_num + new_edges,
                                          src.faces_num + new_faces,
                                          src.corners_num + new_corners);
  BKE_mesh_copy_parameters_for_eval(dst, &src);

  const bke::AttributeAccessor src_attributes = src.attributes();
  bke::MutableAttributeAccessor dst_attributes = dst->attributes_for_write();
  grow_domain(src_attributes, dst_attributes, AttrDomain::Point, src.verts_num, new_verts, attribute_filter);
  grow_domain(src_attributes, dst_attributes, AttrDomain::Edge, src.edges_num, new_edges, attribute_filter);
  grow_domain(src_attributes, dst_attributes, AttrDomain::Face, src.faces_num, new_faces, attribute_filter);
  grow_domain(src_attributes, dst_attributes, AttrDomain::Corner, src.corners_num, new_corners, attribute_filter);

  MutableSpan<float3> positions = dst->vert_positions_for_write();
  positions.drop_front(src.verts_num).copy_from(merged.new_positions);

  MutableSpan<int2> edges = dst->edges_for_write();
  MutableSpan<int> face_offsets = dst->face_offsets_for_write();
  MutableSpan<int> corner_verts = dst->corner_verts_for_write();
  if (!face_offsets.is_empty()) {
    if (src.faces_num > 0) {
      face_offsets.take_front(src.faces_num + 1).copy_from(src.face_offsets());
    }
    else {
      face_offsets[0] = 0;
    }
  }
  int edge = src.edges_num;
  int face = src.faces_num;
  int corner = src.corners_num;
  for (const Span<int> prim : merged.new_prims) {
    if (prim.size() == 2) {
      edges[edge++] = int2(prim[0], prim[1]);
    }
    else {
      corner_verts.slice(corner, prim.size()).copy_from(prim);
      corner += prim.size();
      face_offsets[++face] = corner;
    }
  }
  /* Edges of the new faces, and corner edges for them. */
  if (new_faces > 0) {
    bke::mesh_calc_edges(*dst, true, false);
  }
  return dst;
}

static void apply_to_mesh(GeometrySet &geometry,
                          const MergedCommands &merged,
                          const bke::AttributeFilter &attribute_filter)
{
  const Mesh *mesh = geometry.get_mesh();
  if (!merged.new_positions.is_empty() || !merged.new_prims.is_empty()) {
    geometry.replace_mesh(mesh_add_elements(*mesh, merged, attribute_filter));
    mesh = geometry.get_mesh();
  }
  if (!merged.removed_prims.is_empty()) {
    /* Faces only, keeping their points and edges (Delete Geometry, Only Faces). */
    Array<bool> keep(mesh->faces_num, true);
    for (const int i : merged.removed_prims) {
      keep[i] = false;
    }
    std::optional<Mesh *> result = geometry::mesh_copy_selection_keep_edges(
        *mesh, VArray<bool>::from_span(keep), AttrDomain::Face, attribute_filter);
    if (result) {
      geometry.replace_mesh(*result);
      mesh = geometry.get_mesh();
    }
  }
  if (!merged.removed.is_empty() && mesh != nullptr) {
    Array<bool> keep(mesh->verts_num, true);
    for (const int i : merged.removed) {
      keep[i] = false;
    }
    std::optional<Mesh *> result = geometry::mesh_copy_selection(
        *mesh, VArray<bool>::from_span(keep), AttrDomain::Point, attribute_filter);
    if (result) {
      geometry.replace_mesh(*result);
    }
  }
}

static bool apply_to_pointcloud(GeometrySet &geometry,
                                const MergedCommands &merged,
                                const bke::AttributeFilter &attribute_filter,
                                std::string &r_error)
{
  if (!merged.new_prims.is_empty() || !merged.removed_prims.is_empty()) {
    r_error = "add_prim() and remove_prim() need a mesh, the input is a point cloud";
    return false;
  }
  PointCloud *pointcloud = geometry.get_pointcloud_for_write();
  const int old_num = pointcloud->totpoint;
  if (!merged.new_positions.is_empty()) {
    pointcloud_resize(*pointcloud, old_num + merged.new_positions.size());
    pointcloud->positions_for_write().drop_front(old_num).copy_from(merged.new_positions);
  }
  if (!merged.removed.is_empty()) {
    Array<bool> keep(pointcloud->totpoint, true);
    for (const int i : merged.removed) {
      keep[i] = false;
    }
    IndexMaskMemory memory;
    const IndexMask kept = IndexMask::from_bools(keep, memory);
    PointCloud *dst = bke::pointcloud_new_no_attributes(PointCloudType(pointcloud->type),
                                                        kept.size());
    bke::gather_attributes(pointcloud->attributes(),
                           AttrDomain::Point,
                           AttrDomain::Point,
                           attribute_filter,
                           kept,
                           dst->attributes_for_write());
    geometry.replace_pointcloud(dst);
  }
  return true;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Execution
 * \{ */

struct RunContext {
  const LoadedKernel *kernel = nullptr;
  Field<bool> selection;
  Span<int32_t> params;
  const GeometrySet *input1 = nullptr;
  float frame = 0.0f;
  float time = 0.0f;
  bke::AttributeFilter attribute_filter;
};

/**
 * Run the kernel over one real geometry: the points of the mesh (or the point cloud), the mesh's
 * faces, edges or corners, or once for the whole geometry.
 */
static bool run_on_geometry(GeometrySet &geometry, const RunContext &run, std::string &r_error)
{
  const Signature &sig = run.kernel->signature;
  const std::optional<AttrDomain> domain = run_domain(sig.run_over);
  const bool is_once = sig.run_over == "geometry";
  bool is_mesh = false;
  if (domain == AttrDomain::Point) {
    is_mesh = geometry.has_mesh() && geometry.get_mesh()->verts_num > 0;
    if (!is_mesh && !(geometry.has_pointcloud() && geometry.get_pointcloud()->totpoint > 0)) {
      return true;
    }
  }
  else if (domain) {
    if (!geometry.has_mesh()) {
      return true; /* Faces, edges and corners only exist on meshes. */
    }
    is_mesh = true;
  }
  else {
    /* Geometry: runs once, also on empty geometry, which gets a mesh to build into. */
    if (!geometry.has_mesh() && !geometry.has_pointcloud()) {
      geometry.replace_mesh(BKE_mesh_new_nomain(0, 0, 0, 0));
    }
    is_mesh = geometry.has_mesh();
  }

  PointsSource input1_source;
  if (run.input1) {
    input1_source = points_of(*run.input1);
  }

  /* Validate the signature. */
  for (const SigAttr &attr : sig.attrs) {
    if (!attr_type_from_code(attr.type_code)) {
      r_error = "Unknown attribute type in kernel signature";
      return false;
    }
    if (attr.input >= MAX_INPUTS) {
      r_error = "This node has no input " + std::to_string(attr.input);
      return false;
    }
    if (attr.input == 1 && input1_source.size(attr.domain) == 0 && !attr.create) {
      r_error = "Attribute \"" + attr.name + "\" needs geometry on Input 1";
      return false;
    }
  }
  auto is_written = [](const SigAttr &attr) {
    return attr.input == 0 && attr.write && !attr.snapshot;
  };

  /* Copies of input 0 that must not see the kernel's writes: snapshots, and positions for a
   * KD-tree on input 0. Made before any writer exists. */
  const PointsSource input0_before = points_of(geometry);
  /* Surface queries on input 0 see it as it was before the run: a shared copy makes getting the
   * geometry for write below copy the mesh instead of changing this one. */
  std::optional<GeometrySet> input0_snapshot;
  if (sig.surface_inputs.contains(0)) {
    input0_snapshot = geometry;
  }
  const int64_t points_num = input0_before.size();
  const int64_t prims_num = input0_before.mesh ? input0_before.mesh->faces_num : 0;
  const int64_t domain_size = is_once ? 1 : input0_before.size(*domain);
  if (domain_size == 0) {
    return true;
  }

  /* The Selection, evaluated on the input before the run, as the kernel's mask indices (empty,
   * a null pointer in the context, when everything is selected). Run Over Geometry ignores it. The mask's
   * memory belongs to the evaluator, so the indices are copied out while it exists. */
  Array<int64_t> selected_indices;
  int64_t run_size = domain_size;
  if (!is_once) {
    auto evaluate = [&](const fn::FieldContext &context) {
      fn::FieldEvaluator evaluator(context, domain_size);
      evaluator.add(run.selection);
      evaluator.evaluate();
      const IndexMask selection = evaluator.get_evaluated_as_mask(0);
      run_size = selection.size();
      if (run_size != domain_size) {
        selected_indices.reinitialize(run_size);
        selection.to_indices<int64_t>(selected_indices);
      }
    };
    if (input0_before.mesh) {
      evaluate(bke::MeshFieldContext(*input0_before.mesh, *domain));
    }
    else {
      evaluate(bke::PointCloudFieldContext(*input0_before.pointcloud));
    }
    if (run_size == 0) {
      return true;
    }
  }
  Array<ReadBinding> reads(sig.attrs.size());
  Array<ToastAttr> bindings(std::max<int64_t>(sig.attrs.size(), 1));
  for (const int64_t i : sig.attrs.index_range()) {
    const SigAttr &attr = sig.attrs[i];
    if (attr.input == 0 && attr.snapshot) {
      if (!read_attribute(
              input0_before, attr, *attr_type_from_code(attr.type_code), false, reads[i], r_error))
      {
        return false;
      }
    }
  }

  HostData host_data;
  host_data.npoints[0] = points_num;
  host_data.npoints[1] = input1_source.size();
  host_data.meshes[1] = input1_source.mesh;
  std::array<bke::GAttributeReader, MAX_INPUTS> tree_readers;
  std::array<Array<float3>, MAX_INPUTS> tree_copies;
  std::array<Span<float3>, MAX_INPUTS> tree_positions;
  for (const int input : sig.nearpoints_inputs) {
    if (input < 0 || input >= MAX_INPUTS || !tree_positions[input].is_empty()) {
      continue;
    }
    const PointsSource &source = input == 0 ? input0_before : input1_source;
    if (source.size() == 0) {
      continue;
    }
    bke::GAttributeReader reader = source.attributes()->lookup("position", AttrDomain::Point);
    if (input != 0 && reader.varray.is_span()) {
      tree_positions[input] = reader.varray.get_internal_span().typed<float3>();
      tree_readers[input] = std::move(reader);
    }
    else {
      tree_copies[input] = Array<float3>(source.size());
      array_utils::copy(reader.varray.typed<float3>(), tree_copies[input].as_mutable_span());
      tree_positions[input] = tree_copies[input];
    }
  }

  /* Writers on input 0. Getting the geometry for write un-shares it from Input 1. */
  Mesh *write_mesh = is_mesh ? geometry.get_mesh_for_write() : nullptr;
  bke::MutableAttributeAccessor attributes = is_mesh ?
                                                 write_mesh->attributes_for_write() :
                                                 geometry.get_pointcloud_for_write()->attributes_for_write();
  Vector<bke::GSpanAttributeWriter> writers;
  /* `@color` is a vec3 in Toast and a float4 color in Blender: the kernel writes a vec3 copy that is
   * copied back into the color's RGB after the run, keeping alpha. */
  struct ColorWriteBack {
    int64_t writer;
    Array<float3, 0> rgb; /* No inline buffer: the kernel keeps a pointer across moves. */
  };
  Vector<ColorWriteBack> color_writebacks;
  auto finish_writers = [&]() {
    for (bke::GSpanAttributeWriter &w : writers) {
      w.finish();
    }
  };
  for (const int64_t i : sig.attrs.index_range()) {
    const SigAttr &attr = sig.attrs[i];
    if (!is_written(attr)) {
      continue;
    }
    const AttrType type = *attr_type_from_code(attr.type_code);
    const std::string name = blender_name(attr.name, write_mesh);
    const bool is_color = attr.name == "color" && type == AttrType::Float3;
    const std::optional<bke::AttributeMetaData> meta = attributes.lookup_meta_data(name);
    if (name == "position" && attr.domain != AttrDomain::Point) {
      r_error =
          "@position is a point attribute, it can only be written when running over points";
      finish_writers();
      return false;
    }
    if (meta) {
      const bool ok = is_color ? (meta->domain == attr.domain &&
                                  meta->data_type == AttrType::ColorFloat) :
                                 (meta->domain == attr.domain &&
                                  layout_compatible(meta->data_type, type));
      if (!ok) {
        r_error = is_color ? "@color: color attribute \"" + name +
                                 "\" isn't a float color on the domain the code runs over, which "
                                 "Toast needs to write it. Convert it in Mesh Data > Color "
                                 "Attributes" :
                             "Attribute \"" + attr.name +
                                 "\" exists with another type or domain than the code uses";
        finish_writers();
        return false;
      }
    }
    bke::GSpanAttributeWriter writer = attributes.lookup_or_add_for_write_span(
        name, attr.domain, is_color ? AttrType::ColorFloat : type);
    if (!writer) {
      r_error = "Can't write attribute \"" + attr.name + "\"";
      finish_writers();
      return false;
    }
    if (is_color) {
      MutableSpan<ColorGeometry4f> colors = writer.span.typed<ColorGeometry4f>();
      if (!meta) {
        /* New color: opaque, and the one the viewport shows. */
        colors.fill(ColorGeometry4f(0.0f, 0.0f, 0.0f, 1.0f));
        if (write_mesh) {
          if (!write_mesh->active_color_attribute) {
            write_mesh->active_color_attribute = BLI_strdup(name.c_str());
          }
          if (!write_mesh->default_color_attribute) {
            write_mesh->default_color_attribute = BLI_strdup(name.c_str());
          }
        }
      }
      Array<float3, 0> rgb(colors.size());
      threading::parallel_for(colors.index_range(), 4096, [&](const IndexRange range) {
        for (const int64_t k : range) {
          rgb[k] = float3(colors[k].r, colors[k].g, colors[k].b);
        }
      });
      bindings[i] = {rgb.data(), rgb.size(), attr.type_code, abi_domain(attr.domain)};
      color_writebacks.append({writers.size(), std::move(rgb)});
    }
    else {
      bindings[i] = {writer.span.data(), writer.span.size(), attr.type_code, abi_domain(attr.domain)};
    }
    writers.append(std::move(writer));
  }

  /* Other reads, after the writers so they see the writable data: views where possible. Writes
   * only touch the written attributes' own arrays, and normals stay cached until the position
   * writer finishes after the run. */
  const PointsSource input0 = points_of(geometry);
  host_data.meshes[0] = input0.mesh;
  for (const int64_t i : sig.attrs.index_range()) {
    const SigAttr &attr = sig.attrs[i];
    if (is_written(attr)) {
      continue;
    }
    if (!(attr.input == 0 && attr.snapshot)) {
      const PointsSource &source = attr.input == 0 ? input0 : input1_source;
      if (!read_attribute(
              source, attr, *attr_type_from_code(attr.type_code), true, reads[i], r_error))
      {
        finish_writers();
        return false;
      }
    }
    const int64_t size = (attr.input == 0 ? input0 : input1_source).size(attr.domain);
    bindings[i] = {const_cast<void *>(reads[i].data), size, attr.type_code, abi_domain(attr.domain)};
  }

  /* Spatial queries. */
  std::array<std::unique_ptr<KDTreeNew<float3>>, MAX_INPUTS> trees;
  for (const int input : IndexRange(MAX_INPUTS)) {
    if (!tree_positions[input].is_empty()) {
      trees[input] = std::make_unique<KDTreeNew<float3>>(tree_positions[input]);
      host_data.trees[input] = trees[input].get();
    }
  }

  host_data.bindings = bindings;
  if (input0_snapshot) {
    host_data.surface_meshes[0] = input0_snapshot->get_mesh();
  }
  host_data.surface_meshes[1] = input1_source.mesh;
  if (sig.uses_neighbors) {
    for (const int input : IndexRange(MAX_INPUTS)) {
      if (host_data.meshes[input]) {
        build_neighbors(*host_data.meshes[input],
                        host_data.neighbor_offsets[input],
                        host_data.neighbors[input]);
      }
    }
  }
  ToastHost host = {&host_data,
                    host_nearpoints,
                    host_npoints,
                    host_add_point,
                    host_add_prim,
                    host_remove_point,
                    host_nprims,
                    host_primpoints,
                    host_remove_prim,
                    host_point_faces,
                    host_point_neighbors,
                    host_corner_point,
                    host_corner_face,
                    host_set_attr,
                    host_ray_cast,
                    host_closest_point};

  const int64_t chunks_num = (run_size + GRAIN_SIZE - 1) / GRAIN_SIZE;
  Array<CmdBuffer> buffers(chunks_num);
  threading::parallel_for(IndexRange(chunks_num), 1, [&](const IndexRange range) {
    for (const int64_t chunk : range) {
      CmdBuffer &buffer = buffers[chunk];
      buffer.chunk_id = chunk;
      ToastCtx ctx = {};
      ctx.abi_version = TOAST_ABI_VERSION;
      ctx.begin = chunk * GRAIN_SIZE;
      ctx.end = std::min(run_size, ctx.begin + GRAIN_SIZE);
      ctx.indices = selected_indices.is_empty() ? nullptr : selected_indices.data();
      ctx.domain_size = domain_size;
      ctx.attrs = bindings.data();
      ctx.params = run.params.data();
      ctx.cmd = reinterpret_cast<ToastCmdBuffer *>(&buffer);
      ctx.host = &host;
      ctx.chunk_id = chunk;
      ctx.time = run.time;
      ctx.frame = run.frame;
      run.kernel->fn(&ctx);
    }
  });

  if (sig.uses_cmd && !apply_sets(buffers, bindings, r_error)) {
    for (bke::GSpanAttributeWriter &writer : writers) {
      writer.finish();
    }
    return false;
  }
  for (const ColorWriteBack &wb : color_writebacks) {
    MutableSpan<ColorGeometry4f> colors = writers[wb.writer].span.typed<ColorGeometry4f>();
    threading::parallel_for(colors.index_range(), 4096, [&](const IndexRange range) {
      for (const int64_t k : range) {
        colors[k].r = wb.rgb[k].x;
        colors[k].g = wb.rgb[k].y;
        colors[k].b = wb.rgb[k].z;
      }
    });
  }
  for (bke::GSpanAttributeWriter &writer : writers) {
    writer.finish();
  }

  if (!sig.uses_cmd) {
    return true;
  }
  MergedCommands merged;
  if (!merge_commands(buffers, points_num, prims_num, merged, r_error)) {
    return false;
  }
  if (!merged.has_changes()) {
    return true;
  }
  if (is_mesh) {
    apply_to_mesh(geometry, merged, run.attribute_filter);
    return true;
  }
  return apply_to_pointcloud(geometry, merged, run.attribute_filter, r_error);
}

static const bNodeSocket *find_input(const bNode &node, const StringRef identifier)
{
  for (const bNodeSocket *socket : node.input_sockets()) {
    if (socket->identifier == identifier) {
      return socket;
    }
  }
  return nullptr;
}

/** Fill the kernel's Params block from the param() sockets. */
static bool gather_params(GeoNodeExecParams &params,
                          const Signature &sig,
                          Array<int32_t> &r_buffer,
                          std::string &r_error)
{
  r_buffer = Array<int32_t>(std::max(sig.params_size / 4, 1), 0);
  char *base = reinterpret_cast<char *>(r_buffer.data());
  for (const SigParam &p : sig.params) {
    const std::string identifier = param_identifier(p.name);
    if (find_input(params.node(), identifier) == nullptr) {
      r_error = "The node's sockets are out of date, the code is being recompiled";
      return false;
    }
    const UString id(identifier);
    if (p.type == "float") {
      const float v = params.get_input<float>(id);
      memcpy(base + p.offset, &v, sizeof(v));
    }
    else if (p.type == "int") {
      const int32_t v = params.get_input<int>(id);
      memcpy(base + p.offset, &v, sizeof(v));
    }
    else if (p.type == "bool") {
      const int32_t v = params.get_input<bool>(id) ? 1 : 0;
      memcpy(base + p.offset, &v, sizeof(v));
    }
    else if (p.type == "vec3") {
      const float3 v = params.get_input<float3>(id);
      memcpy(base + p.offset, &v, sizeof(v));
    }
    else if (p.type == "vec4") {
      const ColorGeometry4f v = params.get_input<ColorGeometry4f>(id);
      memcpy(base + p.offset, &v, sizeof(v));
    }
    else {
      r_error = "Unknown parameter type \"" + p.type + "\"";
      return false;
    }
  }
  return true;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const GeometrySet input1 = params.extract_input<GeometrySet>("Input 1"_ustr);
  const NodeGeometryToast &storage = node_storage(params.node());

  auto finish = [&](const std::optional<std::string> &error, const bool is_info = false) {
    if (error) {
      params.error_message_add(is_info ? NodeWarningType::Info : NodeWarningType::Error, *error);
    }
    params.set_output("Geometry"_ustr, std::move(geometry));
  };

  if (storage.error && storage.error[0]) {
    finish(std::string(storage.error));
    return;
  }
  if (!(G.f & G_FLAG_SCRIPT_AUTOEXEC)) {
    finish(std::string(TIP_("Toast code is disabled. Allow Auto Run Python Scripts or trust "
                            "this file to run it")));
    return;
  }
  if (!storage.kernel_path || !storage.kernel_path[0]) {
    finish(std::string(TIP_("No compiled code. Pick a text with Toast code; the Toast add-on "
                            "must be enabled")),
           true);
    return;
  }

  std::string error;
  const LoadedKernel *kernel = load_kernel(storage.kernel_path, error);
  if (!kernel) {
    finish(error);
    return;
  }
  if (!run_domain(kernel->signature.run_over) && kernel->signature.run_over != "geometry") {
    finish(std::string("This Blender can't run kernels over ") + kernel->signature.run_over);
    return;
  }

  Array<int32_t> param_buffer;
  if (!gather_params(params, kernel->signature, param_buffer, error)) {
    finish(error);
    return;
  }

  RunContext run;
  run.kernel = kernel;
  run.selection = params.extract_input<Field<bool>>("Selection"_ustr);
  run.params = param_buffer;
  run.input1 = &input1;
  run.attribute_filter = params.get_attribute_filter("Geometry"_ustr);
  if (const Depsgraph *depsgraph = params.depsgraph()) {
    const Scene *scene = DEG_get_input_scene(depsgraph);
    run.frame = DEG_get_ctime(depsgraph);
    run.time = scene ? float(run.frame / scene->frames_per_second()) : 0.0f;
  }

  std::mutex error_mutex;
  geometry::foreach_real_geometry(geometry, [&](GeometrySet &real_geometry) {
    std::string local_error;
    if (!run_on_geometry(real_geometry, run, local_error)) {
      std::lock_guard lock{error_mutex};
      if (error.empty()) {
        error = local_error;
      }
    }
  });
  finish(error.empty() ? std::nullopt : std::optional<std::string>(error));
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Node Type
 * \{ */

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .description("Geometry the code runs on and writes to (input 0)");
  b.add_input<decl::Bool>("Selection"_ustr)
      .default_value(true)
      .hide_value()
      .evaluated_geometry_field()
      .description("Elements the code runs on, in the Run Over domain. Others are unchanged");
  b.add_input<decl::Geometry>("Input 1"_ustr)
      .description(
          "Geometry the code can read with point_attr(1, ...) and nearby_points(1, ...)");
  b.add_output<decl::Geometry>("Geometry"_ustr).propagate_all();

  const bNode *node = b.node_or_null();
  if (node == nullptr || node->storage == nullptr) {
    return;
  }
  const NodeGeometryToast &storage = node_storage(*node);
  const std::optional<Signature> sig = parse_signature(storage.signature ? storage.signature :
                                                                            "");
  if (!sig) {
    return;
  }
  for (const SigParam &p : sig->params) {
    const UString name(p.name);
    const UString identifier(param_identifier(p.name));
    const double d0 = p.default_value.is_empty() ? 0.0 : p.default_value[0];
    if (p.type == "float") {
      auto &s = b.add_input<decl::Float>(name, identifier).default_value(float(d0));
      if (p.min) {
        s.min(float(*p.min));
      }
      if (p.max) {
        s.max(float(*p.max));
      }
    }
    else if (p.type == "int") {
      auto &s = b.add_input<decl::Int>(name, identifier).default_value(int(d0));
      if (p.min) {
        s.min(int(*p.min));
      }
      if (p.max) {
        s.max(int(*p.max));
      }
    }
    else if (p.type == "bool") {
      b.add_input<decl::Bool>(name, identifier).default_value(d0 != 0.0);
    }
    else if (p.type == "vec3" && p.default_value.size() == 3) {
      b.add_input<decl::Vector>(name, identifier)
          .default_value(float3(p.default_value[0], p.default_value[1], p.default_value[2]));
    }
    else if (p.type == "vec4" && p.default_value.size() == 4) {
      b.add_input<decl::Color>(name, identifier)
          .default_value(ColorGeometry4f(p.default_value[0],
                                         p.default_value[1],
                                         p.default_value[2],
                                         p.default_value[3]));
    }
  }
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  node->storage = MEM_new<NodeGeometryToast>(__func__);
}

static char *dup_or_null(const char *str)
{
  return str ? BLI_strdup(str) : nullptr;
}

static void free_strings(NodeGeometryToast &storage)
{
  for (char **str : {&storage.signature, &storage.kernel_path, &storage.error, &storage.source_hash})
  {
    if (*str) {
      MEM_delete(*str);
      *str = nullptr;
    }
  }
}

static void node_free_storage(bNode *node)
{
  NodeGeometryToast *storage = static_cast<NodeGeometryToast *>(node->storage);
  free_strings(*storage);
  MEM_delete(storage);
}

static void node_copy_storage(bNodeTree * /*dst_tree*/, bNode *dst_node, const bNode *src_node)
{
  const NodeGeometryToast &src = node_storage(*src_node);
  NodeGeometryToast *dst = MEM_new<NodeGeometryToast>(__func__, dna::shallow_copy(src));
  dst->signature = dup_or_null(src.signature);
  dst->kernel_path = dup_or_null(src.kernel_path);
  dst->error = dup_or_null(src.error);
  dst->source_hash = dup_or_null(src.source_hash);
  dst_node->storage = dst;
}

static void node_blend_write(const bNodeTree & /*tree*/, const bNode &node, BlendWriter &writer)
{
  const NodeGeometryToast &storage = node_storage(node);
  for (const char *str :
       {storage.signature, storage.kernel_path, storage.error, storage.source_hash})
  {
    if (str) {
      writer.write_string(str);
    }
  }
}

static void node_blend_read(bNodeTree & /*tree*/, bNode &node, BlendDataReader &reader)
{
  NodeGeometryToast &storage = node_storage(node);
  BLO_read_string(&reader, &storage.signature);
  BLO_read_string(&reader, &storage.kernel_path);
  BLO_read_string(&reader, &storage.error);
  BLO_read_string(&reader, &storage.source_hash);
}

static void node_layout(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.prop(ptr, "text", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
  layout.prop(ptr, "run_over", UI_ITEM_NONE, "", ICON_NONE);
}

/* RNA: the code text (bNode.id) and the strings the add-on writes. */

static PointerRNA rna_text_get(PointerRNA *ptr)
{
  bNode *node = ptr->data_as<bNode>();
  return RNA_id_pointer_create(node->id);
}

static void rna_text_set(PointerRNA *ptr, const PointerRNA value, ReportList * /*reports*/)
{
  bNode *node = ptr->data_as<bNode>();
  ID *new_id = static_cast<ID *>(value.data);
  if (new_id && GS(new_id->name) != ID_TXT) {
    return;
  }
  if (node->id == new_id) {
    return;
  }
  if (node->id) {
    id_us_min(node->id);
  }
  node->id = new_id;
  if (new_id) {
    id_us_plus(new_id);
  }
}

template<char *NodeGeometryToast::*Member>
static std::string rna_string_get(PointerRNA *ptr, PropertyRNA * /*prop*/)
{
  const char *str = node_storage(*ptr->data_as<bNode>()).*Member;
  return str ? str : "";
}

template<char *NodeGeometryToast::*Member>
static int rna_string_length(PointerRNA *ptr, PropertyRNA * /*prop*/)
{
  const char *str = node_storage(*ptr->data_as<bNode>()).*Member;
  return str ? int(strlen(str)) : 0;
}

template<char *NodeGeometryToast::*Member>
static void rna_string_set(PointerRNA *ptr, PropertyRNA * /*prop*/, const std::string &value)
{
  NodeGeometryToast &storage = node_storage(*ptr->data_as<bNode>());
  char *&str = storage.*Member;
  if (str) {
    MEM_delete(str);
    str = nullptr;
  }
  if (!value.empty()) {
    str = BLI_strdup(value.c_str());
  }
  if constexpr (Member == &NodeGeometryToast::signature) {
    const std::optional<Signature> sig = parse_signature(str ? str : "");
    SET_FLAG_FROM_TEST(storage.flag, sig && sig->uses_time, NODE_GEO_TOAST_USES_TIME);
  }
}

template<char *NodeGeometryToast::*Member>
static void def_string(StructRNA *srna,
                       const char *identifier,
                       const char *ui_name,
                       const char *ui_description,
                       RNAPropertyUpdateFunc update)
{
  PropertyRNA *prop = RNA_def_property(srna, identifier, PROP_STRING, PROP_NONE);
  RNA_def_property_string_funcs_runtime(prop,
                                        rna_string_get<Member>,
                                        rna_string_length<Member>,
                                        rna_string_set<Member>,
                                        nullptr,
                                        nullptr);
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);
  RNA_def_property_ui_text(prop, ui_name, ui_description);
  RNA_def_property_update_runtime(prop, update);
}

static void node_rna(StructRNA *srna)
{
  PropertyRNA *prop = RNA_def_property(srna, "text", PROP_POINTER, PROP_NONE);
  RNA_def_property_struct_runtime(srna, prop, RNA_Text);
  RNA_def_property_flag(prop, PROP_EDITABLE);
  RNA_def_property_pointer_funcs_runtime(prop, rna_text_get, rna_text_set, nullptr);
  RNA_def_property_ui_text(prop, "Text", "Text data-block with the Toast code");
  RNA_def_property_update_runtime(prop, rna_Node_update);

  RNA_def_node_enum(srna,
                    "run_over",
                    "Run Over",
                    "Elements the code runs over, once each",
                    run_over_items,
                    NOD_storage_enum_accessors(run_over),
                    int(RunOver::Points));

  def_string<&NodeGeometryToast::signature>(
      srna,
      "signature",
      "Signature",
      "Parameters and attributes of the compiled kernel (JSON, written by the Toast add-on)",
      rna_Node_socket_update);
  def_string<&NodeGeometryToast::kernel_path>(
      srna, "kernel_path", "Kernel Path", "Compiled kernel library", rna_Node_update);
  def_string<&NodeGeometryToast::error>(
      srna, "error", "Error", "Compile error of the code, empty if it compiled", rna_Node_update);
  def_string<&NodeGeometryToast::source_hash>(
      srna, "source_hash", "Source Hash", "Hash of the code the kernel was built from", nullptr);
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeToast"_ustr);
  ntype.ui_name = "Toast";
  ntype.ui_description =
      "Run Python-syntax code on every point, compiled to native code by the Toast add-on";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.initfunc = node_init;
  bke::node_type_storage(ntype, "NodeGeometryToast", node_free_storage, node_copy_storage);
  ntype.blend_write_storage_content = node_blend_write;
  ntype.blend_data_read_storage_content = node_blend_read;
  ntype.draw_buttons = node_layout;
  ntype.geometry_node_execute = node_geo_exec;
  bke::node_register_type(ntype);

  node_rna(ntype.rna_ext.srna);
}
NOD_REGISTER_NODE(node_register)

/** \} */

}  // namespace blender::nodes::node_geo_toast_cc
