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
#include <fstream>
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
#include "BLI_enumerable_thread_specific.hh"
#include "BLI_generic_array.hh"
#include "BLI_kdtree_new.hh"
#include "BLI_map.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_serialize.hh"
#include "BLI_string.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "DNA_curves_types.h"
#include "DNA_mesh_types.h"
#include "DNA_pointcloud_types.h"
#include "DNA_scene_types.h"
#include "DNA_text_types.h"
#include "DNA_volume_types.h"

#include "BKE_appdir.hh"
#include "BKE_attribute.hh"
#include "BKE_bvh.hh"
#include "BKE_curves.hh"
#include "BKE_grease_pencil.hh"
#include "BKE_geometry_fields.hh"
#include "BKE_global.hh"
#include "BKE_instances.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_mesh_mapping.hh"
#include "BKE_pointcloud.hh"
#include "BKE_volume.hh"
#include "BKE_volume_grid.hh"

#ifdef WITH_OPENVDB
#  include "BKE_volume_grid_process.hh"
#  include "BKE_volume_openvdb.hh"
#  include "BLI_math_matrix.hh"
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Morphology.h>
#  include <openvdb/tools/Prune.h>
#endif

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

#include "WM_api.hh"

#include "jit/toast_jit.h"
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
  Curves = 5,
  Instances = 6,
  Voxels = 7,
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
    {int(RunOver::Curves), "CURVES", ICON_NONE, "Curves", "Run the code once for every curve"},
    {int(RunOver::Instances),
     "INSTANCES",
     ICON_NONE,
     "Instances",
     "Run the code once for every instance (of the top-level geometry)"},
    {int(RunOver::Voxels),
     "VOXELS",
     ICON_NONE,
     "Voxels",
     "Run the code once for every active voxel of the volume's grids; @ attributes are grids"},
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
  /* Voxels run as the points of a point cloud, one per voxel (see run_on_volume). */
  if (run_over == "points" || run_over == "voxels") {
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
  if (run_over == "curves") {
    return AttrDomain::Curve;
  }
  if (run_over == "instances") {
    return AttrDomain::Instance;
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
    case AttrDomain::Curve:
      return TOAST_DOMAIN_CURVE;
    case AttrDomain::Instance:
      return TOAST_DOMAIN_INSTANCE;
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
  if (domain == "curve") {
    return AttrDomain::Curve;
  }
  if (domain == "instance") {
    return AttrDomain::Instance;
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
  /** Values of an output() socket, stored as the socket's anonymous attribute. */
  bool output = false;
  AttrDomain domain = AttrDomain::Point;
};

/** An output() socket: `socket` is the socket type (Float, Integer, Boolean, Vector...). */
struct SigOutput {
  std::string name;
  std::string socket;
};

/** A grid read by sample_grid() or grid_gradient(); its slot is its index in the signature. */
struct SigGrid {
  std::string name;
  int input = 0;
  bool vector = false;
  /** voxel_at(): only the grid's transform is used, any type will do. */
  bool any = false;
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
  Vector<SigOutput> outputs;
  Vector<SigGrid> grids;
  int params_size = 0;
  bool uses_cmd = false;
  bool uses_time = false;
  /* add_curve()/extend_curve(): Run Over Geometry runs on curves, made if there are none. */
  bool builds_curves = false;
  bool uses_neighbors = false;
  /* activate_voxel()/deactivate_voxel(). */
  bool uses_voxel_cmd = false;
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
      attr.output = a->lookup_bool("output").value_or(false);
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

  if (const io::serialize::ArrayValue *outputs = dict->lookup_array("outputs")) {
    for (const std::shared_ptr<io::serialize::Value> &item : outputs->elements()) {
      const io::serialize::DictionaryValue *o = item->as_dictionary_value();
      if (o == nullptr) {
        return std::nullopt;
      }
      sig.outputs.append({std::string(o->lookup_str("name").value_or("")),
                          std::string(o->lookup_str("socket").value_or(""))});
    }
  }

  if (const io::serialize::ArrayValue *grids = dict->lookup_array("grids")) {
    for (const std::shared_ptr<io::serialize::Value> &item : grids->elements()) {
      const io::serialize::DictionaryValue *g = item->as_dictionary_value();
      if (g == nullptr) {
        return std::nullopt;
      }
      SigGrid grid;
      grid.name = std::string(g->lookup_str("name").value_or(""));
      grid.input = int(g->lookup_int("input").value_or(0));
      grid.vector = g->lookup_str("type").value_or("float") == "vec3";
      grid.any = g->lookup_str("type").value_or("float") == "any";
      sig.grids.append(std::move(grid));
    }
  }

  if (const io::serialize::DictionaryValue *uses = dict->lookup_dict("uses")) {
    sig.uses_cmd = uses->lookup_bool("cmd").value_or(false);
    sig.uses_time = uses->lookup_bool("time").value_or(false);
    sig.builds_curves = uses->lookup_bool("curves").value_or(false);
    if (const io::serialize::ArrayValue *host = uses->lookup_array("host")) {
      for (const std::shared_ptr<io::serialize::Value> &c : host->elements()) {
        if (const io::serialize::StringValue *v = c->as_string_value()) {
          sig.uses_neighbors |= v->value() == "point_neighbors";
          sig.uses_voxel_cmd |= v->value() == "voxel_cmd";
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

static std::string output_identifier(const StringRef name)
{
  return "o_" + std::string(name);
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

static ToastJitCompileFn jit_compile = nullptr;
static ToastJitSymbolFn jit_symbol = nullptr;

/**
 * The built-in compiler, loaded on first use: `lib/libtoast_jit.so` next to the executable, or on
 * macOS `Blender.app/Contents/Resources/lib/libtoast_jit.dylib` (with Blender's other libraries).
 */
static void *load_jit(std::string &r_error)
{
#ifdef _WIN32
  r_error = "The built-in Toast compiler is not available on Windows yet";
  return nullptr;
#else
  static void *handle = nullptr;
  static std::string error;
  static bool tried = false;
  if (!tried) {
    tried = true;
    const char *program_dir = BKE_appdir_program_dir();
#  ifdef __APPLE__
    const std::string path = std::string(program_dir ? program_dir : ".") +
                             "/../Resources/lib/libtoast_jit.dylib";
#  else
    const std::string path = std::string(program_dir ? program_dir : ".") + "/lib/libtoast_jit.so";
#  endif
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char *err = dlerror();
      error = "This Blender has no built-in Toast compiler (" + std::string(err ? err : path) +
              "); set the Toast add-on to use an external compiler";
    }
    else {
      auto api = reinterpret_cast<ToastJitApiVersionFn>(dlsym(handle, "toast_jit_api_version"));
      jit_compile = reinterpret_cast<ToastJitCompileFn>(dlsym(handle, "toast_jit_compile"));
      jit_symbol = reinterpret_cast<ToastJitSymbolFn>(dlsym(handle, "toast_jit_symbol"));
      if (!api || api() != TOAST_JIT_API_VERSION || !jit_compile || !jit_symbol) {
        error = "The built-in Toast compiler doesn't match this Blender";
        handle = nullptr;
      }
    }
  }
  if (handle == nullptr) {
    r_error = error;
  }
  return handle;
#endif
}

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

  ToastAbiVersionFn abi_fn = nullptr;
  ToastSignatureFn sig_fn = nullptr;
  ToastKernelFn kernel_fn = nullptr;
  void *handle = nullptr;
  if (StringRef(path).endswith(".cpp")) {
    /* C++ source: compile it with the built-in compiler. */
    void *jit = load_jit(r_error);
    if (jit == nullptr) {
      return fail(r_error);
    }
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file) {
      return fail("Can't read kernel source " + std::string(path));
    }
    std::stringstream source;
    source << file.rdbuf();
    const std::string cache_path = std::string(path).substr(0, path.size() - 4);
    char error[4096] = "";
    void *module = jit_compile(source.str().c_str(), cache_path.c_str(), error, sizeof(error));
    if (module == nullptr) {
      return fail(std::string("Kernel compile failed: ") + error);
    }
    abi_fn = reinterpret_cast<ToastAbiVersionFn>(jit_symbol(module, "toast_abi_version"));
    sig_fn = reinterpret_cast<ToastSignatureFn>(jit_symbol(module, "toast_signature"));
    kernel_fn = reinterpret_cast<ToastKernelFn>(jit_symbol(module, "toast_kernel"));
    UNUSED_VARS(jit);
  }
  else {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char *err = dlerror();
      return fail(std::string("Can't load kernel: ") + (err ? err : path.c_str()));
    }
    abi_fn = reinterpret_cast<ToastAbiVersionFn>(dlsym(handle, "toast_abi_version"));
    sig_fn = reinterpret_cast<ToastSignatureFn>(dlsym(handle, "toast_signature"));
    kernel_fn = reinterpret_cast<ToastKernelFn>(dlsym(handle, "toast_kernel"));
  }
  auto close = [&]() {
    if (handle) {
      dlclose(handle);
    }
  };
  if (!abi_fn || !sig_fn || !kernel_fn) {
    close();
    return fail("Not a Toast kernel: " + std::string(path));
  }
  /* Read before any dlclose: the functions are gone once the library is closed. */
  const int32_t abi = abi_fn();
  if (abi != TOAST_ABI_VERSION) {
    close();
    return fail("Kernel built for Toast ABI " + std::to_string(abi) + ", this Blender uses " +
                std::to_string(TOAST_ABI_VERSION) + ". It is recompiled automatically when the "
                "Toast add-on is enabled");
  }
  std::optional<Signature> sig = parse_signature(sig_fn());
  if (!sig) {
    close();
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

enum class CmdKind : int8_t {
  AddPoint,
  AddPrim,
  RemovePoint,
  RemovePrim,
  SetAttr,
  RemoveCurve,
  AddCurve,
  ExtendCurve,
};

struct Cmd {
  CmdKind kind;
  int64_t src;
  /* AddPoint, ExtendCurve: position. */
  float3 position;
  /* RemovePoint: point. RemovePrim: face. SetAttr: element. RemoveCurve, ExtendCurve: curve.
   * AddPrim, AddCurve: range in #CmdBuffer::prim_points. */
  int64_t point = 0;
  IndexRange prim_points;
  /* SetAttr: binding index and the value's bytes in #CmdBuffer::values. */
  int32_t attr = 0;
  IndexRange value;
};

/* -------------------------------------------------------------------- */
/** \name Volume grids
 *
 * sample_grid() reads grids as they were before the run, with the harness's float arithmetic:
 * world to index with the inverted grid matrix (inverted in double, from the float matrix),
 * then trilinear over the 8 voxels around, x first, then y, then z, as `a + (b - a) * t` (just
 * `a` when t is 0, so a voxel center reads one voxel).
 * \{ */

struct RunWarnings;

/** A grid read by sample_grid(), as it was before the run. */
struct GridSlot {
#ifdef WITH_OPENVDB
  bke::VolumeTreeAccessToken token;
  const openvdb::GridBase *vdb = nullptr;
  /* A value accessor per thread (type-erased `GridT::ConstUnsafeAccessor`): it caches the last
   * leaf it visited, so reading neighbor voxels skips the walk down the tree. */
  std::unique_ptr<threading::EnumerableThreadSpecific<std::shared_ptr<void>>> accessors =
      std::make_unique<threading::EnumerableThreadSpecific<std::shared_ptr<void>>>();
#endif
  VolumeGridType type = VOLUME_GRID_UNKNOWN;
  bool vector = false;
  float4x4 to_index = float4x4::identity();
  float3 step = float3(1.0f);
  float3 background = float3(0.0f);
};

#ifdef WITH_OPENVDB

/** The grid's index-to-world matrix in float, like BKE_volume_transform_to_blender. */
static float4x4 grid_matrix(const openvdb::math::Transform &transform)
{
  const openvdb::math::Mat4d m = transform.baseMap()->getAffineMap()->getMat4();
  float4x4 result;
  for (const int col : IndexRange(4)) {
    for (const int row : IndexRange(4)) {
      result[col][row] = float(m(col, row));
    }
  }
  return result;
}

static bool same_matrix(const float4x4 &a, const float4x4 &b)
{
  for (const int col : IndexRange(4)) {
    for (const int row : IndexRange(4)) {
      if (a[col][row] != b[col][row]) {
        return false;
      }
    }
  }
  return true;
}

/** Grid types Toast reads and writes, by value type. */
static bool is_toast_grid_type(const VolumeGridType type)
{
  return ELEM(type,
              VOLUME_GRID_FLOAT,
              VOLUME_GRID_DOUBLE,
              VOLUME_GRID_INT,
              VOLUME_GRID_INT64,
              VOLUME_GRID_BOOLEAN,
              VOLUME_GRID_MASK,
              VOLUME_GRID_VECTOR_FLOAT,
              VOLUME_GRID_VECTOR_DOUBLE,
              VOLUME_GRID_VECTOR_INT);
}

static bool is_vector_grid_type(const VolumeGridType type)
{
  return ELEM(type, VOLUME_GRID_VECTOR_FLOAT, VOLUME_GRID_VECTOR_DOUBLE, VOLUME_GRID_VECTOR_INT);
}

/** Calls `fn.template operator()<GridT>()` for a grid type Toast handles. */
template<typename Fn> static void with_toast_grid_type(const VolumeGridType type, Fn &&fn)
{
  switch (type) {
    case VOLUME_GRID_FLOAT:
      return fn.template operator()<openvdb::FloatGrid>();
    case VOLUME_GRID_DOUBLE:
      return fn.template operator()<openvdb::DoubleGrid>();
    case VOLUME_GRID_INT:
      return fn.template operator()<openvdb::Int32Grid>();
    case VOLUME_GRID_INT64:
      return fn.template operator()<openvdb::Int64Grid>();
    case VOLUME_GRID_BOOLEAN:
      return fn.template operator()<openvdb::BoolGrid>();
    case VOLUME_GRID_MASK:
      return fn.template operator()<openvdb::MaskGrid>();
    case VOLUME_GRID_VECTOR_FLOAT:
      return fn.template operator()<openvdb::Vec3fGrid>();
    case VOLUME_GRID_VECTOR_DOUBLE:
      return fn.template operator()<openvdb::Vec3dGrid>();
    case VOLUME_GRID_VECTOR_INT:
      return fn.template operator()<openvdb::Vec3IGrid>();
    default:
      break;
  }
}

/** A voxel value as up to three floats (bools and masks are 0 or 1). */
template<typename T> static float3 voxel_to_float3(const T &value)
{
  if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, openvdb::ValueMask>) {
    return float3(bool(value) ? 1.0f : 0.0f, 0.0f, 0.0f);
  }
  else if constexpr (openvdb::VecTraits<T>::IsVec) {
    return float3(float(value[0]), float(value[1]), float(value[2]));
  }
  else {
    return float3(float(value), 0.0f, 0.0f);
  }
}

template<typename GridT>
static typename GridT::ConstUnsafeAccessor &slot_accessor(const GridSlot &slot)
{
  using Accessor = typename GridT::ConstUnsafeAccessor;
  std::shared_ptr<void> &local = slot.accessors->local();
  if (!local) {
    local = std::make_shared<Accessor>(
        static_cast<const GridT &>(*slot.vdb).getConstUnsafeAccessor());
  }
  return *static_cast<Accessor *>(local.get());
}

template<typename GridT> static float3 sample_typed(const GridSlot &slot, const float3 &p)
{
  auto &accessor = slot_accessor<GridT>(slot);
  const float4x4 &m = slot.to_index;
  const float x = m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0];
  const float y = m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1];
  const float z = m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2];
  if (!(std::abs(x) < 1e9f && std::abs(y) < 1e9f && std::abs(z) < 1e9f)) {
    return slot.background;
  }
  const float x0 = std::floor(x), y0 = std::floor(y), z0 = std::floor(z);
  const float fx = x - x0, fy = y - y0, fz = z - z0;
  const int i = int(x0), j = int(y0), k = int(z0);
  auto v = [&](const int a, const int b, const int c) {
    return voxel_to_float3(accessor.getValue(openvdb::Coord(a, b, c)));
  };
  /* A lerp with t == 0 is its first value, without reading the second (a voxel center reads one
   * voxel). */
  auto along_x = [&](const int b, const int c) {
    const float3 a = v(i, b, c);
    return fx == 0.0f ? a : a + (v(i + 1, b, c) - a) * fx;
  };
  auto along_y = [&](const int c) {
    const float3 a = along_x(j, c);
    return fy == 0.0f ? a : a + (along_x(j + 1, c) - a) * fy;
  };
  const float3 c0 = along_y(k);
  return fz == 0.0f ? c0 : c0 + (along_y(k + 1) - c0) * fz;
}

static float3 sample_slot(const GridSlot &slot, const float3 &p)
{
  float3 result(0.0f);
  with_toast_grid_type(slot.type, [&]<typename GridT>() { result = sample_typed<GridT>(slot, p); });
  return result;
}

#endif /* WITH_OPENVDB */

static int32_t host_grid_sample(
    void *self, const int32_t grid, const int32_t mode, const float *pos3, float *out);

/** activate_voxel()/deactivate_voxel() calls of a Run Over Voxels run, per thread (sets: the
 * order doesn't matter). Activations are (i, j, k, radius). */
struct VoxelCommands {
  threading::EnumerableThreadSpecific<Vector<int4>> activate;
  threading::EnumerableThreadSpecific<Vector<int3>> deactivate;
};

/**
 * The grids of the signature, found in the volumes of the inputs (null for none). Missing grids
 * and grids of another type read as zero, with a warning.
 */
static void find_grid_slots(const Signature &sig,
                            const std::array<const Volume *, MAX_INPUTS> &volumes,
                            Array<GridSlot> &r_slots,
                            RunWarnings *warnings);

/** \} */

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
  std::array<std::shared_ptr<const struct NeighborTable>, MAX_INPUTS> neighbors;
  /* Bytes per element of each bound attribute, for cmd_set_attr. */
  Span<ToastAttr> bindings;
  /* Meshes for ray_cast/closest_point, as they were before the run. */
  std::array<const Mesh *, MAX_INPUTS> surface_meshes = {};
  /* surface_point(): cumulative triangle areas of each surface mesh, built on first use. */
  mutable std::array<std::once_flag, MAX_INPUTS> area_cdf_once;
  mutable std::array<Array<double>, MAX_INPUTS> area_cdf;
  /* For curve queries; null for inputs without curves. */
  std::array<const bke::CurvesGeometry *, MAX_INPUTS> curves = {};
  /* sample_grid() slots, by the signature's grids. */
  Span<GridSlot> grids;
  /* Run Over Voxels: where activate_voxel()/deactivate_voxel() go (null in other modes). */
  VoxelCommands *voxel_cmds = nullptr;
};

static void host_voxel_cmd(void *self, const int32_t op, const int32_t *ijk, const int32_t radius)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (host.voxel_cmds == nullptr) {
    return;
  }
  if (op == 1) {
    host.voxel_cmds->activate.local().append(int4(ijk[0], ijk[1], ijk[2], radius));
  }
  else {
    host.voxel_cmds->deactivate.local().append(int3(ijk[0], ijk[1], ijk[2]));
  }
}

static int32_t host_grid_sample(
    void *self, const int32_t grid, const int32_t mode, const float *pos3, float *out)
{
  out[0] = out[1] = out[2] = 0.0f;
#ifdef WITH_OPENVDB
  const HostData &host = *static_cast<const HostData *>(self);
  if (grid < 0 || grid >= host.grids.size() || host.grids[grid].vdb == nullptr) {
    return 0;
  }
  const GridSlot &slot = host.grids[grid];
  const float3 p(pos3[0], pos3[1], pos3[2]);
  if (mode == 3) {
    /* The nearest voxel's index. */
    const float4x4 &m = slot.to_index;
    const float x = m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0];
    const float y = m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1];
    const float z = m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2];
    if (!(std::abs(x) < 1e9f && std::abs(y) < 1e9f && std::abs(z) < 1e9f)) {
      return 0;
    }
    out[0] = std::floor(x + 0.5f);
    out[1] = std::floor(y + 0.5f);
    out[2] = std::floor(z + 0.5f);
    return 1;
  }
  if (mode == 2) {
    /* The voxel at index p, rounded: the trilinear read at a whole index reads only it. */
    float3 value = slot.background;
    if (std::abs(p.x) < 1e9f && std::abs(p.y) < 1e9f && std::abs(p.z) < 1e9f) {
      const int3 ijk(int(std::floor(p.x + 0.5f)), int(std::floor(p.y + 0.5f)),
                     int(std::floor(p.z + 0.5f)));
      with_toast_grid_type(slot.type, [&]<typename GridT>() {
        value = voxel_to_float3(
            slot_accessor<GridT>(slot).getValue(openvdb::Coord(ijk.x, ijk.y, ijk.z)));
      });
    }
    out[0] = value.x;
    if (slot.vector) {
      out[1] = value.y;
      out[2] = value.z;
    }
    return 1;
  }
  if (mode == 1) {
    if (slot.vector) {
      return 0;
    }
    for (const int a : IndexRange(3)) {
      const float h = slot.step[a];
      float3 hi = p, lo = p;
      hi[a] = p[a] + h;
      lo[a] = p[a] - h;
      out[a] = (sample_slot(slot, hi).x - sample_slot(slot, lo).x) / (h + h);
    }
    return 1;
  }
  const float3 value = sample_slot(slot, p);
  out[0] = value.x;
  if (slot.vector) {
    out[1] = value.y;
    out[2] = value.z;
  }
  return 1;
#else
  UNUSED_VARS(self, grid, mode, pos3);
  return 0;
#endif
}

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

/**
 * A point on the triangles of an input's mesh, area weighted, as the harness does it: `u` picks
 * the triangle by cumulative (twice) area, summed in double in triangle order; `v` and `w` are its
 * barycentric weights for the second and third corner, folded back when v + w > 1.
 */
static int32_t host_surface_point(
    void *self, const int32_t input, const float u, float v, float w, ToastHit *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.surface_meshes[input] == nullptr) {
    return 0;
  }
  const Mesh &mesh = *host.surface_meshes[input];
  const Span<int3> tris = mesh.corner_tris();
  const Span<int> corner_verts = mesh.corner_verts();
  const Span<float3> positions = mesh.vert_positions();
  std::call_once(host.area_cdf_once[input], [&]() {
    Array<double> cdf(tris.size());
    double total = 0.0;
    for (const int64_t k : tris.index_range()) {
      const float3 &a = positions[corner_verts[tris[k][0]]];
      const float3 &b = positions[corner_verts[tris[k][1]]];
      const float3 &c = positions[corner_verts[tris[k][2]]];
      const double e1x = double(b.x) - double(a.x), e1y = double(b.y) - double(a.y),
                   e1z = double(b.z) - double(a.z);
      const double e2x = double(c.x) - double(a.x), e2y = double(c.y) - double(a.y),
                   e2z = double(c.z) - double(a.z);
      const double cx = e1y * e2z - e1z * e2y;
      const double cy = e1z * e2x - e1x * e2z;
      const double cz = e1x * e2y - e1y * e2x;
      total += std::sqrt(cx * cx + cy * cy + cz * cz);
      cdf[k] = total;
    }
    host.area_cdf[input] = std::move(cdf);
  });
  const Span<double> cdf = host.area_cdf[input];
  if (cdf.is_empty() || !(cdf.last() > 0.0)) {
    return 0;
  }
  const double target = double(u) * cdf.last();
  const int tri = std::min<int>(
      std::upper_bound(cdf.begin(), cdf.end(), target) - cdf.begin(), int(cdf.size()) - 1);
  if (v + w > 1.0f) {
    v = 1.0f - v;
    w = 1.0f - w;
  }
  const float b0 = 1.0f - v - w;
  const float3 &a = positions[corner_verts[tris[tri][0]]];
  const float3 &b = positions[corner_verts[tris[tri][1]]];
  const float3 &c = positions[corner_verts[tris[tri][2]]];
  const float3 position = a * b0 + b * v + c * w;
  return fill_hit(mesh, tri, position, float3(b0, v, w), 0.0f, out);
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
    case TOAST_TYPE_QUAT:
      return 16;
    case TOAST_TYPE_MAT4:
      return 64;
  }
  return 0;
}

/** Points sharing an edge with each point of a mesh, ascending: for point_neighbors(). */
struct NeighborTable {
  Array<int> offsets;
  Array<int> neighbors;
};

static std::shared_ptr<const NeighborTable> build_neighbor_table(const Mesh &mesh)
{
  auto table = std::make_shared<NeighborTable>();
  const Span<int2> edges = mesh.edges();
  Array<int> edge_indices;
  bke::mesh::build_vert_to_edge_map(edges, mesh.verts_num, table->offsets, edge_indices);
  const OffsetIndices<int> offsets(table->offsets);
  table->neighbors.reinitialize(edge_indices.size());
  threading::parallel_for(IndexRange(mesh.verts_num), 2048, [&](const IndexRange range) {
    for (const int v : range) {
      const IndexRange group = offsets[v];
      MutableSpan<int> out = table->neighbors.as_mutable_span().slice(group);
      for (const int k : group.index_range()) {
        const int2 e = edges[edge_indices[group[k]]];
        out[k] = e[0] == v ? e[1] : e[0];
      }
      std::sort(out.begin(), out.end());
    }
  });
  return table;
}

/**
 * Neighbor tables of recent meshes, by their edges. Code that runs again on the same topology
 * (every step of a Repeat Zone, every frame of a simulation that only moves points) reuses the
 * table. Entries hold a weak user of the edges' sharing info: it doesn't keep the edges alive,
 * and an entry matches only while they exist and haven't changed (same version).
 */
static std::shared_ptr<const NeighborTable> neighbor_table(const Mesh &mesh)
{
  const bke::GAttributeReader edges = mesh.attributes().lookup(".edge_verts");
  if (!edges || edges.sharing_info == nullptr) {
    return build_neighbor_table(mesh);
  }
  struct Entry {
    const ImplicitSharingInfo *edges = nullptr;
    int64_t version = 0;
    int verts_num = 0;
    std::shared_ptr<const NeighborTable> table;
    Entry(const ImplicitSharingInfo *edges,
          const int64_t version,
          const int verts_num,
          std::shared_ptr<const NeighborTable> table)
        : edges(edges), version(version), verts_num(verts_num), table(std::move(table))
    {
      edges->add_weak_user();
    }
    Entry(Entry &&other) noexcept
        : edges(std::exchange(other.edges, nullptr)),
          version(other.version),
          verts_num(other.verts_num),
          table(std::move(other.table))
    {
    }
    Entry &operator=(Entry &&other) noexcept
    {
      if (this != &other) {
        this->~Entry();
        new (this) Entry(std::move(other));
      }
      return *this;
    }
    ~Entry()
    {
      if (edges) {
        edges->remove_weak_user_and_delete_if_last();
      }
    }
    bool matches(const ImplicitSharingInfo *info, const int verts) const
    {
      return edges == info && !edges->is_expired() && edges->version() == version &&
             verts_num == verts;
    }
  };
  static std::mutex mutex;
  static Vector<Entry> cache; /* Most recently used last. */
  constexpr int64_t cache_size = 4;
  {
    std::lock_guard lock{mutex};
    for (const int64_t i : cache.index_range()) {
      if (cache[i].matches(edges.sharing_info, mesh.verts_num)) {
        Entry entry = std::move(cache[i]);
        cache.remove(i);
        std::shared_ptr<const NeighborTable> table = entry.table;
        cache.append(std::move(entry));
        return table;
      }
    }
  }
  const int64_t version = edges.sharing_info->version();
  std::shared_ptr<const NeighborTable> table = build_neighbor_table(mesh);
  std::lock_guard lock{mutex};
  if (cache.size() >= cache_size) {
    cache.remove(0);
  }
  cache.append(Entry(edges.sharing_info, version, mesh.verts_num, table));
  return table;
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
  if (input < 0 || input >= MAX_INPUTS || !host.neighbors[input]) {
    return 0;
  }
  const NeighborTable &table = *host.neighbors[input];
  if (point < 0 || point >= table.offsets.size() - 1) {
    return 0;
  }
  return write_indices(table.neighbors.as_span().slice(OffsetIndices<int>(table.offsets)[point]),
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
 * A set_point_attr() on a point added in this run: a handle from add_point(), or an index from
 * the input's point count on (the new points in merged order).
 */
static bool is_new_point_set(const Cmd &cmd, Span<ToastAttr> bindings, Span<SigAttr> sig_attrs)
{
  return sig_attrs[cmd.attr].domain == AttrDomain::Point &&
         (cmd.point < 0 || cmd.point >= bindings[cmd.attr].size);
}

/**
 * Apply set_point_attr()/set_face_attr() into the bound arrays, in (source index, emission order):
 * chunks cover increasing ranges, so that is chunk order. Done before topology changes. Writes to
 * points added in this run wait for the points, see #apply_new_point_sets.
 */
static bool apply_sets(Span<CmdBuffer> buffers,
                       Span<ToastAttr> bindings,
                       Span<SigAttr> sig_attrs,
                       std::string &r_error)
{
  for (const CmdBuffer &buffer : buffers) {
    for (const Cmd &cmd : buffer.cmds) {
      if (cmd.kind != CmdKind::SetAttr || is_new_point_set(cmd, bindings, sig_attrs)) {
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

static int64_t host_ncurves(void *self, const int32_t input)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.curves[input] == nullptr) {
    return 0;
  }
  return host.curves[input]->curves_num();
}

static int32_t host_curve_points(
    void *self, const int32_t input, const int64_t curve, const int32_t max, int64_t *out)
{
  const HostData &host = *static_cast<const HostData *>(self);
  if (input < 0 || input >= MAX_INPUTS || host.curves[input] == nullptr) {
    return 0;
  }
  const bke::CurvesGeometry &curves = *host.curves[input];
  if (curve < 0 || curve >= curves.curves_num()) {
    return 0;
  }
  const IndexRange points = curves.points_by_curve()[curve];
  const int64_t n = std::min<int64_t>(points.size(), std::max(max, 0));
  for (const int64_t k : IndexRange(n)) {
    out[k] = points[k];
  }
  return int32_t(points.size());
}

static void host_remove_curve(void * /*self*/,
                              ToastCmdBuffer *cmd,
                              const int64_t src,
                              const int64_t curve)
{
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  Cmd c{CmdKind::RemoveCurve, src, float3(0.0f)};
  c.point = curve;
  buffer.cmds.append(c);
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

static void host_add_curve(
    void * /*self*/, ToastCmdBuffer *cmd, const int64_t src, const int64_t *pts, const int32_t n)
{
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  const IndexRange range(buffer.prim_points.size(), std::max(n, 0));
  buffer.prim_points.extend(Span<int64_t>(pts, range.size()));
  Cmd c{CmdKind::AddCurve, src, float3(0.0f)};
  c.prim_points = range;
  buffer.cmds.append(c);
}

static int64_t host_extend_curve(void * /*self*/,
                                 ToastCmdBuffer *cmd,
                                 const int64_t src,
                                 const int64_t curve,
                                 const float *pos3)
{
  CmdBuffer &buffer = *reinterpret_cast<CmdBuffer *>(cmd);
  /* A new point, numbered with add_point()'s. */
  const int64_t handle = -((buffer.chunk_id << 32) | buffer.added) - 1;
  buffer.added++;
  Cmd c{CmdKind::ExtendCurve, src, float3(pos3[0], pos3[1], pos3[2])};
  c.point = curve;
  buffer.cmds.append(c);
  return handle;
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
    case TOAST_TYPE_QUAT:
      return AttrType::Quaternion;
    case TOAST_TYPE_MAT4:
      return AttrType::Float4x4;
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
 * active color attribute, and `transform` on instances, which is `instance_transform`. `mesh` is
 * the mesh being read or written, if any.
 */
static std::string blender_name(const StringRef name, const Mesh *mesh, const AttrDomain domain)
{
  if (name == "color") {
    return color_attribute_name(mesh);
  }
  if (name == "transform" && domain == AttrDomain::Instance) {
    return "instance_transform";
  }
  return name;
}

/** Which component of a geometry "points" refer to. */
enum class PointsKind : int8_t { None, Mesh, PointCloud, Curves, Instances };

/**
 * The parts of one geometry the kernel reads. Points are those of `points_kind`; faces, edges
 * and corners are the mesh's; curves and instances their components'.
 */
struct PointsSource {
  const Mesh *mesh = nullptr;
  const PointCloud *pointcloud = nullptr;
  const Curves *curves = nullptr;
  const bke::Instances *instances = nullptr;
  PointsKind points_kind = PointsKind::None;

  int64_t size() const
  {
    switch (points_kind) {
      case PointsKind::Mesh:
        return mesh->verts_num;
      case PointsKind::PointCloud:
        return pointcloud->totpoint;
      case PointsKind::Curves:
        return curves->geometry.point_num;
      case PointsKind::Instances:
        return instances->instances_num();
      case PointsKind::None:
        break;
    }
    return 0;
  }
  /** Attributes of a domain, from the component that has it. */
  std::optional<bke::AttributeAccessor> attributes(const AttrDomain domain) const
  {
    switch (domain) {
      case AttrDomain::Point:
        switch (points_kind) {
          case PointsKind::Mesh:
            return mesh->attributes();
          case PointsKind::PointCloud:
            return pointcloud->attributes();
          case PointsKind::Curves:
            return curves->geometry.wrap().attributes();
          default:
            return std::nullopt;
        }
      case AttrDomain::Curve:
        return curves ? std::optional(curves->geometry.wrap().attributes()) : std::nullopt;
      case AttrDomain::Instance:
        return instances ? std::optional(instances->attributes()) : std::nullopt;
      default:
        return mesh ? std::optional(mesh->attributes()) : std::nullopt;
    }
  }
  /** Element count of a domain. */
  int64_t size(const AttrDomain domain) const
  {
    if (domain == AttrDomain::Point) {
      return this->size();
    }
    if (domain == AttrDomain::Instance) {
      return instances ? instances->instances_num() : 0;
    }
    const std::optional<bke::AttributeAccessor> attributes = this->attributes(domain);
    return attributes && attributes->domain_supported(domain) ? attributes->domain_size(domain) :
                                                                0;
  }
  /** Positions of the points (for instances: their transforms' translation). */
  VArray<float3> positions() const
  {
    if (points_kind == PointsKind::Instances) {
      return bke::instance_position_varray(*instances);
    }
    const std::optional<bke::AttributeAccessor> attributes = this->attributes(AttrDomain::Point);
    if (!attributes) {
      return {};
    }
    return *attributes->lookup_or_default<float3>("position", AttrDomain::Point, float3(0.0f));
  }
};

/** Input 1: every component, with points from the mesh, else the point cloud, curves or instances. */
static PointsSource points_of(const GeometrySet &geometry)
{
  PointsSource source;
  source.mesh = geometry.get_mesh();
  source.pointcloud = geometry.get_pointcloud();
  source.curves = geometry.get_curves();
  source.instances = geometry.get_instances();
  if (source.mesh && source.mesh->verts_num > 0) {
    source.points_kind = PointsKind::Mesh;
  }
  else if (source.pointcloud && source.pointcloud->totpoint > 0) {
    source.points_kind = PointsKind::PointCloud;
  }
  else if (source.curves && source.curves->geometry.point_num > 0) {
    source.points_kind = PointsKind::Curves;
  }
  else if (source.instances && source.instances->instances_num() > 0) {
    source.points_kind = PointsKind::Instances;
  }
  else if (source.mesh) {
    source.points_kind = PointsKind::Mesh;
  }
  return source;
}

/** Input 0: only the component the kernel runs on. */
static PointsSource component_source(const GeometrySet &geometry,
                                     const bke::GeometryComponent::Type type)
{
  PointsSource source;
  switch (type) {
    case bke::GeometryComponent::Type::Mesh:
      source.mesh = geometry.get_mesh();
      source.points_kind = source.mesh ? PointsKind::Mesh : PointsKind::None;
      break;
    case bke::GeometryComponent::Type::PointCloud:
      source.pointcloud = geometry.get_pointcloud();
      source.points_kind = source.pointcloud ? PointsKind::PointCloud : PointsKind::None;
      break;
    case bke::GeometryComponent::Type::Curve:
      source.curves = geometry.get_curves();
      source.points_kind = source.curves ? PointsKind::Curves : PointsKind::None;
      break;
    case bke::GeometryComponent::Type::Instance:
      source.instances = geometry.get_instances();
      source.points_kind = source.instances ? PointsKind::Instances : PointsKind::None;
      break;
    default:
      break;
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
/** Messages that don't stop the node (shown as warnings), collected from every component. */
struct RunWarnings {
  std::mutex mutex;
  VectorSet<std::string> messages;
  void add(std::string message)
  {
    std::lock_guard lock{mutex};
    messages.add(std::move(message));
  }
};

/** A Blender attribute type in Toast's words, and how many numbers it holds. */
static std::pair<StringRefNull, int> toast_type_info(const AttrType type)
{
  switch (type) {
    case AttrType::Bool:
      return {"bool", 1};
    case AttrType::Int8:
    case AttrType::Int32:
      return {"int", 1};
    case AttrType::Float:
      return {"float", 1};
    case AttrType::Int16_2D:
    case AttrType::Int32_2D:
      return {"2D integer vector", 2};
    case AttrType::Float2:
      return {"vec2", 2};
    case AttrType::Float3:
      return {"vec3", 3};
    case AttrType::Float4:
    case AttrType::ColorFloat:
    case AttrType::ColorByte:
      return {"vec4", 4};
    case AttrType::Quaternion:
      return {"quat", 4};
    case AttrType::Float4x4:
      return {"mat4", 16};
    case AttrType::String:
      return {"string", 0};
  }
  return {"?", 0};
}

/**
 * Reading an attribute as another type converts it silently: a vec3 read as a float (the type of
 * an attribute the code doesn't annotate) is wrong without any error. Warn when the conversion
 * loses data: another number of components, or a float read as an int or bool.
 */
static std::optional<std::string> read_type_warning(const SigAttr &attr,
                                                    const AttrType code_type,
                                                    const AttrType geometry_type,
                                                    const std::optional<AttrDomain> run_domain)
{
  if (attr.name == "color" || code_type == geometry_type) {
    return std::nullopt; /* @color is the color attribute's RGB, by design. */
  }
  const auto [code_name, code_size] = toast_type_info(code_type);
  const auto [geo_name, geo_size] = toast_type_info(geometry_type);
  const bool truncates = geometry_type == AttrType::Float &&
                         ELEM(code_type, AttrType::Int32, AttrType::Bool);
  if (code_size == geo_size && !truncates) {
    return std::nullopt;
  }
  std::string message = "\"" + attr.name + "\" is a " + std::string(geo_name) + " on " +
                        (attr.input == 0 ? "the geometry" : "Input 1") +
                        ", but the code reads it as a " + std::string(code_name) + ". ";
  if (geo_size == 0 || geo_name == StringRef("2D integer vector")) {
    return message + "Toast can't read this type";
  }
  if (attr.input == 0 && run_domain && attr.domain == *run_domain) {
    return message + "Give it its type: @" + attr.name + ": " + std::string(geo_name);
  }
  return message + "Give it its type: x: " + std::string(geo_name) + " = ..._attr(" +
         std::to_string(attr.input) + ", \"" + attr.name + "\", ...)";
}

static bool read_attribute(const PointsSource &source,
                           const SigAttr &attr,
                           const AttrType type,
                           const bool allow_view,
                           ReadBinding &r_binding,
                           std::string &r_error,
                           RunWarnings *warnings = nullptr,
                           const std::optional<AttrDomain> run_domain = std::nullopt)
{
  const CPPType &cpp_type = bke::attribute_type_to_cpp_type(type);
  const AttrDomain domain = attr.domain;
  const int64_t size = source.size(domain);
  std::optional<bke::AttributeAccessor> attributes = source.attributes(domain);
  const std::string name = blender_name(attr.name, source.mesh, domain);
  if (attr.name == "position" && domain == AttrDomain::Instance && source.instances &&
      type == AttrType::Float3)
  {
    r_binding.copy = GArray<>(cpp_type, size);
    array_utils::copy(bke::instance_position_varray(*source.instances),
                      r_binding.copy.as_mutable_span().typed<float3>());
    r_binding.data = r_binding.copy.data();
    return true;
  }
  if (attributes && attributes->contains(name)) {
    if (warnings) {
      if (const std::optional<bke::AttributeMetaData> meta = attributes->lookup_meta_data(name)) {
        if (std::optional<std::string> warning = read_type_warning(
                attr, type, meta->data_type, run_domain))
        {
          warnings->add(std::move(*warning));
        }
      }
    }
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

/** set_point_attr() on a point added in this run: binding `attr`, bytes in #MergedCommands. */
struct NewPointSet {
  int point;
  int attr;
  IndexRange value;
};

struct MergedCommands {
  Vector<float3> new_positions;
  /* For each new point: the curve extend_curve() put it at the tip of, or -1 (add_point). */
  Vector<int> new_point_curves;
  Vector<Vector<int>> new_prims;
  Vector<Vector<int>> new_curves;
  Vector<int> removed;
  Vector<int> removed_prims;
  Vector<int> removed_curves;
  Vector<NewPointSet> new_point_sets;
  Vector<uint8_t> new_point_values;
  bool has_changes() const
  {
    return !new_positions.is_empty() || !new_prims.is_empty() || !removed.is_empty() ||
           !removed_prims.is_empty() || !removed_curves.is_empty() || !new_curves.is_empty();
  }
};

/**
 * Merge per-chunk buffers. Chunks cover increasing ranges of the mask and commands inside a
 * chunk are in element order, so chunk order is already (source index, emission order).
 */
static bool merge_commands(Span<CmdBuffer> buffers,
                           Span<ToastAttr> bindings,
                           Span<SigAttr> sig_attrs,
                           const int64_t points_num,
                           const int64_t prims_num,
                           const int64_t curves_num,
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
          r_merged.new_point_curves.append(-1);
          break;
        case CmdKind::ExtendCurve:
          if (curves_num == 0) {
            r_error = "extend_curve() needs curves, the code ran on a mesh or point cloud";
            return false;
          }
          if (cmd.point < 0 || cmd.point >= curves_num) {
            r_error = "extend_curve(): curve " + std::to_string(cmd.point) + " does not exist";
            return false;
          }
          r_merged.new_positions.append(cmd.position);
          r_merged.new_point_curves.append(int(cmd.point));
          break;
        case CmdKind::AddPrim:
        case CmdKind::AddCurve: {
          Vector<int> prim;
          for (const int64_t pt : buffer.prim_points.as_span().slice(cmd.prim_points)) {
            int index;
            if (!resolve(pt, index)) {
              return false;
            }
            prim.append(index);
          }
          if (cmd.kind == CmdKind::AddPrim) {
            r_merged.new_prims.append(std::move(prim));
          }
          else {
            r_merged.new_curves.append(std::move(prim));
          }
          break;
        }
        case CmdKind::SetAttr: {
          /* Others were applied before the merge, see apply_sets. */
          if (is_new_point_set(cmd, bindings, sig_attrs)) {
            int index;
            if (!resolve(cmd.point, index)) {
              return false;
            }
            const IndexRange value(r_merged.new_point_values.size(), cmd.value.size());
            r_merged.new_point_values.extend(buffer.values.as_span().slice(cmd.value));
            r_merged.new_point_sets.append({index, cmd.attr, value});
          }
          break;
        }
        case CmdKind::RemovePrim: {
          if (cmd.point < 0 || cmd.point >= prims_num) {
            r_error = "Face " + std::to_string(cmd.point) + " does not exist";
            return false;
          }
          r_merged.removed_prims.append(int(cmd.point));
          break;
        }
        case CmdKind::RemoveCurve: {
          if (curves_num == 0) {
            r_error = "remove_curve() needs curves, the code ran on a mesh or point cloud";
            return false;
          }
          if (cmd.point < 0 || cmd.point >= curves_num) {
            r_error = "Curve " + std::to_string(cmd.point) + " does not exist";
            return false;
          }
          r_merged.removed_curves.append(int(cmd.point));
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
  else {
    /* Nothing to copy, but keep the attributes: on an empty input (Run Over Geometry building
     * from nothing) they are the ones the code created, and set_point_attr() writes into. */
    src_attributes.foreach_attribute([&](const bke::AttributeIter &iter) {
      if (iter.domain != domain || iter.data_type == bke::AttrType::String ||
          attribute_filter.allow_skip(iter.name))
      {
        return;
      }
      dst_attributes.add(iter.name, domain, iter.data_type, bke::AttributeInitDefaultValue());
    });
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

/**
 * Write set_point_attr() values to points added in this run, once they exist and before
 * removals: in merged order, so the last write wins, like #apply_sets.
 */
static bool apply_new_point_sets(bke::MutableAttributeAccessor attributes,
                                 const MergedCommands &merged,
                                 Span<SigAttr> sig_attrs,
                                 std::string &r_error)
{
  if (merged.new_point_sets.is_empty()) {
    return true;
  }
  Map<int, bke::GSpanAttributeWriter> writers;
  bool ok = true;
  for (const NewPointSet &set : merged.new_point_sets) {
    const SigAttr &attr = sig_attrs[set.attr];
    bke::GSpanAttributeWriter &writer = writers.lookup_or_add_cb(
        set.attr, [&]() { return attributes.lookup_for_write_span(attr.name); });
    if (!writer || writer.domain != AttrDomain::Point ||
        writer.span.type().size != int64_t(set.value.size()) || set.point >= writer.span.size())
    {
      r_error = "set_point_attr(): '" + attr.name + "' can't be written on points added in this run";
      ok = false;
      break;
    }
    memcpy(writer.span[set.point], &merged.new_point_values[set.value.start()], set.value.size());
  }
  for (bke::GSpanAttributeWriter &writer : writers.values()) {
    if (writer) {
      writer.finish();
    }
  }
  return ok;
}

static bool apply_to_mesh(GeometrySet &geometry,
                          const MergedCommands &merged,
                          Span<SigAttr> sig_attrs,
                          const bke::AttributeFilter &attribute_filter,
                          std::string &r_error)
{
  const Mesh *mesh = geometry.get_mesh();
  if (!merged.new_positions.is_empty() || !merged.new_prims.is_empty()) {
    geometry.replace_mesh(mesh_add_elements(*mesh, merged, attribute_filter));
    mesh = geometry.get_mesh();
    if (!apply_new_point_sets(
            geometry.get_mesh_for_write()->attributes_for_write(), merged, sig_attrs, r_error))
    {
      return false;
    }
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
  return true;
}

static bool apply_to_pointcloud(GeometrySet &geometry,
                                const MergedCommands &merged,
                                Span<SigAttr> sig_attrs,
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
    if (!apply_new_point_sets(pointcloud->attributes_for_write(), merged, sig_attrs, r_error)) {
      return false;
    }
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

/**
 * Curves, as the harness does it (see `_merge_curves`). Points are numbered in merged order:
 * the input's, then add_point()/extend_curve() ones. extend_curve() points go at the tip of their
 * curve and copy the tip point's other attributes; add_point() points get default values and must
 * each be in exactly one add_curve() curve, which go after the input's curves. Then
 * set_point_attr() on new points, then removals. Each curve's points are stored together.
 */
static bool apply_to_curves(GeometrySet &geometry,
                            const MergedCommands &merged,
                            Span<SigAttr> sig_attrs,
                            const bke::AttributeFilter &attribute_filter,
                            std::string &r_error)
{
  if (!merged.new_prims.is_empty() || !merged.removed_prims.is_empty()) {
    r_error = "add_edge(), add_face() and remove_face() need a mesh, the code ran on curves";
    return false;
  }
  const Curves &src_id = *geometry.get_curves();
  const bke::CurvesGeometry &src = src_id.geometry.wrap();
  const OffsetIndices<int> src_points_by_curve = src.points_by_curve();
  const int old_points = src.points_num();
  const int old_curves = src.curves_num();
  const int new_num = merged.new_positions.size();
  const int all_points = old_points + new_num;

  /* Which points each new curve uses: add_point() ones exactly once, nothing else. */
  Array<int> used(new_num, 0);
  for (const Span<int> curve : merged.new_curves) {
    for (const int pt : curve) {
      if (pt < old_points || merged.new_point_curves[pt - old_points] != -1) {
        r_error = "add_curve(): point " + std::to_string(pt) + " is already in a curve";
        return false;
      }
      used[pt - old_points]++;
    }
  }
  for (const int k : IndexRange(new_num)) {
    if (merged.new_point_curves[k] == -1 && used[k] != 1) {
      r_error = "Point " + std::to_string(old_points + k) + " from add_point() is in " +
                std::to_string(used[k]) + " curves (add_curve), it must be in one";
      return false;
    }
  }

  /* Points in merged order, with their attributes, in a point cloud used as storage. */
  PointCloud *merged_points = bke::pointcloud_new_no_attributes(PointCloudType::Points, all_points);
  BLI_SCOPED_DEFER([&]() { BKE_id_free(nullptr, merged_points); });
  {
    Array<int> sources(all_points);
    array_utils::fill_index_range<int>(sources.as_mutable_span().take_front(old_points));
    Vector<int> defaults;
    for (const int k : IndexRange(new_num)) {
      const int curve = merged.new_point_curves[k];
      if (curve == -1 || src_points_by_curve[curve].is_empty()) {
        sources[old_points + k] = 0;
        defaults.append(old_points + k);
      }
      else {
        sources[old_points + k] = src_points_by_curve[curve].last();
      }
    }
    bke::MutableAttributeAccessor dst = merged_points->attributes_for_write();
    if (old_points > 0) {
      bke::gather_attributes(
          src.attributes(), AttrDomain::Point, AttrDomain::Point, attribute_filter, sources, dst);
    }
    else {
      src.attributes().foreach_attribute([&](const bke::AttributeIter &iter) {
        if (iter.domain != AttrDomain::Point || iter.data_type == bke::AttrType::String ||
            attribute_filter.allow_skip(iter.name))
        {
          return;
        }
        dst.add(iter.name, AttrDomain::Point, iter.data_type, bke::AttributeInitDefaultValue());
      });
    }
    if (!defaults.is_empty()) {
      IndexMaskMemory memory;
      const IndexMask mask = IndexMask::from_indices(defaults.as_span(), memory);
      dst.foreach_attribute([&](const bke::AttributeIter &iter) {
        if (iter.domain != AttrDomain::Point) {
          return;
        }
        bke::GSpanAttributeWriter writer = dst.lookup_for_write_span(iter.name);
        if (writer) {
          writer.span.type().value_initialize_indices(writer.span.data(), mask);
          writer.finish();
        }
      });
    }
    bke::SpanAttributeWriter<float3> positions = dst.lookup_or_add_for_write_span<float3>(
        "position", AttrDomain::Point);
    positions.span.drop_front(old_points).copy_from(merged.new_positions);
    positions.finish();
    if (!apply_new_point_sets(dst, merged, sig_attrs, r_error)) {
      return false;
    }
  }

  /* Curves: the input's with their new tip points, then the new ones; removals last. */
  Array<bool> keep_point(all_points, true);
  for (const int i : merged.removed) {
    keep_point[i] = false;
  }
  Array<bool> keep_curve(old_curves + merged.new_curves.size(), true);
  for (const int curve : merged.removed_curves) {
    keep_curve[curve] = false;
    keep_point.as_mutable_span().slice(src_points_by_curve[curve]).fill(false);
  }
  Array<Vector<int>> tips(old_curves);
  for (const int k : IndexRange(new_num)) {
    if (merged.new_point_curves[k] != -1) {
      tips[merged.new_point_curves[k]].append(old_points + k);
    }
  }
  Vector<int> point_order;
  Vector<int> curve_order; /* Kept curves: input curve, or old_curves + new curve. */
  Vector<int> offsets = {0};
  auto add_curve = [&](const int curve, auto &&points) {
    const int64_t before = point_order.size();
    for (const int pt : points) {
      if (keep_point[pt]) {
        point_order.append(pt);
      }
    }
    if (keep_curve[curve] && point_order.size() > before) {
      curve_order.append(curve);
      offsets.append(point_order.size());
    }
    else {
      point_order.resize(before);
    }
  };
  for (const int curve : IndexRange(old_curves)) {
    Vector<int> points;
    for (const int pt : src_points_by_curve[curve]) {
      points.append(pt);
    }
    points.extend(tips[curve]);
    add_curve(curve, points);
  }
  for (const int k : merged.new_curves.index_range()) {
    add_curve(old_curves + k, merged.new_curves[k].as_span());
  }

  bke::CurvesGeometry dst(point_order.size(), curve_order.size());
  dst.offsets_for_write().copy_from(offsets);
  bke::gather_attributes(merged_points->attributes(),
                         AttrDomain::Point,
                         AttrDomain::Point,
                         attribute_filter,
                         point_order.as_span(),
                         dst.attributes_for_write());
  /* Curve attributes: the input curves' values; defaults for new curves, which are poly. */
  {
    Vector<int> old_kept;
    for (const int curve : curve_order) {
      if (curve < old_curves) {
        old_kept.append(curve);
      }
    }
    const int added_kept = curve_order.size() - old_kept.size();
    IndexMaskMemory memory;
    const IndexMask old_mask = IndexMask::from_indices(old_kept.as_span(), memory);
    src.attributes().foreach_attribute([&](const bke::AttributeIter &iter) {
      if (iter.domain != AttrDomain::Curve || iter.data_type == bke::AttrType::String ||
          attribute_filter.allow_skip(iter.name))
      {
        return;
      }
      const GVArray values = *iter.get();
      bke::GSpanAttributeWriter writer = dst.attributes_for_write().lookup_or_add_for_write_span(
          iter.name, AttrDomain::Curve, iter.data_type);
      if (!writer) {
        return;
      }
      array_utils::gather(values, old_mask, writer.span.take_front(old_kept.size()));
      writer.span.type().value_initialize_n(writer.span.slice(old_kept.size(), added_kept).data(),
                                            added_kept);
      writer.finish();
    });
    if (added_kept > 0) {
      dst.curve_types_for_write().drop_front(old_kept.size()).fill(CURVE_TYPE_POLY);
      dst.update_curve_types();
    }
  }
  dst.tag_topology_changed();
  Curves *dst_id = bke::curves_new_nomain(std::move(dst));
  bke::curves_copy_parameters(src_id, *dst_id);
  geometry.replace_curves(dst_id);
  return true;
}

static bool apply_to_instances(GeometrySet &geometry,
                               const MergedCommands &merged,
                               const bke::AttributeFilter &attribute_filter,
                               std::string &r_error)
{
  if (!merged.new_positions.is_empty() || !merged.new_prims.is_empty() ||
      !merged.removed_prims.is_empty() || !merged.removed_curves.is_empty())
  {
    r_error = "Only remove_instance() changes instances";
    return false;
  }
  bke::Instances &instances = *geometry.get_instances_for_write();
  Array<bool> keep(instances.instances_num(), true);
  for (const int i : merged.removed) {
    keep[i] = false;
  }
  IndexMaskMemory memory;
  instances.remove(IndexMask::from_bools(keep, memory), attribute_filter);
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
  /** Anonymous attribute names of the output() sockets that are used, by output name. */
  Map<std::string, std::string> output_ids;
  RunWarnings *warnings = nullptr;
  /** Input 0's volume for sample_grid(), when it isn't the geometry's (Run Over Voxels). */
  const Volume *volume0 = nullptr;
  /** Run Over Voxels: collects activate_voxel()/deactivate_voxel(). */
  VoxelCommands *voxel_cmds = nullptr;
};

using ComponentType = bke::GeometryComponent::Type;

/**
 * Run the kernel over one component of a geometry: the points of its mesh, point cloud or curves,
 * the mesh's faces, edges or corners, the curves, the instances, or once for the whole geometry.
 */
static bool run_on_component(GeometrySet &geometry,
                             const ComponentType type,
                             const RunContext &run,
                             std::string &r_error)
{
  const Signature &sig = run.kernel->signature;
  const std::optional<AttrDomain> domain = run_domain(sig.run_over);
  const bool is_once = sig.run_over == "geometry";
  const bool is_mesh = type == ComponentType::Mesh;
  const bool type_is_curves = type == ComponentType::Curve;

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
  const PointsSource input0_before = component_source(geometry, type);
  /* Surface queries on input 0 see it as it was before the run: a shared copy makes getting the
   * geometry for write below copy the mesh instead of changing this one. */
  std::optional<GeometrySet> input0_snapshot;
  if (sig.surface_inputs.contains(0)) {
    input0_snapshot = geometry;
  }
  const int64_t points_num = input0_before.size();
  const int64_t prims_num = input0_before.mesh ? input0_before.mesh->faces_num : 0;
  const int64_t curves_num = input0_before.curves ? input0_before.curves->geometry.curve_num : 0;
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
    switch (type) {
      case ComponentType::Mesh:
        evaluate(bke::MeshFieldContext(*input0_before.mesh, *domain));
        break;
      case ComponentType::PointCloud:
        evaluate(bke::PointCloudFieldContext(*input0_before.pointcloud));
        break;
      case ComponentType::Curve:
        evaluate(bke::CurvesFieldContext(*input0_before.curves, *domain));
        break;
      case ComponentType::Instance:
        evaluate(bke::InstancesFieldContext(*input0_before.instances));
        break;
      default:
        return true;
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
      if (!read_attribute(input0_before,
                          attr,
                          *attr_type_from_code(attr.type_code),
                          false,
                          reads[i],
                          r_error,
                          run.warnings,
                          domain))
      {
        return false;
      }
    }
  }

  HostData host_data;
  host_data.npoints[0] = points_num;
  host_data.npoints[1] = input1_source.size();
  host_data.meshes[1] = input1_source.mesh;
  std::array<VArray<float3>, MAX_INPUTS> tree_readers;
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
    VArray<float3> positions = source.positions();
    if (input != 0 && positions.is_span()) {
      tree_positions[input] = positions.get_internal_span();
      tree_readers[input] = std::move(positions);
    }
    else {
      tree_copies[input] = Array<float3>(source.size());
      array_utils::copy(positions, tree_copies[input].as_mutable_span());
      tree_positions[input] = tree_copies[input];
    }
  }

  /* Writers on input 0. Getting the geometry for write un-shares it from Input 1. */
  Mesh *write_mesh = is_mesh ? geometry.get_mesh_for_write() : nullptr;
  bke::Instances *write_instances = type == ComponentType::Instance ?
                                        geometry.get_instances_for_write() :
                                        nullptr;
  bke::MutableAttributeAccessor attributes = [&]() {
    switch (type) {
      case ComponentType::PointCloud:
        return geometry.get_pointcloud_for_write()->attributes_for_write();
      case ComponentType::Curve:
        return geometry.get_curves_for_write()->geometry.wrap().attributes_for_write();
      case ComponentType::Instance:
        return write_instances->attributes_for_write();
      default:
        return write_mesh->attributes_for_write();
    }
  }();
  /* Instance positions are the translation of their transforms: the kernel writes a copy, set
   * into the transforms after the run (after `@transform`, so the position wins). */
  Array<float3, 0> instance_positions;
  Vector<bke::GSpanAttributeWriter> writers;
  /* `@color` is a vec3 in Toast and a float4 color in Blender: the kernel writes a vec3 copy that is
   * copied back into the color's RGB after the run, keeping alpha. */
  struct ColorWriteBack {
    int64_t writer;
    Array<float3, 0> rgb; /* No inline buffer: the kernel keeps a pointer across moves. */
  };
  Vector<ColorWriteBack> color_writebacks;
  /* Outputs nothing reads still need memory for the kernel to write to. */
  Vector<GArray<>> unused_outputs;
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
    if (attr.output) {
      const int64_t size = input0_before.size(attr.domain);
      const std::string *id = run.output_ids.lookup_ptr(attr.name);
      if (id == nullptr) {
        const CPPType &cpp_type = bke::attribute_type_to_cpp_type(type);
        unused_outputs.append_as(cpp_type, size);
        GArray<> &scratch = unused_outputs.last();
        cpp_type.value_initialize_n(scratch.data(), size);
        bindings[i] = {scratch.data(), size, attr.type_code, abi_domain(attr.domain)};
        continue;
      }
      bke::GSpanAttributeWriter writer = attributes.lookup_or_add_for_write_span(
          *id, attr.domain, type);
      if (!writer) {
        r_error = "Can't store output \"" + attr.name + "\"";
        finish_writers();
        return false;
      }
      bindings[i] = {writer.span.data(), writer.span.size(), attr.type_code, abi_domain(attr.domain)};
      writers.append(std::move(writer));
      continue;
    }
    const std::string name = blender_name(attr.name, write_mesh, attr.domain);
    const bool is_color = attr.name == "color" && type == AttrType::Float3;
    if (name == "position" && attr.domain == AttrDomain::Instance && write_instances &&
        type == AttrType::Float3)
    {
      instance_positions.reinitialize(write_instances->instances_num());
      array_utils::copy(bke::instance_position_varray(*write_instances),
                        instance_positions.as_mutable_span());
      bindings[i] = {
          instance_positions.data(), instance_positions.size(), attr.type_code, TOAST_DOMAIN_INSTANCE};
      continue;
    }
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
    /* Grease Pencil's colors (strokes run as curves) are float colors, as Blender makes them. */
    const bool is_stroke_color = !meta && type == AttrType::Float4 &&
                                 type_is_curves && ELEM(name, "vertex_color", "fill_color");
    bke::GSpanAttributeWriter writer = attributes.lookup_or_add_for_write_span(
        name, attr.domain, (is_color || is_stroke_color) ? AttrType::ColorFloat : type);
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
  const PointsSource input0 = component_source(geometry, type);
  host_data.meshes[0] = input0.mesh;
  host_data.curves[0] = input0.curves ? &input0.curves->geometry.wrap() : nullptr;
  host_data.curves[1] = input1_source.curves ? &input1_source.curves->geometry.wrap() : nullptr;
  for (const int64_t i : sig.attrs.index_range()) {
    const SigAttr &attr = sig.attrs[i];
    if (is_written(attr)) {
      continue;
    }
    if (!(attr.input == 0 && attr.snapshot)) {
      const PointsSource &source = attr.input == 0 ? input0 : input1_source;
      if (!read_attribute(source,
                          attr,
                          *attr_type_from_code(attr.type_code),
                          true,
                          reads[i],
                          r_error,
                          run.warnings,
                          domain))
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
  host_data.voxel_cmds = run.voxel_cmds;
  Array<GridSlot> grid_slots;
  if (!sig.grids.is_empty()) {
    find_grid_slots(sig,
                    {run.volume0 ? run.volume0 : geometry.get_volume(),
                     run.input1 ? run.input1->get_volume() : nullptr},
                    grid_slots,
                    run.warnings);
    host_data.grids = grid_slots;
  }
  if (input0_snapshot) {
    host_data.surface_meshes[0] = input0_snapshot->get_mesh();
  }
  host_data.surface_meshes[1] = input1_source.mesh;
  if (sig.uses_neighbors) {
    for (const int input : IndexRange(MAX_INPUTS)) {
      if (host_data.meshes[input]) {
        host_data.neighbors[input] = neighbor_table(*host_data.meshes[input]);
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
                    host_closest_point,
                    host_ncurves,
                    host_curve_points,
                    host_remove_curve,
                    host_add_curve,
                    host_extend_curve,
                    host_surface_point,
                    host_grid_sample,
                    host_voxel_cmd};

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

  if (sig.uses_cmd && !apply_sets(buffers, bindings, sig.attrs, r_error)) {
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
  if (!instance_positions.is_empty()) {
    MutableSpan<float4x4> transforms = write_instances->transforms_for_write();
    threading::parallel_for(transforms.index_range(), 4096, [&](const IndexRange range) {
      for (const int64_t k : range) {
        transforms[k].location() = instance_positions[k];
      }
    });
  }

  if (!sig.uses_cmd) {
    return true;
  }
  MergedCommands merged;
  if (!merge_commands(
          buffers, bindings, sig.attrs, points_num, prims_num, curves_num, merged, r_error))
  {
    return false;
  }
  if (!merged.has_changes()) {
    return true;
  }
  switch (type) {
    case ComponentType::Curve:
      return apply_to_curves(geometry, merged, sig.attrs, run.attribute_filter, r_error);
    case ComponentType::Instance:
      return apply_to_instances(geometry, merged, run.attribute_filter, r_error);
    default:
      break;
  }
  if (!merged.removed_curves.is_empty()) {
    r_error = "remove_curve() needs curves, the code ran on a mesh or point cloud";
    return false;
  }
  if (!merged.new_curves.is_empty()) {
    r_error = "add_curve() needs curves, the code ran on a mesh or point cloud";
    return false;
  }
  if (is_mesh) {
    return apply_to_mesh(geometry, merged, sig.attrs, run.attribute_filter, r_error);
  }
  return apply_to_pointcloud(geometry, merged, sig.attrs, run.attribute_filter, r_error);
}

/**
 * Grease Pencil: each layer's strokes on the current frame are curves. They run like a Curves
 * component (Run Over Points and Curves, and Geometry when the code makes curves), one layer
 * after the other, and go back into the drawing.
 */
static bool run_on_grease_pencil(GeometrySet &geometry, const RunContext &run, std::string &r_error)
{
  GreasePencil *grease_pencil = geometry.get_grease_pencil_for_write();
  if (grease_pencil == nullptr) {
    return true;
  }
  for (const int layer_index : grease_pencil->layers().index_range()) {
    bke::greasepencil::Drawing *drawing = grease_pencil->get_eval_drawing(
        grease_pencil->layer(layer_index));
    if (drawing == nullptr) {
      continue;
    }
    GeometrySet strokes = GeometrySet::from_curves(
        bke::curves_new_nomain(std::move(drawing->strokes_for_write())));
    const bool ok = run_on_component(strokes, ComponentType::Curve, run, r_error);
    /* The strokes go back even after an error (then unchanged). */
    Curves *result = strokes.get_curves_for_write();
    drawing->strokes_for_write() = result ? std::move(result->geometry.wrap()) :
                                            bke::CurvesGeometry();
    drawing->tag_topology_changed();
    if (!ok) {
      return false;
    }
  }
  return true;
}

static void find_grid_slots(const Signature &sig,
                            const std::array<const Volume *, MAX_INPUTS> &volumes,
                            Array<GridSlot> &r_slots,
                            RunWarnings *warnings)
{
  r_slots.reinitialize(sig.grids.size());
#ifdef WITH_OPENVDB
  for (const int64_t i : sig.grids.index_range()) {
    const SigGrid &want = sig.grids[i];
    GridSlot &slot = r_slots[i];
    const std::string where = want.input == 0 ? "the geometry" : "Input " + std::to_string(want.input);
    const Volume *volume = want.input >= 0 && want.input < MAX_INPUTS ? volumes[want.input] :
                                                                         nullptr;
    const bke::VolumeGridData *grid = volume ? BKE_volume_grid_find(volume, want.name) : nullptr;
    if (grid == nullptr) {
      if (warnings) {
        warnings->add("sample_grid(): " + where + " has no grid \"" + want.name +
                      "\", it reads as zero");
      }
      continue;
    }
    const VolumeGridType type = grid->grid_type();
    if (!is_toast_grid_type(type) || (!want.any && is_vector_grid_type(type) != want.vector)) {
      if (warnings) {
        warnings->add("sample_grid(): grid \"" + want.name + "\" isn't a " +
                      (want.vector ? "vector" : "float") +
                      " grid, it reads as zero (annotate vector reads: v: vec3 = sample_grid(...))");
      }
      continue;
    }
    slot.type = type;
    slot.vector = want.vector;
    slot.vdb = &grid->grid(slot.token);
    const float4x4 matrix = grid_matrix(slot.vdb->transform());
    slot.to_index = float4x4(math::invert(double4x4(matrix)));
    for (const int a : IndexRange(3)) {
      slot.step[a] = std::sqrt(matrix[a][0] * matrix[a][0] + matrix[a][1] * matrix[a][1] +
                               matrix[a][2] * matrix[a][2]);
    }
    with_toast_grid_type(type, [&]<typename GridT>() {
      slot.background = voxel_to_float3(static_cast<const GridT &>(*slot.vdb).background());
    });
  }
#else
  UNUSED_VARS(volumes);
  if (warnings && !sig.grids.is_empty()) {
    warnings->add("sample_grid(): this Blender was built without OpenVDB, grids read as zero");
  }
#endif
}

#ifdef WITH_OPENVDB

/** Toast attribute types a grid can hold, and the grid type that holds each. */
static std::optional<VolumeGridType> grid_type_for(const AttrType type)
{
  switch (type) {
    case AttrType::Float:
      return VOLUME_GRID_FLOAT;
    case AttrType::Int32:
      return VOLUME_GRID_INT;
    case AttrType::Bool:
      return VOLUME_GRID_BOOLEAN;
    case AttrType::Float3:
      return VOLUME_GRID_VECTOR_FLOAT;
    default:
      return std::nullopt;
  }
}

/**
 * A voxel value converted to the attribute's type, the way numpy's assignment converts it. Vector
 * grids only go to vec3 attributes and scalar grids to scalar ones (checked before).
 */
template<typename AttrT, typename T> static AttrT voxel_to_attr(const T &value)
{
  if constexpr (std::is_same_v<AttrT, float3>) {
    return voxel_to_float3(value);
  }
  else if constexpr (openvdb::VecTraits<T>::IsVec) {
    return AttrT();
  }
  else if constexpr (std::is_same_v<T, openvdb::ValueMask>) {
    return AttrT(true);
  }
  else if constexpr (std::is_same_v<AttrT, bool>) {
    return value != T(0);
  }
  else {
    return AttrT(value);
  }
}

/** An attribute value as a voxel of the grid type made for that attribute type. */
template<typename ValueT, typename AttrT> static ValueT attr_to_voxel(const AttrT &value)
{
  if constexpr (std::is_same_v<AttrT, float3> && openvdb::VecTraits<ValueT>::IsVec) {
    return ValueT(value.x, value.y, value.z);
  }
  else if constexpr (std::is_same_v<AttrT, float3> || openvdb::VecTraits<ValueT>::IsVec ||
                     std::is_same_v<ValueT, openvdb::ValueMask>)
  {
    return ValueT();
  }
  else {
    return ValueT(value);
  }
}

/**
 * activate_voxel()/deactivate_voxel() on every grid of the volume with the run's transform:
 * deactivations first, of voxels nothing activates (set to the background, then inactive nodes
 * pruned), then activations (cubes of `radius` by dilation), as the harness's _apply_voxel_cmds.
 */
static void apply_voxel_commands(Volume &volume, const float4x4 &matrix, VoxelCommands &cmds)
{
  constexpr int max_radius = 16;
  Array<openvdb::MaskTree> by_radius(max_radius + 1);
  bool any = false;
  for (const Vector<int4> &list : cmds.activate) {
    for (const int4 &c : list) {
      const int r = std::clamp(c.w, 0, max_radius);
      by_radius[r].setValueOn(openvdb::Coord(c.x, c.y, c.z));
      any = true;
    }
  }
  openvdb::MaskTree activate;
  if (any) {
    for (const int r : by_radius.index_range()) {
      if (by_radius[r].empty()) {
        continue;
      }
      if (r > 0) {
        openvdb::tools::dilateActiveValues(by_radius[r], r, openvdb::tools::NN_FACE_EDGE_VERTEX);
      }
      activate.topologyUnion(by_radius[r]);
    }
  }
  openvdb::MaskTree deactivate;
  for (const Vector<int3> &list : cmds.deactivate) {
    for (const int3 &c : list) {
      deactivate.setValueOn(openvdb::Coord(c.x, c.y, c.z));
    }
  }
  deactivate.topologyDifference(activate);
  if (activate.empty() && deactivate.empty()) {
    return;
  }
  for (const int i : IndexRange(BKE_volume_num_grids(&volume))) {
    bke::VolumeGridData *grid = BKE_volume_grid_get_for_write(&volume, i);
    if (!is_toast_grid_type(grid->grid_type()) ||
        !same_matrix(grid_matrix(grid->transform()), matrix))
    {
      continue;
    }
    bke::VolumeTreeAccessToken token;
    openvdb::GridBase &base = grid->grid_for_write(token);
    with_toast_grid_type(grid->grid_type(), [&]<typename GridT>() {
      auto &tree = static_cast<GridT &>(base).tree();
      if (!deactivate.empty()) {
        auto accessor = tree.getAccessor();
        for (auto it = deactivate.cbeginValueOn(); it; ++it) {
          accessor.setValueOff(it.getCoord(), tree.background());
        }
        openvdb::tools::pruneInactive(tree);
      }
      if (!activate.empty()) {
        tree.topologyUnion(activate);
      }
    });
  }
}

/**
 * Run Over Voxels: the voxels active in any grid the code uses as an @ attribute (in all the
 * volume's grids when none of those exists yet), with tiles split into voxels, become the points
 * of a point cloud at the voxels' centers, ordered by leaf (origin x, y, z) and then by offset in
 * the leaf, as the harness orders them. The grids are its attributes; the code runs on it like
 * Run Over Points, and the attributes it writes go back into grids: an existing grid of the same
 * type and transform gains the voxels it lacked, any other gets replaced by a new grid on these
 * voxels. Grids whose transform differs from the first one's are left out, with a warning.
 */
static bool run_on_volume(GeometrySet &geometry, const RunContext &run, std::string &r_error)
{
  const Volume *volume = geometry.get_volume();
  if (volume == nullptr) {
    return true;
  }
  const Signature &sig = run.kernel->signature;
  Vector<std::string> used;
  for (const SigAttr &attr : sig.attrs) {
    if (attr.input == 0 && !attr.output && attr.name != "position" &&
        !StringRef(attr.name).startswith("__") && !used.contains(attr.name))
    {
      used.append(attr.name);
    }
  }
  Vector<const bke::VolumeGridData *> topology;
  for (const std::string &name : used) {
    if (const bke::VolumeGridData *grid = BKE_volume_grid_find(volume, name)) {
      topology.append(grid);
    }
  }
  if (topology.is_empty()) {
    for (const int i : IndexRange(BKE_volume_num_grids(volume))) {
      topology.append(BKE_volume_grid_get(volume, i));
    }
  }
  if (topology.is_empty()) {
    return true;
  }
  const openvdb::math::Transform::Ptr transform = topology[0]->transform().copy();
  const float4x4 matrix = grid_matrix(*transform);
  auto same_transform = [&](const bke::VolumeGridData &grid) {
    return same_matrix(grid_matrix(grid.transform()), matrix);
  };

  openvdb::MaskTree mask;
  for (const bke::VolumeGridData *grid : topology) {
    if (!same_transform(*grid)) {
      run.warnings->add("Run Over Voxels: grid \"" + grid->name() +
                        "\" has another voxel size or place than \"" + topology[0]->name() +
                        "\", it was left out");
      continue;
    }
    bke::VolumeTreeAccessToken token;
    const openvdb::GridBase &base = grid->grid(token);
    with_toast_grid_type(grid->grid_type(), [&]<typename GridT>() {
      mask.topologyUnion(static_cast<const GridT &>(base).tree());
    });
  }
  mask.voxelizeActiveTiles();

  using MaskLeaf = openvdb::MaskTree::LeafNodeType;
  Vector<const MaskLeaf *> leaves;
  for (auto it = mask.cbeginLeaf(); it; ++it) {
    leaves.append(it.getLeaf());
  }
  std::sort(leaves.begin(), leaves.end(), [](const MaskLeaf *a, const MaskLeaf *b) {
    return a->origin() < b->origin();
  });
  Array<int64_t> offsets(leaves.size() + 1);
  offsets[0] = 0;
  for (const int64_t i : leaves.index_range()) {
    offsets[i + 1] = offsets[i] + int64_t(leaves[i]->onVoxelCount());
  }
  const int64_t voxels_num = offsets.last();
  if (voxels_num == 0) {
    return true;
  }
  if (voxels_num > int64_t(std::numeric_limits<int>::max())) {
    r_error = "Run Over Voxels: too many voxels";
    return false;
  }
  auto foreach_leaf = [&](auto &&fn) {
    threading::parallel_for(leaves.index_range(), 16, [&](const IndexRange range) {
      for (const int64_t i : range) {
        fn(*leaves[i], offsets[i]);
      }
    });
  };

  PointCloud *pointcloud = BKE_pointcloud_new_nomain(PointCloudType::Points, int(voxels_num));
  {
    MutableSpan<float3> positions = pointcloud->positions_for_write();
    const float4x4 &t = matrix;
    foreach_leaf([&](const MaskLeaf &leaf, int64_t k) {
      for (auto it = leaf.cbeginValueOn(); it; ++it, ++k) {
        const openvdb::Coord c = it.getCoord();
        const float i = float(c.x()), j = float(c.y()), l = float(c.z());
        positions[k] = float3(t[0][0] * i + t[1][0] * j + t[2][0] * l + t[3][0],
                              t[0][1] * i + t[1][1] * j + t[2][1] * l + t[3][1],
                              t[0][2] * i + t[1][2] * j + t[2][2] * l + t[3][2]);
      }
    });
  }
  /* The grids' values as attributes, converted to the type the code uses them as. */
  bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
  /* @voxel and @voxel_size. */
  for (const SigAttr &attr : sig.attrs) {
    if (attr.input != 0 || attributes.contains(attr.name)) {
      continue;
    }
    if (attr.name == "__voxel") {
      bke::SpanAttributeWriter<float3> writer =
          attributes.lookup_or_add_for_write_only_span<float3>(attr.name, AttrDomain::Point);
      foreach_leaf([&](const MaskLeaf &leaf, int64_t k) {
        for (auto it = leaf.cbeginValueOn(); it; ++it, ++k) {
          const openvdb::Coord c = it.getCoord();
          writer.span[k] = float3(float(c.x()), float(c.y()), float(c.z()));
        }
      });
      writer.finish();
    }
    else if (attr.name == "__voxel_size") {
      float3 size;
      for (const int a : IndexRange(3)) {
        size[a] = std::sqrt(matrix[a][0] * matrix[a][0] + matrix[a][1] * matrix[a][1] +
                            matrix[a][2] * matrix[a][2]);
      }
      bke::SpanAttributeWriter<float3> writer =
          attributes.lookup_or_add_for_write_only_span<float3>(attr.name, AttrDomain::Point);
      writer.span.fill(size);
      writer.finish();
    }
  }
  /* grid_voxel() fast path: copies of grids on the voxels, read at each voxel's index in the
   * grid's own index space (as grid_voxel() reads), and face neighbor tables, as the harness. */
  std::optional<openvdb::Int32Tree> numbers;
  for (const SigAttr &attr : sig.attrs) {
    const StringRef name = attr.name;
    if (attr.input != 0 || attributes.contains(name)) {
      continue;
    }
    if (name.startswith("__grid_")) {
      const bool vector = attr.type_code == TOAST_TYPE_FLOAT3;
      bke::GSpanAttributeWriter writer = attributes.lookup_or_add_for_write_span(
          name, AttrDomain::Point, vector ? AttrType::Float3 : AttrType::Float);
      const bke::VolumeGridData *grid = BKE_volume_grid_find(volume, name.drop_prefix(7));
      if (grid && is_toast_grid_type(grid->grid_type()) &&
          is_vector_grid_type(grid->grid_type()) == vector)
      {
        bke::VolumeTreeAccessToken token;
        const openvdb::GridBase &base = grid->grid(token);
        with_toast_grid_type(grid->grid_type(), [&]<typename GridT>() {
          const GridT &typed = static_cast<const GridT &>(base);
          foreach_leaf([&](const MaskLeaf &leaf, int64_t k) {
            auto accessor = typed.getConstUnsafeAccessor();
            for (auto it = leaf.cbeginValueOn(); it; ++it, ++k) {
              const float3 v = voxel_to_float3(accessor.getValue(it.getCoord()));
              if (vector) {
                writer.span.typed<float3>()[k] = v;
              }
              else {
                writer.span.typed<float>()[k] = v.x;
              }
            }
          });
        });
      }
      writer.finish();
    }
    else if (name.size() == 4 && name.startswith("__n") && name[3] >= '0' && name[3] <= '5') {
      if (!numbers) {
        /* Each voxel's number, -1 elsewhere. */
        numbers.emplace(-1);
        numbers->topologyUnion(mask);
        foreach_leaf([&](const MaskLeaf &leaf, int64_t k) {
          openvdb::Int32Tree::LeafNodeType *target = numbers->probeLeaf(leaf.origin());
          for (auto it = leaf.cbeginValueOn(); it; ++it, ++k) {
            target->setValueOnly(it.pos(), int32_t(k));
          }
        });
      }
      static const int3 steps[6] = {
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      const int3 d = steps[name[3] - '0'];
      bke::SpanAttributeWriter<int> writer = attributes.lookup_or_add_for_write_only_span<int>(
          name, AttrDomain::Point);
      foreach_leaf([&](const MaskLeaf &leaf, int64_t k) {
        auto accessor = numbers->getConstUnsafeAccessor();
        for (auto it = leaf.cbeginValueOn(); it; ++it, ++k) {
          writer.span[k] = accessor.getValue(it.getCoord().offsetBy(d.x, d.y, d.z));
        }
      });
      writer.finish();
    }
  }
  for (const SigAttr &attr : sig.attrs) {
    if (attr.input != 0 || attr.output || attr.name == "position" ||
        attributes.contains(attr.name))
    {
      continue;
    }
    const bke::VolumeGridData *grid = BKE_volume_grid_find(volume, attr.name);
    const AttrType type = *attr_type_from_code(attr.type_code);
    if (grid == nullptr || !same_transform(*grid) || !grid_type_for(type) ||
        !is_toast_grid_type(grid->grid_type()) ||
        is_vector_grid_type(grid->grid_type()) != (type == AttrType::Float3))
    {
      continue;
    }
    bke::GSpanAttributeWriter writer = attributes.lookup_or_add_for_write_only_span(
        attr.name, AttrDomain::Point, type);
    bke::VolumeTreeAccessToken token;
    const openvdb::GridBase &base = grid->grid(token);
    with_toast_grid_type(grid->grid_type(), [&]<typename GridT>() {
      const GridT &typed = static_cast<const GridT &>(base);
      bke::attribute_math::to_static_type(type, [&]<typename AttrT>() {
        if constexpr (std::is_same_v<AttrT, float> || std::is_same_v<AttrT, int> ||
                      std::is_same_v<AttrT, bool> || std::is_same_v<AttrT, float3>)
        {
          MutableSpan<AttrT> values = writer.span.typed<AttrT>();
          foreach_leaf([&](const MaskLeaf &leaf, int64_t k) {
            auto accessor = typed.getConstUnsafeAccessor();
            for (auto it = leaf.cbeginValueOn(); it; ++it, ++k) {
              values[k] = voxel_to_attr<AttrT>(accessor.getValue(it.getCoord()));
            }
          });
        }
      });
    });
    writer.finish();
  }

  GeometrySet voxels = GeometrySet::from_pointcloud(pointcloud);
  RunContext inner = run;
  inner.volume0 = volume;
  VoxelCommands voxel_cmds;
  inner.voxel_cmds = &voxel_cmds;
  if (!run_on_component(voxels, ComponentType::PointCloud, inner, r_error)) {
    return false;
  }

  /* Written attributes back into grids. */
  const PointCloud *result = voxels.get_pointcloud();
  Volume *volume_for_write = geometry.get_volume_for_write();
  Vector<std::string> done;
  for (const SigAttr &attr : sig.attrs) {
    if (attr.input != 0 || !attr.write || attr.output || attr.name == "position" ||
        StringRef(attr.name).startswith("__") || done.contains(attr.name))
    {
      continue;
    }
    done.append(attr.name);
    const bke::GAttributeReader reader = result->attributes().lookup(attr.name);
    const AttrType type = *attr_type_from_code(attr.type_code);
    const std::optional<VolumeGridType> grid_type = grid_type_for(type);
    if (!reader || !grid_type) {
      run.warnings->add("@" + attr.name + ": a grid can't hold this type (use float, int, bool or vec3), it was not stored");
      continue;
    }
    const GVArraySpan values(*reader);
    bke::VolumeGridData *existing = BKE_volume_grid_find_for_write(volume_for_write, attr.name);
    openvdb::GridBase::Ptr new_grid;
    openvdb::GridBase *target = nullptr;
    bke::VolumeTreeAccessToken token;
    if (existing && existing->grid_type() == *grid_type && same_transform(*existing)) {
      target = &existing->grid_for_write(token);
    }
    else {
      if (existing) {
        BKE_volume_grid_remove(volume_for_write, existing);
      }
      new_grid = bke::volume_grid::create_grid_with_topology(mask, *transform, *grid_type);
      target = new_grid.get();
    }
    with_toast_grid_type(*grid_type, [&]<typename GridT>() {
      using ValueT = typename GridT::ValueType;
      GridT &typed = static_cast<GridT &>(*target);
      typed.tree().topologyUnion(mask);
      typed.tree().voxelizeActiveTiles();
      bke::attribute_math::to_static_type(type, [&]<typename AttrT>() {
        if constexpr (std::is_same_v<AttrT, float> || std::is_same_v<AttrT, int> ||
                      std::is_same_v<AttrT, bool> || std::is_same_v<AttrT, float3>)
        {
          if constexpr (!std::is_same_v<ValueT, openvdb::ValueMask>) {
            const Span<AttrT> typed_values = values.typed<AttrT>();
            foreach_leaf([&](const MaskLeaf &leaf, int64_t k) {
              typename GridT::TreeType::LeafNodeType *target_leaf = typed.tree().probeLeaf(
                  leaf.origin());
              if (target_leaf == nullptr) {
                return;
              }
              for (auto it = leaf.cbeginValueOn(); it; ++it, ++k) {
                target_leaf->setValueOn(it.pos(), attr_to_voxel<ValueT>(typed_values[k]));
              }
            });
          }
        }
      });
    });
    if (new_grid) {
      bke::GVolumeGrid grid(std::move(new_grid));
      grid.get_for_write().set_name(attr.name);
      grid->add_user();
      BKE_volume_grid_add(volume_for_write, grid.get());
    }
  }
  apply_voxel_commands(*volume_for_write, matrix, voxel_cmds);
  return true;
}

#endif /* WITH_OPENVDB */

/** The components of one real geometry the kernel runs on, by Run Over. */
static Vector<ComponentType> components_to_run(GeometrySet &geometry,
                                               const StringRef run_over,
                                               const bool builds_curves)
{
  Vector<ComponentType> types;
  const Mesh *mesh = geometry.get_mesh();
  const PointCloud *pointcloud = geometry.get_pointcloud();
  const Curves *curves = geometry.get_curves();
  if (run_over == "points") {
    if (mesh && mesh->verts_num > 0) {
      types.append(ComponentType::Mesh);
    }
    if (pointcloud && pointcloud->totpoint > 0) {
      types.append(ComponentType::PointCloud);
    }
    if (curves && curves->geometry.point_num > 0) {
      types.append(ComponentType::Curve);
    }
  }
  else if (run_over == "curves") {
    if (curves && curves->geometry.curve_num > 0) {
      types.append(ComponentType::Curve);
    }
  }
  else if (run_over == "geometry") {
    /* Runs once, also on empty geometry, which gets a mesh to build into (curves when the code
     * makes curves). */
    if (builds_curves || (!mesh && !pointcloud && curves)) {
      if (!curves) {
        if (geometry.has_grease_pencil()) {
          return types; /* The code draws into the layers instead, see run_on_grease_pencil. */
        }
        geometry.replace_curves(bke::curves_new_nomain(0, 0));
      }
      types.append(ComponentType::Curve);
      return types;
    }
    if (!mesh && !pointcloud) {
      geometry.replace_mesh(BKE_mesh_new_nomain(0, 0, 0, 0));
    }
    types.append(geometry.has_mesh() ? ComponentType::Mesh : ComponentType::PointCloud);
  }
  else if (mesh) {
    types.append(ComponentType::Mesh); /* Faces, edges and corners. */
  }
  return types;
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
static const bNodeSocket *find_output(const bNode &node, const StringRef identifier)
{
  for (const bNodeSocket *socket : node.output_sockets()) {
    if (socket->identifier == identifier) {
      return socket;
    }
  }
  return nullptr;
}

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

  RunWarnings warnings;
  auto finish = [&](const std::optional<std::string> &error, const bool is_info = false) {
    if (error) {
      params.error_message_add(is_info ? NodeWarningType::Info : NodeWarningType::Error, *error);
    }
    for (const std::string &warning : warnings.messages) {
      params.error_message_add(NodeWarningType::Warning, warning);
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
  for (const SigOutput &output : kernel->signature.outputs) {
    const std::string identifier = output_identifier(output.name);
    if (find_output(params.node(), identifier) == nullptr) {
      finish(std::string("The node's sockets are out of date, the code is being recompiled"));
      return;
    }
    if (std::optional<std::string> id = params.get_output_anonymous_attribute_id_if_needed(
            UString(identifier)))
    {
      run.output_ids.add(output.name, std::move(*id));
    }
  }
  run.kernel = kernel;
  run.selection = params.extract_input<Field<bool>>("Selection"_ustr);
  run.params = param_buffer;
  run.input1 = &input1;
  run.attribute_filter = params.get_attribute_filter("Geometry"_ustr);
  run.warnings = &warnings;
  if (const Depsgraph *depsgraph = params.depsgraph()) {
    const Scene *scene = DEG_get_input_scene(depsgraph);
    run.frame = DEG_get_ctime(depsgraph);
    run.time = scene ? float(run.frame / scene->frames_per_second()) : 0.0f;
  }

  const StringRef run_over = kernel->signature.run_over;
  if (run_over == "instances") {
    /* The instances of the top-level geometry, like Blender's instance nodes. */
    const bke::Instances *instances = geometry.get_instances();
    if (instances && instances->instances_num() > 0) {
      run_on_component(geometry, ComponentType::Instance, run, error);
    }
    finish(error.empty() ? std::nullopt : std::optional<std::string>(error));
    return;
  }
  std::mutex error_mutex;
  geometry::foreach_real_geometry(geometry, [&](GeometrySet &real_geometry_ref) {
    GeometrySet &real_geometry = real_geometry_ref;
#ifdef WITH_OPENVDB
    /* activate_voxel()/deactivate_voxel() outside Run Over Voxels: collected over this
     * geometry's components, then applied to its volume (see below). */
    RunContext run_here = run;
    VoxelCommands voxel_cmds;
    if (kernel->signature.uses_voxel_cmd && run_over != "voxels") {
      run_here.voxel_cmds = &voxel_cmds;
    }
    const RunContext &run = run_here;
    BLI_SCOPED_DEFER([&]() {
      if (run_here.voxel_cmds && real_geometry.has_volume()) {
        Volume *volume = real_geometry.get_volume_for_write();
        if (BKE_volume_num_grids(volume) > 0) {
          const float4x4 matrix = grid_matrix(BKE_volume_grid_get(volume, 0)->transform());
          apply_voxel_commands(*volume, matrix, voxel_cmds);
        }
      }
    });
#endif
    if (run_over == "voxels") {
#ifdef WITH_OPENVDB
      std::string local_error;
      if (!run_on_volume(real_geometry, run, local_error)) {
        std::lock_guard lock{error_mutex};
        if (error.empty()) {
          error = local_error;
        }
      }
#else
      std::lock_guard lock{error_mutex};
      error = "Run Over Voxels needs a Blender built with OpenVDB";
#endif
      return;
    }
    for (const ComponentType type :
         components_to_run(real_geometry, run_over, kernel->signature.builds_curves))
    {
      std::string local_error;
      if (!run_on_component(real_geometry, type, run, local_error)) {
        std::lock_guard lock{error_mutex};
        if (error.empty()) {
          error = local_error;
        }
        return;
      }
    }
    if (real_geometry.has_grease_pencil() &&
        (run_over == "points" || run_over == "curves" ||
         (run_over == "geometry" && kernel->signature.builds_curves)))
    {
      std::string local_error;
      if (!run_on_grease_pencil(real_geometry, run, local_error)) {
        std::lock_guard lock{error_mutex};
        if (error.empty()) {
          error = local_error;
        }
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
  for (const SigOutput &o : sig->outputs) {
    const UString name(o.name);
    const UString identifier(output_identifier(o.name));
    if (o.socket == "Float") {
      b.add_output<decl::Float>(name, identifier).anonymous_attribute_output();
    }
    else if (o.socket == "Integer") {
      b.add_output<decl::Int>(name, identifier).anonymous_attribute_output();
    }
    else if (o.socket == "Boolean") {
      b.add_output<decl::Bool>(name, identifier).anonymous_attribute_output();
    }
    else if (o.socket == "Vector") {
      b.add_output<decl::Vector>(name, identifier).anonymous_attribute_output();
    }
    else if (o.socket == "Color") {
      b.add_output<decl::Color>(name, identifier).anonymous_attribute_output();
    }
    else if (o.socket == "Rotation") {
      b.add_output<decl::Rotation>(name, identifier).anonymous_attribute_output();
    }
    else if (o.socket == "Matrix") {
      b.add_output<decl::Matrix>(name, identifier).anonymous_attribute_output();
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
  /* The Toast add-on's compile operator (the add-on compiles; without it there is nothing to
   * call). */
  if (WM_operatortype_find("TOAST_OT_compile_node", true)) {
    const bNode &node = *ptr->data_as<bNode>();
    PointerRNA op_ptr = layout.op("TOAST_OT_compile_node", IFACE_("Compile"), ICON_FILE_REFRESH);
    RNA_string_set(&op_ptr, "tree", ptr->owner_id->name + 2);
    RNA_string_set(&op_ptr, "node", node.name);
  }
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
