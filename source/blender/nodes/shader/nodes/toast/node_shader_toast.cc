/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup shdnodes
 *
 * Toast fork: a per-shading-point procedural written in Toast code. The Toast add-on compiles the
 * code (the Text in `bNode.id`) into a GPU shader function and a signature, and writes both into
 * the node storage. The node never calls Python.
 *
 * The function's arguments are the param() sockets, then the shading inputs the code reads
 * (`@position`, `@uv`, named attributes, ...) in signature order, then the output() sockets.
 * Shading inputs come from the same library functions as the Geometry, Texture Coordinate and
 * Scene Time nodes, so they have the same values as those nodes. Named attributes are passed as
 * float4, like to the Attribute node.
 *
 * EEVEE and the viewport use `gpu_source`. Cycles (CPU) calls the C++ function in
 * `kernel_path`, see `toast_shader_info()` (NOD_toast.hh); on GPU devices the outputs are zero.
 * As with the geometry node, the code only runs when Auto Run Python Scripts allows it;
 * otherwise the outputs are their default values.
 */

#include <sstream>
#include <string>

#include "BLI_map.hh"
#include "BLI_set.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_serialize.hh"
#include "BLI_string.hh"
#include "BLI_vector.hh"

#include "DNA_customdata_types.h"
#include "DNA_node_types.h"
#include "DNA_text_types.h"

#include "BKE_global.hh"
#include "BKE_lib_id.hh"

#include "BLO_read_write.hh"

#include "NOD_rna_define.hh"
#include "NOD_toast.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"
#include "RNA_prototypes.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "WM_api.hh"

#include "node_shader_util.hh"

namespace blender {

namespace nodes::node_shader_toast_cc {

NODE_STORAGE_FUNCS(NodeShaderToast)

/* -------------------------------------------------------------------- */
/** \name Signature
 * \{ */

struct SigParam {
  std::string name;
  std::string type;
  Vector<double, 4> default_value;
  std::optional<double> min;
  std::optional<double> max;
};

/** A value the function reads from the shading point: a built-in one (`kind` "builtin": position,
 * uv, ...) or a named attribute (`kind` "attribute"). */
struct SigInput {
  std::string name;
  std::string kind;
  std::string type;
};

struct SigOutput {
  std::string name;
  std::string type;
};

struct Signature {
  std::string function;
  Vector<SigParam> params;
  Vector<SigInput> inputs;
  Vector<SigOutput> outputs;
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

static std::optional<Signature> parse_signature(const char *json_str)
{
  if (json_str == nullptr || json_str[0] == '\0') {
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
  if (dict == nullptr || dict->lookup_str("run_over").value_or("") != "shader") {
    return std::nullopt;
  }

  Signature sig;
  sig.function = std::string(dict->lookup_str("function").value_or(""));
  if (sig.function.empty()) {
    return std::nullopt;
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
  if (const io::serialize::ArrayValue *inputs = dict->lookup_array("inputs")) {
    for (const std::shared_ptr<io::serialize::Value> &item : inputs->elements()) {
      const io::serialize::DictionaryValue *i = item->as_dictionary_value();
      if (i == nullptr) {
        return std::nullopt;
      }
      sig.inputs.append({std::string(i->lookup_str("name").value_or("")),
                         std::string(i->lookup_str("kind").value_or("")),
                         std::string(i->lookup_str("type").value_or(""))});
    }
  }
  if (const io::serialize::ArrayValue *outputs = dict->lookup_array("outputs")) {
    for (const std::shared_ptr<io::serialize::Value> &item : outputs->elements()) {
      const io::serialize::DictionaryValue *o = item->as_dictionary_value();
      if (o == nullptr) {
        return std::nullopt;
      }
      sig.outputs.append({std::string(o->lookup_str("name").value_or("")),
                          std::string(o->lookup_str("type").value_or(""))});
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

static std::optional<GPUType> gpu_type(const StringRef type)
{
  if (type == "float") {
    return GPU_FLOAT;
  }
  if (type == "int") {
    return GPU_INT;
  }
  if (type == "bool") {
    return GPU_BOOL;
  }
  if (type == "vec2") {
    return GPU_VEC2;
  }
  if (type == "vec3") {
    return GPU_VEC3;
  }
  if (type == "vec4") {
    return GPU_VEC4;
  }
  return std::nullopt;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name GPU
 * \{ */

/** Built-in shading inputs, from the outputs of library functions (each called once). */
class ShadingInputs {
  GPUMaterial *mat_;
  Map<std::string, GPUNodeLink *> links_;
  /* Links that already went through the Bump offset. */
  Set<std::string> done_;

  void add_geometry()
  {
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GPUNodeLink *out[9] = {};
    GPU_link(mat_,
             "node_geometry",
             GPU_constant(zero),
             GPU_kernel_globals(),
             GPU_shading_data(),
             &out[0],
             &out[1],
             &out[2],
             &out[3],
             &out[4],
             &out[5],
             &out[6],
             &out[7],
             &out[8]);
    links_.add("position", out[0]);
    links_.add("normal", out[1]);
    links_.add("true_normal", out[3]);
    links_.add("incoming", out[4]);
    links_.add("parametric", out[5]);
    links_.add("backfacing", out[6]);
  }

  void add_tex_coord()
  {
    float dummy_matrix[4][4] = {};
    /* As the Texture Coordinate node without an object: the render object's matrix. */
    dummy_matrix[3][3] = 0.0f;
    GPUNodeLink *out[7] = {};
    GPU_link(mat_,
             "node_tex_coord",
             GPU_uniform(&dummy_matrix[0][0]),
             GPU_attribute(mat_, CD_ORCO, ""),
             GPU_attribute(mat_, CD_AUTO_FROM_NAME, ""),
             GPU_kernel_globals(),
             GPU_shading_data(),
             &out[0],
             &out[1],
             &out[2],
             &out[3],
             &out[4],
             &out[5],
             &out[6]);
    links_.add("generated", out[0]);
    links_.add("uv", out[2]);
    links_.add("object", out[3]);
    links_.add("camera", out[4]);
    links_.add("window", out[5]);
    links_.add("reflection", out[6]);
  }

  void add_time()
  {
    GPU_material_flag_set(mat_, GPU_MATFLAG_SCENE_TIME);
    GPUNodeLink *seconds = nullptr, *frame = nullptr;
    GPU_link(mat_, "node_scene_time", GPU_kernel_globals(), &seconds, &frame);
    links_.add("time", seconds);
    links_.add("frame", frame);
  }

 public:
  explicit ShadingInputs(GPUMaterial *mat) : mat_(mat) {}

  GPUNodeLink *get(bNode *node, const StringRef name)
  {
    if (GPUNodeLink *link = links_.lookup_default(name, nullptr)) {
      return link;
    }
    if (ELEM(name, "position", "normal", "true_normal", "incoming", "parametric", "backfacing")) {
      add_geometry();
    }
    else if (ELEM(name, "generated", "uv", "object", "camera", "window", "reflection")) {
      add_tex_coord();
    }
    else if (ELEM(name, "time", "frame")) {
      add_time();
    }
    else if (name == "color") {
      links_.add("color", GPU_attribute_default_color(mat_));
      return links_.lookup("color");
    }
    else {
      return nullptr;
    }
    /* As the Geometry and Texture Coordinate nodes: offsets for the Bump node, normalized
     * directions after them. Each link once: only the group just added. */
    for (const char *vec : {"position",
                            "normal",
                            "true_normal",
                            "incoming",
                            "parametric",
                            "generated",
                            "uv",
                            "object",
                            "camera",
                            "window",
                            "reflection"})
    {
      GPUNodeLink **link = links_.lookup_ptr(vec);
      if (link == nullptr || *link == nullptr || done_.contains(vec)) {
        continue;
      }
      done_.add(vec);
      node_shader_gpu_bump_tex_coord(mat_, node, link);
      if (ELEM(StringRef(vec), "normal", "true_normal", "incoming", "reflection")) {
        GPU_link(mat_, "vector_math_normalize", *link, *link, *link, *link, link, nullptr);
      }
    }
    return links_.lookup_default(name, nullptr);
  }
};

static int node_shader_gpu(GPUMaterial *mat,
                           bNode *node,
                           bNodeExecData * /*execdata*/,
                           GPUNodeStack *in,
                           GPUNodeStack *out)
{
  /* Like the geometry node: Toast code only runs in trusted files (the source is put into the
   * material's shader as it is stored in the file). */
  if (!(G.f & G_FLAG_SCRIPT_AUTOEXEC)) {
    return 0;
  }
  const NodeShaderToast &storage = node_storage(*node);
  if (storage.gpu_source == nullptr || storage.gpu_source[0] == '\0' ||
      (storage.error && storage.error[0] != '\0'))
  {
    return 0;
  }
  const std::optional<Signature> sig = parse_signature(storage.signature);
  if (!sig) {
    return 0;
  }
  /* The sockets must be the ones the function was compiled for. */
  if (sig->params.size() != BLI_listbase_count(&node->inputs) ||
      sig->outputs.size() != BLI_listbase_count(&node->outputs))
  {
    return 0;
  }

  ShadingInputs shading(mat);
  Vector<GPUNodeLink *> links;
  Vector<GPUType> types;
  for (const SigInput &input : sig->inputs) {
    const std::optional<GPUType> type = gpu_type(input.type);
    if (!type) {
      return 0;
    }
    GPUNodeLink *link = nullptr;
    if (input.kind == "builtin") {
      link = shading.get(node, input.name);
    }
    else if (input.kind == "attribute") {
      link = GPU_attribute(mat, CD_AUTO_FROM_NAME, input.name.c_str());
    }
    if (link == nullptr) {
      return 0;
    }
    links.append(link);
    /* Named attributes are float4 in the material (as for the Attribute node); the function
     * converts them to the type the code reads. */
    types.append(input.kind == "attribute" ? GPU_VEC4 : *type);
  }
  return GPU_stack_link_source(
      mat, node, sig->function.c_str(), storage.gpu_source, in, out, links, types);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Node Type
 * \{ */

static void node_declare(NodeDeclarationBuilder &b)
{
  const bNode *node = b.node_or_null();
  if (node == nullptr || node->storage == nullptr) {
    return;
  }
  const std::optional<Signature> sig = parse_signature(node_storage(*node).signature);
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
    if (o.type == "float") {
      b.add_output<decl::Float>(name, identifier);
    }
    else if (o.type == "int") {
      b.add_output<decl::Int>(name, identifier);
    }
    else if (o.type == "bool") {
      b.add_output<decl::Bool>(name, identifier);
    }
    else if (o.type == "vec3") {
      b.add_output<decl::Vector>(name, identifier);
    }
    else if (o.type == "vec4") {
      b.add_output<decl::Color>(name, identifier);
    }
  }
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  node->storage = MEM_new<NodeShaderToast>(__func__);
}

static char *dup_or_null(const char *str)
{
  return str ? BLI_strdup(str) : nullptr;
}

static void node_free_storage(bNode *node)
{
  NodeShaderToast *storage = static_cast<NodeShaderToast *>(node->storage);
  for (char *str : {storage->signature,
                     storage->gpu_source,
                     storage->error,
                     storage->source_hash,
                     storage->kernel_path})
  {
    if (str) {
      MEM_delete(str);
    }
  }
  MEM_delete(storage);
}

static void node_copy_storage(bNodeTree * /*dst_tree*/, bNode *dst_node, const bNode *src_node)
{
  const NodeShaderToast &src = node_storage(*src_node);
  NodeShaderToast *dst = MEM_new<NodeShaderToast>(__func__, dna::shallow_copy(src));
  dst->signature = dup_or_null(src.signature);
  dst->gpu_source = dup_or_null(src.gpu_source);
  dst->error = dup_or_null(src.error);
  dst->source_hash = dup_or_null(src.source_hash);
  dst->kernel_path = dup_or_null(src.kernel_path);
  dst_node->storage = dst;
}

static void node_blend_write(const bNodeTree & /*tree*/, const bNode &node, BlendWriter &writer)
{
  const NodeShaderToast &storage = node_storage(node);
  for (const char *str : {storage.signature,
                           storage.gpu_source,
                           storage.error,
                           storage.source_hash,
                           storage.kernel_path})
  {
    if (str) {
      writer.write_string(str);
    }
  }
}

static void node_blend_read(bNodeTree & /*tree*/, bNode &node, BlendDataReader &reader)
{
  NodeShaderToast &storage = node_storage(node);
  BLO_read_string(&reader, &storage.signature);
  BLO_read_string(&reader, &storage.gpu_source);
  BLO_read_string(&reader, &storage.error);
  BLO_read_string(&reader, &storage.source_hash);
  BLO_read_string(&reader, &storage.kernel_path);
}

static void node_layout(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.prop(ptr, "text", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
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

template<char *NodeShaderToast::*Member>
static std::string rna_string_get(PointerRNA *ptr, PropertyRNA * /*prop*/)
{
  const char *str = node_storage(*ptr->data_as<bNode>()).*Member;
  return str ? str : "";
}

template<char *NodeShaderToast::*Member>
static int rna_string_length(PointerRNA *ptr, PropertyRNA * /*prop*/)
{
  const char *str = node_storage(*ptr->data_as<bNode>()).*Member;
  return str ? int(strlen(str)) : 0;
}

template<char *NodeShaderToast::*Member>
static void rna_string_set(PointerRNA *ptr, PropertyRNA * /*prop*/, const std::string &value)
{
  char *&str = node_storage(*ptr->data_as<bNode>()).*Member;
  if (str) {
    MEM_delete(str);
    str = nullptr;
  }
  if (!value.empty()) {
    str = BLI_strdup(value.c_str());
  }
}

template<char *NodeShaderToast::*Member>
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

  def_string<&NodeShaderToast::signature>(
      srna,
      "signature",
      "Signature",
      "Parameters, shading inputs and outputs of the compiled function (JSON, written by the "
      "Toast add-on)",
      rna_Node_socket_update);
  def_string<&NodeShaderToast::gpu_source>(srna,
                                           "gpu_source",
                                           "GPU Source",
                                           "Generated shader function for EEVEE and the viewport",
                                           rna_Node_update);
  def_string<&NodeShaderToast::error>(
      srna, "error", "Error", "Compile error of the code, empty if it compiled", rna_Node_update);
  def_string<&NodeShaderToast::kernel_path>(srna,
                                            "kernel_path",
                                            "Kernel Path",
                                            "The function compiled for the CPU (Cycles)",
                                            rna_Node_update);
  def_string<&NodeShaderToast::source_hash>(srna,
                                            "source_hash",
                                            "Source Hash",
                                            "Hash of the code the function was built from",
                                            nullptr);
}

/** \} */

}  // namespace nodes::node_shader_toast_cc

namespace nodes {

std::optional<ToastShaderInfo> toast_shader_info(const bNode &node, std::string &r_error)
{
  using namespace node_shader_toast_cc;
  if (!(G.f & G_FLAG_SCRIPT_AUTOEXEC) || node.storage == nullptr) {
    return std::nullopt;
  }
  const NodeShaderToast &storage = node_storage(node);
  if ((storage.error && storage.error[0]) || !storage.kernel_path || !storage.kernel_path[0]) {
    return std::nullopt;
  }
  const std::optional<Signature> sig = parse_signature(storage.signature);
  if (!sig) {
    return std::nullopt;
  }
  ToastShaderInfo info;
  info.function = toast_load_shader_function(storage.kernel_path, r_error);
  if (info.function == nullptr) {
    return std::nullopt;
  }
  for (const SigParam &p : sig->params) {
    info.params.append({param_identifier(p.name), p.type, ""});
  }
  for (const SigInput &i : sig->inputs) {
    info.inputs.append({i.name, i.type, i.kind});
  }
  for (const SigOutput &o : sig->outputs) {
    info.outputs.append({output_identifier(o.name), o.type, ""});
  }
  return info;
}

}  // namespace nodes

void register_node_type_sh_toast()
{
  namespace file_ns = nodes::node_shader_toast_cc;

  static bke::bNodeType ntype;

  sh_node_type_base(&ntype, "ShaderNodeToast"_ustr);
  ntype.ui_name = "Toast";
  ntype.ui_description =
      "Python-syntax code run for every shading point, compiled to a shader by the Toast add-on "
      "(EEVEE)";
  ntype.nclass = NODE_CLASS_TEXTURE;
  ntype.declare = file_ns::node_declare;
  ntype.initfunc = file_ns::node_init;
  bke::node_type_storage(
      ntype, "NodeShaderToast", file_ns::node_free_storage, file_ns::node_copy_storage);
  ntype.blend_write_storage_content = file_ns::node_blend_write;
  ntype.blend_data_read_storage_content = file_ns::node_blend_read;
  ntype.draw_buttons = file_ns::node_layout;
  ntype.gpu_fn = file_ns::node_shader_gpu;
  ntype.default_width = bke::NodeWidth::_160;
  bke::node_register_type(ntype);

  file_ns::node_rna(ntype.rna_ext.srna);
}

}  // namespace blender
