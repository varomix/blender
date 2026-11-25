/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Toast benchmark plugin: noise displace written by hand in C++, as a reference for the Toast
 * node. Per point: n = perlin(P * freq); P += N * n * amp; height = n. Same arithmetic and
 * chunking as the Toast kernel, so the outputs are bit-identical.
 */

#include "BLI_noise.hh"
#include "BLI_task.hh"

#include "BKE_attribute.hh"
#include "BKE_mesh.hh"

#include "NOD_toast_plugins.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_noise_displace_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr).supported_type(GeometryComponent::Type::Mesh);
  b.add_input<decl::Float>("Amp"_ustr).default_value(0.2f);
  b.add_input<decl::Float>("Freq"_ustr).default_value(4.0f);
  b.add_output<decl::Geometry>("Geometry"_ustr).propagate_all();
}

static void displace(Mesh &mesh, const float amp, const float freq)
{
  const Span<float3> normals = mesh.vert_normals();
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  bke::SpanAttributeWriter<float3> positions = attributes.lookup_or_add_for_write_span<float3>(
      "position", bke::AttrDomain::Point);
  bke::SpanAttributeWriter<float> height = attributes.lookup_or_add_for_write_span<float>(
      "height", bke::AttrDomain::Point);
  MutableSpan<float3> P = positions.span;
  MutableSpan<float> h = height.span;
  threading::parallel_for(P.index_range(), 1024, [&](const IndexRange range) {
    for (const int64_t i : range) {
      const float n = noise::perlin(P[i] * freq);
      P[i] = P[i] + normals[i] * n * amp;
      h[i] = n;
    }
  });
  positions.finish();
  height.finish();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const float amp = params.extract_input<float>("Amp"_ustr);
  const float freq = params.extract_input<float>("Freq"_ustr);
  if (Mesh *mesh = geometry.get_mesh_for_write()) {
    displace(*mesh, amp, freq);
  }
  params.set_output("Geometry"_ustr, std::move(geometry));
}

static bke::bNodeType ntype;
static bool registered = false;

static bool register_nodes()
{
  if (!toast_plugins::geo_node_type_base(ntype, "GeometryNodeToastNoiseDisplaceCpp")) {
    return false;
  }
  ntype.ui_name = "Noise Displace (C++)";
  ntype.ui_description = "Hand-written C++ noise displace, the benchmark reference for Toast";
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

}  // namespace blender::nodes::node_geo_noise_displace_cc

TOAST_PLUGIN_EXPORT const ToastPluginInfo *toast_plugin_info()
{
  namespace ns = blender::nodes::node_geo_noise_displace_cc;
  static const ToastPluginInfo info = {
      TOAST_PLUGIN_API_VERSION, "Noise Displace (C++)", ns::register_nodes, ns::unregister_nodes};
  return &info;
}
