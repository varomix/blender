/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Toast fork: C++ node plugins, loaded from shared libraries at startup and on "Reload Plugins".
 *
 * A plugin is a shared library built against this exact Blender build (same commit, compiler and
 * flags, the HDK rule), which exports `toast_plugin_info()`. Its register function defines node
 * types with the helpers below instead of `geo_node_type_base`/`node_register_type`:
 *
 * \code{.cc}
 * static bke::bNodeType ntype;
 * static bool register_nodes()
 * {
 *   if (!toast_plugins::geo_node_type_base(ntype, "GeometryNodeMyNode")) {
 *     return false;
 *   }
 *   ntype.ui_name = "My Node";
 *   ...
 *   toast_plugins::register_node_type(ntype);
 *   return true;
 * }
 * static void unregister_nodes()
 * {
 *   toast_plugins::unregister_node_type(ntype);
 * }
 * TOAST_PLUGIN_EXPORT const ToastPluginInfo *toast_plugin_info()
 * {
 *   static const ToastPluginInfo info = {
 *       TOAST_PLUGIN_API_VERSION, "My Plugin", register_nodes, unregister_nodes};
 *   return &info;
 * }
 * \endcode
 *
 * Plugin libraries are never unloaded: node types stay referenced by depsgraph copies after
 * they are unregistered. A reload loads a fresh copy of each library instead.
 */

#pragma once

#include <string>

#include "BLI_function_ref.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

struct Main;
struct StructRNA;

namespace blender::bke {
struct bNodeType;
}

/** Bump when this API or the meaning of #ToastPluginInfo changes. */
#define TOAST_PLUGIN_API_VERSION 1

#if defined(_WIN32)
#  define TOAST_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#  define TOAST_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

extern "C" {
struct ToastPluginInfo {
  int api_version;
  const char *name;
  /** Define and register the plugin's node types. False if any failed (see the console). */
  bool (*register_nodes)();
  /** Unregister every type that #register_nodes registered. */
  void (*unregister_nodes)();
};
typedef const ToastPluginInfo *(*ToastPluginInfoFn)();
}

namespace blender::nodes::toast_plugins {

/* -------------------------------------------------------------------- */
/** \name For Plugins
 * \{ */

/**
 * Like #geo_node_type_base, for a node type in a plugin: also defines its RNA struct, which
 * built-in types get from `makesrna`. Returns false (and reports) if the idname is taken.
 */
bool geo_node_type_base(bke::bNodeType &ntype, StringRef idname);

/** Register a type set up with #geo_node_type_base. Nodes using it in open files come back. */
void register_node_type(bke::bNodeType &ntype);

/** Unregister a type and free its RNA struct. Nodes using it become undefined. */
void unregister_node_type(bke::bNodeType &ntype);

/** \} */

/* -------------------------------------------------------------------- */
/** \name For Blender
 * \{ */

/** Folders searched for plugins: `toast_plugins` next to the executable, then `$TOAST_PLUGIN_PATH`
 * (separated by ':'). */
Vector<std::string> plugin_dirs();

/** Load and register every plugin. Called once at startup; errors are printed. */
void load_all();

struct ReloadResult {
  Vector<std::string> loaded;
  Vector<std::string> errors;
};

/**
 * Unregister every plugin's node types, then load the current plugin files again and register
 * them. `before_srna_free` runs for each RNA struct about to be freed, so the UI can drop
 * references to it. The caller must update node trees afterwards.
 */
ReloadResult reload_all(FunctionRef<void(StructRNA *)> before_srna_free);

/** \} */

}  // namespace blender::nodes::toast_plugins
