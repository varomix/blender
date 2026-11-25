/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Toast fork: loader for C++ node plugins. See `NOD_toast_plugins.hh`.
 */

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#ifndef _WIN32
#  include <dlfcn.h>
#  include <unistd.h>
#endif

#include "BKE_appdir.hh"
#include "BKE_global.hh"
#include "BKE_main.hh"
#include "BKE_node_runtime.hh"
#include "BKE_node.hh"

#include "BLI_listbase.hh"

#include "DNA_node_types.h"

#include "RNA_access.hh"
#include "RNA_define.hh"
#include "RNA_prototypes.hh"
/* For #BlenderRNA::structs_map, see #geo_node_type_base. */
#include "../../../../makesrna/intern/rna_internal_types.hh"

#include "NOD_toast_plugins.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::toast_plugins {

namespace fs = std::filesystem;

struct LoadedPlugin {
  std::string path;
  const ToastPluginInfo *info = nullptr;
};

/** Plugins whose node types are registered now. */
static Vector<LoadedPlugin> &loaded_plugins()
{
  static Vector<LoadedPlugin> plugins;
  return plugins;
}

/** Set during #reload_all, so #unregister_node_type can let the UI drop RNA references. */
static FunctionRef<void(StructRNA *)> before_srna_free_fn;

/** Errors reported by the helpers while a plugin registers. */
static Vector<std::string> *current_errors = nullptr;

static void report(std::string message)
{
  if (current_errors) {
    current_errors->append(message);
  }
  fprintf(stderr, "Toast plugins: %s\n", message.c_str());
}

/* -------------------------------------------------------------------- */
/** \name Helpers for Plugins
 * \{ */

bool geo_node_type_base(bke::bNodeType &ntype, const StringRef idname)
{
  const std::string id = idname;
  if (RNA_struct_find(id.c_str()) != nullptr) {
    report("node type \"" + id + "\" already exists, the plugin can't define it");
    return false;
  }
  BlenderRNA &brna = RNA_blender_rna_get();
  StructRNA *srna = RNA_def_struct_ptr(&brna, id.c_str(), RNA_GeometryNode);
  /* At run time, RNA_def_struct_ptr only makes a struct findable by name when its base asks for
   * it. Node types need it: #node_type_base looks their struct up, and so does `bpy.types`.
   * #RNA_struct_free removes the entry again. */
  RNA_def_struct_flag(srna, STRUCT_PUBLIC_NAMESPACE);
  brna.structs_map.add(srna->identifier, srna);
  blender::geo_node_type_base(&ntype, UString(id));
  return true;
}

/** Call #fn for every node in G_MAIN using this type. Plugin types are geometry node types, so
 * they only appear in node groups. */
template<typename Fn> static void foreach_node_of_type(const bke::bNodeType &ntype, Fn &&fn)
{
  if (G_MAIN == nullptr) {
    return;
  }
  for (bNodeTree &ntree : G_MAIN->nodetrees) {
    for (bNode *node : ntree.all_nodes()) {
      if (node->typeinfo == &ntype) {
        fn(ntree, *node);
      }
    }
  }
}

void register_node_type(bke::bNodeType &ntype)
{
  /* Gives nodes in open files (undefined until now) this type. */
  bke::node_register_type(ntype);
  /* Their declaration was cleared when the previous type was unregistered. */
  foreach_node_of_type(ntype, [](bNodeTree &ntree, bNode &node) {
    bke::node_declaration_ensure(ntree, node);
  });
}

void unregister_node_type(bke::bNodeType &ntype)
{
  StructRNA *srna = ntype.rna_ext.srna;
  if (srna) {
    if (before_srna_free_fn) {
      before_srna_free_fn(srna);
    }
    RNA_struct_free(&RNA_blender_rna_get(), srna);
  }
  /* Nodes point at the type's static declaration, which unregistering deletes. Built-in types are
   * never unregistered while Blender runs, so nothing else clears these. */
  foreach_node_of_type(ntype, [](bNodeTree & /*ntree*/, bNode &node) {
    node.runtime->declaration = nullptr;
    for (bNodeSocket *socket : node.input_sockets()) {
      socket->runtime->declaration = nullptr;
    }
    for (bNodeSocket *socket : node.output_sockets()) {
      socket->runtime->declaration = nullptr;
    }
  });
  /* Also tells nodes in open files that the type is gone. The type itself stays allocated (it is
   * static in the plugin library, which is never unloaded). */
  bke::node_unregister_type(ntype);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Loading
 * \{ */

Vector<std::string> plugin_dirs()
{
  Vector<std::string> dirs;
  const char *program_dir = BKE_appdir_program_dir();
  if (program_dir && program_dir[0]) {
    dirs.append((fs::path(program_dir) / "toast_plugins").string());
  }
  if (const char *env = getenv("TOAST_PLUGIN_PATH")) {
    StringRef rest = env;
    while (!rest.is_empty()) {
      const int64_t sep = rest.find(':');
      const StringRef dir = sep == StringRef::not_found ? rest : rest.substr(0, sep);
      if (!dir.is_empty()) {
        dirs.append(dir);
      }
      rest = sep == StringRef::not_found ? StringRef() : rest.drop_prefix(sep + 1);
    }
  }
  return dirs;
}

static Vector<std::string> find_plugin_files()
{
  Vector<std::string> files;
  for (const std::string &dir : plugin_dirs()) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
      continue;
    }
    Vector<std::string> in_dir;
    for (const fs::directory_entry &entry : fs::directory_iterator(dir, ec)) {
      if (entry.is_regular_file(ec) && entry.path().extension() == ".so") {
        in_dir.append(entry.path().string());
      }
    }
    /* Deterministic order within a folder. */
    std::sort(in_dir.begin(), in_dir.end());
    files.extend(in_dir);
  }
  return files;
}

#ifndef _WIN32
/**
 * Load a private copy of the library: a rebuild then never writes into a mapped file, and a
 * reload gets a new copy even though the old one stays loaded (dlopen returns the already loaded
 * library for the same path).
 */
static void *open_copy(const std::string &path, std::string &r_error)
{
  static std::atomic<int> counter = 0;
  std::error_code ec;
  const fs::path tmp_dir = fs::temp_directory_path(ec) /
                           ("blender_toast_plugins_" + std::to_string(getpid()));
  fs::create_directories(tmp_dir, ec);
  const fs::path copy = tmp_dir / (fs::path(path).stem().string() + "." +
                                   std::to_string(counter++) + ".so");
  fs::copy_file(path, copy, fs::copy_options::overwrite_existing, ec);
  if (ec) {
    r_error = "can't copy " + path + ": " + ec.message();
    return nullptr;
  }
  void *handle = dlopen(copy.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char *err = dlerror();
    r_error = err ? err : ("can't load " + path);
  }
  /* The mapping stays valid after the file is removed. */
  fs::remove(copy, ec);
  return handle;
}
#endif

static void load_files(Vector<std::string> &r_loaded, Vector<std::string> &r_errors)
{
#ifdef _WIN32
  UNUSED_VARS(r_loaded, r_errors);
#else
  current_errors = &r_errors;
  for (const std::string &path : find_plugin_files()) {
    std::string error;
    void *handle = open_copy(path, error);
    if (handle == nullptr) {
      report(error);
      continue;
    }
    auto info_fn = reinterpret_cast<ToastPluginInfoFn>(dlsym(handle, "toast_plugin_info"));
    const ToastPluginInfo *info = info_fn ? info_fn() : nullptr;
    if (info == nullptr) {
      /* Not a plugin; it can't be unloaded safely either way, so leave it. */
      report(path + " is not a Toast plugin (no toast_plugin_info)");
      continue;
    }
    if (info->api_version != TOAST_PLUGIN_API_VERSION) {
      report(path + ": plugin API " + std::to_string(info->api_version) + ", Blender has " +
             std::to_string(TOAST_PLUGIN_API_VERSION) + ", rebuild it");
      continue;
    }
    if (!info->register_nodes()) {
      report(path + ": some node types failed to register");
    }
    loaded_plugins().append({path, info});
    r_loaded.append(info->name ? info->name : path);
  }
  current_errors = nullptr;
#endif
}

void load_all()
{
  Vector<std::string> loaded;
  Vector<std::string> errors;
  load_files(loaded, errors);
  for (const std::string &name : loaded) {
    printf("Toast plugins: loaded %s\n", name.c_str());
  }
}

ReloadResult reload_all(const FunctionRef<void(StructRNA *)> before_srna_free)
{
  ReloadResult result;
  before_srna_free_fn = before_srna_free;
  for (const LoadedPlugin &plugin : loaded_plugins()) {
    plugin.info->unregister_nodes();
  }
  before_srna_free_fn = {};
  loaded_plugins().clear();
  load_files(result.loaded, result.errors);
  return result;
}

/** \} */

}  // namespace blender::nodes::toast_plugins
