# Toast fork geometry nodes

Native C++ geometry nodes added by the Toast fork. They live in this folder so the fork stays easy
to rebase onto new Blender releases. This folder will also hold the generic `GeometryNodeToast`
kernel node. Outside this folder, the fork changes only:

| File | Change |
| --- | --- |
| `source/blender/nodes/geometry/CMakeLists.txt` | one `include()` of `toast_nodes.cmake` |
| `source/blender/makesrna/intern/rna_nodetree.cc` | one `#include` of `toast_rna_defines.inc` |
| `scripts/startup/bl_ui/node_add_menu_geometry.py` | "Toast" entry in the Add/Swap menus |
| `scripts/startup/bl_ui/node_add_menu_toast.py` | new file, the menu contents |
| `build_files/toast/package_portable.sh` | new file, portable archive packaging |
| `source/blender/makesdna/DNA_node_types.h` | `NodeGeometryToast` storage |
| `source/blender/nodes/intern/dependencies.cc` | Toast kernels using `@Frame`/`@Time` are time dependent |
| `source/blender/nodes/NOD_toast_plugins.hh` | new file, C++ node plugin API |
| `source/blender/blenkernel/intern/node.cc` | static declarations freed with their (deferred) node type |
| `source/creator/creator.cc`, `CMakeLists.txt` | load plugins at startup; install the Toast add-on as a core add-on |
| `toast/addon` (submodule), `toast/README.md`, `.gitmodules` | the Toast add-on ([varomix/toast](https://github.com/varomix/toast)) |
| `.lfsconfig` | fetch Blender's Git LFS files from projects.blender.org |
| `source/creator/symbols_unix.map` | export Blender's C++ API for plugins |
| `source/blender/editors/space_node/node_toast_plugins.cc` (+ `node_ops.cc`, `node_intern.hh`, `CMakeLists.txt`) | Reload Node Plugins operator |

## Adding a node

1. Create `node_geo_<name>.cc` here (copy `node_geo_bricker.cc` for the structure).
   Use the idname `GeometryNode<Name>` and a unique namespace `node_geo_<name>_cc`. Check that
   upstream Blender has no node with that idname.
2. Add the file to `toast_nodes.cmake` and its idname to `toast_rna_defines.inc`
   (adding an RNA struct rebuilds a large part of Blender, expect ~10 minutes).
3. Add the idname to `TOAST_GEOMETRY_NODES` in `scripts/startup/bl_ui/node_add_menu_toast.py`.
4. Build: `cmake --build ../build_linux_5.3 --target install -j$(nproc)`.

## C++ node plugins

Full C++ nodes can live in shared libraries instead of Blender itself: `plugins/<name>/` here,
listed in `toast_add_node_plugins()` in `toast_nodes.cmake`. They build into
`bin/toast_plugins/` (also searched: `$TOAST_PLUGIN_PATH`), load at startup, and
Toast > Reload Plugins in the Add menu picks up a rebuilt one in a running Blender:

```
cd ../build_linux_5.3 && make toast_plugin_bricker   # ~10 s, Blender is not relinked
```

A plugin must be built from this exact tree (the HDK rule); its API is in
`NOD_toast_plugins.hh`. Plugin libraries are never unloaded, since depsgraph copies keep
referencing their node types; each reload loads a fresh copy. Bricker is the first plugin.

## Nodes

### Bricker (`GeometryNodeBricker`, plugin `toast_plugin_bricker`)

Divides the selected faces into bricks: every face is cut by three families of parallel planes,
spaced by **Size** along each axis, shifted by **Offset** and tilted by **Angle**. Points created
on an edge are shared by the faces using it, so closed meshes stay closed; **Fix Shared Edges**
also inserts those points into unselected neighbor faces to avoid T-junctions. Concave faces are
supported. Point and corner attributes are interpolated, face attributes copied.

## Tests

`tests/verify_bricker.py` compares the node against reference JSON files (input meshes and the
expected divided meshes, with attributes):

```
../build_linux_5.3/bin/blender -b --factory-startup --python tests/verify_bricker.py -- <ref_dir> result.json
```

## Distribution

`build_files/toast/package_portable.sh` builds, installs and writes
`../packages/blender-<version>-toast-<hash>-<os>-<arch>.tar.xz` (or `.zip` with `FORMAT=zip`):
the build without its build tools, the C++ node plugins, the Toast add-on in `addons_core`
(enabled by default: it is in the factory add-on list in `blendfile.cc`), a `portable` folder so
settings stay inside the package, `TOAST_GUIDE.md` and `README_TOAST.txt`. Users extract it and
run `blender` inside; they need g++ or clang++ for Toast to compile kernels. Build Windows and
macOS packages on those platforms (the Toast node is Linux/macOS only for now).

Files saved with these nodes open in stock Blender with the nodes missing (shown as undefined).
