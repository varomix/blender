# Blender with Toast

This is a fork of [Blender](https://projects.blender.org/blender/blender) that adds
**Toast**: Python-syntax code compiled to native kernels, running inside a geometry node, plus
C++ node plugins that load without rebuilding Blender. The `toast-<version>` branches hold the
fork for each Blender release, as a short series of commits on top of Blender's release tag;
`toast` points at the newest one.

- `toast/addon`: the Toast add-on and compiler ([varomix/toast](https://github.com/varomix/toast)),
  a git submodule, installed as a core add-on and enabled by default.
- `source/blender/nodes/geometry/nodes/toast/`: the Toast node, the plugin loader and the
  plugins; its `README.md` lists everything the fork changes.
- `toast/addon/docs/guide.md`: how to write Toast code.

## Building

Build it like Blender ([build instructions](https://developer.blender.org/docs/handbook/building_blender/)),
from this branch:

```bash
git clone -b toast https://github.com/varomix/blender.git
cd blender
git submodule update --init toast/addon
make update
make
```

Toast compiles kernels with a C++ compiler at run time: install g++ or clang++. The Toast node
runs on Linux and macOS for now.

`build_files/toast/package_portable.sh` makes a portable archive with the add-on, the plugins
and a self-contained settings folder.

For add-on development, point the build at a checkout and link instead of copying:
`cmake -DTOAST_ADDON_DIR=/path/to/toast -DTOAST_ADDON_LINK=ON <build dir>`.

## License

GPL-2.0-or-later, like Blender.
