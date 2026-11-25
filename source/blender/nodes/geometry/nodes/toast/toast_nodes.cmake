# SPDX-FileCopyrightText: 2026 varomix
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Toast fork geometry nodes.
# Included from `source/blender/nodes/geometry/CMakeLists.txt`, appends to its `SRC` list so the
# nodes are compiled into `bf_nodes_geometry` and picked up by node discovery
# (`NOD_REGISTER_NODE`). To add a built-in node: create `node_geo_<name>.cc` here, list it below
# and add its idname to `scripts/startup/bl_ui/node_add_menu_toast.py`.

list(APPEND SRC
  nodes/toast/node_geo_toast.cc
  nodes/toast/toast_abi.h
)
