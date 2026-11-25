# SPDX-FileCopyrightText: 2026 varomix
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Add/Swap menu entries for the Toast fork geometry nodes."""

import bpy
from bl_ui import node_add_menu


# Node idnames shown in the "Toast" menu, in display order. Node types from C++ plugins are shown
# only while their plugin is loaded.
TOAST_GEOMETRY_NODES = (
    "GeometryNodeToast",
    "GeometryNodeBricker",
)


class NODE_MT_gn_toast_base(node_add_menu.NodeMenu):
    bl_label = "Toast"

    def draw(self, _context):
        layout = self.layout
        for idname in TOAST_GEOMETRY_NODES:
            if getattr(bpy.types, idname, None) is not None:
                self.node_operator(layout, idname)


        self.draw_assets_for_catalog(layout, self.bl_label)
