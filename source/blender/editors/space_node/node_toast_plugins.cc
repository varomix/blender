/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup spnode
 *
 * Toast fork: operator to reload C++ node plugins (see `NOD_toast_plugins.hh`).
 */

#include "DNA_node_types.h"

#include "BLI_listbase.hh"

#include "BKE_context.hh"
#include "BKE_main.hh"
#include "BKE_main_invariants.hh"
#include "BKE_node_tree_update.hh"
#include "BKE_report.hh"

#include "DEG_depsgraph_build.hh"

#include "NOD_toast_plugins.hh"

#include "UI_interface_c.hh"

#ifdef WITH_PYTHON
#  include "BPY_extern.hh"
#endif

#include "WM_api.hh"
#include "WM_types.hh"

#include "node_intern.hh"

namespace blender::ed::space_node {

static wmOperatorStatus node_toast_reload_plugins_exec(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  const nodes::toast_plugins::ReloadResult result = nodes::toast_plugins::reload_all(
      [&](StructRNA *srna) {
        ui::refresh_for_srna_unregister(bmain, srna);
#ifdef WITH_PYTHON
        /* `bpy.types` may hold a class for the type; release it before the struct is freed, as
         * operator types do. */
        BPY_free_srna_pytype(srna);
#endif
      });

  /* Rebuild everything that may refer to the old node types. */
  for (bNodeTree &ntree : bmain->nodetrees) {
    BKE_ntree_update_tag_all(&ntree);
  }
  BKE_main_ensure_invariants(*bmain);
  DEG_relations_tag_update(bmain);
  WM_main_add_notifier(NC_NODE | NA_EDITED, nullptr);

  for (const std::string &error : result.errors) {
    BKE_report(op->reports, RPT_ERROR, error.c_str());
  }
  BKE_reportf(op->reports,
              result.errors.is_empty() ? RPT_INFO : RPT_WARNING,
              "Reloaded %d node plugin(s)",
              int(result.loaded.size()));
  return OPERATOR_FINISHED;
}

void NODE_OT_toast_reload_plugins(wmOperatorType *ot)
{
  ot->name = "Reload Node Plugins";
  ot->description =
      "Load the C++ node plugins again from their folders, picking up rebuilt plugin libraries";
  ot->idname = "NODE_OT_toast_reload_plugins";

  ot->exec = node_toast_reload_plugins_exec;

  ot->flag = OPTYPE_REGISTER;
}

}  // namespace blender::ed::space_node
