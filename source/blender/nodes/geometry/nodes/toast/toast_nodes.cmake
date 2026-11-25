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
  nodes/toast/toast_plugins.cc
)

# C++ node plugins: shared modules written to `bin/toast_plugins`, loaded at startup and by
# Reload Plugins (see `NOD_toast_plugins.hh`). They compile with the include directories,
# definitions and options of `bf_nodes_geometry` but don't link Blender's libraries: those
# symbols come from the blender executable when the plugin is loaded. Rebuild one plugin with
# `make <name>` in the build folder, no Blender relink needed.
function(toast_add_node_plugin name)
  add_library(${name} MODULE ${ARGN})
  target_include_directories(${name} PRIVATE
    $<TARGET_PROPERTY:bf_nodes_geometry,INCLUDE_DIRECTORIES>
  )
  target_compile_definitions(${name} PRIVATE
    $<TARGET_PROPERTY:bf_nodes_geometry,COMPILE_DEFINITIONS>
  )
  target_compile_options(${name} PRIVATE
    $<TARGET_PROPERTY:bf_nodes_geometry,COMPILE_OPTIONS>
  )
  get_target_property(_libs bf_nodes_geometry LINK_LIBRARIES)
  foreach(_lib ${_libs})
    if(TARGET ${_lib})
      target_link_libraries(${name} PRIVATE $<COMPILE_ONLY:${_lib}>)
    endif()
  endforeach()
  add_dependencies(${name} bf_nodes_geometry)
  set_target_properties(${name} PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/toast_plugins
  )
endfunction()

function(toast_add_node_plugins)
  toast_add_node_plugin(toast_plugin_bricker
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/toast/plugins/bricker/node_geo_bricker.cc
  )
  # Hand-written C++ reference for the Toast benchmark.
  toast_add_node_plugin(toast_plugin_noise_displace
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/toast/plugins/noise_displace/node_geo_noise_displace.cc
  )
endfunction()

# Deferred so the plugins see the final settings of `bf_nodes_geometry`.
cmake_language(DEFER CALL toast_add_node_plugins)
