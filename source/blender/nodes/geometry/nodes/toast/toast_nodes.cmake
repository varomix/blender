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

# The built-in kernel compiler: `libtoast_jit`, statically linked with clang and LLVM from the
# precompiled libraries, loaded by the Toast node when a kernel is C++ source. Only its own
# functions are exported, so its LLVM can't clash with other copies in the process (OSL's).
set(TOAST_JIT_LLVM_DIR "${LIBDIR}/llvm" CACHE PATH "LLVM with clang libraries for the Toast JIT")
mark_as_advanced(TOAST_JIT_LLVM_DIR)
if(UNIX AND NOT APPLE AND EXISTS "${TOAST_JIT_LLVM_DIR}/bin/llvm-config" AND
   EXISTS "${TOAST_JIT_LLVM_DIR}/lib/libclangCodeGen.a")
  execute_process(
    COMMAND "${TOAST_JIT_LLVM_DIR}/bin/llvm-config" --libfiles all
    OUTPUT_VARIABLE _toast_llvm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND "${TOAST_JIT_LLVM_DIR}/bin/llvm-config" --system-libs
    OUTPUT_VARIABLE _toast_llvm_system_libs OUTPUT_STRIP_TRAILING_WHITESPACE)
  separate_arguments(_toast_llvm_libs)
  separate_arguments(_toast_llvm_system_libs)
  file(GLOB _toast_clang_libs "${TOAST_JIT_LLVM_DIR}/lib/libclang*.a")
  add_library(toast_jit SHARED ${CMAKE_CURRENT_SOURCE_DIR}/nodes/toast/jit/toast_jit.cc)
  target_include_directories(toast_jit SYSTEM PRIVATE "${TOAST_JIT_LLVM_DIR}/include")
  target_link_libraries(toast_jit PRIVATE
    -Wl,--start-group ${_toast_clang_libs} ${_toast_llvm_libs} -Wl,--end-group
    ${_toast_llvm_system_libs})
  target_link_options(toast_jit PRIVATE -Wl,--exclude-libs,ALL -Wl,--gc-sections)
  set_target_properties(toast_jit PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/lib
  )
else()
  message(STATUS "Toast: no clang/LLVM libraries in ${TOAST_JIT_LLVM_DIR}, "
                 "kernels need an external C++ compiler")
endif()

