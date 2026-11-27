/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Toast fork: what render engines need from the shader editor's Toast node (`ShaderNodeToast`).
 *
 * The Toast add-on compiles the node's code into a C++ function for the CPU (the node's
 * `kernel_path`):
 *
 * \code{.cc}
 * extern "C" void toast_shader(const float *in, float *out);
 * \endcode
 *
 * `in` holds the parameters, then the shading inputs, `out` the outputs, each as floats in
 * signature order: float, int and bool take one float, vec3 three, vec4 four.
 */

#pragma once

#include <optional>
#include <string>

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

namespace blender {
struct bNode;
}

namespace blender::nodes {

using ToastShaderFn = void (*)(const float *in, float *out);

/** A parameter, shading input or output of a Toast shader function. */
struct ToastShaderSocket {
  /** The node's socket identifier for parameters ("p_<name>") and outputs ("o_<name>"); the
   * attribute name for shading inputs. */
  std::string identifier;
  /** Toast type: "float", "int", "bool", "vec3" or "vec4". */
  std::string type;
  /** Shading inputs: "builtin" (position, uv, ...) or "attribute" (a named attribute). */
  std::string kind;

  /** Floats this value takes in `in` / `out`. */
  int size() const
  {
    return type == "vec3" ? 3 : (type == "vec4" ? 4 : 1);
  }
};

struct ToastShaderInfo {
  ToastShaderFn function = nullptr;
  /** The function for GPUs, in Blender's portable shader dialect (the node's `gpu_source`), and
   * its name. Arguments: the params, then the shading inputs (named attributes as float4, as
   * the Attribute node gets them), then the outputs as references. */
  std::string gpu_source;
  std::string gpu_function;
  Vector<ToastShaderSocket> params;
  Vector<ToastShaderSocket> inputs;
  Vector<ToastShaderSocket> outputs;
};

/**
 * The compiled CPU function of a `ShaderNodeToast` and its sockets, or nothing (with
 * `r_error` set when there is something to tell): no code, a compile error, an untrusted file
 * (Auto Run Python Scripts), or no built-in compiler. Compiles on first use, then cached.
 * Thread-safe.
 */
std::optional<ToastShaderInfo> toast_shader_info(const bNode &node, std::string &r_error);

/**
 * Load the shader function `toast_shader` from a kernel file: C++ source (`.cpp`, compiled by
 * the built-in compiler) or a shared library. Cached per path for the session. Defined with the
 * geometry node, which shares the loader.
 */
ToastShaderFn toast_load_shader_function(StringRefNull path, std::string &r_error);

}  // namespace blender::nodes
