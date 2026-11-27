/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Toast fork: NODE_TOAST calls a function compiled from Toast code (the shader editor's Toast
 * node). The function only exists on the CPU; other devices write zero outputs. */

#pragma once

#include "kernel/svm/node_types.h"
#include "kernel/svm/util.h"

CCL_NAMESPACE_BEGIN

ccl_device_noinline void svm_node_toast(ccl_private float *ccl_restrict stack,
                                        const ccl_global SVMNodeToast &ccl_restrict node)
{
#ifndef __KERNEL_GPU__
  /* Each value takes at most four floats. */
  float in[SVM_TOAST_MAX_INPUTS * 4];
  float out[SVM_TOAST_MAX_OUTPUTS * 4];
  int k = 0;
  for (int i = 0; i < node.num_inputs; i++) {
    const uint offset = node.input_offset[i];
    switch (node.input_type[i]) {
      case NODE_TOAST_FLOAT:
        in[k++] = stack_load_float(stack, offset);
        break;
      case NODE_TOAST_INT:
        in[k++] = float(stack_load_int(stack, offset));
        break;
      case NODE_TOAST_FLOAT3:
      case NODE_TOAST_COLOR4: {
        const float3 v = stack_load_float3(stack, offset);
        in[k++] = v.x;
        in[k++] = v.y;
        in[k++] = v.z;
        if (node.input_type[i] == NODE_TOAST_COLOR4) {
          in[k++] = 1.0f;
        }
        break;
      }
    }
  }
  for (int i = 0; i < SVM_TOAST_MAX_OUTPUTS * 4; i++) {
    out[i] = 0.0f;
  }
  using ToastShaderFn = void (*)(const float *, float *);
  const uint64_t address = uint64_t(node.function_lo) | (uint64_t(node.function_hi) << 32);
  reinterpret_cast<ToastShaderFn>(address)(in, out);
#endif
  int k_out = 0;
  for (int i = 0; i < node.num_outputs; i++) {
    const uint offset = node.output_offset[i];
    const NodeToastValue type = node.output_type[i];
#ifndef __KERNEL_GPU__
    const float *v = out + k_out;
#else
    const float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
    k_out += (type == NODE_TOAST_FLOAT3) ? 3 : ((type == NODE_TOAST_COLOR4) ? 4 : 1);
    if (!stack_valid(offset)) {
      continue;
    }
    switch (type) {
      case NODE_TOAST_FLOAT:
        stack_store_float(stack, offset, v[0]);
        break;
      case NODE_TOAST_INT:
        stack_store_int(stack, offset, int(v[0]));
        break;
      case NODE_TOAST_FLOAT3:
      case NODE_TOAST_COLOR4:
        stack_store_float3(stack, offset, make_float3(v[0], v[1], v[2]));
        break;
    }
  }
}

CCL_NAMESPACE_END
