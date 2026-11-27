/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Toast fork: built with `__TOAST_NODES__` instead of OSL for scenes with Toast nodes: the same
 * modules shading in OptiX, with SVM calling the Toast functions (see kernel/svm/toast.h). */
#ifndef __TOAST_NODES__
#  define WITH_OSL
#endif

/* Volume shading raygens for OSL, loaded as a separate optix module so they
 * can be compiled in parallel with the base OSL module and skipped for
 * scenes without volumes. */

#include "kernel/device/optix/compat.h"
#include "kernel/device/optix/globals.h"
#ifdef __TOAST_NODES__
#  include "kernel/device/gpu/image.h" /* Toast fork: SVM looks up textures. */
#endif

#include "kernel/film/data_passes.h"

#include "kernel/integrator/shade_volume.h"

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_volume()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_volume(nullptr, path_index, kernel_params.render_buffer);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_volume_ray_marching()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_volume_ray_marching(nullptr, path_index, kernel_params.render_buffer);
}
