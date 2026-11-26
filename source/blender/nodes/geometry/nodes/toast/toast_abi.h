/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * C ABI between `GeometryNodeToast` and Toast kernels. This is a copy of the ABI section of
 * `toast_rt.h` in the Toast add-on; both must stay identical. Change it only together with
 * TOAST_ABI_VERSION, since kernels compiled against an older ABI are rejected by version.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#define TOAST_ABI_VERSION 9

extern "C" {

struct ToastCmdBuffer;

enum {
  TOAST_TYPE_FLOAT = 0,
  TOAST_TYPE_INT = 1,
  TOAST_TYPE_BOOL = 2,
  TOAST_TYPE_FLOAT2 = 3,
  TOAST_TYPE_FLOAT3 = 4,
  TOAST_TYPE_FLOAT4 = 5,
  TOAST_TYPE_QUAT = 6,
  TOAST_TYPE_MAT4 = 7,
};

enum {
  TOAST_DOMAIN_POINT = 0,
  TOAST_DOMAIN_FACE = 1,
  TOAST_DOMAIN_EDGE = 2,
  TOAST_DOMAIN_CORNER = 3,
  TOAST_DOMAIN_CURVE = 4,
  TOAST_DOMAIN_INSTANCE = 5,
};

typedef struct ToastHit {
  float position[3];
  float normal[3];
  float bary[3];
  float distance;
  int32_t face;
  int32_t tri_points[3];
  int32_t hit;
} ToastHit;

typedef struct ToastAttr {
  void *data;
  int64_t size;
  int32_t type;
  int32_t domain;
} ToastAttr;

typedef struct ToastHost {
  void *self;
  int32_t (*nearpoints)(
      void *self, int32_t input, const float *pos3, float radius, int32_t max, int64_t *out);
  int64_t (*npoints)(void *self, int32_t input);
  int64_t (*cmd_add_point)(void *self, ToastCmdBuffer *cmd, int64_t src, const float *pos3);
  void (*cmd_add_prim)(
      void *self, ToastCmdBuffer *cmd, int64_t src, const int64_t *pts, int32_t n);
  void (*cmd_remove_point)(void *self, ToastCmdBuffer *cmd, int64_t src, int64_t pt);
  /* ABI 2. */
  int64_t (*nprims)(void *self, int32_t input);
  int32_t (*primpoints)(void *self, int32_t input, int64_t prim, int32_t max, int64_t *out);
  void (*cmd_remove_prim)(void *self, ToastCmdBuffer *cmd, int64_t src, int64_t prim);
  /* ABI 3. */
  int32_t (*point_faces)(void *self, int32_t input, int64_t point, int32_t max, int64_t *out);
  int32_t (*point_neighbors)(void *self, int32_t input, int64_t point, int32_t max, int64_t *out);
  int64_t (*corner_point)(void *self, int32_t input, int64_t corner);
  int64_t (*corner_face)(void *self, int32_t input, int64_t corner);
  void (*cmd_set_attr)(
      void *self, ToastCmdBuffer *cmd, int64_t src, int32_t attr, int64_t elem, const void *value);
  /* ABI 4. */
  int32_t (*ray_cast)(void *self,
                      int32_t input,
                      const float *origin,
                      const float *dir,
                      float max_dist,
                      ToastHit *out);
  int32_t (*closest_point)(
      void *self, int32_t input, const float *pos, float max_dist, ToastHit *out);
  /* ABI 5. */
  int64_t (*ncurves)(void *self, int32_t input);
  int32_t (*curve_points)(void *self, int32_t input, int64_t curve, int32_t max, int64_t *out);
  void (*cmd_remove_curve)(void *self, ToastCmdBuffer *cmd, int64_t src, int64_t curve);
  /* ABI 6. */
  void (*cmd_add_curve)(
      void *self, ToastCmdBuffer *cmd, int64_t src, const int64_t *pts, int32_t n);
  int64_t (*cmd_extend_curve)(
      void *self, ToastCmdBuffer *cmd, int64_t src, int64_t curve, const float *pos3);
  /* ABI 7. */
  int32_t (*surface_point)(
      void *self, int32_t input, float u, float v, float w, ToastHit *out);
  /* ABI 8. */
  int32_t (*grid_sample)(void *self, int32_t grid, int32_t mode, const float *pos3, float *out);
  /* ABI 9. */
  void (*voxel_cmd)(void *self, int32_t op, const int32_t *ijk, int32_t radius);
} ToastHost;

typedef struct ToastCtx {
  uint32_t abi_version, _pad;
  int64_t begin, end;
  const int64_t *indices;
  int64_t domain_size;
  ToastAttr *attrs;
  const void *params;
  ToastCmdBuffer *cmd;
  const ToastHost *host;
  int64_t chunk_id;
  float time, frame;
} ToastCtx;

typedef void (*ToastKernelFn)(const ToastCtx *ctx);
typedef int32_t (*ToastAbiVersionFn)();
typedef const char *(*ToastSignatureFn)();

}  // extern "C"

/* Layout checks; the same values are asserted by the harness through ctypes. */
static_assert(sizeof(ToastAttr) == 24);
static_assert(sizeof(ToastHost) == 192);
static_assert(sizeof(ToastHit) == 60);
static_assert(offsetof(ToastCtx, begin) == 8);
static_assert(offsetof(ToastCtx, attrs) == 40);
static_assert(offsetof(ToastCtx, chunk_id) == 72);
static_assert(sizeof(ToastCtx) == 88);
