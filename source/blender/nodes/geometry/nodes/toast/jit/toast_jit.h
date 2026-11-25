/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Toast fork: C interface of `libtoast_jit`, the built-in kernel compiler (see toast_jit.cc).
 * The Toast node loads the library at run time, so these are also its function pointer types.
 */

#pragma once

#define TOAST_JIT_API_VERSION 1

extern "C" {
int toast_jit_api_version();
const char *toast_jit_llvm_version();
void *toast_jit_compile(const char *source, const char *cache_path, char *error, int error_size);
void *toast_jit_symbol(void *module, const char *name);

typedef int (*ToastJitApiVersionFn)();
typedef void *(*ToastJitCompileFn)(const char *source,
                                   const char *cache_path,
                                   char *error,
                                   int error_size);
typedef void *(*ToastJitSymbolFn)(void *module, const char *name);
}
