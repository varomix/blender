/* SPDX-FileCopyrightText: 2026 varomix
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Toast fork: the built-in kernel compiler. Compiles a Toast kernel (self-contained C++, the
 * runtime header inlined) in memory with clang, to an object for the host CPU, and links it with
 * LLVM's ORC JIT. Built as its own shared library (`libtoast_jit`), statically linked with the
 * clang and LLVM libraries from Blender's precompiled libraries, exporting only the functions
 * below so its LLVM can't clash with other LLVM copies in the process (OSL has one).
 *
 * Kernels are compiled with the same floating point rules as the external compiler path (no fast
 * math, no FMA contraction, wrapping signed integers), so both give identical results.
 */

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PreprocessorOptions.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include "toast_jit.h"

#define TOAST_JIT_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

struct Module {
  std::unique_ptr<llvm::orc::LLJIT> jit;
};

std::once_flag init_flag;

void init_llvm()
{
  std::call_once(init_flag, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
}

void set_error(char *error, const int error_size, const std::string &message)
{
  if (error && error_size > 0) {
    std::strncpy(error, message.c_str(), error_size - 1);
    error[error_size - 1] = '\0';
  }
}

/* Source -> optimized LLVM IR, with clang in memory. */
std::unique_ptr<llvm::Module> compile_to_ir(const std::string &source,
                                            llvm::LLVMContext &context,
                                            std::string &r_error)
{
  auto ci = std::make_unique<clang::CompilerInstance>();
  clang::DiagnosticOptions diag_opts;
  std::string diagnostics;
  llvm::raw_string_ostream diag_stream(diagnostics);
  auto *printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
  ci->createDiagnostics(printer, true);

  const std::vector<std::string> args_s = {
      "-x", "c++", "-std=c++17", "-O3",
      /* Same floating point rules as the external compiler path. */
      "-ffp-contract=off", "-fwrapv",
      "-triple", llvm::sys::getProcessTriple(),
      "-target-cpu", std::string(llvm::sys::getHostCPUName()),
      /* Kernels are self-contained: no system or clang headers. */
      "-nostdsysteminc", "-nobuiltininc",
      "toast_kernel.cpp"};  // clang-format: keep
  std::vector<const char *> args;
  for (const std::string &a : args_s) {
    args.push_back(a.c_str());
  }
  if (!clang::CompilerInvocation::CreateFromArgs(ci->getInvocation(), args, ci->getDiagnostics()))
  {
    r_error = "invalid compiler arguments: " + diagnostics;
    return nullptr;
  }
  ci->getPreprocessorOpts().addRemappedFile(
      "toast_kernel.cpp", llvm::MemoryBuffer::getMemBufferCopy(source).release());

  clang::EmitLLVMOnlyAction action(&context);
  if (!ci->ExecuteAction(action)) {
    r_error = "compile failed:\n" + diagnostics;
    return nullptr;
  }
  return action.takeModule();
}

/* IR -> object file for the host CPU. */
std::unique_ptr<llvm::MemoryBuffer> compile_to_object(llvm::Module &module, std::string &r_error)
{
  auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!jtmb) {
    r_error = llvm::toString(jtmb.takeError());
    return nullptr;
  }
  jtmb->setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
  llvm::TargetOptions options;
  options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
  jtmb->getOptions() = options;
  auto tm = jtmb->createTargetMachine();
  if (!tm) {
    r_error = llvm::toString(tm.takeError());
    return nullptr;
  }
  module.setDataLayout((*tm)->createDataLayout());
  module.setTargetTriple((*tm)->getTargetTriple());

  llvm::SmallVector<char, 0> object;
  llvm::raw_svector_ostream stream(object);
  llvm::legacy::PassManager passes;
  if ((*tm)->addPassesToEmitFile(passes, stream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
    r_error = "the target can't emit object files";
    return nullptr;
  }
  passes.run(module);
  return std::make_unique<llvm::SmallVectorMemoryBuffer>(std::move(object), false);
}

}  // namespace

TOAST_JIT_EXPORT int toast_jit_api_version()
{
  return TOAST_JIT_API_VERSION;
}

TOAST_JIT_EXPORT const char *toast_jit_llvm_version()
{
  return LLVM_VERSION_STRING;
}

/**
 * Compile `source` and link it. With `cache_path` (a path without extension, unique for the
 * source), the object is cached next to it, per CPU and LLVM version: a cached object is used
 * instead of compiling, and a compiled one is written. Returns an opaque module, or null with a
 * message in `error`.
 */
TOAST_JIT_EXPORT void *toast_jit_compile(const char *source,
                                         const char *cache_path,
                                         char *error,
                                         const int error_size)
{
  init_llvm();
  std::string message;
  std::unique_ptr<llvm::MemoryBuffer> object;
  std::string object_path;
  if (cache_path && cache_path[0]) {
    object_path = std::string(cache_path) + "." + std::string(llvm::sys::getHostCPUName()) +
                  ".llvm" + LLVM_VERSION_STRING + ".o";
  }

  if (!object_path.empty()) {
    auto cached = llvm::MemoryBuffer::getFile(object_path);
    if (cached) {
      object = std::move(*cached);
    }
  }
  if (!object) {
    auto context = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> module = compile_to_ir(source, *context, message);
    if (!module) {
      set_error(error, error_size, message);
      return nullptr;
    }
    object = compile_to_object(*module, message);
    if (!object) {
      set_error(error, error_size, message);
      return nullptr;
    }
    if (!object_path.empty()) {
      /* Write next to the final name and rename, so readers never see a partial file. */
      const std::string tmp = object_path + ".tmp";
      {
        std::ofstream out(tmp, std::ios::binary);
        out.write(object->getBufferStart(), object->getBufferSize());
      }
      std::rename(tmp.c_str(), object_path.c_str());
    }
  }

  auto jit = llvm::orc::LLJITBuilder().create();
  if (!jit) {
    set_error(error, error_size, llvm::toString(jit.takeError()));
    return nullptr;
  }
  /* Kernels call libm and memcpy: resolve those from the process. */
  auto generator = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      (*jit)->getDataLayout().getGlobalPrefix());
  if (!generator) {
    set_error(error, error_size, llvm::toString(generator.takeError()));
    return nullptr;
  }
  (*jit)->getMainJITDylib().addGenerator(std::move(*generator));
  if (auto err = (*jit)->addObjectFile(std::move(object))) {
    set_error(error, error_size, llvm::toString(std::move(err)));
    return nullptr;
  }
  auto *result = new Module();
  result->jit = std::move(*jit);
  return result;
}

/** Address of a kernel function, or null. Modules are never freed: kernels stay callable. */
TOAST_JIT_EXPORT void *toast_jit_symbol(void *module, const char *name)
{
  auto *m = static_cast<Module *>(module);
  auto address = m->jit->lookup(name);
  if (!address) {
    llvm::consumeError(address.takeError());
    return nullptr;
  }
  return address->toPtr<void *>();
}
