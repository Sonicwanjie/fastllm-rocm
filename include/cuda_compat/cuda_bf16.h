#pragma once
// cuda_bf16.h - MSVC-compatible bfloat16 definitions for host-side compilation
//
// This header is found via the include path before ROCm's hip_bfloat16.h.
// It provides MSVC-compatible types so that host code can handle BF16 data
// without needing the Clang-specific constructs in amd_hip_bf16.h.

#ifndef __HIPCC__
// MSVC path - use our own lightweight BF16 type
// The actual cuda_bf16.h from cuda_shim will be found next via include path
// This file exists to intercept the include before ROCm's version is found
#include "cuda_bf16.h"  // forwards to cuda_shim version
#else
// hipcc path - use real HIP headers
#include <hip/hip_bfloat16.h>
#endif
