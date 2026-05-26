#pragma once
#include <hip/hip_bfloat16.h>

// Type aliases for CUDA bfloat16 types -> HIP equivalents
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;

// CUDA math functions for bfloat16 -> HIP equivalents
inline __half __float2half(const float f) { return __float2half(f); }
inline float __half2float(const __half h) { return __half2float(h); }

// __store_global and __load_global are CUDA-specific, use regular stores/loads on HIP
template<typename T>
__device__ __forceinline__ void __store_global(T* addr, const T& val) {
    *addr = val;
}

template<typename T>
__device__ __forceinline__ T __load_global(const T* addr) {
    return *addr;
}
