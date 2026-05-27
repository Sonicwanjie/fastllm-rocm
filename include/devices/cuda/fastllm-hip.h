#pragma once

// only take effect when compiling with HIP
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

// MSVC compatibility: make HIP device/host attributes no-ops when compiling with MSVC
// (hipcc defines __HIPCC__, MSVC does not)
#ifndef __HIPCC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
// Override __align__ to use MSVC syntax (HIP defines it as __attribute__((aligned))
// which MSVC cannot parse)
#ifdef _MSC_VER
#undef __align__
#define __align__(x) __declspec(align(x))
#endif
#endif // __HIPCC__

#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>

// BF16 headers: only include with hipcc (Clang-based).
// MSVC cannot parse __attribute__((aligned)), __host__/__device__ etc. in amd_hip_bf16.h
#ifdef __HIPCC__
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>
#else
// MSVC: prevent downstream from including these headers
#ifndef __HIP_BFLOAT16_H__
#define __HIP_BFLOAT16_H__
#endif
#ifndef __HIP_BF16_H__
#define __HIP_BF16_H__
#endif
#endif

// __ldg compatibility
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr) { return *ptr; }

// rocwmma macros
#ifndef ROCWMMA_HOST_DEVICE
#define ROCWMMA_HOST_DEVICE __host__ __device__
#endif
#ifndef ROCWMMA_HOST
#define ROCWMMA_HOST __host__
#endif
#ifndef ROCWMMA_DEVICE
#define ROCWMMA_DEVICE __device__
#endif

// rocwmma only works with hipcc
#if defined(USE_ROCM) && !defined(HIP_NO_TENSOR_CORE) && defined(__HIPCC__)
#include <rocwmma/rocwmma.hpp>
#endif

// ========== SIMD intrinsics (hipcc only) ==========
#ifdef __HIPCC__
typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > 127) tmp = 127;
        if(tmp < -128) tmp = -128;
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    return c;
}
#endif // __HIPCC__

// ========== bfloat16 types (CUDA compat) ==========
#ifdef __HIPCC__
struct __nv_bfloat16 {
    uint16_t __x;
    __host__ __device__ __forceinline__ __nv_bfloat16() : __x(0) {}
    __host__ __device__ __forceinline__ __nv_bfloat16(float f) { __x = hip_bfloat16(f).data; }
    __host__ __device__ __forceinline__ __nv_bfloat16(hip_bfloat16 f) { __x = f.data; }
    __host__ __device__ __forceinline__ __nv_bfloat16(const __nv_bfloat16&) = default;
    __host__ __device__ __forceinline__ __nv_bfloat16& operator=(const __nv_bfloat16&) = default;
    __host__ __device__ __forceinline__ __nv_bfloat16& operator=(float f) { __x = hip_bfloat16(f).data; return *this; }
    __device__ __forceinline__ operator float() const { hip_bfloat16 r; r.data = __x; return static_cast<float>(r); }
    __device__ __forceinline__ operator hip_bfloat16() const { hip_bfloat16 r; r.data = __x; return r; }
};
using __nv_bfloat162 = __hip_bfloat162;
#else
// MSVC host-side: simple POD type matching __hip_bfloat16 layout
struct __nv_bfloat16 {
    uint16_t __x;
    __nv_bfloat16() : __x(0) {}
    __nv_bfloat16(float f) {
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        bits = (bits + (0x7fff + ((bits >> 16) & 1))) >> 16;
        __x = (uint16_t)bits;
    }
    __nv_bfloat16(const __nv_bfloat16&) = default;
    __nv_bfloat16& operator=(const __nv_bfloat16&) = default;
    __nv_bfloat16& operator=(float f) { *this = __nv_bfloat16(f); return *this; }
    operator float() const {
        uint32_t bits = (uint32_t)__x << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    }
};
struct __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
    __nv_bfloat162() : x(), y() {}
    __nv_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) : x(a), y(b) {}
};
#endif // __HIPCC__

// bfloat16 conversion helpers
static inline float __nvbf16_to_float_nv(const __nv_bfloat16& v) {
    return static_cast<float>(v);
}
static inline __nv_bfloat16 __float2nvbf16(float f) {
    return __nv_bfloat16(f);
}

#ifdef __HIPCC__
static __host__ __device__ __forceinline__ float2 __bfloat1622float2_impl(const __nv_bfloat162& v) {
    float2 r;
    r.x = static_cast<float>(v.x);
    r.y = static_cast<float>(v.y);
    return r;
}
#define __bfloat1622float2 __bfloat1622float2_impl
#endif

// FP8 type stubs
#ifdef __HIPCC__
#if defined(__CUDA_ARCH__)
#include <hip/hip_fp8.h>
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
#else
struct __nv_fp8_e4m3 { unsigned char __x; __host__ __device__ __nv_fp8_e4m3() : __x(0) {} __host__ __device__ __nv_fp8_e4m3(float) : __x(0) {} __host__ __device__ __nv_fp8_e4m3(int) : __x(0) {} __host__ __device__ operator float() const { return 0.f; } __host__ __device__ operator half() const { return __float2half(0.f); } __host__ __device__ operator __nv_bfloat16() const { return __nv_bfloat16(0.f); } };
struct __nv_fp8_e5m2 { unsigned char __x; __host__ __device__ __nv_fp8_e5m2() : __x(0) {} __host__ __device__ __nv_fp8_e5m2(float) : __x(0) {} __host__ __device__ __nv_fp8_e5m2(int) : __x(0) {} __host__ __device__ operator float() const { return 0.f; } __host__ __device__ operator half() const { return __float2half(0.f); } __host__ __device__ operator __nv_bfloat16() const { return __nv_bfloat16(0.f); } };
#endif
#else
struct __nv_fp8_e4m3 { unsigned char __x; __nv_fp8_e4m3() : __x(0) {} __nv_fp8_e4m3(float) : __x(0) {} operator float() const { return 0.f; } };
struct __nv_fp8_e5m2 { unsigned char __x; __nv_fp8_e5m2() : __x(0) {} __nv_fp8_e5m2(float) : __x(0) {} operator float() const { return 0.f; } };
#endif

// ========== Union types with guards ==========
#ifndef _FASTLLM_UNION_TYPES_DEFINED
#define _FASTLLM_UNION_TYPES_DEFINED
typedef union __align__(16) {
    uint2 in;
    uint8_t out[8];
} union_char8;

typedef union __align__(16) {
    uint32_t in;
    uint8_t out[4];
} union_char4;

typedef union __align__(16) _union_half_4 {
    uint2 in;
    half out[4];
    half2 out2[2];
    __device__ _union_half_4() {}
} union_half4;

typedef union __align__(16) _union_half_8 {
    uint4 in;
    half out[8];
    half2 out2[4];
    __device__ _union_half_8() {}
} union_half8;
#endif // _FASTLLM_UNION_TYPES_DEFINED

// BF16 union types
#ifndef _UNION_BF16_TYPES_DEFINED
#define _UNION_BF16_TYPES_DEFINED
typedef union __align__(16) _union_bf16_4 {
    uint2 in;
    __nv_bfloat16 out[4];
    __nv_bfloat162 out2[2];
    __device__ _union_bf16_4() {}
} union_bf16_4;

#ifndef _UNION_BF16_4_FP16_DEFINED
#define _UNION_BF16_4_FP16_DEFINED
typedef union __align__(16) _union_bf16_4_fp16 {
    uint2 in;
    __nv_bfloat16 out[4];
    __nv_bfloat162 out2[2];
    __device__ _union_bf16_4_fp16() {}
} union_bf16_4_fp16;
#endif

typedef union __align__(16) _union_bf16_8 {
    uint4 in;
    __nv_bfloat16 out[8];
    __nv_bfloat162 out2[4];
    __device__ _union_bf16_8() {}
} union_bf16_8;
#endif // _UNION_BF16_TYPES_DEFINED

// ========== hipblas wrappers ==========
namespace fastllm_hip {

    inline const hipblasHalf* ToHipblasHalfConst(const half* x) {
        return reinterpret_cast<const hipblasHalf*>(x);
    }
    inline hipblasHalf* ToHipblasHalf(half* x) {
        return reinterpret_cast<hipblasHalf*>(x);
    }

    inline hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle,
        hipblasOperation_t transA, hipblasOperation_t transB,
        int m, int n, int k,
        const void* alpha, const void* A, hipDataType aType, int lda,
        const void* B, hipDataType bType, int ldb,
        const void* beta, void* C, hipDataType cType, int ldc,
        hipblasComputeType_t computeType, hipblasGemmAlgo_t algo) {
        return ::hipblasGemmEx(handle, transA, transB, m, n, k, alpha, A, aType, lda, B, bType, ldb, beta, C, cType, ldc, computeType, algo);
    }

    inline hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t handle,
        hipblasOperation_t transA, hipblasOperation_t transB,
        int m, int n, int k,
        const half* alpha, const half* AP, int lda, long long strideA,
        const half* BP, int ldb, long long strideB,
        const half* beta, half* CP, int ldc, long long strideC,
        int batchCount) {
        return ::hipblasHgemmStridedBatched(handle, transA, transB, m, n, k,
            ToHipblasHalfConst(alpha), ToHipblasHalfConst(AP), lda, strideA,
            ToHipblasHalfConst(BP), ldb, strideB,
            ToHipblasHalfConst(beta), ToHipblasHalf(CP), ldc, strideC, batchCount);
    }

    inline hipblasStatus_t hipblasHgemm(hipblasHandle_t handle,
        hipblasOperation_t transA, hipblasOperation_t transB,
        int m, int n, int k,
        const half *alpha, const half *AP, int lda,
        const half *BP, int ldb,
        const half *beta, half *CP, int ldc) {
        return ::hipblasHgemm(handle, transA, transB, m, n, k,
            ToHipblasHalfConst(alpha), ToHipblasHalfConst(AP), lda,
            ToHipblasHalfConst(BP), ldb,
            ToHipblasHalfConst(beta), ToHipblasHalf(CP), ldc);
    }
} // namespace fastllm_hip

#define hipblasGemmEx fastllm_hip::hipblasGemmEx
#define hipblasHgemmStridedBatched fastllm_hip::hipblasHgemmStridedBatched
#define hipblasHgemm fastllm_hip::hipblasHgemm

// ========== CUDA error compat ==========
#define checkCudaErrors(msg, val) showError(val, msg, __FILE__, __LINE__)
#define cudaErrorNotSupported hipErrorNotSupported

#ifndef HIPBLAS_GEMM_DEFAULT_TENSOR_OP
#define HIPBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#endif

// ========== Other CUDA->HIP mappings ==========
#define __grid_constant__
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

#ifdef __HIPCC__
static __device__ __forceinline__ half2 __hmul2_fn(half2 a, half2 b) {
    return __hmul2(a, b);
}
static __device__ __forceinline__ float __fmaf_ieee_rn_fn(float a, float b) {
    return a * b;
}
#endif

#endif // __HIP_PLATFORM_AMD__
