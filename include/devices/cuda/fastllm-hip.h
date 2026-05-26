#pragma once

// only take effect when compiling with HIP
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>

// __ldg compatibility - on AMD GPUs, regular loads are sufficient
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr) { return *ptr; }

// Ensure rocwmma macros are defined before any rocwmma header is pulled in
// (e.g. via hip_fp8.h -> rocwmma/internal/float8.hpp)
#ifndef ROCWMMA_HOST_DEVICE
#define ROCWMMA_HOST_DEVICE __host__ __device__
#endif
#ifndef ROCWMMA_HOST
#define ROCWMMA_HOST __host__
#endif
#ifndef ROCWMMA_DEVICE
#define ROCWMMA_DEVICE __device__
#endif

#if defined(USE_ROCM) && !defined(HIP_NO_TENSOR_CORE) // support tensor core
#include <rocwmma/rocwmma.hpp>
#endif

// ========== Warp shuffle macros ==========
// Use inline functions to handle both calling conventions




// ========== SIMD intrinsics ==========
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

// ========== bfloat16 types (CUDA compat) ==========
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

// __nv_bfloat162: use __hip_bfloat162 from ROCm SDK
// Provides x, y members (__hip_bfloat16 type) and full operator support
using __nv_bfloat162 = __hip_bfloat162;


// bfloat16 conversion helpers
// __nv_bfloat16 has operator float() and operator hip_bfloat16() for implicit conversion
// So HIP native __bfloat162float(hip_bfloat16) and __float2bfloat16(float) work via implicit conversion
// But for explicit __nv_bfloat16 arguments, we provide these helpers:
static __host__ __device__ __forceinline__ float __nvbf16_to_float_nv(const __nv_bfloat16& v) {
    return static_cast<float>(v);
}
static __host__ __device__ __forceinline__ __nv_bfloat16 __float2nvbf16(float f) {
    return __nv_bfloat16(f);
}
// Override HIP bfloat16 functions to also accept __nv_bfloat16
// (these are separate overloads, not macros, to avoid ambiguity)

static __host__ __device__ __forceinline__ float2 __bfloat1622float2_impl(const __nv_bfloat162& v) {
    float2 r;
    r.x = static_cast<float>(v.x);
    r.y = static_cast<float>(v.y);
    return r;
}
#define __bfloat1622float2 __bfloat1622float2_impl
// FP8 type forward declarations
// amd_hip_bf16.h has Clang-incompatible 'static' operators
// Only include hip_fp8.h during actual device code generation (not host C++ compile)
#if defined(__CUDA_ARCH__)
#include <hip/hip_fp8.h>
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
#else
// Forward declare FP8 types for host/side compilation
struct __nv_fp8_e4m3 { unsigned char __x; __host__ __device__ __nv_fp8_e4m3() : __x(0) {} __host__ __device__ __nv_fp8_e4m3(float) : __x(0) {} __host__ __device__ __nv_fp8_e4m3(int) : __x(0) {} __host__ __device__ operator float() const { return 0.f; } __host__ __device__ operator half() const { return __float2half(0.f); } __host__ __device__ operator __nv_bfloat16() const { return __nv_bfloat16(0.f); } };
struct __nv_fp8_e5m2 { unsigned char __x; __host__ __device__ __nv_fp8_e5m2() : __x(0) {} __host__ __device__ __nv_fp8_e5m2(float) : __x(0) {} __host__ __device__ __nv_fp8_e5m2(int) : __x(0) {} __host__ __device__ operator float() const { return 0.f; } __host__ __device__ operator half() const { return __float2half(0.f); } __host__ __device__ operator __nv_bfloat16() const { return __nv_bfloat16(0.f); } };
#endif

// ========== Union types with guards ==========
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
#endif // _UNION_BF16_4_FP16_DEFINED

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

// Use inline namespace wrappers instead of 'using' to avoid conflicts with hipblas declarations
// Code should call fastllm_hip::hipblasGemmEx() etc. directly, or we use #define redirects
#define hipblasGemmEx fastllm_hip::hipblasGemmEx
#define hipblasHgemmStridedBatched fastllm_hip::hipblasHgemmStridedBatched
#define hipblasHgemm fastllm_hip::hipblasHgemm

// ========== CUDA error compat ==========
#define checkCudaErrors(msg, val) showError(val, msg, __FILE__, __LINE__)
#define cudaErrorNotSupported hipErrorNotSupported

// HIPBLAS_GEMM_DEFAULT_TENSOR_OP may not be defined in older hipBLAS
#ifndef HIPBLAS_GEMM_DEFAULT_TENSOR_OP
#define HIPBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#endif

// ========== Other CUDA->HIP mappings ==========
#define __grid_constant__
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

// __hmul2 must be inline function, not macro
static __device__ __forceinline__ half2 __hmul2_fn(half2 a, half2 b) {
    return __hmul2(a, b);
}

// __fmaf_ieee_rn -> simple multiply
static __device__ __forceinline__ float __fmaf_ieee_rn_fn(float a, float b) {
    return a * b;
}

#endif // __HIP_PLATFORM_AMD__







