#pragma once
// cublas compatibility shim for HIP builds
#include <hipblas/hipblas.h>
// #include <hipblas/hipblas_v2.h>  // not available in ROCm SDK

// Type aliases
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasOperation_t hipblasOperation_t
#define cublasGemmAlgo_t hipblasGemmAlgo_t
#define cublasComputeType_t hipblasComputeType_t
#define cublasDataType_t hipDataType
#define cublasFillMode_t hipblasFillMode_t
#define cublasDiagType_t hipblasDiagType_t
#define cublasSideMode_t hipblasSideMode_t
#define cublasPointerMode_t hipblasPointerMode_t
#define cublasAtomicsMode_t hipblasAtomicsMode_t
#define cublasMath_t hipblasMath_t
#define cublasGemmEx hipblasGemmEx
#define cublasSgemm hipblasSgemm
#define cublasHgemm hipblasHgemm
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasSetPointerMode hipblasSetPointerMode
#define cublasGetPointerMode hipblasGetPointerMode
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasHgemmStridedBatched hipblasHgemmStridedBatched

// Constants
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT_TENSOR_OP

// Data types
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_16BF HIP_R_16BF
