#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(128) dequant_gemv_kernel(half_t* __restrict__ A, signed char* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 2048));
  float C_local[64];
  float acc[1];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int ko = 0; ko < 64; ++ko) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)A_shared)[((int)threadIdx.x)] = A[((ko * 64) + ((int)threadIdx.x))];
    }
    *(int4*)(((signed char*)B_shared) + (((int)threadIdx.x) * 16)) = *(int4*)(B + ((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 1) * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 1) * 16)));
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      acc[0] = 0.000000e+00f;
      for (int ki = 0; ki < 64; ++ki) {
        float q_val = ((float)((half_t*)A_shared)[ki]);
        signed char w_bits = ((signed char*)B_shared)[((i_1 * 32) + (ki >> 1))];
        half_t w_val = ((half_t)((w_bits >> ((signed char)((ki & 1) * 4))) & (signed char)15));
        acc[0] = (acc[0] + (q_val * ((float)((half_t)((w_bits >> ((signed char)((ki & 1) * 4))) & (signed char)15)))));
      }
      C_local[i_1] = (C_local[i_1] + acc[0]);
    }
  }
  if ((((int)threadIdx.x) >> 6) == 0) {
    C[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63))] = ((half_t)C_local[(((int)threadIdx.x) & 63)]);
  }
}

