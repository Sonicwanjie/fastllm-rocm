#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(128) dequant_gemm_kernel(half_t* __restrict__ A, signed char* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  float C_local[32];
  half_t B_dequant[128];
  signed char B_shared_local_cast[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int ko = 0; ko < 64; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      half_t broadcast_var_1 = half_t(0.000000e+00f);
      uint4 condval;
      if ((i_1 < 2)) {
        condval = *(uint4*)(A + ((((i_1 * 65536) + ((((int)threadIdx.x) >> 3) * 4096)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      } else {
        condval = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(((half_t*)A_shared) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
    }
    *(int4*)(((signed char*)B_shared) + (((int)threadIdx.x) * 16)) = *(int4*)(B + ((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 1) * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 1) * 16)));
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      *(char2*)(B_shared_local_cast + 0) = make_char2(((signed char*)B_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((i_2 & 15) >> 3) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((i_2 >> 4) * 8)) + (i_2 & 7))], ((signed char*)B_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((i_2 & 15) >> 3) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((i_2 >> 4) * 8)) + (i_2 & 7))]);
      signed char broadcast_var_2 = (signed char)15;
      uint1 __1;
      char2 __2;
        char2 __3;
          char2 v_ = *(char2*)(B_shared_local_cast + 0);
          char2 v__1 = make_char2(((signed char)0)+((signed char)4*0), ((signed char)0)+((signed char)4*1));
          __3.x=((v_.x >> v__1.x));
          __3.y=((v_.y >> v__1.y));
        char2 v__2 = make_char2(broadcast_var_2, broadcast_var_2);
        __2.x=((__3.x & v__2.x));
        __2.y=((__3.y & v__2.y));
      *((half_t*)(&(((half2*)(&(__1.x)))->x))) = (half_t)(__2.x);
      *((half_t*)(&(((half2*)(&(__1.x)))->y))) = (half_t)(__2.y);
      *(uint1*)(B_dequant + (i_2 * 2)) = __1;
    }
    {
      half_t B_local[32];
      for (int ki = 0; ki < 4; ++ki) {
        for (int j = 0; j < 2; ++j) {
          for (int col = 0; col < 2; ++col) {
            *(uint4*)(B_local + ((j * 16) + (col * 8))) = *(uint4*)(((half_t*)A_shared) + (((((((((int)threadIdx.x) >> 6) * 2048) + (j * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((col + (((int)threadIdx.x) & 1)) & 1) * 8)));
          }
        }
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            {
      typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 tl_v16f16;
      typedef __attribute__((__vector_size__(8 * sizeof(float)))) float tl_v8f32;
      *((tl_v8f32*)C_local + ((i_3 * 2) + j_1)) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          *((tl_v16f16*)B_dequant + ((ki * 2) + i_3)),
          *((tl_v16f16*)B_local + j_1),
          *((tl_v8f32*)C_local + ((i_3 * 2) + j_1)));
    };
          }
        }
      }
    }
  }
  if ((((int)threadIdx.x) & 63) < 32) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      C[((((((((i_4 >> 4) * 65536) + ((i_4 & 7) * 8192)) + (((((int)threadIdx.x) & 31) >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_4 & 15) >> 3) * 16)) + (((int)threadIdx.x) & 15))] = ((half_t)C_local[i_4]);
    }
  }
}

