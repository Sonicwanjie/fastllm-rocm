#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void flash_decode_kernel(half_t* __restrict__ K, half_t* __restrict__ O, half_t* __restrict__ Q, half_t* __restrict__ V) {
  float Q_local[128];
  half_t Q_local_cast[8];
  float acc_o[128];
  float logsum[1];
  float scores_max[1];
  half_t K_local[128];
  float acc_s[1];
  float scores_max_prev[1];
  half_t V_local[128];
  half_t O_local_cast_1[4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(uint4*)(Q_local_cast + 0) = *(uint4*)(Q + ((((int)blockIdx.x) * 128) + (i * 8)));
    for (int vec = 0; vec < 2; ++vec) {
      float4 __1;
      uint2 v_ = *(uint2*)(Q_local_cast + (vec * 4));
      __1.x = (float)(((half2*)(&(v_.x)))->x);
      __1.y = (float)(((half2*)(&(v_.x)))->y);
      __1.z = (float)(((half2*)(&(v_.y)))->x);
      __1.w = (float)(((half2*)(&(v_.y)))->y);
      *(float4*)(Q_local + ((i * 8) + (vec * 4))) = __1;
    }
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  logsum[0] = 0.000000e+00f;
  scores_max[0] = -HUGE_VALF;
  for (int ko = 0; ko < 4096; ++ko) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      *(uint4*)(K_local + (i_2 * 8)) = *(uint4*)(K + ((((((int)blockIdx.x) >> 2) * 524288) + (ko * 128)) + (i_2 * 8)));
    }
    acc_s[0] = 0.000000e+00f;
    for (int d = 0; d < 128; ++d) {
      acc_s[0] = (acc_s[0] + (Q_local[d] * ((float)K_local[d])));
    }
    acc_s[0] = (acc_s[0] * 8.838835e-02f);
    scores_max_prev[0] = scores_max[0];
    scores_max[0] = max(scores_max[0], acc_s[0]);
    #pragma unroll
    for (int i_3 = 0; i_3 < 128; ++i_3) {
      acc_o[i_3] = (acc_o[i_3] * exp2f((scores_max_prev[0] - scores_max[0])));
    }
    acc_s[0] = exp2f((acc_s[0] - scores_max[0]));
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      *(uint4*)(V_local + (i_4 * 8)) = *(uint4*)(V + ((((((int)blockIdx.x) >> 2) * 524288) + (ko * 128)) + (i_4 * 8)));
    }
    for (int d_1 = 0; d_1 < 128; ++d_1) {
      acc_o[d_1] = (acc_o[d_1] + (acc_s[0] * ((float)V_local[d_1])));
    }
    logsum[0] = (logsum[0] * exp2f((scores_max_prev[0] - scores_max[0])));
    logsum[0] = (logsum[0] + acc_s[0]);
  }
  uint2 __2;
  float4 __3;
    float4 v__1 = *(float4*)(acc_o + (((int)threadIdx.x) * 4));
    float4 v__2 = make_float4((logsum[0] + 1.000000e-10f), (logsum[0] + 1.000000e-10f), (logsum[0] + 1.000000e-10f), (logsum[0] + 1.000000e-10f));
    __3.x = (v__1.x/v__2.x);
    __3.y = (v__1.y/v__2.y);
    __3.z = (v__1.z/v__2.z);
    __3.w = (v__1.w/v__2.w);
  *((half_t*)(&(((half2*)(&(__2.x)))->x))) = (half_t)(__3.x);
  *((half_t*)(&(((half2*)(&(__2.x)))->y))) = (half_t)(__3.y);
  *((half_t*)(&(((half2*)(&(__2.y)))->x))) = (half_t)(__3.z);
  *((half_t*)(&(((half2*)(&(__2.y)))->y))) = (half_t)(__3.w);
  *(uint2*)(O_local_cast_1 + 0) = __2;
  *(uint2*)(O + ((((int)blockIdx.x) * 128) + (((int)threadIdx.x) * 4))) = *(uint2*)(O_local_cast_1 + 0);
}

