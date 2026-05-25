#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void mla_decode_kernel(half_t* __restrict__ K, half_t* __restrict__ O, half_t* __restrict__ Q, half_t* __restrict__ V) {
  float Q_local[192];
  half_t Q_local_cast[8];
  float acc_o[6];
  float logsum[1];
  float scores_max[1];
  half_t K_proj[192];
  float acc_s[1];
  half_t V_row[6];
  half_t O_local_cast_1[2];
  #pragma unroll
  for (int i = 0; i < 24; ++i) {
    *(uint4*)(Q_local_cast + 0) = *(uint4*)(Q + ((((int)blockIdx.x) * 192) + (i * 8)));
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
  for (int i_1 = 0; i_1 < 3; ++i_1) {
    float broadcast_var = 0.000000e+00f;
    *(float2*)(acc_o + (i_1 * 2)) = make_float2(broadcast_var, broadcast_var);
  }
  logsum[0] = 0.000000e+00f;
  scores_max[0] = -HUGE_VALF;
  for (int ko = 0; ko < 4096; ++ko) {
    for (int di = 0; di < 192; ++di) {
      K_proj[di] = K[((ko * 512) + di)];
    }
    acc_s[0] = 0.000000e+00f;
    for (int di_1 = 0; di_1 < 192; ++di_1) {
      acc_s[0] = (acc_s[0] + (Q_local[di_1] * ((float)K_proj[di_1])));
    }
    acc_s[0] = (acc_s[0] * 7.216878e-02f);
    float scores_max_prev = scores_max[0];
    scores_max[0] = max(scores_max[0], acc_s[0]);
    float correction = exp2f((scores_max_prev - scores_max[0]));
    #pragma unroll
    for (int i_2 = 0; i_2 < 6; ++i_2) {
      acc_o[i_2] = (acc_o[i_2] * correction);
    }
    acc_s[0] = exp2f((acc_s[0] - scores_max[0]));
    for (int di_2 = 0; di_2 < 192; ++di_2) {
      if (((int)threadIdx.x) <= ((di_2 & 63) >> 1)) {
        V_row[(((di_2 >> 6) * 2) + (di_2 & 1))] = V[((ko * 512) + di_2)];
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 6; ++i_3) {
      acc_o[i_3] = (acc_o[i_3] + (acc_s[0] * ((float)V_row[i_3])));
    }
    logsum[0] = (logsum[0] * correction);
    logsum[0] = (logsum[0] + acc_s[0]);
  }
  float inv_sum = (1.000000e+00f / (logsum[0] + 1.000000e-10f));
  #pragma unroll
  for (int i_4 = 0; i_4 < 3; ++i_4) {
    uint1 __2;
    float2 __3;
      float2 v__1 = *(float2*)(acc_o + (i_4 * 2));
      float2 v__2 = make_float2(inv_sum, inv_sum);
      __3.x = (v__1.x*v__2.x);
      __3.y = (v__1.y*v__2.y);
    *((half_t*)(&(((half2*)(&(__2.x)))->x))) = (half_t)(__3.x);
    *((half_t*)(&(((half2*)(&(__2.x)))->y))) = (half_t)(__3.y);
    *(uint1*)(O_local_cast_1 + 0) = __2;
    *(uint1*)(O + (((((int)blockIdx.x) * 192) + (i_4 * 64)) + (((int)threadIdx.x) * 2))) = *(uint1*)(O_local_cast_1 + 0);
  }
}

