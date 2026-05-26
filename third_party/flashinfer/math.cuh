/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of ex2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  return exp2f(x);
}

/*!
 * \brief Wrapper of log2 instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  return log2f(x);
}

/*!
 * \brief Computes 2^x for half2
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  float lo = __low2float(x);
  float hi = __high2float(x);
  return __floats2half2_rn(exp2f(lo), exp2f(hi));
}

/*!
 * \brief Computes 2^x for half
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  return __float2half(exp2f(__half2float(x)));
}

/*!
 * \brief Computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  return __frcp_rn(x);
}

/*!
 * \brief Performs a butterfly shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return __shfl_xor(x, lane_mask);
}

/*!
 * \brief Performs a butterfly shuffle on half2
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor(x, lane_mask);
}

/*!
 * \brief Computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  return __frsqrt_rn(x);
}

/*!
 * \brief Computes tanh(x) for float
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  return tanhf(x);
}

/*!
 * \brief Computes tanh(x) for half2
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  float lo = __low2float(x);
  float hi = __high2float(x);
  return __floats2half2_rn(tanhf(lo), tanhf(hi));
}

/*!
 * \brief Computes tanh(x) for half
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  return __float2half(tanhf(__half2float(x)));
}

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
