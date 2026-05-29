// FastLLM wrapper for rocm-cpp Flash Decoding Attention
// Based on rocm-cpp kv_cache_attn_decode kernel
// Supports head_dim up to 256

#pragma once

#include "fastllm-cuda.cuh"

#ifdef USE_ROCM

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cmath>

namespace fastllm {
namespace rocm_attn {

constexpr int FLASH_DECODE_BLOCK = 128;
constexpr int FLASH_DECODE_WARP = 32;
constexpr int FLASH_DECODE_MAX_HEAD_DIM = 256;

// Replicate the kernel from rocm-cpp
__global__ __launch_bounds__(FLASH_DECODE_BLOCK)
void FlashDecodingKernel(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* output,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale)
{
    const int h = blockIdx.x;
    if (h >= num_q_heads) return;

    const int tid = threadIdx.x;
    const int lane = tid & (FLASH_DECODE_WARP - 1);
    const int wid = tid / FLASH_DECODE_WARP;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = h / gqa_ratio;

    __shared__ __half Q_shared[FLASH_DECODE_MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += FLASH_DECODE_BLOCK) {
        Q_shared[d] = Q[(size_t)h * head_dim + d];
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    const int elems_per_thread = (head_dim + FLASH_DECODE_BLOCK - 1) / FLASH_DECODE_BLOCK;
    float o_local[8] = {0};

    __shared__ float s_max[FLASH_DECODE_BLOCK / FLASH_DECODE_WARP];
    __shared__ float s_sum[FLASH_DECODE_BLOCK / FLASH_DECODE_WARP];

    for (int t = 0; t < seq_len; ++t) {
        const __half* K_row = K + ((size_t)t * num_kv_heads + kv_head) * head_dim;

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += FLASH_DECODE_BLOCK) {
            partial += (float)Q_shared[d] * (float)K_row[d];
        }
        
        #pragma unroll
        for (int o = FLASH_DECODE_WARP / 2; o > 0; o >>= 1) {
            partial += __shfl_xor(partial, o);
        }
        if (lane == 0) s_sum[wid] = partial;
        __syncthreads();
        
        if (wid == 0) {
            float v = (lane < FLASH_DECODE_BLOCK / FLASH_DECODE_WARP) ? s_sum[lane] : 0.0f;
            #pragma unroll
            for (int o = (FLASH_DECODE_BLOCK / FLASH_DECODE_WARP) / 2; o > 0; o >>= 1) {
                v += __shfl_xor(v, o);
            }
            if (lane == 0) s_max[0] = v * scale;
        }
        __syncthreads();
        const float s = s_max[0];

        const float m_new = (s > m) ? s : m;
        const float alpha = expf(m - m_new);
        const float beta = expf(s - m_new);
        l = l * alpha + beta;
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) o_local[k] *= alpha;
        
        const __half* V_row = V + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * FLASH_DECODE_BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * (float)V_row[d];
            }
        }
        m = m_new;
    }

    const float inv_l = 1.0f / l;
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = tid + ei * FLASH_DECODE_BLOCK;
        if (d < head_dim) {
            output[(size_t)h * head_dim + d] = __float2half(o_local[ei] * inv_l);
        }
    }
}

// Launcher for flash decoding attention
// Supports head_dim up to 256
bool FlashDecodingAttention(
    const fastllm::Data& q,      // [batch, num_heads, head_dim] or [num_heads, head_dim]
    const fastllm::Data& k,      // [seq_len, num_kv_heads, head_dim]
    const fastllm::Data& v,      // [seq_len, num_kv_heads, head_dim]
    fastllm::Data& output,       // [num_heads, head_dim]
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale, hipStream_t stream)
{
    if (head_dim > FLASH_DECODE_MAX_HEAD_DIM) {
        printf("FlashDecodingAttention: head_dim %d > MAX %d\n", head_dim, FLASH_DECODE_MAX_HEAD_DIM);
        return false;
    }
    
    const __half* Q_dev = (const __half*)FastllmCudaPrepareInput(q);
    const __half* K_dev = (const __half*)FastllmCudaPrepareInput(k);
    const __half* V_dev = (const __half*)FastllmCudaPrepareInput(v);
    __half* O_dev = (__half*)FastllmCudaPrepareOutput(output);
    
    dim3 grid(num_q_heads, 1, 1);
    dim3 block(FLASH_DECODE_BLOCK, 1, 1);
    
    FlashDecodingKernel<<<grid, block, 0, stream>>>(
        Q_dev, K_dev, V_dev, O_dev,
        num_q_heads, num_kv_heads, head_dim, seq_len, scale);
    
    FastllmCudaFinishInput(q, (void*)Q_dev);
    FastllmCudaFinishInput(k, (void*)K_dev);
    FastllmCudaFinishInput(v, (void*)V_dev);
    FastllmCudaFinishOutput(output, O_dev);
    
    return true;
}

} // namespace rocm_attn
} // namespace fastllm

#endif // USE_ROCM
