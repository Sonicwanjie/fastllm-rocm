// Flash Decoding Attention for gfx1151 (RDNA 3.5)
// Supports head_dim = 64/128/256/512

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace fastllm {
namespace hip {

// Flash Decoding Attention Kernel
// Grid: (batch, num_heads)
// Block: 128 threads
// Uses online softmax for numerical stability
template <typename T, int HEAD_DIM>
__global__ void FlashDecodingKernel(
    const T* __restrict__ Q,           // [batch, num_heads, head_dim]
    const T* __restrict__ K,           // [total_kv_len, num_kv_heads, head_dim]  
    const T* __restrict__ V,           // [total_kv_len, num_kv_heads, head_dim]
    const int* __restrict__ cu_seqlens, // [batch + 1]
    T* __restrict__ output,             // [batch, num_heads, head_dim]
    int num_q_heads, int num_kv_heads, int max_kv_len, float scale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    int kv_groups = num_q_heads / num_kv_heads;
    int kv_head = head_idx / kv_groups;
    
    int seq_start = cu_seqlens[batch_idx];
    int seq_end = cu_seqlens[batch_idx + 1];
    int kv_len = seq_end - seq_start;
    
    // Shared memory
    extern __shared__ char smem[];
    T* Q_shared = (T*)smem;
    T* K_shared = Q_shared + HEAD_DIM;
    float* scores = (float*)(K_shared + HEAD_DIM);
    float* logsum = scores + HEAD_DIM;
    
    // Load Q
    const T* q_row = Q + head_idx * HEAD_DIM;
    if (tid < HEAD_DIM) {
        Q_shared[tid] = q_row[tid];
    }
    
    // Init softmax state
    if (tid < HEAD_DIM) {
        scores[tid] = -INFINITY;
        logsum[tid] = 0.0f;
    }
    __syncthreads();
    
    // Main loop over KV tiles
    const int TILE_N = 128;
    int num_tiles = (kv_len + TILE_N - 1) / TILE_N;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int k_start = seq_start + tile * TILE_N;
        int k_len = min(TILE_N, kv_len - tile * TILE_N);
        
        // Load K tile
        const T* k_tile = K + k_start * num_kv_heads + kv_head * HEAD_DIM;
        for (int i = tid; i < TILE_N * HEAD_DIM; i += blockDim.x) {
            int n = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            K_shared[i] = (n < k_len) ? k_tile[i] : (T)0;
        }
        __syncthreads();
        
        // Compute attention scores and online softmax
        if (tid < HEAD_DIM) {
            float q = (float)Q_shared[tid];
            float max_prev = scores[tid];
            float sum = 0.0f;
            
            #pragma unroll 4
            for (int n = 0; n < TILE_N; n++) {
                if (n < k_len) {
                    float k = (float)K_shared[n * HEAD_DIM + tid];
                    float s = q * k * scale;
                    float exp_s = expf(s - max_prev);
                    sum += exp_s;
                }
            }
            
            // Online softmax update
            logsum[tid] = logsum[tid] + sum;
            scores[tid] = max_prev + logf(sum + 1e-10f);
        }
        __syncthreads();
    }
    
    // Write output
    T* o_row = output + head_idx * HEAD_DIM;
    if (tid < HEAD_DIM) {
        o_row[tid] = (T)logsum[tid];
    }
}

// Launcher with DISPATCH_HEAD_DIM
#define DISPATCH_HEAD_DIM_DECODE(HEAD_DIM, ...) \
    switch (HEAD_DIM) { \
        case 64: { constexpr int HD = 64; __VA_ARGS__(64); break; } \
        case 128: { constexpr int HD = 128; __VA_ARGS__(128); break; } \
        case 256: { constexpr int HD = 256; __VA_ARGS__(256); break; } \
        case 512: { constexpr int HD = 512; __VA_ARGS__(512); break; } \
        default: return hipErrorInvalidValue; \
    }

template <typename T>
hipError_t LaunchFlashDecodingAttention(
    const void* Q_dev, const void* K_dev, const void* V_dev,
    const int* cu_seqlens_dev, void* output_dev,
    int batch, int num_q_heads, int num_kv_heads,
    int head_dim, int max_kv_len, float scale,
    hipStream_t stream)
{
    dim3 grid(batch, num_q_heads);
    dim3 block(128);
    size_t smem = HEAD_DIM * 4 * sizeof(float);
    
    switch (head_dim) {
        case 64:
            FlashDecodingKernel<T, 64><<<grid, block, smem, stream>>>(
                (const T*)Q_dev, (const T*)K_dev, (const T*)V_dev,
                cu_seqlens_dev, (T*)output_dev,
                num_q_heads, num_kv_heads, max_kv_len, scale);
            break;
        case 128:
            FlashDecodingKernel<T, 128><<<grid, block, smem, stream>>>(
                (const T*)Q_dev, (const T*)K_dev, (const T*)V_dev,
                cu_seqlens_dev, (T*)output_dev,
                num_q_heads, num_kv_heads, max_kv_len, scale);
            break;
        case 256:
            FlashDecodingKernel<T, 256><<<grid, block, smem, stream>>>(
                (const T*)Q_dev, (const T*)K_dev, (const T*)V_dev,
                cu_seqlens_dev, (T*)output_dev,
                num_q_heads, num_kv_heads, max_kv_len, scale);
            break;
        case 512:
            FlashDecodingKernel<T, 512><<<grid, block, smem, stream>>>(
                (const T*)Q_dev, (const T*)K_dev, (const T*)V_dev,
                cu_seqlens_dev, (T*)output_dev,
                num_q_heads, num_kv_heads, max_kv_len, scale);
            break;
        default:
            return hipErrorInvalidValue;
    }
    
    return hipGetLastError();
}

} // namespace hip
} // namespace fastllm
