#ifndef FASTLLM_FLASH_DECODE_HIP
#define FASTLLM_FLASH_DECODE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template <int HEAD_DIM, int KV_TILE, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
FlashDecodeKernel(const half* Q, const half* K, const half* V, half* O,
                  int num_qo_heads, int num_kv_heads, int seq_len, float scale) {
    const int head_idx = blockIdx.x;
    const int group_size = num_qo_heads / num_kv_heads;
    const int kv_head = head_idx / group_size;
    const int tid = threadIdx.x;
    const half* Qh = Q + head_idx * HEAD_DIM;
    half* Oh = O + head_idx * HEAD_DIM;
    const half* Kh = K + kv_head * seq_len * HEAD_DIM;
    const half* Vh = V + kv_head * seq_len * HEAD_DIM;

    const int QPT = (HEAD_DIM + NUM_THREADS - 1) / NUM_THREADS;
    float q[QPT];
    for (int i = 0; i < QPT; i++) {
        int idx = tid * QPT + i;
        q[i] = (idx < HEAD_DIM) ? __half2float(__ldg(Qh + idx)) : 0.0f;
    }

    float out_acc[QPT];
    float max_val = -1e30f;
    float sum_val = 0.0f;
    for (int i = 0; i < QPT; i++) out_acc[i] = 0.0f;

    __shared__ half Kt[KV_TILE * HEAD_DIM];
    __shared__ half Vt[KV_TILE * HEAD_DIM];
    int num_tiles = (seq_len + KV_TILE - 1) / KV_TILE;

    for (int t = 0; t < num_tiles; t++) {
        int ks = t * KV_TILE;
        int valid = ks + KV_TILE < seq_len ? KV_TILE : seq_len - ks;
        int tile_sz = KV_TILE * HEAD_DIM;
        for (int idx = tid; idx < tile_sz; idx += NUM_THREADS) {
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
            Kt[idx] = (r < valid) ? __ldg(Kh + (ks + r) * HEAD_DIM + c) : __float2half(0.0f);
            Vt[idx] = (r < valid) ? __ldg(Vh + (ks + r) * HEAD_DIM + c) : __float2half(0.0f);
        }
        __syncthreads();

        for (int r = 0; r < valid; r++) {
            float dot = 0.0f;
            for (int i = 0; i < QPT; i++) {
                int ci = tid * QPT + i;
                if (ci < HEAD_DIM) dot += q[i] * __half2float(Kt[r * HEAD_DIM + ci]);
            }
            for (int o = 32; o > 0; o >>= 1) dot += __shfl_down(dot, o);
            float score = (tid < 64 ? dot : 0.0f) * scale;
            score = __shfl(score, 0);
            float new_max = fmaxf(max_val, score);
            float corr = expf(max_val - new_max);
            float s = expf(score - new_max);
            for (int i = 0; i < QPT; i++) {
                int ci = tid * QPT + i;
                float vv = (ci < HEAD_DIM) ? __half2float(Vt[r * HEAD_DIM + ci]) : 0.0f;
                out_acc[i] = out_acc[i] * corr + s * vv;
            }
            sum_val = sum_val * corr + s;
            max_val = new_max;
        }
        __syncthreads();
    }
    for (int i = 0; i < QPT; i++) {
        int idx = tid * QPT + i;
        if (idx < HEAD_DIM) Oh[idx] = __float2half(out_acc[i] / sum_val);
    }
}

inline bool LaunchFlashDecodeAttention(const half* Q, const half* K, const half* V, half* O,
    int num_qo_heads, int num_kv_heads, int seq_len, int head_dim,
    float scale, int batch_size, hipStream_t stream) {
    dim3 grid(num_qo_heads * batch_size);
    if (head_dim == 64)       hipLaunchKernelGGL((FlashDecodeKernel<64,64,256>),  grid, dim3(256), 0, stream, Q,K,V,O,num_qo_heads,num_kv_heads,seq_len,scale);
    else if (head_dim == 128) hipLaunchKernelGGL((FlashDecodeKernel<128,32,256>), grid, dim3(256), 0, stream, Q,K,V,O,num_qo_heads,num_kv_heads,seq_len,scale);
    else if (head_dim == 256) hipLaunchKernelGGL((FlashDecodeKernel<256,16,256>), grid, dim3(256), 0, stream, Q,K,V,O,num_qo_heads,num_kv_heads,seq_len,scale);
    else if (head_dim == 512) hipLaunchKernelGGL((FlashDecodeKernel<512,8,256>),  grid, dim3(256), 0, stream, Q,K,V,O,num_qo_heads,num_kv_heads,seq_len,scale);
    else return false;
    return true;
}

#endif
