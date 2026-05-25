// fastllm-tilelang-bridge.h
// Bridge layer between fastllm and tilelang-generated kernels
// Provides AOT-compiled kernels for:
//   - Flash Attention Prefill (GQA, MFMA-optimized)
//   - Fused Dequant GEMV (INT4 -> FP16, decode M=1)
//   - Fused Dequant GEMM (INT4 -> FP16, prefill M>1)
//   - Fused MoE (routing + expert GEMM)

#ifndef FASTLLM_TILELANG_BRIDGE_H
#define FASTLLM_TILELANG_BRIDGE_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Flash Attention Prefill
// ============================================================================

typedef struct {
    int batch;
    int num_qo_heads;
    int num_kv_heads;
    int seq_q;
    int seq_kv;
    int head_dim;
    int is_causal;
    float scale;
} TileLangFlashAttnParams;

bool TileLangFlashAttentionPrefill(
    const half* Q,
    const half* K,
    const half* V,
    half* Out,
    const TileLangFlashAttnParams* params,
    hipStream_t stream
);

bool TileLangFlashAttentionSupported(
    int num_qo_heads, int num_kv_heads,
    int head_dim, int is_causal
);

// ============================================================================
// Fused Dequantize GEMV (decode: M=1)
// ============================================================================

typedef struct {
    int N;           // output dimension (rows of weight)
    int K;           // input dimension (cols of weight)
    int num_bits;    // 4 for int4
    int group_size;  // quantization group size (e.g., 128)
} TileLangDequantGemvParams;

// INT4 group-quantized GEMV: C = dequant(B) * A, where B is packed int4
// A: [K] (fp16), B: [N, K/2] (packed int4), C: [N] (fp16)
// scales/mins: [N, K/group_size] (fp16)
bool TileLangDequantGemvInt4Group(
    const half* A,
    const uint8_t* B,
    const half* scales,
    const half* mins,
    half* C,
    const TileLangDequantGemvParams* params,
    hipStream_t stream
);

// INT4 flat (no group) GEMV
bool TileLangDequantGemvInt4(
    const half* A,
    const uint8_t* B,
    const half* scales,
    const half* mins,
    half* C,
    int N, int K,
    hipStream_t stream
);

// ============================================================================
// Fused Dequantize GEMM (prefill: M>1)
// ============================================================================

typedef struct {
    int M;           // batch dimension
    int N;           // output dimension
    int K;           // input dimension
    int num_bits;    // 4 for int4
    int group_size;  // quantization group size
} TileLangDequantGemmParams;

bool TileLangDequantGemmInt4Group(
    const half* A,
    const uint8_t* B,
    const half* scales,
    const half* mins,
    half* C,
    const TileLangDequantGemmParams* params,
    hipStream_t stream
);

// ============================================================================
// Fused MoE
// ============================================================================

typedef struct {
    int num_experts;
    int top_k;
    int hidden_size;
    int inter_size;
    int batch;       // number of tokens
} TileLangFusedMoEParams;

// Fused MoE: routing + expert compute
// input: [batch, hidden_size]
// gate_up_weight: [num_experts, inter_size * 2, hidden_size / 2] (int4 packed)
// down_weight: [num_experts, hidden_size, inter_size / 2] (int4 packed)
// output: [batch, hidden_size]
bool TileLangFusedMoE(
    const half* input,
    const half* router_weight,     // [hidden_size, num_experts]
    const uint8_t* gate_up_weight, // int4 packed expert weights
    const uint8_t* down_weight,    // int4 packed expert weights
    const half* gate_up_scales,
    const half* gate_up_mins,
    const half* down_scales,
    const half* down_mins,
    half* output,
    const TileLangFusedMoEParams* params,
    hipStream_t stream
);

bool TileLangFusedMoESupported(int hidden_size, int num_experts, int top_k);

#ifdef __cplusplus
}
#endif

#endif // FASTLLM_TILELANG_BRIDGE_H
