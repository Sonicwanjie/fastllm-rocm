# Baseline Benchmark: Gemma-4-e2b on GFX1151 (RDNA3.5)

## Hardware
- GPU: AMD Radeon 8060S (gfx1151, RDNA3.5 APU)
- VRAM: 69 GB (unified memory, GTT-mapped)
- CPU: AMD Ryzen AI Max (Strix Halo)
- OS: Windows 11

## Software
- Compiler: hipcc (HIP SDK 7.x) + MSVC 2026
- Runtime: fastllm-rocm (current main)

## Model: Gemma-4-e2b-it (BF16)
- Parameters: ~10B
- Architecture: 60 layers, 8 Q heads / 1 KV head (sliding), 32 Q heads / 4 KV heads (global)
- Sliding head_dim: 256, Global head_dim: 512
- PLE: enabled (hidden_size_per_layer_input > 0)
- KV Shared Layers: enabled (last 20 layers)

## Current Performance (2026-05-27)

| Metric | Value |
|--------|-------|
| Prefill speed | ~7-12 tok/s |
| Decode speed (BF16) | ~20-23 tok/s |
| Decode speed (INT4) | Not yet tested |

## Bottleneck Analysis

From KERNEL_COMPARISON.md:

1. **Linear/Decode GEMV** (~60% of decode time):
   - WMMA 16x16x16 with M=1 uses only 1/16 of tile capacity
   - FastllmGemvFp16Fp16Kernel2MultiRow: 256 threads, scalar dot product
   - No vectorized loads or software pipelining

2. **Attention Decode** (~30% of decode time):
   - 3-pass: Q×K? (hipBLAS) ? Softmax ? Attn×V (hipBLAS)
   - hipBLAS may not be optimized for M=1 on gfx1151
   - FlashInfer disabled, no flash decoding

3. **Memory bandwidth** (~10%):
   - Unified memory architecture (GTT-backed)
   - No explicit async copy/prefetch

## Targets

| Metric | Current | Phase 2+3 Target | Final Target |
|--------|---------|------------------|--------------|
| Decode BF16 | ~20 tok/s | >30 tok/s | >40 tok/s |
| Decode INT4 | N/A | >50 tok/s | >80 tok/s |
| Prefill | ~10 tok/s | >20 tok/s | >50 tok/s |

## New Kernels (Phase 2+3)

| Kernel | File | Status |
|--------|------|--------|
| MFMA Decode GEMV (FP16) | `linear/fastllm-linear-gemv-mfma.hip` | Created |
| MFMA Decode GEMV (BF16) | `linear/fastllm-linear-gemv-mfma.hip` | Created |
| Fused INT4 Dequant GEMV | `linear/fastllm-linear-int4gemv-mfma.hip` | Created |
| Flash Decode Attention | `attention/fastllm-flash-decode.hip` | Created |

## Dispatch Changes

- `cudadevice.cpp`: FP16 weight + n < 8 ? `LaunchFastllmGemvFp16Opt()`
- `cudadevice.cpp`: INT4_GROUP + n < 8 ? `LaunchFastllmInt4GroupGemvFused()`
- `fastllm-attention.hip`: decode (q1==1) ? `LaunchFlashDecodeAttention()`
- CMakeLists.txt: Added new .hip files for Linux builds
