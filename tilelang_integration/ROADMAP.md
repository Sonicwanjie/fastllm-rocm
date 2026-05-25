# tilelang-fastllm integration roadmap

## 方向 1: 量化推理内核 (Dequantize GEMM)
- [x] Generator script: `generate_kernels.py` — `dequant_gemv_int4`, `dequant_gemm_int4_batch`
- [x] Bridge API: `TileLangDequantGemvInt4Group()`, `TileLangDequantGemmInt4Group()`
- [x] Bridge kernel: HIP native GEMV with online int4 dequant
- [ ] AOT: tilelang compile → .hip source → replace bridge reference impl
- [ ] Wire bridge into `cudadevice.cpp` dispatch

## 方向 2: Attention 内核 (Flash Decoding)
- [x] Flash decode kernel: `fastllm-flash-decode.hip`
  - Supported configs: 8Q/1KV/256dim, **32Q/32KV/256dim** (Gemma 4 sliding), **32Q/4KV/512dim** (Gemma 4 global)
  - Generic `LoadChunk`/`StoreChunk` helpers for any HEAD_DIM
  - Fused: single kernel launch replaces GEMM+softmax+GEMM 3-step
- [x] Attention guard updated in `fastllm-attention.hip` to allow head_dim=512, num_qo_heads=32
- [x] Tilelang flash attention bridge: `TileLangFlashAttentionPrefill()`
- [ ] AOT: tilelang MFMA-optimized prefill kernel

## 方向 3: MoE 推理内核 (FusedMoE)
- [x] Generator script: `generate_kernels.py` — `fused_moe`
- [x] Bridge API: `TileLangFusedMoE()`, `TileLangFusedMoESupported()`
- [x] Bridge kernel: Reference fused MoE (routing + expert GEMM)
- [ ] AOT: tilelang generate optimized MoE kernel
- [ ] Wire into Gemma 4 MoE dispatch

## 方向 4: DeepSeek 专用内核 (MLA/NSA)
- [x] Generator script: `generate_kernels.py` — `mla_decode`
- [ ] Bridge kernel implementation
- [ ] AOT generation

## 整合策略
tilelang 是 Python JIT -> 生成 CUDA/HIP 源码
fastllm 是纯 C++ -> 需要预编译的 kernel
桥接: tilelang 预编译 → 导出 .hip → 编译为 .obj → fastllm 链接

### AOT Pipeline
1. `python generate_kernels.py --target hip --output-dir generated` — 生成 HIP 源码
2. `compile_hip_msvc2.py` 自动包含 `generated/*.hip` 和 bridge `.cpp`
3. Bridge API (`fastllm-tilelang-bridge.h`) 提供 C 接口给 fastllm
4. 当 tilelang 不可用时，bridge 内含参考 HIP kernel 作为 fallback

### Build Integration
- `compile_hip_msvc2.py`: 包含 `tilelang_integration/tilelang_bridge/*.cpp` + `tilelang_integration/generated/*.hip`
- CMakeLists.txt: 包含 `tilelang_integration/tilelang_bridge/fastllm-tilelang-bridge.cpp`
- Include path: `tilelang_integration/tilelang_bridge/`

## 目录结构
  tilelang_kernels/        - tilelang Python 脚本, 生成 kernel 源码
  tilelang_bridge/         - C++ 桥接层, 连接 tilelang kernel 到 fastllm
  generated/               - tilelang 导出的 HIP 源码 (gitignore)
