# tilelang vs fastllm Kernel Comparison

## 测试环境
- GPU: AMD Radeon 8060S (gfx1151, RDNA3.5)
- VRAM: 69 GB
- Model: Gemma-4-e2b-it (BF16, heads_q=8, heads_kv=1, head_dim=256, 35 layers)
- OS: Windows, ROCm

## 当前 fastllm 性能 (Gemma-4-e2b, batch=1)
- Prefill: ~7-12 tok/s
- Decode: ~20-23 tok/s
- Model size: ~10 GB (safetensors)

---

## 1. Attention Kernel 对比

### fastllm (hip/attention/fastllm-attention.hip)

**Decode 路径** (DoFastllmCudaAttentionBatch):
- 三步分解: Q×Kᵀ → Softmax → Attn×V
- Q×Kᵀ: FastllmHalfMatMulTransBBatchKernel 手写 WMMA kernel
  - 128 threads/block
  - rocwmma 16×16×16 fragments (GFX11 only supports this size)
  - 每个 head 独立 launch 一个 block
- Softmax: FastllmSoftmaxKernelBatchInner1 根据 seq_len 选 32/64/128 threads
- Attn×V: FastllmHalfMatMulKernel 类似 Q×Kᵀ 的 WMMA kernel
- **问题**: decode 时 M=1, WMMA 利用率极低 (只用了 1 行 out of 16)

**Prefill 路径** (FastllmCudaHalfPagedAttentionBatch):
- 使用 FlashInfer 库的 paged attention
- 支持 FlashInfer 的 PrefillPlan/DecodeRun
- **问题**: FlashInfer 对 gfx1151 的支持不完善, 可能回退到非-tiled 实现

**GQA 处理**: 
- Host 端 pointer array 批量调度
- 每个 KV head 对应 group_size 个 Q heads
- 无 kernel-level GQA fusion

### tilelang (bench_flash_attn.py)

**Flash Attention GQA** (flashattn_gqa):
- 单 kernel fused attention: Q×Kᵀ + softmax + Attn×V
- Block tiled: block_M=128, block_N=128
- 使用 T.gemm() → 自动 lowering 到 MFMA intrinsic
- Online softmax with log-sum-exp correction
- 2-stage software pipeline (Pipelined K/V loading)
- **GQA native**: kernel 内部计算 kv_head = by // group_size
- Causal mask 内联在 kernel 中

**ROCm 后端** (tilelang/rocm/op/gemm/gemm_mfma.py):
- MatrixCoreIntrinEmitter 直接生成 MFMA 指令
- Swizzled shared memory layout
- Warp partition auto-tuning

### 对比表

| 维度 | fastllm | tilelang | 优势方 |
|------|---------|----------|--------|
| Decode GEMM 效率 | 低 (M=1 WMMA 利用率差) | N/A (无 decode kernel) | - |
| Prefill attention | FlashInfer (可能 fallback) | Fused tiled (MFMA) | tilelang |
| GQA 融合 | Host-side batching | Kernel 内置 | tilelang |
| Pipeline | 无 | 2-stage | tilelang |
| WMMA vs MFMA | WMMA 16×16×16 | MFMA (direct) | tilelang |
| Paged KV cache | 支持 (FlashInfer) | 不支持 | fastllm |
| BF16 支持 | 部分转换 | 原生 | tilelang |

---

## 2. Linear/GEMM Kernel 对比

### fastllm (hip/linear/fastllm-linear-fp16.hip)

- FastllmHalfMatMulTransBBatchKernel: 手写 WMMA GEMV
- FastllmHalfMatMulKernel: 手写 WMMA GEMM
- 支持 FP16, BF16, FP8, INT8, INT4 多种量化格式
- **问题**: decode 时 M=1, WMMA 16×16 只用了 1 行

### tilelang (generate_kernels.py)

- make_dequant_gemv_int4: 专为 decode (M=1) 设计的 GEMV kernel
  - Block_N=64, Block_K=64, 2-stage pipeline
  - 内置 INT4 → FP16 反量化
- make_dequant_gemm_int4_batch: Prefill GEMM (M>1)
  - Block_M=64, Block_N=64, Block_K=64, 3-stage pipeline
- make_flash_decode: Flash Decoding for GQA
  - 专门为 GQA decode 优化的 kernel
  - Block split + reduction

### 对比表

| 维度 | fastllm | tilelang | 优势方 |
|------|---------|----------|--------|
| Decode GEMV | 通用 WMMA (M=1 低效) | 专门 GEMV kernel | tilelang |
| INT4 反量化 | 分离 (dequant + GEMM) | Fused (dequant in GEMM) | tilelang |
| Prefill GEMM | hipBLAS + 自定义 | Tiled MFMA | 可比 |
| Pipeline | 无 | 2-3 stage | tilelang |

---

## 3. MoE Kernel 对比

### fastllm (hip/moe/fastllm-moe-fp8.hip)

- MoE routing + expert computation 分离
- 支持 FP8 expert 权重

### tilelang (generate_kernels.py: make_fused_moe)

- make_fused_moe: Fused MoE kernel
  - Top-K routing + expert GEMM fused
  - num_experts=64, top_k=8
  - 支持 permutation-based load balancing

---

## 4. 优化建议

### 短期 (fastllm 内部优化)
1. **Decode GEMV**: M=1 时 WMMA 利用率极低, 需要专门的 GEMV kernel (SIMT 或 reduced tile)
2. **Flash Decoding**: 多 KV head + 长序列时, 实现 flash decoding (split-K + reduction)
3. **BF16 原生**: 当前 BF16 → FP16 转换有性能和精度损失

### 中期 (tilelang 集成)
1. **Fused Attention**: 用 tilelang 的 flash attention 替换 3-pass decode attention
2. **Fused Dequant GEMM**: 用 tilelang 的 INT4 dequant GEMV 替换分离的 dequant + GEMM
3. **Fused MoE**: 用 tilelang 的 fused MoE 替换分离的 routing + expert

### 长期
1. tilelang JIT → AOT 预编译 → 静态链接到 fastllm
2. 针对 gfx1151 的 MFMA auto-tuning
3. Paged KV cache + tilelang flash attention 结合

---

## 5. 模型输出问题

当前 Gemma-4-e2b-it 输出乱码, 可能原因:
1. Chat template 处理不正确 (thinking mode tokens)
2. BF16 → FP16 转换精度损失 (Gemma 4 对精度敏感)
3. Tokenizer 编码/解码问题 (262K vocab)
4. RoPE 参数 (sliding/full attention 有不同的 rope_theta)

需要进一步调试模型前向传播的中间结果。
