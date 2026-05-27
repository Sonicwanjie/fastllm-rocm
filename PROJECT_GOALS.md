# fastllm-windows-rocm 项目目标

## 项目愿景

构建 **Windows 原生 ROCm 推理引擎**——在 AMD GPU (gfx1151) 上高效运行主流开源 LLM，
支持 GGUF 模型格式，达到 Linux 同级性能水平。

---

## 目标环境

| 维度 | 规格 |
|------|------|
| **OS** | Windows 11 |
| **GPU** | AMD Radeon 8060S (RDNA3.5, gfx1151) |
| **VRAM** | 69 GB |
| **ROCm** | HIP SDK 7.x (`C:\rocm`) |
| **编译器** | MSVC 2026 (host) + hipcc (kernel) |
| **构建** | CMake + MSBuild / Ninja |

---

## 核心目标

### 1. Windows + ROCm 稳定编译 & 运行
- [x] 15 个 HIP kernel 文件全部通过 hipcc 编译
- [x] MSVC host 代码与 HIP kernel 链接成功
- [x] `main.exe` 可运行，GPU 推理正常工作
- [ ] CMake 一键配置 (无需手动拆分 .hip → .obj 流程)
- [ ] CI/build 脚本自动化编译验证

### 2. 模型支持

| 模型 | GPU 推理 | 输出正确 | 性能达标 | 备注 |
|------|---------|---------|---------|------|
| **Gemma 4 E2B** | ✅ 可推理 | ❌ 乱码 | ~10 tok/s | 缺 PLE、KV共享层、partial rotary |
| **Gemma 4 E2B INT4** | 待测 | ⬜ | ⬜ | 量化模型已下载 |
| **Qwen3 系列** | ✅ 已集成 | ⬜ 待测 | ⬜ | Qwen3-30B-A3B MoE / Qwen3.5 / Qwen3-Next |
| **Qwen3.6** | ⬜ | ⬜ | ⬜ | 新模型，架构待适配 |
| **Llama 系列** | ✅ 已集成 | ⬜ | ⬜ | LLaMA/LLaMA2/3 基础模型 |
| **DeepSeek V2/V4** | ✅ kernel 已有 | ⬜ | ⬜ | MoE 模型 |

**Gemma 4 修复优先级最高**：
1. Per-Layer Embedding (PLE) — 每层独立 embedding 输入
2. KV Shared Layers — 后 20 层共享 KV cache
3. Partial Rotary Factor — full_attention 层 partial_rotary_factor=0.25
4. Chat Template — thinking mode token 处理

### 3. GGUF 格式完整支持
- [x] GGUF 读取/解析 (`third_party/gguf/gguf.cpp`)
- [x] GGUF 反量化 (`ggml-quant.cpp`, `ggml-dequantize.cpp`)
- [x] GGUF → fastllm 适配层 (`gguf-adapter.cpp`)
- [ ] Gemma 4 GGUF 推理验证
- [ ] Qwen3 GGUF 推理验证
- [ ] 量化格式全覆盖 (Q4_0, Q4_K_M, Q6_K, IQ 系列等)

### 4. 推理性能目标

| 场景 | 当前 (Gemma 4) | 目标 |
|------|---------------|------|
| **Prefill** | ~10 tok/s | >50 tok/s |
| **Decode** | ~10 tok/s | >40 tok/s |
| **INT4 Decode** | ⬜ | >80 tok/s |

**优化路径**：
1. **Kernel 层**：WMMA → MFMA 指令 (tilelang 生成)
2. **Decode GEMV**：M=1 专用 kernel (WMMA 16x16 利用率极低)
3. **Fused Attention**：3-pass → 1-pass Fused Flash Attention
4. **BF16 原生**：减少 BF16↔FP16 转换损耗
5. **Fused MoE**：routing + expert GEMM 融合
6. **FlashInfer HIP 移植**：paged KV cache + tiled attention

### 5. Tilelang 集成
- [ ] Tilelang JIT → AOT 预编译 → 静态链接
- [x] Fused Attention Kernel (Flash Attention GQA) — astllm-flash-decode.hip created, head_dim 64/128/256/512
- [x] Fused Dequant GEMV (INT4 decode 专用) — astllm-linear-int4gemv-mfma.hip created
- [ ] Fused MoE Kernel
- [ ] gfx1151 MFMA auto-tuning

---

## 技术架构

```
fastllm-windows-rocm/
├── include/           # 头文件 (fastllm.h, model.h, device.h 等)
│   └── devices/cuda/  # CUDA/HIP 设备抽象层
├── src/
│   ├── devices/cpu/   # CPU 后端 (含 AliveThreadPool)
│   ├── devices/cuda/  # HIP kernel wrapper (*.hip → hipcc, *.cpp → MSVC)
│   └── models/        # 模型实现 (gemma4.cpp, qwen3.cpp, llama.cpp 等)
├── third_party/
│   ├── gguf/          # GGUF 解析 & 反量化
│   ├── flashinfer/    # FlashInfer (ROCm port, 目前 disabled)
│   ├── gpu_iface/     # ROCm GPU 抽象层
│   └── tilelang_kernels/  # Tilelang 生成的 AOT kernel
├── tools/             # CLI 工具 (ftllm)
├── cmake/             # CMake 模块 (WindowsHIP.cmake 等)
├── build_scripts/     # Python 编译脚本
└── build-rocm-msvc2/  # 当前构建目录
```

**编译拆分策略** (Windows HIP 核心问题)：
- `.hip` 文件 → `hipcc` → `.obj` (纯 GPU kernel + extern "C" 启动函数)
- host `.cpp` 文件 → `MSVC` → `.obj` (STL + fastllm::Data 封装)
- 最终 MSVC linker 合并所有 `.obj`

---

## 当前状态 (2026-05-26)

### ✅ 已完成
- HIP 编译框架 (`WindowsHIP.cmake`, `compile_hip.py`)
- 15/15 HIP kernel 编译通过 (attention, linear fp16/bf16/fp8/int4, ggml, multihip, moe)
- `hip_fastllm.lib` (10.6 MB) 生成
- `main.exe` 链接 & 运行成功 (Gemma 4 GPU 推理)
- Gemma 4 模型加载 (9.54 GB safetensors)
- Tokenizer 修复 (ByteFallback decode, 与 HF 对齐)
- GGUF 库集成编译
- ROCm FlashInfer headers 替换 (`gpu_iface` 抽象层)
- Qwen3/Qwen3-MoE/Qwen3.5/Qwen3-Next 模型代码
- INT4/INT8 量化 kernel

### ❌ 待修复
- **Gemma 4 PLE**：Per-layer embedding 未实现 → 输出乱码
- **Gemma 4 KV shared layers**：后 20 层共享 KV 未实现
- **Gemma 4 partial rotary**：full_attention 层 RoPE 参数
- **Qwen3 GGUF**：未验证
- **FlashInfer attention disabled**：使用 hipBLAS fallback
- **性能**：~10 tok/s，远低于目标

### ⬜ 待开发
- **Gemma 4 完整修复** (PLE + KV shared + partial rotary)
- **Qwen3.6 适配** (新架构)
- **GGUF 端到端验证** (Gemma 4 GGUF + Qwen3 GGUF)
- **Tilelang kernel 集成**
- **CMake 一键构建**
- **性能优化** (MFMA, Fused Attention, GEMV)

---

## 里程碑

### M1: Gemma 4 推理正确 ✅→🔧
- [ ] 实现 Per-Layer Embedding
- [ ] 实现 KV Shared Layers
- [ ] 修复 Partial Rotary Factor
- [ ] Chat template 适配
- [ ] 验证输出与 HF 对齐

### M2: GGUF 端到端
- [ ] Gemma 4 GGUF 正确加载 & 推理
- [ ] Qwen3 GGUF 正确加载 & 推理
- [ ] 主流量化格式全覆盖

### M3: Qwen3 系列验证
- [ ] Qwen3-30B-A3B (MoE) 推理验证
- [ ] Qwen3.5 推理验证
- [ ] Qwen3.6 适配 & 验证

### M4: 性能达标
- [ ] Decode >40 tok/s (FP16/BF16)
- [ ] INT4 Decode >80 tok/s
- [ ] Tilelang fused kernel 集成
- [ ] MFMA 指令替换 WMMA

### M5: 工程完善
- [ ] CMake 一键构建
- [ ] 构建脚本自动化
- [ ] CI 编译验证
- [ ] 文档补全

---

## 参考

- [BUILD_SPLIT_PLAN.md](./BUILD_SPLIT_PLAN.md) — HIP/主机代码拆分方案
- [FLASHINFER_ANALYSIS.md](./FLASHINFER_ANALYSIS.md) — FlashInfer HIP 移植分析
- [tilelang_integration/KERNEL_COMPARISON.md](./tilelang_integration/KERNEL_COMPARISON.md) — fastllm vs tilelang kernel 对比
- [docs/rocm.md](./docs/rocm.md) — ROCm 编译指南
- [docs/qwen3.md](./docs/qwen3.md) — Qwen3 使用指南
- [memory/](./memory/) — 开发日志 (每日记录)
