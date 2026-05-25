## FlashInfer 移植分析

### 现状
- **fastllm 的 FlashInfer fork** (`third_party/flashinfer/`) 已有完整的 attention 头文件
- **ROCm flashinfer `amd-integration` 分支** 提供了 `gpu_iface` 抽象层（CUDA→HIP 运行时映射）
- `gpu_iface` 已复制到 `third_party/flashinfer/gpu_iface/`
- **但 attention 模块**（`prefill.cuh`, `decode.cuh`, `variants.cuh` 等）**没有用 `gpu_iface`**——它们直接 `#include <cuda_runtime.h>`, `<cuda_bf16.h>`, `<cuda_fp16.h>`

### 问题规模
FlashInfer attention 的 HIP 移植需要：
1. 替换 ~10 个核心文件中的 CUDA include → gpu_iface include
2. 替换 `cudaError_t` → `gpuError_t`, `cudaLaunchKernel` → `gpuLaunchKernel` 等数百个 API 调用
3. 处理 `cooperative_groups`（HIP 有 `hip_cooperative_groups`）
4. 处理 MMA/WGMA 操作（Hopper 特有，ROCm 不支持——但 generic attention 可能不依赖这些）
5. 处理 `__half`/`__nv_bfloat16` 类型差异

### 估计工作量
- 纯 include 替换 + `gpu_iface` 重写：~500-1000 行修改
- 测试编译通过：可能还需要解决更多兼容性问题
- **高风险**：FlashInfer 的 attention kernel 大量使用 WMMA/MMA 指令，在 ROCm 上可能不兼容

### 建议
1. **短期**：保持 FlashInfer attention disabled（用 hipBLAS fallback），先完成库的编译和链接
2. **中期**：用 `gpu_iface` 做 FlashInfer attention 的完整 HIP 移植
3. **替代方案**：直接用 ROCm 的 aiter attention（`flashinfer-amd-int` 里的 `attention/aiter/` 目录）替代 fastllm 的 FlashInfer 调用

### WMMA 问题
- fastllm 的 `HalfFC` 和 `FastllmHalfMatMulKernel` 使用 `8x32x16` WMMA fragment
- rocwmma 在 `MmaConfig` 中检查 `is_layout_same_v`，对 `8x32x16` 的 matrix_a 和 matrix_b layout 不匹配
- 需要查看 `rocm-libraries` 中 rocwmma 的最新版本是否有修复
