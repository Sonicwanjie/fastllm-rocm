#!/usr/bin/env python3
"""Apply GFX1151 optimization patches after hipify runs.
This script patches the hipified files to add our optimized kernel dispatch.
"""
import sys
import os

PROJ = os.path.dirname(os.path.abspath(__file__))

def patch_cudadevice():
    """Add extern declarations and dispatch to cudadevice.cpp"""
    path = os.path.join(PROJ, "src", "devices", "hip", "cudadevice.cpp")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add extern declarations before namespace fastllm (only once)
    extern_block = '''// GFX1151 optimized kernel launchers (defined in .hip files, linked via hip_fastllm.lib)
extern "C" {
    void LaunchFastllmGemvFp16Opt(void *input, void *weight, void *output, void *bias, int n, int m, int k, bool addTo);
    void LaunchFastllmInt4GroupGemvFused(void *input, void *weight, void *scales, void *mins, void *bias, void *output, int n, int m, int k, int group, int groupCnt, bool addTo);
}

'''
    
    if "LaunchFastllmGemvFp16Opt" not in content:
        print("  Adding extern declarations...")
        content = content.replace("namespace fastllm {", extern_block + "namespace fastllm {")
    
    # Patch FP16 dispatch: add n < 8 fast path
    old_fp16 = '''            } else if (weight.dataType == DataType::FLOAT16) {
                FastllmCudaHalfMatMulFloat16(input, weight, bias, output, n, m, k);'''
    
    new_fp16 = '''            } else if (weight.dataType == DataType::FLOAT16) {
                if (n < 8) {
                    LaunchFastllmGemvFp16Opt(FastllmCudaPrepareInput(input), weight.cudaData, FastllmCudaPrepareOutput(output), bias.dims.size() == 0 ? nullptr : weight.extraCudaHalfData[0], n, m, k, false);
                } else {
                    FastllmCudaHalfMatMulFloat16(input, weight, bias, output, n, m, k);
                }'''
    
    if old_fp16 in content and "LaunchFastllmGemvFp16Opt" not in content.split("namespace fastllm")[0]:
        print("  Patching FP16 dispatch...")
        content = content.replace(old_fp16, new_fp16, 1)
    
    # Patch INT4_GROUP dispatch
    old_int4 = '''            } else if (weight.dataType == DataType::INT4_GROUP) {
                FastllmCudaHalfMatMulFloatInt4Group(input, weight, bias, output, n, m, k);'''
    
    new_int4 = '''            } else if (weight.dataType == DataType::INT4_GROUP) {
                if (n < 8) {
                    LaunchFastllmInt4GroupGemvFused(FastllmCudaPrepareInput(input), weight.cudaData, weight.extraCudaHalfData[0], weight.extraCudaHalfData[1], bias.dims.size() == 0 ? nullptr : weight.extraCudaHalfData[1], FastllmCudaPrepareOutput(output), n, m, k, weight.group, weight.groupCnt, false);
                } else {
                    FastllmCudaHalfMatMulFloatInt4Group(input, weight, bias, output, n, m, k);
                }'''
    
    if old_int4 in content and "LaunchFastllmInt4GroupGemvFused" not in content.split("namespace fastllm")[0]:
        print("  Patching INT4 dispatch...")
        content = content.replace(old_int4, new_int4, 1)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Patched {path}")

def patch_attention():
    """Add flash decode include and dispatch to fastllm-attention.hip"""
    path = os.path.join(PROJ, "src", "devices", "hip", "attention", "fastllm-attention.hip")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add include
    if "fastllm-flash-decode.hip" not in content:
        # Find last #include line and add after it
        lines = content.split("\n")
        last_include = 0
        for i, line in enumerate(lines):
            if line.startswith("#include"):
                last_include = i
        lines.insert(last_include + 1, '')
        lines.insert(last_include + 2, '// GFX1151 optimized flash decode attention')
        lines.insert(last_include + 3, '#include "fastllm-flash-decode.hip"')
        content = "\n".join(lines)
        print("  Added flash decode include")
    
    # Add flash decode dispatch at the beginning of DoFastllmCudaAttentionBatch
    flash_dispatch = '''    // === GFX1151 Flash Decode Attention for decode path ===
    {
        int q1 = q[0]->dims[1];
        if (q1 == 1) {
            int num_qo_heads = q[0]->dims[0];
            int num_kv_heads = k[0]->dims[0];
            int seq_len = k[0]->dims[1];
            int head_dim = q[0]->dims[2];
            half *Q_data = (half*)q[0]->cudaData;
            half *K_data = (half*)k[0]->cudaData;
            half *V_data = (half*)v[0]->cudaData;
            half *O_data = (half*)FastllmCudaPrepareOutput(*output[0]);
            if (LaunchFlashDecodeAttention(Q_data, K_data, V_data, O_data,
                                          num_qo_heads, num_kv_heads, seq_len, head_dim,
                                          scale, batch, 0)) {
                return true;
            }
        }
    }

'''
    
    target = "bool DoFastllmCudaAttentionBatch(fastllm::Data **q"
    if target in content and "GFX1151 Flash Decode" not in content:
        idx = content.index(target)
        # Find the opening brace
        brace_idx = content.index("{", idx)
        content = content[:brace_idx + 1] + "\n" + flash_dispatch + content[brace_idx + 1:]
        print("  Added flash decode dispatch")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Patched {path}")


def ensure_new_kernel_files():
    """Copy GFX1151 kernel files to hip/ directory (hipify may have deleted them)"""
    import shutil
    src_dir = os.path.join(PROJ, "gfx1151_kernels")
    dst_files = [
        ("linear/fastllm-linear-gemv-mfma.hip", "src/devices/hip/linear/fastllm-linear-gemv-mfma.hip"),
        ("linear/fastllm-linear-int4gemv-mfma.hip", "src/devices/hip/linear/fastllm-linear-int4gemv-mfma.hip"),
        ("attention/fastllm-flash-decode.hip", "src/devices/hip/attention/fastllm-flash-decode.hip"),
    ]
    for src_rel, dst_rel in dst_files:
        src = os.path.join(src_dir, src_rel)
        dst = os.path.join(PROJ, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Copied {dst_rel}")

if __name__ == "__main__":
    print("Applying GFX1151 optimization patches...")
    ensure_new_kernel_files()
    patch_cudadevice()
    patch_attention()
    print("Done!")
