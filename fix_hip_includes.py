import glob, os, re

header_replacements = {
    '<cuda_fp8.h>': '<hip/hip_fp8.h>',
    '<cuda_bf16.h>': '<hip/hip_bfloat16.h>',
    '<rccl/rccl.h>': '<hip/hip_runtime.h>',
}

# Additional CUDA->HIP replacements missed by hipify
additional_replacements = {
    'CUBLAS_COMPUTE_32F': 'HIPBLAS_COMPUTE_32F',
    'CUBLAS_GEMM_DEFAULT_TENSOR_OP': 'HIPBLAS_GEMM_DEFAULT_TENSOR_OP',
}

# Regex patterns for duplicate typedefs (already in fastllm-hip.h)
typedef_patterns_to_remove = [
    r'typedef\s+union\s+__align__\(16\)\s+_union_bf16_4_fp16\s*\{[^}]*\}\s*union_bf16_4_fp16\s*;',
    r'typedef\s+union\s+__align__\(16\)\s+_union_bf16_4\s*\{[^}]*\}\s*union_bf16_4\s*;',
    r'typedef\s+union\s+__align__\(16\)\s+_union_bf16_8\s*\{[^}]*\}\s*union_bf16_8\s*;',
]

# CUDA->HIP API macros for flashinfer compatibility
cuda_hip_macros = r'''
// CUDA->HIP API compatibility macros (injected by fix_hip_includes.py)
#ifndef CUDA_HIP_MACROS_INCLUDED
#define CUDA_HIP_MACROS_INCLUDED
#define cudaGetDevice hipGetDevice
#define cudaLaunchKernel hipLaunchKernel
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
// PDL (Programmatic Dependent Launch) is CUDA-only - provide stubs
struct cudaLaunchAttribute { unsigned id; union { int cooperative; } val; cudaLaunchAttribute() : id(0) {} };
#define cudaLaunchAttributeProgrammaticStreamSerialization 0
struct cudaLaunchConfig_t {
    cudaLaunchAttribute* attrs; unsigned numAttrs; dim3 gridDim; dim3 blockDim;
    size_t dynamicSmemBytes; hipStream_t stream;
    cudaLaunchConfig_t() : attrs(nullptr), numAttrs(0), gridDim(1), blockDim(1), dynamicSmemBytes(0), stream(0) {}
};
// Stub: cudaLaunchKernelEx falls back to regular hipLaunchKernel
// Use inline wrapper to pack args into void*[]
template<typename KernT, typename... Args>
static inline __host__ hipError_t cudaLaunchKernelEx(cudaLaunchConfig_t* cfg, KernT kernel, Args... args) {
    void* arg_ptrs[] = {(void*)&args...};
    return hipLaunchKernel((void*)kernel, cfg->gridDim, cfg->blockDim, arg_ptrs, cfg->dynamicSmemBytes, cfg->stream);
}
#endif
'''

def fix_hip_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content

    for old, new in header_replacements.items():
        content = content.replace(old, new)

    content = content.replace('CUDA_R_16BF', 'HIP_R_16BF')

    content = re.sub(
        r'hipDataType\s+(\w+)\s*=\s*(HIP_R_\w+)\s*,\s*(\w+)\s*=\s*(HIP_R_\w+)\s*,\s*(\w+)\s*=\s*(HIP_R_\w+)\s*,\s*ComputeType\s*=\s*(HIP_R_\w+)',
        r'hipDataType \1 = \2, \3 = \4, \5 = \6; hipblasComputeType_t ComputeType = HIPBLAS_COMPUTE_32F',
        content
    )

    content = content.replace('__ballot(', '__ballot64(')

    content = re.sub(r'__bfloat162float\s*\(\s*([^)]+)\s*\)', r'((float)(__nv_bfloat16(\1)))', content)
    content = re.sub(r'__float2bfloat16_rn\s*\(\s*([^)]+)\s*\)', r'__nv_bfloat16(\1)', content)

    content = content.replace('0xffffffffu', '0xffffffffffffffffULL')
    content = re.sub(r'__shfl_xor_sync\s*\(\s*0xffffffff\b', '__shfl_xor_sync(0xffffffffffffffffULL', content)
    content = re.sub(r'__shfl_sync\s*\(\s*0xffffffff\b', '__shfl_sync(0xffffffffffffffffULL', content)
    content = re.sub(r'__shfl_up_sync\s*\(\s*0xffffffff\b', '__shfl_up_sync(0xffffffffffffffffULL', content)
    content = re.sub(r'__shfl_down_sync\s*\(\s*0xffffffff\b', '__shfl_down_sync(0xffffffffffffffffULL', content)

    content = content.replace('const unsigned int warpMask =', 'const uint64_t warpMask =')

    # Apply additional CUDA->HIP replacements
    for old, new in additional_replacements.items():
        content = content.replace(old, new)

    # Remove duplicate typedef definitions
    for pattern in typedef_patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    content = re.sub(r'\n{3,}', '\n\n', content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Fixed: {filepath}')


def inject_cuda_hip_macros(filepath):
    if 'fastllm-attention' not in filepath:
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'CUDA_HIP_MACROS_INCLUDED' in content:
        return
    content = content.replace(
        '#include "hip/hip_runtime.h"',
        '#include "hip/hip_runtime.h"\n' + cuda_hip_macros,
        1
    )
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Injected CUDA->HIP macros: {filepath}')


for pattern in ['src/devices/hip/**/*.hip', 'src/devices/multihip/**/*.hip']:
    for f in glob.glob(pattern, recursive=True):
        fix_hip_file(f)

# Separate pass: inject CUDA->HIP macros into attention .hip files
for f in glob.glob('src/devices/hip/**/*.hip', recursive=True):
    inject_cuda_hip_macros(f)
for f in glob.glob('src/devices/multihip/**/*.hip', recursive=True):
    inject_cuda_hip_macros(f)
