import glob, os, re

header_replacements = {
    '<cuda_fp8.h>': '<hip/hip_fp8.h>',
    '<cuda_bf16.h>': '<hip/hip_bfloat16.h>',
    '<rccl/rccl.h>': '<hip/hip_runtime.h>',
}

def fix_hip_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content

    for old, new in header_replacements.items():
        content = content.replace(old, new)

    content = content.replace('CUDA_R_16BF', 'HIP_R_16BF')

    # Fix ALL hipDataType ComputeType patterns (any value)
    content = re.sub(
        r'hipDataType\s+(\w+)\s*=\s*(HIP_R_\w+)\s*,\s*(\w+)\s*=\s*(HIP_R_\w+)\s*,\s*(\w+)\s*=\s*(HIP_R_\w+)\s*,\s*ComputeType\s*=\s*(HIP_R_\w+)',
        r'hipDataType \1 = \2, \3 = \4, \5 = \6; hipblasComputeType_t ComputeType = HIPBLAS_COMPUTE_32F',
        content
    )

    content = content.replace('__ballot(', '__ballot64(')

    # Fix __bfloat162float and __float2bfloat16_rn
    content = re.sub(r'__bfloat162float\s*\(\s*([^)]+)\s*\)', r'((float)(__nv_bfloat16(\1)))', content)
    content = re.sub(r'__float2bfloat16_rn\s*\(\s*([^)]+)\s*\)', r'__nv_bfloat16(\1)', content)

    # Fix warp shuffle mask: 32-bit -> 64-bit
    content = content.replace('0xffffffffu', '0xffffffffffffffffULL')
    content = re.sub(r'__shfl_xor_sync\s*\(\s*0xffffffff\b', '__shfl_xor_sync(0xffffffffffffffffULL', content)
    content = re.sub(r'__shfl_sync\s*\(\s*0xffffffff\b', '__shfl_sync(0xffffffffffffffffULL', content)
    content = re.sub(r'__shfl_up_sync\s*\(\s*0xffffffff\b', '__shfl_up_sync(0xffffffffffffffffULL', content)
    content = re.sub(r'__shfl_down_sync\s*\(\s*0xffffffff\b', '__shfl_down_sync(0xffffffffffffffffULL', content)

    # Fix warpMask variable type
    content = content.replace('const unsigned int warpMask =', 'const uint64_t warpMask =')

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Fixed: {filepath}')

for pattern in ['src/devices/hip/**/*.hip', 'src/devices/multihip/**/*.hip']:
    for f in glob.glob(pattern, recursive=True):
        fix_hip_file(f)
