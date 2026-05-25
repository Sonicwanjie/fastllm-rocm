import pathlib
p = pathlib.Path(r'C:\Users\q\.openclaw\workspace\fastllm-rocm\src\devices\hip\linear\fastllm-linear-bf16.hip')
content = p.read_text(encoding='utf-8')

# Remove the optimized kernel block
start_marker = '// --- Optimized GEMV kernel for AMD APU (warp shuffle reduction) ---'
end_marker = '// --- End optimized GEMV kernel ---'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)
if start_idx >= 0 and end_idx >= 0:
    end_idx += len(end_marker)
    content = content[:start_idx] + content[end_idx:]
    print(f'Removed optimized GEMV kernel block ({start_idx}:{end_idx})')

# Restore original LaunchFastllmGemmFp32Bf16
old_launch = 'void LaunchFastllmGemmFp32Bf16(float *input, __nv_bfloat16 *weight, float *output, float *bias, int n, int m, int k) {\n    if (n == 1) {\n        hipLaunchKernelGGL(HIP_KERNEL_NAME(FastllmGemvFp32Bf16Warp<256>), dim3(k), dim3(256), 0, 0,\n                           input, weight, output, bias, m, k);\n    } else if (n <= 4) {\n        for (int i = 0; i < n; i++) {\n            hipLaunchKernelGGL(HIP_KERNEL_NAME(FastllmGemvFp32Bf16Warp<256>), dim3(k), dim3(256), 0, 0,\n                               input + (size_t)i * m, weight, output + (size_t)i * k, bias, m, k);\n        }\n    } else {\n        for (int i = 0; i < n; i++) {\n            hipLaunchKernelGGL(HIP_KERNEL_NAME(FastllmGemvFp32Bf16Warp<256>), dim3(k), dim3(256), 0, 0,\n                               input + (size_t)i * m, weight, output + (size_t)i * k, bias, m, k);\n        }\n    }\n}'

new_launch = 'void LaunchFastllmGemmFp32Bf16(float *input, __nv_bfloat16 *weight, float *output, float *bias, int n, int m, int k) {\n    if (n == 1) {\n       hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 1>) , dim3(k), dim3(256), 0, 0, input, weight, output, bias, m, k);\n    } else if (n == 2) {\n       hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 2>) , dim3(k), dim3(256), 0, 0, input, weight, output, bias, m, k);\n    } else if (n == 3) {\n       hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 3>) , dim3(k), dim3(256), 0, 0, input, weight, output, bias, m, k);\n    } else if (n == 4) {\n       hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 4>) , dim3(k), dim3(256), 0, 0, input, weight, output, bias, m, k);\n    } else if (n == 5) {\n       hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 5>) , dim3(k), dim3(256), 0, 0, input, weight, output, bias, m, k);\n    } else if (n == 6) {\n       hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 6>) , dim3(k), dim3(256), 0, 0, input, weight, output, bias, m, k);\n    } else if (n == 7) {\n       hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 7>) , dim3(k), dim3(256), 0, 0, input, weight, output, bias, m, k);\n    } else {\n        for (int i = 0; i < n; i++) {\n           hipLaunchKernelGGL(( FastllmGemvFp32Bf16Kernel2MultiRow<256, 1>) , dim3(k), dim3(256), 0, 0, input + i * m, weight, output + i * k, bias, m, k);\n        }\n    }\n}'

if old_launch in content:
    content = content.replace(old_launch, new_launch)
    print('Restored original LaunchFastllmGemmFp32Bf16')
else:
    print('LaunchFastllmGemmFp32Bf16 not found or already original')

p.write_text(content, encoding='utf-8')
print('Reverted bf16.hip to original kernels')
