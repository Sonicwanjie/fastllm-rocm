import os, glob

# All HIP objects
hip_obj_dir = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\hip_obj"
hip_objs = sorted(glob.glob(os.path.join(hip_obj_dir, "*.obj")))

# All MSVC objects
msvc_obj_dir = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\fastllm_tools.dir\Release"
msvc_objs = sorted(glob.glob(os.path.join(msvc_obj_dir, "*.obj")))

# Output
out_dir = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release"
out_lib = os.path.join(out_dir, "fastllm_tools.lib")
out_dll = os.path.join(out_dir, "fastllm_tools.dll")

# Libraries
hipblas_lib = r"C:\rocm\lib\hipblas.lib"
amdhip64_lib = r"C:\rocm\lib\amdhip64.lib"
clang_rt = r"C:\Python314\Lib\site-packages\_rocm_sdk_devel\lib\llvm\lib\clang\23\lib\windows\clang_rt.builtins-x86_64.lib"

# Build response file
rsp_path = os.path.join(out_dir, "link.rsp")
with open(rsp_path, "w") as f:
    f.write(f'/OUT:"{out_dll}"\n')
    f.write(f'/IMPLIB:"{out_lib}"\n')
    f.write('/NOLOGO\n')
    f.write('/DLL\n')
    f.write('/MACHINE:X64\n')
    f.write('/NODEFAULTLIB:LIBCMT\n')
    f.write('/NODEFAULTLIB:LIBCPMT\n')
    for obj in msvc_objs:
        f.write(f'"{obj}"\n')
    for obj in hip_objs:
        f.write(f'"{obj}"\n')
    f.write(f'"{hipblas_lib}"\n')
    f.write(f'"{amdhip64_lib}"\n')
    f.write(f'"{clang_rt}"\n')
    f.write('kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib\n')

print(f"Response file written to: {rsp_path}")
print(f"Total MSVC objs: {len(msvc_objs)}")
print(f"Total HIP objs: {len(hip_objs)}")

# Show first few lines
with open(rsp_path) as f:
    lines = f.readlines()
print(f"\nFirst 10 lines of {rsp_path}:")
for line in lines[:10]:
    print(f"  {line.rstrip()}")
print(f"  ... ({len(lines)} total lines)")
