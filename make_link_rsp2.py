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

# Windows SDK library paths (x64)
win_sdk_root = r"C:\Program Files (x86)\Windows Kits\10\lib"
win_lib_paths = []
for ver in os.listdir(win_sdk_root):
    lib_path = os.path.join(win_sdk_root, ver, "um", "x64")
    ucrt_path = os.path.join(win_sdk_root, ver, "ucrt", "x64")
    if os.path.exists(lib_path):
        win_lib_paths.append(lib_path)
    if os.path.exists(ucrt_path):
        win_lib_paths.append(ucrt_path)

# MSVC library path
msvc_lib = r"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231\lib\x64"

# Build response file
rsp_path = os.path.join(out_dir, "link.rsp")
with open(rsp_path, "w") as f:
    f.write(f'/OUT:"{out_dll}"\n')
    f.write(f'/IMPLIB:"{out_lib}"\n')
    f.write('/NOLOGO\n')
    f.write('/DLL\n')
    f.write('/MACHINE:X64\n')
    f.write('/FORCE:MULTIPLE\n')  # Ignore duplicate symbol LNK2005 errors
    f.write('/SUBSYSTEM:CONSOLE\n')
    # Library paths (order matters: specific first, then MSVC, then Windows SDK)
    f.write(f'/LIBPATH:"{msvc_lib}"\n')
    for wp in win_lib_paths:
        f.write(f'/LIBPATH:"{wp}"\n')
    # System libs
    f.write('kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib\n')
    # ROCm libs
    f.write(f'"{hipblas_lib}"\n')
    f.write(f'"{amdhip64_lib}"\n')
    f.write(f'"{clang_rt}"\n')
    # Main fastllm static library
    fastllm_lib = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\fastllm.dir\Release\fastllm.lib"
    f.write(f'"{fastllm_lib}"\n')
    # HIP objects first (so real implementations override stubs from MSVC objects)
    for obj in hip_objs:
        f.write(f'"{obj}"\n')
    for obj in msvc_objs:
        f.write(f'"{obj}"\n')

print(f"Response file: {rsp_path}")
print(f"Win SDK lib paths: {win_lib_paths}")
print(f"MSVC lib path: {msvc_lib}")
print(f"Total objects: {len(msvc_objs)} C++ + {len(hip_objs)} HIP = {len(msvc_objs)+len(hip_objs)}")
