# Find all HIP .obj files from build-rocm-msvc2/hip_obj
import os, glob

hip_obj_dir = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\hip_obj"
hip_objs = sorted(glob.glob(os.path.join(hip_obj_dir, "*.obj")))
print("HIP objects:")
for o in hip_objs:
    print(f"  {o}")
print(f"Total: {len(hip_objs)}")

# Find the MSVC object files
msvc_obj_dir = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\fastllm_tools.dir\Release"
msvc_objs = sorted(glob.glob(os.path.join(msvc_obj_dir, "*.obj")))
print(f"\nMSVC objects: {len(msvc_objs)}")

# Libraries
rocm_lib = r"C:\rocm\lib"
hipblas_lib = os.path.join(rocm_lib, "hipblas.lib")
amdhip64_lib = os.path.join(rocm_lib, "amdhip64.lib")
clang_rt = r"C:\Python314\Lib\site-packages\_rocm_sdk_devel\lib\llvm\lib\clang\23\lib\windows\clang_rt.builtins-x86_64.lib"

print(f"\nhipblas: {hipblas_lib}")
print(f"amdhip64: {amdhip64_lib}")
print(f"clang_rt: {clang_rt}")
