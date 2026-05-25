import os

hip_obj_dir = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-windows\hip_obj"
obj_files = sorted(os.listdir(hip_obj_dir))

print("HIP object files found:")
for f in obj_files:
    print(f"  {f}")

print("\nCMake code to add:")
print("    # Pass HIP .obj files directly to MSVC linker (bypass broken ar-lib)")
hip_dir_for_cmake = hip_obj_dir.replace("\\", "/").replace("C:", "/c")
for f in obj_files:
    cmake_path = f"\"{hip_dir_for_cmake}/{f}\""
    print(f"    {cmake_path}")
