# Fix the vcxproj to use build-rocm-msvc2 hip_obj files instead of build-rocm-windows
import re

vcxproj_path = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\fastllm_tools.vcxproj"

with open(vcxproj_path, "r", encoding="utf-8") as f:
    content = f.read()

# The wrong path
wrong_path = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-windows\hip_obj"
# The correct path
correct_path = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\hip_obj"

# Count occurrences before
count_before = content.count(wrong_path)
print(f"Occurrences of build-rocm-windows hip_obj path: {count_before}")

# Replace
new_content = content.replace(wrong_path, correct_path)

count_after = new_content.count(wrong_path)
print(f"After replacement: {count_after}")

# Also fix the path for the multihip file (has a slightly different path)
wrong_multihip = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-windows\hip_obj\src_devices_multihip"
correct_multihip = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\hip_obj\src_devices_multihip"
count_multihip = new_content.count(wrong_multihip)
print(f"Occurrences of build-rocm-windows multihip path: {count_multihip}")
new_content = new_content.replace(wrong_multihip, correct_multihip)

with open(vcxproj_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done - vcxproj updated")

# Verify
with open(vcxproj_path, "r", encoding="utf-8") as f:
    verify = f.read()
remaining_windows = verify.count(wrong_path)
remaining_windows_multihip = verify.count(wrong_multihip)
print(f"Verification - remaining build-rocm-windows hip_obj: {remaining_windows}")
print(f"Verification - remaining build-rocm-windows multihip: {remaining_windows_multihip}")
correct = verify.count(correct_path)
print(f"Occurrences of build-rocm-msvc2 hip_obj path: {correct}")
