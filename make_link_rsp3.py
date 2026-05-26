import os, glob
bd = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2"
objs = glob.glob(f"{bd}/fastllm.dir/Release/*.obj")
objs = [o for o in objs if "_cuda" not in os.path.basename(o) and "_multicuda" not in os.path.basename(o) and "deepseekv4" not in os.path.basename(o)]
rsp = f"{bd}/link_main.rsp"
lines = [
    "/NOLOGO", "/MACHINE:X64", "/SUBSYSTEM:CONSOLE",
    f"/OUT:{bd}/Release/main.exe",
    f'"{bd}/main.dir/Release/main.obj"'
]
for o in sorted(objs):
    lines.append(f'"{o}"')
hip_objs = glob.glob(f"{bd}/hip_obj/*.obj")
for o in sorted(hip_objs):
    lines.append(f'"{o}"')
lines.append("C:/rocm/lib/hipblas.lib")
lines.append("C:/rocm/lib/amdhip64.lib")
lines.append('"C:/Python314/Lib/site-packages/_rocm_sdk_devel/lib/llvm/lib/clang/23/lib/windows/clang_rt.builtins-x86_64.lib"')
open(rsp, "w").write("\n".join(lines))
print(f"Wrote {rsp}: {len(lines)-4} entries")
