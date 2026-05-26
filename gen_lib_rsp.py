import os, sys
sys.stdout.reconfigure(encoding='utf-8')
d = r'C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\fastllm.dir\Release'
objs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.obj')]
with open(r'C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release\lib_content.rsp', 'w') as f:
    for o in objs:
        f.write(f'"{o}"\n')
print(f'Found {len(objs)} obj files')
