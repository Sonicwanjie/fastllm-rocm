import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('build-rocm-msvc2/Release/link_main_final.rsp', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('build-rocm-msvc2/Release/link_nofastllm.rsp', 'w', encoding='utf-8') as f:
    for line in lines:
        if 'fastllm.lib' not in line:
            f.write(line)
print('Done')
