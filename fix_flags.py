import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('rebuild_full.bat', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('/arch:AVX2', '')
with open('rebuild_full.bat', 'w', encoding='utf-8') as f:
    f.write(content)
print('Removed /arch:AVX2')
