import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('src/model.cpp', 'r', encoding='utf-8') as f:
    content = f.read()
# Remove debug lines
import re
content = re.sub(r'\s*printf\("basellm::InitParams entry\\n"\); fflush\(stdout\);\n', '\n', content)
content = re.sub(r'\s*printf\("DEBUG: Before InitParams.*?\n', '\n', content)
content = re.sub(r'\s*printf\("DEBUG: After InitParams.*?\n', '\n', content)
with open('src/model.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
print('Cleaned')
