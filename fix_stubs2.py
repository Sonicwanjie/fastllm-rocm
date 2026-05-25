import pathlib
p = pathlib.Path(r'C:\Users\q\.openclaw\workspace\fastllm-rocm\src\devices\hip\attention\fastllm-attention.hip')
content = p.read_text(encoding='utf-8')

# Remove previous inline stubs
bad = '    // TileLang stubs (disabled for now)\n'
idx = content.find(bad)
if idx >= 0:
    # Find the end of the stubs block
    end = content.find('\n\n    // ---- TileLang', idx)
    if end >= 0:
        content = content[:idx] + content[end+2:]
        print('Removed inline stubs')

# Add namespace-scope stubs before DoCudaAttentionReshape
marker = 'void DoCudaAttentionReshape'
stubs = '// TileLang stubs (disabled)\nstatic bool TileLangFlashAttentionSupported(int, int, int, bool) { return false; }\nstatic bool TileLangFlashAttentionPrefill(void*, void*, void*, void*, void*, void*) { return false; }\n\n'
content = content.replace(marker, stubs + marker)
p.write_text(content, encoding='utf-8')
print('Added namespace-scope stubs')
