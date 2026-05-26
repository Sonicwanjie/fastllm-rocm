import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('include/fastllm.h', 'r', encoding='utf-8') as f:
    content = f.read()
func_code = '''
// Strip unsupported Jinja macro blocks from a template string
inline void StripJinjaMacros(std::string &tpl) {
    for (size_t p = 0; p < tpl.size(); ) {
        size_t ms = tpl.find("{% macro", p);
        if (ms == std::string::npos) ms = tpl.find("{%- macro", p);
        if (ms == std::string::npos) break;
        size_t me = tpl.find("{% endmacro", ms + 2);
        if (me == std::string::npos) me = tpl.find("{%- endmacro", ms + 2);
        if (me == std::string::npos) break;
        size_t cl = tpl.find("%}", me + 2);
        if (cl == std::string::npos) { p = me + 2; continue; }
        tpl.erase(ms, cl + 2 - ms);
        p = ms;
    }
}
'''
content = content.replace('#endif //TEST_FASTLLM_H', func_code + '\n#endif //TEST_FASTLLM_H')
with open('include/fastllm.h', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
