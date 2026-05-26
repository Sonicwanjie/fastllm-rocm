import sys
sys.stdout.reconfigure(encoding='utf-8')

strip_code_fastllm = '''
            // Strip unsupported Jinja macro blocks
            {
                std::string& tpl = tokenizer.chatTemplate;
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

strip_code_model = '''
                // Strip unsupported Jinja macro blocks
                {
                    std::string& tpl = model->weight.tokenizer.chatTemplate;
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

with open('src/fastllm.cpp', 'r', encoding='utf-8') as f:
    content = f.read()
old = '            tokenizer.chatTemplate = this->dicts["chat_template"];'
if strip_code_fastllm.strip() not in content:
    content = content.replace(old, old + strip_code_fastllm, 1)
    with open('src/fastllm.cpp', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed fastllm.cpp')
else:
    print('fastllm.cpp already fixed')

with open('src/model.cpp', 'r', encoding='utf-8') as f:
    content = f.read()
old2 = '                model->weight.tokenizer.chatTemplate = params["tokenizer.chat_template"].string_value();'
if strip_code_model.strip() not in content:
    content = content.replace(old2, old2 + strip_code_model, 1)
    with open('src/model.cpp', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed model.cpp')
else:
    print('model.cpp already fixed')
