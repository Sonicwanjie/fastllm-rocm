import sys
sys.stdout.reconfigure(encoding='utf-8')

# Fix tokenizer.cpp
with open('src/tokenizer.cpp', 'r', encoding='utf-8') as f:
    content = f.read()
# Replace the inline macro stripping with a call to StripJinjaMacros
old = '''            // Strip unsupported Jinja macro blocks before parsing
            std::string& tpl = this->chatTemplate;
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
            }'''
new = '            StripJinjaMacros(this->chatTemplate);'
content = content.replace(old, new)
with open('src/tokenizer.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
print('tokenizer.cpp fixed')

# Fix fastllm.cpp - add StripJinjaMacros call after chatTemplate assignment
with open('src/fastllm.cpp', 'r', encoding='utf-8') as f:
    content = f.read()
old = '            tokenizer.chatTemplate = this->dicts["chat_template"];'
new = old + '\n            StripJinjaMacros(tokenizer.chatTemplate);'
content = content.replace(old, new, 1)
with open('src/fastllm.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
print('fastllm.cpp fixed')

# Fix model.cpp - add StripJinjaMacros call after chatTemplate assignment  
with open('src/model.cpp', 'r', encoding='utf-8') as f:
    content = f.read()
old = '                model->weight.tokenizer.chatTemplate = params["tokenizer.chat_template"].string_value();'
new = old + '\n                StripJinjaMacros(model->weight.tokenizer.chatTemplate);'
content = content.replace(old, new, 1)
with open('src/model.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
print('model.cpp fixed')
