import pathlib
p = pathlib.Path(r'C:\Users\q\.openclaw\workspace\fastllm-rocm\src\devices\hip\attention\fastllm-attention.hip')
content = p.read_text(encoding='utf-8')

old = '''    // ---- TileLang Flash Attention Prefill (MFMA-optimized) ----
    if (q1 > 1 && actual_batch >= 1 && q2 == 256 && v2 == 256) {
        TileLangFlashAttnParams tl_params;
        tl_params.batch = actual_batch;
        tl_params.num_qo_heads = num_qo_heads;
        tl_params.num_kv_heads = num_kv_heads;
        tl_params.seq_q = q1;
        tl_params.seq_kv = k1;
        tl_params.head_dim = q2;
        tl_params.is_causal = (maskType == 0);
        tl_params.scale = scale;
        if (TileLangFlashAttentionSupported(num_qo_heads, num_kv_heads, q2, tl_params.is_causal)) {
            bool tl_ok = TileLangFlashAttentionPrefill(qd, kd, vd, od, &tl_params, nullptr);
            if (tl_ok) {
                DeviceSync();
                return true;
            }
        }
    }'''

new = '''    // ---- TileLang Flash Attention disabled ----
    // TileLang prefill not yet integrated''' 

if old in content:
    content = content.replace(old, new)
    p.write_text(content, encoding='utf-8')
    print('Disabled TileLang block')
else:
    print('ERROR: block not found')
