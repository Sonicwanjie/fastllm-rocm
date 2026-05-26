import struct

path = r"C:\Users\q\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf"
keys_of_interest = ['per_layer','kv_shared','layer_type','partial_rotary','hidden_size_per','num_kv','ple','sliding','global','rope','norm_eps','head_dim']

with open(path, 'rb') as f:
    magic = f.read(4).decode('utf-8')
    version = struct.unpack('<I', f.read(4))[0]
    print(f"GGUF v{version}, magic={magic}")
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    print(f"{n_tensors} tensors, {n_kv} meta keys")
    
    for i in range(min(n_kv, 200)):
        key_len = struct.unpack('<Q', f.read(8))[0]
        if key_len > 1000:
            f.read(key_len)
            key = "[TOO_LONG]"
        else:
            key = f.read(key_len).decode('utf-8', errors='replace')
        val_type = struct.unpack('<I', f.read(4))[0]
        should_print = any(w in key.lower() for w in keys_of_interest)
        
        val = ""
        if val_type == 8:
            s_len = struct.unpack('<Q', f.read(8))[0]
            if s_len < 10000:
                val = f.read(s_len).decode('utf-8', errors='replace')[:300]
            else:
                f.read(s_len)
                val = f"[string len {s_len}]"
        elif val_type == 4:
            val = str(struct.unpack('<I', f.read(4))[0])
        elif val_type == 5:
            val = str(struct.unpack('<i', f.read(4))[0])
        elif val_type == 6:
            val = str(struct.unpack('<f', f.read(4))[0])
        elif val_type == 12:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_len > 10000:
                f.read(arr_len * (1 if arr_type in (0,1,7) else 2 if arr_type in (2,3) else 4))
                val = f"[array len {arr_len}]"
            elif arr_type == 5:
                vals = [struct.unpack('<i', f.read(4))[0] for _ in range(arr_len)]
                val = str(vals[:30])
            else:
                elem_size = 4
                f.read(arr_len * elem_size)
                val = f"[arr{arr_len}]"
        elif val_type == 7:
            val = str(struct.unpack('<?', f.read(1))[0])
        else:
            if val_type in (0,1): f.read(1)
            elif val_type in (2,3): f.read(2)
            elif val_type in (9,10,11): f.read(8)
        if should_print:
            print(f"  {key} = {val}")
