import struct

path = r"C:\Users\q\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf"
keys_of_interest = ['per_layer','kv_shared','layer_type','partial_rotary','hidden_size_per','num_kv','ple','sliding','global','rope','norm_eps','head_dim']

with open(path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    for i in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8', errors='replace')
        val_type = struct.unpack('<I', f.read(4))[0]
        should_print = any(w in key.lower() for w in keys_of_interest)
        
        if val_type == 8:
            s_len = struct.unpack('<Q', f.read(8))[0]
            val = f.read(s_len).decode('utf-8', errors='replace')[:300]
        elif val_type == 4:
            val = struct.unpack('<I', f.read(4))[0]
        elif val_type == 5:
            val = struct.unpack('<i', f.read(4))[0]
        elif val_type == 6:
            val = struct.unpack('<f', f.read(4))[0]
        elif val_type == 12:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 5:
                vals = [struct.unpack('<i', f.read(4))[0] for _ in range(arr_len)]
                val = str(vals[:50])
            else:
                elem_size = 4 if arr_type in (4,5,6) else 2 if arr_type in (2,3) else 1
                f.read(arr_len * elem_size)
                val = f"[array of {arr_len} elements type {arr_type}]"
        elif val_type == 7:
            val = struct.unpack('<?', f.read(1))[0]
        else:
            if val_type in (0,1): f.read(1)
            elif val_type in (2,3): f.read(2)
            elif val_type == 9: f.read(8)
            elif val_type in (10,11): f.read(8)
            val = f"[type {val_type}]"
        if should_print:
            print(f"  {key} = {val}")
