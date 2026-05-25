import struct, os

# Check both directories
dirs = [
    r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-windows\hip_obj",
    r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\hip_obj"
]

for dir_path in dirs:
    print(f"\n{'='*60}")
    print(f"Directory: {os.path.basename(os.path.dirname(dir_path))}/{os.path.basename(dir_path)}")
    print('='*60)
    for obj_name in sorted(os.listdir(dir_path)):
        obj_path = os.path.join(dir_path, obj_name)
        with open(obj_path, "rb") as f:
            data = f.read()
        
        machine = struct.unpack_from("<H", data, 0)[0]
        num_sections = struct.unpack_from("<H", data, 2)[0]
        symtab_off = struct.unpack_from("<I", data, 8)[0]
        num_syms = struct.unpack_from("<I", data, 12)[0]
        
        has_bf16 = b"FastllmCudaBFloat16MatMulInt4Group" in data
        has_bf16128 = b"FastllmCudaBFloat16MatMulInt4Group128" in data
        
        # Check string table for these names
        strtab_off = symtab_off + num_syms * 18
        strtab_size = struct.unpack_from("<I", data, strtab_off)[0] if strtab_off + 4 <= len(data) else 0
        strtab = data[strtab_off+4:strtab_off+strtab_size] if strtab_off + strtab_size <= len(data) else b''
        
        in_strtab = b"FastllmCudaBFloat16MatMulInt4Group" in strtab
        
        print(f"  {obj_name}: {len(data)//1024}KB, AMD64={machine==0x8664}, bf16={has_bf16}, bf16128={has_bf16128}, in_strtab={in_strtab}")
