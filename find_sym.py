import struct

# Search for mangled C++ symbols in the COFF object
with open(r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-windows\hip_obj\src_devices_hip_linear_fastllm-linear-int4group_hip.obj", "rb") as f:
    data = f.read()

import re

# Find all readable strings that look like C++ mangled names or function names
strings = re.findall(b'[A-Za-z_][A-Za-z0-9_]{6,}', data)
cpp_symbols = []
for s in strings:
    try:
        decoded = s.decode('ascii')
        # Look for things that might be C++ mangled symbols or function names
        if 'Fastllm' in decoded or 'BFloat' in decoded or 'MatMul' in decoded or 'Int4' in decoded:
            cpp_symbols.append(decoded)
    except:
        pass

print("Relevant symbols found:")
for s in sorted(set(cpp_symbols))[:30]:
    print(f"  {s}")

# Also check for any symbol starting with ? (C++ mangled)
mangled = re.findall(b'\?[A-Za-z0-9_@?$]+', data)
print(f"\nC++ mangled symbols count: {len(mangled)}")
for m in mangled[:10]:
    try:
        print(f"  {m.decode('ascii')}")
    except:
        pass
