import ctypes, os

# Try loading hipblas.dll first to see if it works
hipblas = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release\hipblas.dll"
amdhip = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release\amdhip64_7.dll"

for dll_path in [hipblas, amdhip]:
    print(f"\nTrying to load: {os.path.basename(dll_path)}")
    try:
        dll = ctypes.CDLL(dll_path)
        print(f"  SUCCESS! Handle: {dll._handle}")
    except OSError as e:
        print(f"  FAILED: {e}")
    except Exception as e:
        print(f"  Error ({type(e).__name__}): {e}")

# Now try fastllm_tools with LOAD_WITH_ALTERED_SEARCH_PATH
print("\nTrying fastllm_tools.dll with altered search path:")
dll_path = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release\fastllm_tools.dll"
try:
    # Try setting the DLL directory
    import ctypes.wintypes
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
    
    dll = ctypes.windll.LoadLibraryEx(dll_path, None, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
    print(f"  SUCCESS! Handle: {dll}")
except Exception as e:
    print(f"  FAILED ({type(e).__name__}): {e}")
