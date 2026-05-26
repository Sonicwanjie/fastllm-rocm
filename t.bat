@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cl.exe /nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_CUDA /DUSE_NUMAS /source-charset:utf-8 /arch:AVX2 /I"include" /I"include\utils" /I"include\models" /I"include\blocks" /I"include\devices\cpu" /I"include\devices\disk" /I"include\devices\cuda" /I"third_party\json11" /I"third_party\gguf" /I"third_party\flashinfer" /I"third_party\gpu_iface" /Fo"build-rocm-msvc2\fastllm_tools.dir\Release\tokenizer.obj" "src\tokenizer.cpp"
