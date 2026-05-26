@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d "C:\Users\q\.openclaw\workspace\fastllm-rocm"

echo === Step 1: Compile gemma4.cpp ===
call _build_gemma4.bat

echo === Step 2: Build fastllm_new.lib ===
lib.exe /NOLOGO /OUT:build-rocm-msvc2\fastllm.dir\Release\fastllm_new.lib build-rocm-msvc2\fastllm.dir\Release\*.obj

echo === Step 3: Link main.exe with /FORCE ===
link.exe /NOLOGO /MACHINE:X64 /SUBSYSTEM:CONSOLE /FORCE:UNRESOLVED /OUT:build-rocm-msvc2\Release\main.exe build-rocm-msvc2\main.dir\Release\main.obj build-rocm-msvc2\fastllm.dir\Release\fastllm_new.lib build-rocm-msvc2\hip_fastllm.lib C:\rocm\lib\hipblas.lib C:\rocm\lib\amdhip64.lib "C:\Python314\Lib\site-packages\_rocm_sdk_devel\lib\llvm\lib\clang\23\lib\windows\clang_rt.builtins-x86_64.lib"
echo === Done ===
