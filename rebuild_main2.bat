@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d "C:\Users\q\.openclaw\workspace\fastllm-rocm"

echo === Step 1: Compile gemma4.cpp ===
call _build_gemma4.bat

echo === Step 2: Rebuild fastllm.lib ===
REM The fastllm.dir has all the individual .obj files already.
REM gemma4.obj was updated in place by _build_gemma4.bat.
REM Rebuild the static lib from all objs in fastllm.dir\Release\
lib.exe /NOLOGO /OUT:build-rocm-msvc2\fastllm.dir\Release\fastllm_new.lib build-rocm-msvc2\fastllm.dir\Release\*.obj
if errorlevel 1 (
    echo FAILED to build fastllm_new.lib
    exit /b 1
)

echo === Step 3: Link main.exe ===
link.exe /NOLOGO /MACHINE:X64 /SUBSYSTEM:CONSOLE /OUT:build-rocm-msvc2\Release\main.exe build-rocm-msvc2\main.dir\Release\main.obj build-rocm-msvc2\fastllm.dir\Release\fastllm_new.lib build-rocm-msvc2\hip_fastllm.lib C:\rocm\lib\hipblas.lib C:\rocm\lib\amdhip64.lib "C:\Python314\Lib\site-packages\_rocm_sdk_devel\lib\llvm\lib\clang\23\lib\windows\clang_rt.builtins-x86_64.lib"
echo === Done ===
