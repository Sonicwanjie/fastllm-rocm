@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=%PROJ%\build-rocm-msvc2

echo [1/3] Compiling tokenizer.cpp...
cl.exe /nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_NUMAS /source-charset:utf-8 /I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface" /Fo"%BDIR%\fastllm_tools.dir\Release\tokenizer.obj" "%PROJ%\src\tokenizer.cpp"

if errorlevel 1 (
    echo [FAIL] tokenizer.cpp compile failed
    exit /b 1
)
echo tokenizer.obj OK

echo [2/3] Copying to fastllm.dir...
copy /Y "%BDIR%\fastllm_tools.dir\Release\tokenizer.obj" "%BDIR%\fastllm.dir\Release\tokenizer.obj" >nul

echo [3/3] Linking main.exe...
link.exe @"%BDIR%\Release\link_main_final.rsp"

if errorlevel 1 (
    echo [FAIL] Link failed
    exit /b 1
)
echo main.exe linked OK
