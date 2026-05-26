@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=%PROJ%\build-rocm-msvc2
set CFLAGS=/nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_NUMAS /source-charset:utf-8
set INCLUDES=/I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface"

echo [1/5] Compiling tokenizer.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\tokenizer.obj" "%PROJ%\src\tokenizer.cpp"
if errorlevel 1 ( echo [FAIL] tokenizer && exit /b 1 )

echo [2/5] Compiling template.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\template.obj" "%PROJ%\src\template.cpp"
if errorlevel 1 ( echo [FAIL] template && exit /b 1 )

echo [3/5] Compiling basellm.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\basellm.obj" "%PROJ%\src\models\basellm.cpp"
if errorlevel 1 ( echo [FAIL] basellm && exit /b 1 )

echo [4/5] Compiling gemma4.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\gemma4.obj" "%PROJ%\src\models\gemma4.cpp"
if errorlevel 1 ( echo [FAIL] gemma4 && exit /b 1 )

echo [5/5] Linking main.exe...
link.exe @"%BDIR%\Release\link_main_final.rsp"
if errorlevel 1 ( echo [FAIL] link && exit /b 1 )

echo ALL DONE
