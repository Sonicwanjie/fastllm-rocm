@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=%PROJ%\build-rocm-msvc2
set CFLAGS=/nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_CUDA /DUSE_NUMAS /source-charset:utf-8 /arch:AVX2
set INCLUDES=/I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\include\devices\cuda" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface"

echo [1/8] fastllm.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\fastllm.obj" "%PROJ%\src\fastllm.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL fastllm && exit /b 1 )
echo OK

echo [2/8] model.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\model.obj" "%PROJ%\src\model.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL model && exit /b 1 )
echo OK

echo [3/8] tokenizer.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\tokenizer.obj" "%PROJ%\src\tokenizer.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL tokenizer && exit /b 1 )
echo OK

echo [4/8] template.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\template.obj" "%PROJ%\src\template.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL template && exit /b 1 )
echo OK

echo [5/8] basellm.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\basellm.obj" "%PROJ%\src\models\basellm.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL basellm && exit /b 1 )
echo OK

echo [6/8] gemma4.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\gemma4.obj" "%PROJ%\src\models\gemma4.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL gemma4 && exit /b 1 )
echo OK

echo [7/8] gguf.cpp + adapter...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\gguf.obj" "%PROJ%\third_party\gguf\gguf.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL gguf && exit /b 1 )
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\gguf-adapter.obj" "%PROJ%\third_party\gguf\gguf-adapter.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL gguf-adapter && exit /b 1 )
echo OK

echo [8/8] main.cpp + link...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\main.dir\Release\main.obj" "%PROJ%\main.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL main && exit /b 1 )
link.exe @"%BDIR%\Release\link_main_final.rsp" >nul 2>&1
if errorlevel 1 ( echo FAIL link && exit /b 1 )

echo ALL DONE
