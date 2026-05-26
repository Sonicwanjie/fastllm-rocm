@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=%PROJ%\build-rocm-msvc2
set CFLAGS=/nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_CUDA /DUSE_NUMAS /source-charset:utf-8 /arch:AVX2
set INCLUDES=/I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\include\devices\cuda" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface"

echo [1/5] tokenizer.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\tokenizer.obj" "%PROJ%\src\tokenizer.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )
echo OK

echo [2/5] model.cpp (with USE_CUDA)...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\model.obj" "%PROJ%\src\model.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )
echo OK

echo [3/5] fastllm.cpp (with USE_CUDA)...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\fastllm.obj" "%PROJ%\src\fastllm.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )
echo OK

echo [4/5] main.cpp (with USE_CUDA)...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\main.dir\Release\main.obj" "%PROJ%\main.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )
echo OK

echo [5/5] Linking...
link.exe @"%BDIR%\Release\link_main_final.rsp"
if errorlevel 1 ( echo FAIL LINK && exit /b 1 )

echo ALL DONE
