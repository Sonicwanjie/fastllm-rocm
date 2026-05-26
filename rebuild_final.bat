@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=%PROJ%\build-rocm-msvc2
set CFLAGS=/nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_CUDA /DUSE_NUMAS /source-charset:utf-8
set INCLUDES=/I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\include\devices\cuda" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface"

echo [1/3] tokenizer.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\tokenizer.obj" "%PROJ%\src\tokenizer.cpp"
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [2/3] main.cpp...
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\main.dir\Release\main.obj" "%PROJ%\main.cpp"
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [3/3] Linking...
link.exe @"%BDIR%\Release\link_main_final.rsp"
if errorlevel 1 ( echo FAIL LINK && exit /b 1 )

echo ALL DONE
