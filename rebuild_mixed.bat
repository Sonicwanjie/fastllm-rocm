@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=%PROJ%\build-rocm-msvc2
set CFLAGS_CUDA=/nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_CUDA /DUSE_NUMAS /source-charset:utf-8 /arch:AVX2
set CFLAGS_NOCUDA=/nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_NUMAS /source-charset:utf-8 /arch:AVX2
set INCLUDES_CUDA=/I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\include\devices\cuda" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface"
set INCLUDES_NO=/I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface"

echo [1/7] fastllm.cpp (USE_CUDA)...
cl.exe %CFLAGS_CUDA% %INCLUDES_CUDA% /Fo"%BDIR%\fastllm_tools.dir\Release\fastllm.obj" "%PROJ%\src\fastllm.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [2/7] model.cpp (USE_CUDA)...
cl.exe %CFLAGS_CUDA% %INCLUDES_CUDA% /Fo"%BDIR%\fastllm_tools.dir\Release\model.obj" "%PROJ%\src\model.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [3/7] tokenizer.cpp (USE_CUDA)...
cl.exe %CFLAGS_CUDA% %INCLUDES_CUDA% /Fo"%BDIR%\fastllm_tools.dir\Release\tokenizer.obj" "%PROJ%\src\tokenizer.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [4/7] template.cpp (USE_CUDA)...
cl.exe %CFLAGS_CUDA% %INCLUDES_CUDA% /Fo"%BDIR%\fastllm_tools.dir\Release\template.obj" "%PROJ%\src\template.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [5/7] basellm.cpp (NO CUDA)...
cl.exe %CFLAGS_NOCUDA% %INCLUDES_NO% /Fo"%BDIR%\fastllm_tools.dir\Release\basellm.obj" "%PROJ%\src\models\basellm.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [6/7] gemma4.cpp (NO CUDA)...
cl.exe %CFLAGS_NOCUDA% %INCLUDES_NO% /Fo"%BDIR%\fastllm_tools.dir\Release\gemma4.obj" "%PROJ%\src\models\gemma4.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )

echo [7/7] main.cpp (USE_CUDA) + link...
cl.exe %CFLAGS_CUDA% %INCLUDES_CUDA% /Fo"%BDIR%\main.dir\Release\main.obj" "%PROJ%\main.cpp" >nul 2>&1
if errorlevel 1 ( echo FAIL && exit /b 1 )
link.exe @"%BDIR%\Release\link_main_final.rsp"
if errorlevel 1 ( echo FAIL LINK && exit /b 1 )

echo ALL DONE
