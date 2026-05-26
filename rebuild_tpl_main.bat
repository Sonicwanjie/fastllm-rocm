@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=%PROJ%\build-rocm-msvc2
set CFLAGS=/nologo /c /MD /O2 /Ob2 /DNDEBUG /std:c++20 /EHsc /GR /DNOMINMAX /DWIN32 /D_WINDOWS /DUSE_NUMAS /source-charset:utf-8
set INCLUDES=/I"%PROJ%\include" /I"%PROJ%\include\utils" /I"%PROJ%\include\models" /I"%PROJ%\include\blocks" /I"%PROJ%\include\devices\cpu" /I"%PROJ%\include\devices\disk" /I"%PROJ%\third_party\json11" /I"%PROJ%\third_party\gguf" /I"%PROJ%\third_party\flashinfer" /I"%PROJ%\third_party\gpu_iface"
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\fastllm_tools.dir\Release\template.obj" "%PROJ%\src\template.cpp" >nul 2>&1
cl.exe %CFLAGS% %INCLUDES% /Fo"%BDIR%\main.dir\Release\main.obj" "%PROJ%\main.cpp" >nul 2>&1
link.exe @"%BDIR%\Release\link_nofastllm.rsp"
