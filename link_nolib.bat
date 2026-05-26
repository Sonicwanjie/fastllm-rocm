@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set BDIR=C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2
link.exe @"%BDIR%\Release\link_test_nolib.rsp"
if errorlevel 1 ( echo FAIL && exit /b 1 )
echo DONE
