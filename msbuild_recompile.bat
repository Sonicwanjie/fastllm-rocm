@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2
msbuild fastllm_tools.vcxproj /p:Configuration=Release /p:Platform=x64 /t:ClCompile /p:SelectedFiles="src\model.cpp;src\fastllm.cpp;src\tokenizer.cpp" /v:minimal
