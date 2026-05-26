@echo off
chcp 65001 >nul
set PATH=C:\rocm\bin;C:\rocm\hip\bin;%PATH%
cd /d C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release
(echo hello && echo stop) | main.exe --path "C:\Users\q\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf" --threads 4
echo EXIT=%ERRORLEVEL%
