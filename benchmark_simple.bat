@echo off
setlocal
set MODEL_PATH=%1
if "%MODEL_PATH%"=="" set MODEL_PATH=models\gemma-4-e2b-it
set DTYPE=%2
if "%DTYPE%"=="" set DTYPE=bfloat16
set ATYPE=%3
if "%ATYPE%"=="" set ATYPE=float16

echo === FastLLM Benchmark ===
echo Model: %MODEL_PATH%
echo dtype: %DTYPE%  atype: %ATYPE%
echo.

echo %time% Starting...
echo Hello! Please write a short paragraph about artificial intelligence. | .\build-rocm-ninja\main.exe -p %MODEL_PATH% -t 4 --dtype %DTYPE% --atype %ATYPE%
echo.
echo %time% Done.
endlocal
