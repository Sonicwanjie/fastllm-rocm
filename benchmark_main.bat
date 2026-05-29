
@echo off
REM Simple benchmark using main.exe
REM Usage: benchmark_main.bat
REM Types a prompt, waits for response, measures time

setlocal
set MODEL_PATH=%1
if "%MODEL_PATH%"=="" set MODEL_PATH=models\gemma-4-e2b-it
set DTYPE=%2
if "%DTYPE%"=="" set DTYPE=bfloat16
set ATYPE=%3
if "%ATYPE%"=="" set ATYPE=float16

echo Running benchmark with:
echo   Model: %MODEL_PATH%
echo   dtype: %DTYPE%
echo   atype: %ATYPE%
echo.

echo Starting timer...
set START=%TIME%

echo Hello! Please write a short paragraph about artificial intelligence. | .\build-rocm-ninja\main.exe -p %MODEL_PATH% -t 4 --dtype %DTYPE% --atype %ATYPE%

set END=%TIME%
echo.
echo Start: %START%
echo End:   %END%
endlocal
