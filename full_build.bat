@echo off
setlocal enabledelayedexpansion
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PROJ=C:\Users\q\.openclaw\workspace\fastllm-rocm
set BDIR=C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2-full
set ROCM=C:\rocm

echo ========================================
echo  Full CMake build (skip hipify)
echo ========================================

if exist "%BDIR%" rmdir /s /q "%BDIR%"
mkdir "%BDIR%"
cd /d "%BDIR%"

cmake "%PROJ%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_ROCM=ON -DUSE_NUMAS=OFF -DUSE_NUMA=OFF -DUSE_MMAP=OFF -DUSE_SENTENCEPIECE=OFF -DBUILD_CLI=OFF -DPY_API=OFF -DCMAKE_VERBOSE_MAKEFILE=ON -DROCM_ARCH=gfx1151 -DROCM_PATH="%ROCM%" -DCMAKE_PREFIX_PATH="%ROCM%" -DCMAKE_HIP_COMPILER="%ROCM%\bin\hipcc.exe" 2>&1

if %ERRORLEVEL% neq 0 (
    echo CMake FAILED
    exit /b 1
)

echo.
echo Building...
cmake --build . --target fastllm_tools -j16 2>&1

if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
    exit /b 1
)

echo.
echo BUILD SUCCESS
dir Release\main.exe
