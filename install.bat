@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo           Efface Magique LR - Automated Setup Script (Windows)
echo ======================================================================
echo.

:: Prefer standard Python to create a digitally signed virtual environment (prevents Windows Smart App Control blocks)
where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Found system Python. Creating digitally signed virtual environment...
    python -m venv .venv
    goto :INSTALL_PACKAGES
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Found 'uv' package manager. Setting up virtual environment...
    uv venv .venv --python 3.11
    if !errorlevel! neq 0 (
        echo [*] Falling back to default uv python...
        uv venv .venv
    )
    goto :INSTALL_PACKAGES
)

echo [!] Error: Neither 'python' nor 'uv' was found in your system PATH.
echo Please install Python 3.10+ (python.org) or Astral uv: https://astral.sh/uv
pause
exit /b 1

:INSTALL_PACKAGES
:: Check if uv is available for ultra-fast pip installation
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Installing dependencies with fast uv pip...
    uv pip install -r requirements.txt --python .venv\Scripts\python.exe

    where nvidia-smi >nul 2>nul
    if !errorlevel! equ 0 (
        echo [*] NVIDIA GPU detected! Installing PyTorch with CUDA 12.4 hardware acceleration...
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --python .venv\Scripts\python.exe
    )
    goto :INSTALL_COMPLETE
)

echo [*] Installing dependencies with standard pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

where nvidia-smi >nul 2>nul
if !errorlevel! equ 0 (
    echo [*] NVIDIA GPU detected! Upgrading PyTorch with CUDA 12.4 hardware acceleration...
    pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124
)
goto :INSTALL_COMPLETE

:INSTALL_COMPLETE
echo.
echo ======================================================================
echo                    Installation Succeeded!
echo ======================================================================
echo.
echo [1] Lightroom Classic Plugin:
echo     - In Lightroom, go to: File > Plug-in Manager
echo     - Click 'Add' and choose the folder:
echo       %~dp0plugin\ai_eraser.lrplugin
echo.
echo [2] Standalone Companion App Test:
echo     .venv\Scripts\python.exe -m companion.app
echo.
echo ======================================================================
pause
