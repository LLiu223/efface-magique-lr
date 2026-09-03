@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo           Efface Magique LR - Automated Setup Script (Windows)
echo ======================================================================
echo.

:: Check for uv or python
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Found 'uv' package manager. Setting up fast virtual environment...
    uv venv .venv --python 3.11
    if %errorlevel% neq 0 (
        echo [*] Falling back to default uv python...
        uv venv .venv
    )
    echo [*] Installing dependencies with uv pip...
    uv pip install -r requirements.txt --python .venv\Scripts\python.exe
    goto :INSTALL_COMPLETE
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Setting up standard virtual environment with Python...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt

    :: Detect NVIDIA GPU and upgrade to CUDA 12.4 hardware accelerated PyTorch
    where nvidia-smi >nul 2>nul
    if !errorlevel! equ 0 (
        echo [*] NVIDIA GPU detected! Upgrading PyTorch with CUDA 12.4 hardware acceleration...
        pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124
    )
    goto :INSTALL_COMPLETE
)

echo [!] Error: Neither 'uv' nor 'python' was found in your system PATH.
echo Please install Python 3.10+ or Astral uv: https://astral.sh/uv
pause
exit /b 1

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
