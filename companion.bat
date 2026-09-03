@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtual environment not found at "%PYTHON_EXE%".
    echo Please run install.bat first to set up the environment.
    pause
    exit /b 1
)

if "%~1"=="" (
    start "" "%PYTHON_EXE%" -m companion.app
) else (
    "%PYTHON_EXE%" -m companion.app --input "%~1" --output "%~1"
)
