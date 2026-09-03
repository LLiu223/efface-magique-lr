@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo Packaging Efface Magique LR for distribution...
"%PYTHON_EXE%" "%SCRIPT_DIR%package_release.py"

if exist "%SCRIPT_DIR%dist\Efface-Magique-LR.zip" (
    echo.
    echo ========================================================
    echo  Success! Distribution ZIP created at:
    echo  dist\Efface-Magique-LR.zip
    echo ========================================================
    echo.
    explorer /select,"%SCRIPT_DIR%dist\Efface-Magique-LR.zip"
) else (
    echo [ERROR] Packaging failed.
)
pause
