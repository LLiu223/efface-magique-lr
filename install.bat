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
    if !errorlevel! neq 0 goto :CORE_FAILED
    uv pip install --no-deps simple-lama-inpainting==0.1.2 --python .venv\Scripts\python.exe
    if !errorlevel! neq 0 goto :CORE_FAILED

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
if !errorlevel! neq 0 goto :CORE_FAILED
pip install --no-deps simple-lama-inpainting==0.1.2
if !errorlevel! neq 0 goto :CORE_FAILED

where nvidia-smi >nul 2>nul
if !errorlevel! equ 0 (
    echo [*] NVIDIA GPU detected! Upgrading PyTorch with CUDA 12.4 hardware acceleration...
    pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124
)
goto :INSTALL_COMPLETE

:CORE_FAILED
echo.
echo [ERROR] Installing the core Python packages failed; see the message above.
echo     The app will not run until this is fixed. Nothing else was changed.
pause
exit /b 1

:INSTALL_COMPLETE
:: ---------------------------------------------------------------------
:: Optional models: each one asks y/N (default N), downloads into models\ and verifies SHA256. Safe to re-run: verified files are skipped.
:: ---------------------------------------------------------------------
set "MODELS_DIR=%~dp0models"
set "FETCH_ERRORS=0"

echo.
set "ANSWER="
set /p "ANSWER=[?] Download MI-GAN, a small fast eraser, 28 MB, MIT licence? [y/N]: "
if /i "!ANSWER!"=="y" call :INSTALL_EXTRAS requirements-migan.txt MI-GAN
if /i "!ANSWER!"=="y" if "!PIP_OK!"=="1" (
    call :FETCH "https://huggingface.co/andraniksargsyan/migan/resolve/406830d0fa60666da0071c342ad2fbc8f30c5c64/migan_pipeline_v2.onnx" "!MODELS_DIR!\migan_pipeline_v2.onnx" 6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b
)

echo.
echo [i] Stable Diffusion inpainting is prompt-driven but slow on CPU. Download size: about 2.1 GB,
echo     plus the diffusers/transformers/accelerate packages. Licence: CreativeML OpenRAIL-M, use restrictions apply.
echo     The diffusers safety checker is turned off; see README "Optional models" for why and how to turn it on.
set "ANSWER="
set /p "ANSWER=[?] Download Stable Diffusion inpainting, about 2.1 GB? [y/N]: "
if /i "!ANSWER!"=="y" call :INSTALL_EXTRAS requirements-sd.txt "Stable Diffusion"
if /i "!ANSWER!"=="y" if "!PIP_OK!"=="1" (
    set "SD_URL=https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
    call :FETCH_SD model_index.json 636d700de8970a70f02b91d433ad949dc2ceb750f39ba0cf2cf39ef0b0b9d1e4
    call :FETCH_SD feature_extractor/preprocessor_config.json 2a1da83b5e1032aaeef397552ddb408dca0d8cd1dc58f61bf6abf38d6f33a0a2
    call :FETCH_SD scheduler/scheduler_config.json 8b457848b74deeeab2a967b75048dd2cb6f393cfc17751eca1b5ec655cff0183
    call :FETCH_SD text_encoder/config.json 845df614cb9327ae7bbea027316246fae917827407da6df13572e41b5f93b4cc
    call :FETCH_SD text_encoder/model.fp16.safetensors 77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e
    call :FETCH_SD tokenizer/merges.txt 9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a
    call :FETCH_SD tokenizer/special_tokens_map.json c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac
    call :FETCH_SD tokenizer/tokenizer_config.json 00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882
    call :FETCH_SD tokenizer/vocab.json e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349
    call :FETCH_SD unet/config.json 669f6581087fac5b16c44e42c58b359a35eadf2af5691d7945480bd7ceac8e6d
    call :FETCH_SD unet/diffusion_pytorch_model.fp16.safetensors 24b788b4a777748377cc20364eea4ae113c8c42f4468c16bc8c02fdae5492af9
    call :FETCH_SD vae/config.json 4cd0ce4803567837175fbb8412689fe77d069c1ef8171beb83979db309a8bad5
    call :FETCH_SD vae/diffusion_pytorch_model.fp16.safetensors 4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c
)

echo.
echo [i] The assistant can find any object you name ("power lines", "the sign on the left") with CLIPSeg.
echo     Without it, it only finds 20 common object classes (person, car, dog, ...).
set "ANSWER="
set /p "ANSWER=[?] Enable finding any named object (CLIPSeg, about 600 MB, Apache-2.0)? [y/N]: "
if /i "!ANSWER!"=="y" call :INSTALL_EXTRAS requirements-clipseg.txt CLIPSeg
if /i "!ANSWER!"=="y" if "!PIP_OK!"=="1" (
    set "CS_URL=https://huggingface.co/CIDAS/clipseg-rd64-refined/resolve/999e0328d9e10b484360c477313983f9afdd7050"
    call :FETCH "!CS_URL!/config.json" "!MODELS_DIR!\clipseg\config.json" c023375966d31b3b1392764f7bd91df47098ce19f62f11b0263d8eedcf708bcd
    call :FETCH "!CS_URL!/preprocessor_config.json" "!MODELS_DIR!\clipseg\preprocessor_config.json" 4fb09ebcfd7651205ca8299b993c30088e5535ef350a72d46c5c4580eeac0440
    call :FETCH "!CS_URL!/tokenizer_config.json" "!MODELS_DIR!\clipseg\tokenizer_config.json" 4c75d57fd9bd0be8478ad2d6f8b9cebdd4a45338eb108547329c2b6333476ca6
    call :FETCH "!CS_URL!/special_tokens_map.json" "!MODELS_DIR!\clipseg\special_tokens_map.json" c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac
    call :FETCH "!CS_URL!/vocab.json" "!MODELS_DIR!\clipseg\vocab.json" e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349
    call :FETCH "!CS_URL!/merges.txt" "!MODELS_DIR!\clipseg\merges.txt" 9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a
    call :FETCH "!CS_URL!/model.safetensors" "!MODELS_DIR!\clipseg\model.safetensors" d00ca85d6b859f9d07b7cfb8ef26fe9771cb275b34c9368f2ecf603139307f55
)

if not "!FETCH_ERRORS!"=="0" (
    echo.
    echo [ERROR] !FETCH_ERRORS! optional model step^(s^) failed: package install, download or verification.
    echo     Partial and bad downloads were deleted.
    echo     The core app works without them. Re-run install.bat to retry.
)

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
exit /b 0

:: ---------------------------------------------------------------------
:: Subroutines
:: ---------------------------------------------------------------------
:INSTALL_EXTRAS
:: %1 = requirements file, %2 = model label. Sets PIP_OK=1 on success; on failure the caller skips the weights.
set "PIP_OK=0"
echo [*] Installing %~2 packages from %~1...
where uv >nul 2>nul
if !errorlevel! equ 0 (
    uv pip install -r "%~1" --python .venv\Scripts\python.exe
) else (
    .venv\Scripts\python.exe -m pip install -r "%~1"
)
if !errorlevel! neq 0 (
    echo [ERROR] Installing %~1 failed. Skipping the %~2 weight download: the model could not run without its packages.
    set /a FETCH_ERRORS+=1
    exit /b 1
)
set "PIP_OK=1"
exit /b 0

:FETCH_SD
:: %1 = file path inside the pinned HF revision, %2 = expected SHA256
set "SD_REL=%~1"
call :FETCH "!SD_URL!/!SD_REL!" "!MODELS_DIR!\sd_inpaint\!SD_REL:/=\!" %~2
exit /b

:FETCH
:: %1 = URL, %2 = destination file, %3 = expected SHA256. Downloads to <dest>.part and moves it into place only
:: after the hash matches, so an interrupted or failed download never leaves a file at the final path.
set "FETCH_DEST=%~2"
set "FETCH_PART=%~2.part"
if exist "!FETCH_DEST!" (
    call :HASH_MATCHES "!FETCH_DEST!" %~3
    if "!HASH_OK!"=="1" (
        echo [=] Already present and verified: !FETCH_DEST!
        exit /b 0
    )
    echo [*] Existing file failed verification, downloading it again: !FETCH_DEST!
    del /f /q "!FETCH_DEST!"
)
for %%D in ("!FETCH_DEST!") do if not exist "%%~dpD" mkdir "%%~dpD"
if exist "!FETCH_PART!" del /f /q "!FETCH_PART!"
echo [*] Downloading %~1
curl.exe -L --fail --retry 3 -o "!FETCH_PART!" "%~1"
if !errorlevel! neq 0 (
    echo [ERROR] Download failed: %~1
    if exist "!FETCH_PART!" del /f /q "!FETCH_PART!"
    set /a FETCH_ERRORS+=1
    exit /b 1
)
call :HASH_MATCHES "!FETCH_PART!" %~3
if not "!HASH_OK!"=="1" (
    echo [ERROR] SHA256 mismatch, download deleted: !FETCH_PART!
    del /f /q "!FETCH_PART!"
    set /a FETCH_ERRORS+=1
    exit /b 1
)
move /y "!FETCH_PART!" "!FETCH_DEST!" >nul
if !errorlevel! neq 0 (
    echo [ERROR] Could not move the verified download into place: !FETCH_DEST!
    del /f /q "!FETCH_PART!"
    set /a FETCH_ERRORS+=1
    exit /b 1
)
echo [+] Verified: !FETCH_DEST!
exit /b 0

:HASH_MATCHES
:: %1 = file, %2 = expected SHA256. Sets HASH_OK=1 on a match. certutil's header and footer
:: lines contain ':' (in every UI language), the hash line does not.
set "HASH_OK=0"
set "HASH_ACTUAL="
for /f "delims=" %%H in ('certutil -hashfile "%~1" SHA256 ^| findstr /v ":"') do (
    if not defined HASH_ACTUAL set "HASH_ACTUAL=%%H"
)
if defined HASH_ACTUAL set "HASH_ACTUAL=!HASH_ACTUAL: =!"
if /i "!HASH_ACTUAL!"=="%~2" set "HASH_OK=1"
exit /b 0
