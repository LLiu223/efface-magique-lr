#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "          Efface Magique LR - Automated Setup Script (macOS/Linux)     "
echo "======================================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
    echo "[*] Found 'uv' package manager. Setting up virtual environment..."
    uv venv .venv
    uv pip install -r requirements.txt --python .venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    echo "[*] Setting up virtual environment with python3..."
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[!] Error: Neither 'uv' nor 'python3' was found."
    echo "Please install Python 3.10+ or Astral uv: https://astral.sh/uv"
    exit 1
fi

echo ""
echo "======================================================================"
echo "                   Installation Succeeded!                            "
echo "======================================================================"
echo ""
echo "[1] Lightroom Classic Plugin:"
echo "    - In Lightroom, open: File > Plug-in Manager"
echo "    - Click 'Add' and select the folder:"
echo "      ${SCRIPT_DIR}/plugin/ai_eraser.lrplugin"
echo ""
echo "[2] Run Standalone Companion:"
echo "    .venv/bin/python -m companion.app"
echo ""
echo "======================================================================"
