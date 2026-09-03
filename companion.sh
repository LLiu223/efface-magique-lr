#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "[ERROR] Python virtual environment not found at $PYTHON_BIN"
    echo "Please run ./install.sh first."
    exit 1
fi

if [ -z "$1" ]; then
    "$PYTHON_BIN" -m companion.app &
else
    "$PYTHON_BIN" -m companion.app --input "$1" --output "$1" &
fi
