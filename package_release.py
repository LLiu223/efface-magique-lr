"""
package_release.py
Automated packaging script for Efface Magique LR.
Creates a clean, portable distribution ZIP archive for other users.
"""

import os
import zipfile
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DIST_DIR = os.path.join(_PROJECT_ROOT, "dist")
_ZIP_NAME = "Efface-Magique-LR.zip"

# Items to include in the distribution
INCLUDE_DIRS = [
    "plugin",
    "companion",
]

INCLUDE_FILES = [
    "install.bat",
    "install.sh",
    "companion.bat",
    "companion.sh",
    "requirements.txt",
    "README.md",
    "package_release.bat",
]

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    ".coverage",
}

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".git",
    ".vscode",
    ".tmp",
    "dist",
}

def create_release_zip() -> str:
    """Bundle all necessary files into dist/Efface-Magique-LR.zip."""
    os.makedirs(_DIST_DIR, exist_ok=True)
    zip_path = os.path.join(_DIST_DIR, _ZIP_NAME)

    print(f"[*] Packaging Efface Magique LR distribution into: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add explicit files
        for filename in INCLUDE_FILES:
            file_path = os.path.join(_PROJECT_ROOT, filename)
            if os.path.isfile(file_path):
                arcname = os.path.join("Efface-Magique-LR", filename)
                zf.write(file_path, arcname)
                print(f"  + Added {filename}")

        # Add directories recursively
        for dir_name in INCLUDE_DIRS:
            dir_path = os.path.join(_PROJECT_ROOT, dir_name)
            if not os.path.isdir(dir_path):
                continue

            for root, dirs, files in os.walk(dir_path):
                # Filter out excluded directories in-place
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in EXCLUDE_EXTS or f.startswith("."):
                        continue

                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, _PROJECT_ROOT)
                    arcname = os.path.join("Efface-Magique-LR", rel_path)
                    zf.write(full_path, arcname)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[OK] Release package created successfully: {zip_path} ({size_mb:.2f} MB)")
    return zip_path

if __name__ == "__main__":
    out = create_release_zip()
    print(f"\nReady to distribute: {out}")
