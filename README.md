# 🪄 Efface Magique LR (Lightroom AI Generative Eraser)

> **Free, open-source, private, and local alternative to Adobe Lightroom's cloud-based Generative Remove.**  
> Seamlessly integrates into Adobe Lightroom Classic via a native Lua SDK plugin and a high-performance Python AI companion app powered by PyQt6 and Simple-LaMa.

---

## ✨ Features

- ⚡ **Native Lightroom Classic Integration**: Launch directly from Lightroom via `File > Plug-in Extras > AI Generative Eraser` or Library context menu.
- 🔄 **Automatic Stacking & Reimport**: Exports a full 16-bit lossless TIFF preserving color profiles (ProPhoto RGB / Adobe RGB), launches the companion app, and stacks the edited version adjacent to your original photo.
- 🧠 **Simple-LaMa AI Inpainting**: State-of-the-art fast Fourier convolution inpainting running 100% locally on your machine (NVIDIA CUDA / Apple Silicon MPS / multi-threaded CPU).
- 🎯 **High-Res Crop & Feather Blend**:
  - Automatically calculates the minimum bounding box with a 25% safety margin.
  - Feathers the edited patch back onto the full-resolution original with a Gaussian alpha mask.
  - Guarantees **zero blur or artifact degradation** on untouched image areas.
- 🎨 **Modern Lightroom-Style Companion GUI**:
  - High-performance canvas supporting 24MP–60MP RAW/TIFF exports.
  - Interactive red mask brush with adjustable size, opacity, and eraser mode.
  - Smooth pan (middle click / spacebar drag) and zoom (mouse wheel).
  - Full Undo (`Ctrl+Z`) and Redo (`Ctrl+Y`) action history.
  - Non-destructive Before/After comparison (Hold `Spacebar` or toggle button).

---

## 🚀 Quick Start

### 1. Installation

#### Windows:
Run the one-click setup script in PowerShell or Command Prompt:
```cmd
install.bat
```

#### macOS / Linux:
```bash
chmod +x install.sh
./install.sh
```

---

### 2. Add Plugin to Lightroom Classic

1. Open **Adobe Lightroom Classic**.
2. Go to **File > Plug-in Manager** (`Ctrl+Alt+Shift+,` on Windows / `Cmd+Opt+Shift+,` on Mac).
3. Click the **Add** button in the lower-left corner.
4. Select the folder:
   ```
   efface-magique-lr/plugin/ai_eraser.lrplugin
   ```
5. Confirm that the status displays **"Installed and running"** with a green dot.
6. Click **Done**.

---

### 3. Usage Workflow

1. In Lightroom Classic's **Library** or **Develop** module, select any photo.
2. Click **File > Plug-in Extras > AI Generative Eraser...** (or right-click the photo > *Plug-in Extras*).
3. The companion app opens with your full-resolution photo.
4. Paint over the object(s) you want to remove using the red brush.
5. Click **✨ Erase Object**.
6. Once satisfied, click **✔ Save & Return to Lightroom**.
7. The edited image is automatically imported into your Lightroom catalog and stacked right next to your original!

---

## ⌨ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `[` / `]` | Decrease / Increase Brush Radius |
| `Ctrl + Z` | Undo (Brush stroke or inpainting step) |
| `Ctrl + Y` / `Ctrl + Shift + Z` | Redo |
| `Spacebar` (Hold) | Compare Before (Original) vs After (Edited) / Pan View |
| `Middle Mouse Drag` | Pan Image Canvas |
| `Mouse Wheel` | Smooth Zoom (Centered under cursor) |

---

## 🧪 Standalone Testing (Without Lightroom)

You can run the companion app independently on any photo:
```bash
# Windows
.venv\Scripts\python -m companion.app --input path\to\photo.jpg --output path\to\edited.jpg

# macOS / Linux
.venv/bin/python -m companion.app --input path/to/photo.jpg --output path/to/edited.jpg
```

---

## 📁 Project Structure

```
efface-magique-lr/
├── GEMINI.md                            # Architecture and source-of-truth blueprint
├── README.md                            # User documentation and setup guide
├── requirements.txt                     # Pinned Python package dependencies
├── install.bat                          # Automated Windows installer
├── install.sh                           # Automated macOS/Linux installer
│
├── plugin/
│   └── ai_eraser.lrplugin/              # Adobe Lightroom Classic Plugin Bundle
│       ├── Info.lua                     # Plugin manifest and menu hooks
│       ├── GenerativeEraser.lua         # Export -> Launch Companion -> Reimport & Stack
│       └── PluginUtils.lua              # Environment discovery & CLI launcher helper
│
├── companion/                           # Python AI Companion Application
│   ├── app.py                           # PyQt6 Main Window and toolbar
│   ├── canvas.py                        # Interactive high-res canvas with brush & zoom
│   ├── inpainting_engine.py             # Simple-LaMa model & high-res feather blend
│   └── __main__.py                      # Package entry point
│
└── tests/
    └── test_engine.py                   # Automated test suite
```

---

## 🛠 Troubleshooting

- **"Python not found" in Lightroom**:
  Ensure you ran `install.bat` (or `install.sh`) so `.venv` is created in the project directory. The plugin automatically detects `.venv\Scripts\python.exe`.
- **GPU Acceleration**:
  If you have an NVIDIA GPU, PyTorch will automatically use CUDA. On Apple Silicon, it utilizes Metal Performance Shaders (MPS). Otherwise, multi-threaded CPU fallback is used.
- **Color Profiles**:
  The plugin exports 16-bit TIFFs in ProPhoto RGB to maintain maximum dynamic range and color accuracy.

---

## 📄 License
MIT License. Free and open-source for personal and commercial photography workflows.
