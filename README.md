# 🪄 Efface Magique LR (Lightroom AI Generative Eraser)

> **Free, open-source, private, and local alternative to Adobe Lightroom's cloud-based Generative Remove.**  
> Seamlessly integrates into Adobe Lightroom Classic via a native Lua SDK plugin and a high-performance Python AI companion app powered by PyQt6 and Simple-LaMa.

---

## ✨ Features

- ⚡ **Seamless Live Window Sync**: Launch once via `File > Plug-in Extras > ⚡ AI Generative Eraser (Live Window)...`. The companion window stays open on a second monitor or side-by-side and automatically updates whenever you navigate or select photos in Lightroom!
- 🔄 **Non-Blocking Auto-Stacking**: Click **⚡ Save & Sync to Lightroom** (`Ctrl+S`) to losslessly save 16-bit TIFFs and stack them directly into your Lightroom catalog without closing or relaunching the companion app!
- 📌 **Always-on-Top Floating Mode**: Pin the companion window (`Ctrl+T`) to keep it floating smoothly above Lightroom.
- 🧠 **Dual Inpainting Engine**:
  - ✨ **Firefly Generative Mode**: Multi-seed diffusion generating 3 distinct candidate variations with prompt guidance.
  - ⚡ **Fast Spot Mode**: Simple-LaMa GAN for instantaneous blemish, dust, and wire removal.
- 🎯 **Photographic Quality & Fidelity**:
  - Context-aware crop with >= 35% margin (default 85%).
  - Camera sensor noise estimation & ISO grain matching.
  - Smooth sigmoid alpha transition with **zero pixel bleed or blur** on untouched regions.
  - 16-bit ProPhoto RGB, Adobe RGB, and EXIF metadata preservation.

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

#### ⚡ Option A: Seamless Live Window Sync (Recommended)
1. In Lightroom Classic, select **File > Plug-in Extras > ⚡ AI Generative Eraser (Live Window)...**
2. The companion window opens and displays your selected photo.
3. Switch photos in Lightroom (using arrow keys or clicking in the Filmstrip) — the companion window **updates dynamically and seamlessly**!
4. Paint over unwanted objects with the red brush and click **✨ Erase Object**.
5. Pick your favorite of 3 Firefly candidate variations.
6. Click **⚡ Save & Sync to Lightroom** (`Ctrl+S`) — the photo is losslessly saved and automatically stacked in your Lightroom catalog, and the companion window stays open ready for your next photo!

#### 🪄 Option B: Single Photo Mode
1. In Lightroom, select photo > **File > Plug-in Extras > 🪄 AI Generative Eraser (Single Photo)...**
2. Edit in companion, click **⚡ Save & Sync to Lightroom** (`Ctrl+S`), and close the window when finished.

---

## ⌨ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl + S` / `Ctrl + Shift + S` | **Save & Sync to Lightroom** (Auto-stacks into catalog and stays open) |
| `Ctrl + T` | **Toggle Always on Top (Pin)** |
| `[` / `]` | Decrease / Increase Brush Radius |
| `Ctrl + Z` | Undo (Brush stroke or inpainting step) |
| `Ctrl + Y` / `Ctrl + Shift + Z` | Redo |
| `Spacebar` (Hold) / `\` | Compare Before (Original) vs After (Active Variation) |
| `Y` | Interactive Before/After Split-Screen Slider |
| `Ctrl + 0` / `F` | Fit Image to Screen Viewport |
| `Ctrl + 1` | 100% 1:1 Pixel Scale View |
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
