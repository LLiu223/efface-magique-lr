# GEMINI.md — Project Blueprint: Efface Magique LR (Adobe Firefly Grade)

## 1. Executive Summary & Architecture

**Efface Magique LR** is an open-source, local, private, and high-performance alternative to Adobe Lightroom's cloud-based Generative Remove / Magic Eraser, engineered to match and rival Adobe Firefly (Generative Fill). It integrates directly into Adobe Lightroom Classic via a native Lua SDK plugin (`.lrplugin`) and a high-performance local Python companion application built with PyQt6, PyTorch, and a dual-tier inpainting pipeline.

### End-to-End Workflow Architecture
```
+---------------------------------------------------------------------------------------------------+
|                                      ADOBE LIGHTROOM CLASSIC                                      |
|                                                                                                   |
| 1. User selects target photo in Library or Develop module.                                        |
| 2. User invokes: File > Plug-in Extras > "🪄 AI Generative Eraser" (or Library Context Menu,       |
|    or Ctrl+Alt+E external editing preset).                                                        |
| 3. GenerativeEraser.lua:                                                                          |
|    - Validates target photo and acquires export session.                                          |
|    - Exports full-resolution 16-bit TIFF with embedded color profile (ProPhoto RGB) to workspace. |
|    - Launches local companion app via LrTasks.execute / LrShell, passing file arguments.         |
|    - Yields task and listens for companion exit code.                                             |
| 4. On companion exit code 0:                                                                      |
|    - Reads exported & inpainted TIFF back into Lightroom Catalog using LrCatalog:addPhoto().     |
|    - Stacks the new photo adjacent to the original source photo automatically.                    |
+-------------------------------------------------+-------------------------------------------------+
                                                  | Process invocation (CLI arguments)
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        PYTHON COMPANION APP (PyQt6 - Firefly Experience)                          |
|                                                                                                   |
| - High-performance hardware-accelerated canvas supporting 24MP-60MP raw/tiff exports.             |
| - Dual Inpainting Engine:                                                                         |
|   * ✨ Generative Firefly Mode: Diffusion-based inpainting generating 3 candidate variations     |
|     with distinct random seeds and optional text prompt conditioning.                             |
|   * ⚡ Fast Mode: Simple-LaMa GAN for instantaneous spot, dust, and wire removal.                 |
| - Variations Carousel: Clickable thumbnail cards at bottom to instantly preview & switch          |
|   between generated variations, plus "🔄 More Variations" button with seed re-roll.               |
| - High-Fidelity Photographic Blending Pipeline:                                                   |
|   1. Intelligent context-aware crop & padding with >= 35% margin (default 85%).                   |
|   2. Camera sensor noise profiling (Laplacian variance & per-channel residuals).                   |
|   3. Synthetic sensor ISO grain matching to eliminate plastic smoothing.                         |
|   4. Seamless sigmoid / outer-feather alpha blending (zero untouched pixel bleed).                |
| - Brush mask tools: Auto-Erase on stroke release, adjustable radius ([ and ]), eraser mode.       |
| - Hold-to-Compare: Hold \ (Backslash) or Spacebar for instant Before/After comparison.            |
| - 16-bit Color & Metadata: Lossless ProPhoto RGB ICC profile, EXIF, and DPI tag preservation.    |
| - "Save & Return to Lightroom": Writes active variation losslessly to TIFF and exits code 0.      |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Technology Stack

- **Lightroom Classic Plugin**:
  - Lightroom Classic SDK (Lua 5.1 / SDK version 11.0+)
  - Lightroom APIs: `LrApplication`, `LrCatalog`, `LrTasks`, `LrExportSession`, `LrShell`, `LrPathUtils`, `LrFileUtils`, `LrDialogs`, `LrProgressScope`
- **Companion Application**:
  - Python 3.10+ (Tested on 3.11 / 3.12)
  - **GUI Toolkit**: PyQt6 (High-DPI aware, OpenGL viewport canvas, dark studio aesthetic)
  - **Dual Inference Engine**:
    * Firefly Diffusion Inpainting with multi-seed variation & prompt guidance
    * Simple-LaMa (Fast Fourier Convolutions) for rapid spot healing
  - **GPU Hardware Acceleration & Optimizations**:
    * NVIDIA CUDA GPU acceleration with cuDNN auto-tuning (`torch.backends.cudnn.benchmark = True`)
    * TensorFloat-32 (`TF32`) computation on NVIDIA Ampere (RTX 3070) Tensor Cores
    * Direct in-VRAM GPU spatial frequency synthesis (PyTorch CUDA tensors) eliminating CPU bus transfer overhead
    * Pre-warmed CUDA kernels on initialization to eliminate first-stroke latency
    * Auto-scaling neural inference (`max_dim = 1024` with Lanczos high-res reconstruction): Prevents VRAM exhaustion and thread freezes when inpainting large objects on 24MP-60MP images
    * Automatic VRAM garbage collection (`torch.cuda.empty_cache()`) to leave VRAM available for Lightroom Classic
  - **Photographic Pipeline**:
    * Sensor noise profiling & ISO grain matching (`companion.utils.blending`)
    * Softened sigmoid transition alpha composite
    * Bounding box margin scaling ($\ge 35\%$ margin)
  - **Deep Learning Runtime**: PyTorch (CUDA 12.4 / Apple Silicon MPS / multi-threaded CPU fallback)
  - **Image Processing**: OpenCV (`opencv-python`), Pillow (`PIL`), NumPy
- **Packaging & Environment**:
  - `uv` / standard `venv` Python environment management
  - Cross-platform launchers with automatic NVIDIA GPU detection (`companion.bat`, `companion.sh`, `install.bat`, `install.sh`)

---

## 3. Directory Layout

```
efface-magique-lr/
├── GEMINI.md                            # Complete architectural blueprint & project guidelines
├── README.md                            # End-user setup and installation manual
├── requirements.txt                     # Production Python dependencies
├── requirements-dev.txt                 # Testing & QA dependencies (pytest, pytest-qt, etc.)
├── run_tests.py                         # Unified test runner with coverage and formatted reports
├── companion.bat                        # Windows external editor launcher for Ctrl+Alt+E workflow
├── companion.sh                         # Unix/macOS external editor launcher
├── install.bat                          # One-click Windows setup script
├── install.sh                           # One-click macOS/Linux setup script
├── .gitignore                           # Git ignore rules for venvs, temp TIFFs, and caches
│
├── plugin/
│   └── ai_eraser.lrplugin/              # Adobe Lightroom Classic Plugin Bundle
│       ├── Info.lua                     # Plugin manifest, IDs, SDK version & multi-module menu hooks
│       ├── GenerativeEraser.lua         # Export -> Launch Companion -> Await -> Re-import & Stack
│       └── PluginUtils.lua              # Cross-platform path resolver & Python launcher helper
│
├── companion/                           # Python AI Companion Application
│   ├── __init__.py                      # Package descriptor
│   ├── __main__.py                      # Entry point for python -m companion
│   ├── app.py                           # PyQt6 Main Window, toolbars, prompt bar, carousel, headless CLI
│   ├── canvas.py                        # Interactive high-res canvas, brush, auto-fit, hold-to-compare (\)
│   ├── inpainting_engine.py             # Dual Engine (EngineMode.FIREFLY & FAST) + multi-seed variations
│   └── utils/
│       ├── __init__.py                  # Utils package marker
│       └── blending.py                  # High-fidelity photographic pipeline (crop, noise, grain, sigmoid)
│
└── tests/                               # Comprehensive Automated Test Suite (38 Tests)
    ├── test_engine.py                   # Inpainting benchmarks, bounds, 24MP & 16-bit TIFF tests (10 tests)
    ├── test_gui.py                      # Headless PyQt6 canvas, tools, zoom, and signals (7 tests)
    ├── test_plugin.py                   # Lua static AST analysis & Lightroom SDK mock runner (11 tests)
    ├── test_e2e.py                      # End-to-end CLI headless pipeline verification (3 tests)
    └── test_firefly_pipeline.py         # Firefly 3 variations, prompt, sensor grain, 24MP, metadata (7 tests)
```

---

## 4. Operational Specifications

### Installation & Environment Setup
```bash
# Windows (PowerShell / Command Prompt)
.\install.bat

# macOS / Linux
chmod +x install.sh && ./install.sh
```

### Standalone Companion Run
```bash
# Using virtualenv
.venv\Scripts\python -m companion.app --input path/to/sample.tif --output path/to/sample_edited.tif

# Or without arguments for GUI file picker mode
.venv\Scripts\python -m companion.app

# With prompt guidance in headless automated mode
.venv\Scripts\python -m companion.app --headless --input photo.tif --output out.tif --prompt "clear sky" --test-mask-rect "100,100,200,200"
```

### Lightroom Classic Integration Options
1. **Plug-in Extras Menu:** Select photo > **File > Plug-in Extras > 🪄 AI Generative Eraser...**
2. **Right-Click Context Menu:** Right-click photo in Library Grid > **Plug-in Extras > 🪄 AI Generative Eraser...**
3. **Instant External Editor (Ctrl + Alt + E):**
   - Open **Edit > Preferences > External Editing**.
   - Under *Additional External Editor*, select `companion.bat` (or `companion.sh` on macOS).
   - Format: **TIFF**, Color Space: **ProPhoto RGB**, Bit Depth: **16 bits/component**.
   - Save preset as **Efface Magique**.
   - Press **Ctrl + Alt + E** on any photo to jump directly into the AI eraser!

---

## 5. Coding & Quality Standards

- **Zero Artifact Bleed**: Never blur or resample untouched photo regions. Inpainting strictly blends back into the full-res source image with bit-exact 0-difference outside the context crop.
- **Sensor Grain Matching**: Never leave plastic, overly-smooth patches. Synthesize and match the local camera sensor noise profile ($\sigma_r, \sigma_g, \sigma_b$) and Laplacian variance.
- **Non-blocking UI**: All inpainting tasks run on background `QThread` workers with continuous progress updates.
- **Color Profile & Metadata Preservation**: Preserve 16-bit ProPhoto RGB ICC profile bytes, EXIF metadata tags, and resolution DPI across export and save roundtrips.

---

## 6. Testing Strategy & Quality Gate

The project enforces an automated five-tier testing pyramid to guarantee reliability and prevent regressions:

### Test Suite Architecture (38 Automated Tests)
1. **Engine Benchmarks (`tests/test_engine.py` - 10 Tests)**:
   - Synthetic 24MP (6000x4000) and 16-bit TIFF test patterns.
   - Strict edge-case boundaries (empty masks, full canvas, border pixels, multi-island masks, 1-pixel micro-masks).
   - Numerical 0-diff bit-exact integrity on untouched regions.
   - Hardware detection & CPU/CUDA fallback.
2. **Headless GUI Logic (`tests/test_gui.py` - 7 Tests)**:
   - Screen-to-scene coordinate mapping under 25%, 100%, and 400% zoom with pan offsets.
   - Undo / Redo history state consistency across 5+ sequential operations.
   - Brush radius hotkey (`[` and `]`) clamping logic.
   - Eraser tool clearing behavior and Stroke Finished signal emission.
   - Fit-to-screen and automatic viewport sizing.
   - Headless Save & Exit lifecycle and exit codes.
3. **Lua Plugin Static Analysis & Mocking (`tests/test_plugin.py` - 11 Tests)**:
   - Complete AST syntax compilation of all `.lua` scripts via ANTLR4 parser.
   - Lightroom Classic SDK workflow mocking (missing selections, cancelled dialogs, export failures, exit code 130 handling).
4. **End-to-End Pipeline (`tests/test_e2e.py` - 3 Tests)**:
   - Command-line subprocess execution with `--headless` and `--test-mask-rect`.
   - File validity, dimension preservation, and compression integrity.
5. **Firefly Generative Pipeline (`tests/test_firefly_pipeline.py` - 7 Tests)**:
   - 3-variation batch inference with distinct random seeds.
   - Dual engine mode switching (`EngineMode.FIREFLY` vs `EngineMode.FAST`).
   - Prompt-conditioned synthesis vs context-aware unguided removal.
   - Context-aware crop margin calculation enforcing $\ge 35\%$ margin.
   - Camera sensor noise profile estimation and ISO grain synthesis.
   - 24MP full-resolution bit-exact untouched preservation.
   - 16-bit TIFF ICC profile and EXIF tag preservation.

### Execution Commands
```bash
# Run full 38-test suite with code coverage report
python run_tests.py

# Run specific test suites
python run_tests.py firefly
python run_tests.py engine
python run_tests.py gui
python run_tests.py plugin
python run_tests.py e2e

# Run without code coverage for ultra-fast execution
python run_tests.py --no-cov

# Direct pytest invocation
pytest tests/ -v --cov=companion
```
