# GEMINI.md — Project Blueprint: Efface Magique LR

## 1. Executive Summary & Architecture

**Efface Magique LR** is an open-source, local, private, and high-performance alternative to Adobe Lightroom's cloud-based Generative Remove / Magic Eraser. It integrates directly into Adobe Lightroom Classic via a native Lua SDK plugin (`.lrplugin`) and a high-performance local Python companion application built with PyQt6, PyTorch, and a dual-tier inpainting pipeline.

### End-to-End Workflow Architecture
```
+---------------------------------------------------------------------------------------------------+
|                                      ADOBE LIGHTROOM CLASSIC                                      |
|                                                                                                   |
| 1. User selects target photo in Library or Develop module.                                        |
| 2. User invokes: File > Plug-in Extras > "⚡ AI Generative Eraser" (Live Window).                 |
| 3. LiveBridge.lua:                                                                                |
|    - Observes photo selection, debounces exports, and communicates with Companion via IPC bridge. |
|    - Exports 16-bit TIFF with embedded color profile (kept in memory as 16-bit by companion).     |
|    - Sends selection event to local companion app via HTTP IPC API.                               |
| 4. On companion save & sync:                                                                      |
|    - Reads edited TIFF back into Lightroom Catalog using LrCatalog:addPhoto().                    |
|    - Auto-stacks the new photo adjacent to original photo and stays live.                         |
+-------------------------------------------------+-------------------------------------------------+
                                                  | Process invocation (CLI arguments)
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                    PYTHON COMPANION APP (PyQt6)                                   |
|                                                                                                   |
| - High-performance hardware-accelerated canvas supporting 24MP-60MP raw/tiff exports.             |
| - Dual Inpainting Engine:                                                                         |
|   * ✨ LaMa Variations: LaMa inpainting with slight seeded variations (3 candidates, distinct     |
|     seeds). The optional prompt only adds a deterministic colour tint of a few levels.            |
|   * ⚡ Fast Mode: Simple-LaMa GAN for instantaneous spot, dust, and wire removal.                 |
| - Variations Carousel: Clickable thumbnail cards at bottom to instantly preview & switch          |
|   between generated variations, plus "🔄 More Variations" button with seed re-roll.               |
| - High-Fidelity Photographic Blending Pipeline:                                                   |
|   1. Intelligent context-aware crop & padding with >= 35% margin (default 85%).                   |
|   2. Camera sensor noise profiling (Laplacian variance & per-channel residuals).                  |
|   3. Synthetic sensor ISO grain matching to eliminate plastic smoothing.                          |
|   4. Seamless sigmoid / outer-feather alpha blending (zero untouched pixel bleed).                |
| - Brush mask tools: Auto-Erase on stroke release, adjustable radius ([ and ]), eraser mode.       |
| - Hold-to-Compare: Hold \ (Backslash) or Spacebar for instant Before/After comparison.            |
| - Color & Metadata: ProPhoto RGB ICC profile and DPI tag preservation. 16-bit TIFFs: untouched    |
|   pixels keep full 16-bit precision; the erased area is computed in 8-bit and merged back;        |
|   preview is 8-bit.                                                                               |
| - "Save & Return to Lightroom": Writes active variation to a TIFF (16-bit for 16-bit input).      |
+---------------------------------------------------------------------------------------------------+
```

---

### Inpainting Models (registry)
The pipeline (crop, dilation, grain, feather blend, layers) is independent of the network that fills the hole.
That network is an `InpaintModel` (`companion/models/base.py`) looked up by name in
`companion/models/registry.py`; LaMa (`companion/models/lama.py`, name `"lama"`) is the first and default entry.
The **Model** dropdown next to the Fast/Generative dropdown lists every registered model (unavailable ones are
greyed out with a tooltip); the choice is saved in `QSettings` (`model/name`) and restored at startup (unknown or
unavailable names fall back to `lama`). `InpaintingEngine.set_model(name)` unloads the old model and frees device
memory under the load lock before the new one loads lazily, so only one model is ever resident in VRAM.
The dropdown is locked while an erase runs. The bridge accepts an optional `"model"` string in `POST /api/select`
(absent keeps the current choice; non-string, unknown or unavailable names get `400 {"error": ...}`).

| Name (`name`) | Weights | Speed | Prompt | Licence | Install |
|---|---|---|---|---|---|
| LaMa (`lama`, default) | ~200 MB (package cache) | fast | no | Apache-2.0 | `requirements.txt` |
| MI-GAN (`migan`) | 28 MB ONNX, `models/migan_pipeline_v2.onnx` | fastest, also on CPU | no | MIT (code and weights) | install.bat optional step + `requirements-migan.txt` |
| Stable Diffusion (`sd_inpaint`) | ~2.1 GB fp16 safetensors, `models/sd_inpaint/` | slow: seconds on GPU, minutes on CPU | yes (prompt field) | CreativeML OpenRAIL-M | install.bat optional step + `requirements-sd.txt` |

Optional weights live in the repo-root `models/` folder (`CONFIG.MODELS_DIR`, gitignored, never packaged).
install.bat asks y/N per model (default N) and installs only that model's pinned requirements file
(`requirements-migan.txt`: onnxruntime; `requirements-sd.txt`: diffusers, transformers, accelerate). If that pip
install fails, the model's weights are not downloaded. Downloads are pinned (HF revision), saved as `<file>.part`,
checked against their SHA256 and only then moved to the final name, so an interrupted download never shows up as
an installed model.
Models whose package or weights are missing stay greyed out in the Model dropdown with a tooltip saying what to install.
The prompt field next to the Model dropdown is enabled only for models with `uses_prompt` (Stable Diffusion).
A prompt the assistant sets with its `set_prompt` tool shows up in that field, and the field's text is the
starting prompt for the assistant.
Stable Diffusion runs with `local_files_only=True` and the diffusers safety checker disabled.

How to add a model:
```python
# companion/models/my_model.py, then add `from companion.models import my_model` to companion/models/__init__.py
import importlib.util
from PIL import Image
from companion.models.base import InpaintModel
from companion.models.registry import register

@register
class MyModel(InpaintModel):
    name = "my_model"          # stable id (settings, bridge)
    label = "My Model"         # dropdown text
    uses_prompt = False
    steps = None

    @classmethod
    def is_available(cls):     # cheap check only: never load weights here
        return importlib.util.find_spec("my_model_pkg") is not None

    def load(self, device, progress_callback=None):
        import my_model_pkg    # lazy import keeps startup fast
        self._net = my_model_pkg.load(device)

    def unload(self):
        self._net = None       # drop references; the engine then frees device memory

    def inpaint(self, crop, mask, seed, prompt):
        return self._net(crop, mask).convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
```

### Assistant (optional, `companion/agent/`)
Type a request such as "remove the person on the left" in the **Assistant** box in the top bar and press **Ask**.
The assistant finds and **selects** what you named; it never erases. A LangChain agent
(`langchain.agents.create_agent`) works only through five tools and never replaces the engine:

```
Assistant box -> AgentWorker (QThread, harness.py: max steps / token budget / timeout / cancel / allow-list / 1 retry)
              -> tools.py: list_models, select_model, set_prompt, detect_regions, select_regions
              -> registry.get + InpaintingEngine.set_model / get_segmentation_model / CLIPSeg (clipseg_detector)
              -> selected regions added to the red brush mask (one Ctrl+Z step) -> user clicks Erase as usual
```

- Install with `pip install -r requirements-agent.txt` (pinned versions; `requirements.txt` is unchanged). Without
  it the app runs as before and the assistant fields are disabled with the tooltip "install requirements-agent.txt".
- Providers: `ollama` (default, `http://localhost:11434`), `anthropic`, `openai`, `gemini`. Keys are read only from
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`; they are never written to settings, logs or the usage DB.
- `detect_regions(query)` finds any named object with the optional CLIPSeg detector; without it (or without a
  query) only the 20 VOC classes (person, car, bottle, dog, chair...) are found.
- The selection is added to any strokes you already painted, as one undo step (Ctrl+Z removes it). The status bar
  says what was selected, e.g. "Assistant selected: power lines (r1). Adjust with the brush, then click Erase."
  Then erase with any model, variations and options as usual. If the photo or layers changed while the assistant
  ran, or a Lightroom preview is shown, the selection is dropped and the status bar says so. **Stop** cancels.
  `select_model` and `set_prompt` still prepare the model and prompt your Erase will use (the Model dropdown
  and prompt field show them).
- **Usage** opens the usage log (SQLite in the app data folder: tokens, latency, estimated cost per call, CSV export,
  Clear) and the assistant settings (Ollama URL, token budget, opt-in "store prompt text").
- **Privacy:** images are never sent to an LLM in v1, nor are file names or EXIF; only your request text, the model
  list and region labels/positions. Ollama on localhost keeps that text on this computer (a non-localhost Ollama URL
  shows a warning). Anthropic, OpenAI and Gemini are cloud services: the text and region labels leave the device.

## 2. Technology Stack

- **Lightroom Classic Plugin**:
  - Lightroom Classic SDK (Lua 5.1 / SDK version 11.0+)
  - Lightroom APIs: `LrApplication`, `LrCatalog`, `LrTasks`, `LrExportSession`, `LrShell`, `LrPathUtils`, `LrFileUtils`, `LrDialogs`, `LrProgressScope`
- **Companion Application**:
  - Python 3.10+ (Tested on 3.11 / 3.12)
  - **GUI Toolkit**: PyQt6 (High-DPI aware, OpenGL viewport canvas, dark studio aesthetic)
  - **Dual Inference Engine**:
    * LaMa Variations mode (UI label, formerly "Generative AI"): LaMa inpainting with slight seeded variations (prompt only adds a few levels of colour tint)
    * Simple-LaMa (Fast Fourier Convolutions) for rapid spot healing
  - **GPU Hardware Acceleration & Optimizations**:
    * NVIDIA CUDA GPU acceleration with cuDNN auto-tuning (`torch.backends.cudnn.benchmark = True`)
    * TensorFloat-32 (`TF32`) computation on NVIDIA Ampere (RTX 3070) Tensor Cores
    * Direct in-VRAM GPU spatial frequency synthesis (PyTorch CUDA tensors) eliminating CPU bus transfer overhead
    * Pre-warmed CUDA kernels on initialization to eliminate first-stroke latency
    * Auto-scaling neural inference (`max_dim = 1024` with Lanczos high-res reconstruction): Prevents VRAM exhaustion and thread freezes when inpainting large objects on 24MP-60MP images
    * Automatic VRAM garbage collection (`torch.cuda.empty_cache()`) to leave VRAM available for Lightroom Classic
  - **Photographic Pipeline**:
    * Sensor noise profiling & ISO grain matching (`companion.pipeline`)
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
├── requirements-migan.txt               # Optional MI-GAN dependency (onnxruntime), pinned
├── requirements-sd.txt                  # Optional Stable Diffusion dependencies (diffusers, transformers, accelerate), pinned
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
│       ├── LiveBridge.lua               # Live Window: Persistent selection observer & seamless IPC sync
│       └── PluginUtils.lua              # Cross-platform path resolver & background process launcher
│
├── companion/                           # Python AI Companion Application
│   ├── __init__.py                      # Package descriptor
│   ├── __main__.py                      # Entry point for python -m companion
│   ├── app.py                           # PyQt6 Main Window, toolbars, prompt bar, carousel, headless CLI
│   ├── canvas.py                        # Interactive high-res canvas, brush, auto-fit, hold-to-compare (\)
│   ├── model_engine.py                  # Engine around the active registry model (EngineMode.FIREFLY & FAST) + seeded variations
│   ├── models/                          # InpaintModel ABC (base.py), registry.py, lama.py (default "lama"), migan.py, sd_inpaint.py
│   ├── agent/                           # Optional assistant: llm.py, tools.py, harness.py, usage.py, usage_dialog.py
│   ├── pipeline.py                      # Photographic pipeline (crop, noise, grain, sigmoid blending)
│   ├── live_bridge.py                   # High-performance local HTTP IPC bridge (port discovery, REST endpoints)
│   └── utils/
│       ├── __init__.py                  # Utils package marker
│       └── subject_detector.py          # Subject / object awareness and segmentation
│
└── tests/                               # Comprehensive Automated Test Suite (72 Tests)
    ├── test_engine.py                   # Inpainting benchmarks, bounds, 24MP & 16-bit TIFF tests (10 tests)
    ├── test_gui.py                      # Headless PyQt6 canvas, tools, zoom, and signals (17 tests)
    ├── test_plugin.py                   # Lua static AST analysis & Lightroom SDK mock runner (13 tests)
    ├── test_e2e.py                      # End-to-end CLI headless pipeline verification (3 tests)
    ├── test_firefly_pipeline.py         # Firefly 3 variations, prompt, sensor grain, 24MP, metadata (14 tests)
    ├── test_subject_detector.py         # Subject boundary and object awareness (4 tests)
    └── test_live_bridge.py              # Live Bridge REST API, signals, always-on-top, and sync (11 tests)
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

# With a prompt (only adds a slight deterministic colour tint) in headless automated mode
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
- **Color Profile & Metadata Preservation**: Preserve ProPhoto RGB ICC profile bytes (pixels are processed at 8 bits per channel), EXIF metadata tags, and resolution DPI across export and save roundtrips.

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
   - Prompt colour tint vs no-prompt removal.
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
