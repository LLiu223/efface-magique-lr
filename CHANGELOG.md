# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 16-bit TIFF round trip: when the input is a 16-bit RGB TIFF saved as TIFF, pixels the edit did not change keep full 16-bit precision. The erased area is computed in 8-bit and merged back, and the preview is 8-bit. A stdlib deflate TIFF writer (`companion/utils/tiff16.py`) keeps the ICC profile and DPI and writes through a temp file plus `os.replace`. (006)
- GitHub Actions CI (`.github/workflows/ci.yml`): `ruff check .` and headless pytest on Ubuntu with CPU-only torch and Lua 5.1. (007)
- `ruff.toml` with a conservative rule set (`E4`, `E7`, `E9`, `F`). (007)
- `tests/lua/test_json_util.lua`, which runs the plugin JSON helpers under a real Lua interpreter. (007)

### Security

- Live bridge: removed the CORS headers. Any request with an `Origin` header, or with a `Host` other than `127.0.0.1:<port>`, is rejected with 403, which blocks cross-site requests and DNS rebinding. POST bodies over 64 KiB get 413, and an invalid `Content-Length` gets 400. (001)
- Plugin: JSON request bodies built in Lua now escape their string fields (paths, photo IDs, titles). (002)

### Fixed

- Pending-import responses are sent as raw UTF-8 (`ensure_ascii=False`), and the Lua parser handles non-ASCII paths and `]`, `{`, `}` inside paths. (001)
- The live export cache is capped at 5 `live_*.tif` files per session and reused only when the photo's `lastEditTime` is unchanged. (002)
- Import polling is throttled to every 10th loop tick (about 300 ms) instead of every 30 ms. (002)
- Imports are guarded with `LrTasks.pcall`. The temp file is deleted only after a successful `addPhoto`, and `/api/import_done` is sent even on failure, so an item is not retried forever. (002)
- Each photo gets its own output path, so switching photos no longer saves over the previous photo's file. (003)
- A live selection that arrives during inpainting is deferred until the worker finishes. (003)
- Switching photos with unsynced layers asks before discarding them. The prompt never stacks, and only the newest selection is applied. (003)
- Re-selecting the photo that is already loaded is skipped, so its layers are kept. (003)
- Fast mode raises an error when the LaMa model failed to load instead of silently returning the unchanged image. (004)
- `InpaintingEngine.load_model` holds a lock, so the pre-warm thread and the first erase never load the model twice. (004)
- `gemini.md` box border alignment. (005)
- Unused imports dropped, and tests that need the real LaMa model skip when it is not installed. (007)

### Removed

- Dead diffusion code path, plus the `inpainting_engine.py` and `utils/blending.py` shims. (004)
- `.coverage` is no longer tracked. (005)
- Dead `_on_save_and_exit` handler, and the worker cancel/terminate in `closeEvent`. (003)

### Changed

- The "Generative AI" mode is described as what it is: LaMa inpainting with slight seeded variations, where the optional prompt only adds a small colour tint. (004)
- Documented the 8-bit processing limit. (005)
- The plugin JSON helpers moved from `LiveBridge.lua` to `JsonUtil.lua`, which has no Lightroom SDK dependency. (007)
- Pillow pinned to `>=10,<13`. (004)
