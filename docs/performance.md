# Performance

Measured with `tools/benchmark.py`. It times each stage of `InpaintingEngine.generate_variations` on synthetic images (random RGB noise, seed 1234, one 300x200 ellipse mask at 60% / 40% of the frame), then the 8-bit and 16-bit TIFF saves the app does on **Save & Sync**. Output files go to a temp dir and are deleted after each size.

```
python tools/benchmark.py --stub                 # stub engine, no LaMa weights needed
python tools/benchmark.py                        # real LaMa, needs the cached big-lama.pt
python tools/benchmark.py --stub --sizes 12,24   # limit sizes; --mode fast|firefly (default firefly)
```

Read these numbers with the following in mind:
- **Stub rows are not LaMa timings.** `--stub` replaces the LaMa network with `cv2.inpaint` and is labelled `STUB` in every row. Every other stage runs the real code.
- **Peak memory is a lower bound.** tracemalloc sees only Python and numpy allocations. Pillow, OpenCV and torch native buffers, including the full-size PIL images and all GPU memory, are not counted. Inputs are created before tracing starts, so the peak counts pipeline work only.
- One variation, Generative AI (firefly) mode, `detect_subject=False`, `enable_grain=False`. The app asks for 3 variations in this mode, and each extra one repeats inpaint, blend and paste.
- Saves use random noise, which deflate cannot compress. Real photos compress better, so expect the save rows to be pessimistic.
- Single runs, not averaged. Expect a few percent of variation between runs.

## Machine

- CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD
- OS: Windows-10-10.0.26200-SP0 (Windows 11 Home)
- Python: 3.11.9
- numpy 1.26.4, opencv 4.11.0, torch 2.6.0+cu124, Pillow 9.5.0
- Device: cuda (NVIDIA GeForce RTX 3070)

## Stub engine (`python tools/benchmark.py --stub`), after the fixes below

| engine | size | stage | ms |
|---|---|---|---:|
| STUB | 12 MP (4240x2832) | dilate | 70.3 |
| STUB | 12 MP (4240x2832) | crop | 25.2 |
| STUB | 12 MP (4240x2832) | noise_profile | 31.8 |
| STUB | 12 MP (4240x2832) | convert_base | 0.0 |
| STUB | 12 MP (4240x2832) | inpaint | 183.1 |
| STUB | 12 MP (4240x2832) | blend | 366.6 |
| STUB | 12 MP (4240x2832) | paste | 8.7 |
| STUB | 12 MP (4240x2832) | save_tiff8 | 986.9 |
| STUB | 12 MP (4240x2832) | merge16 | 248.6 |
| STUB | 12 MP (4240x2832) | save_tiff16 | 2330.9 |
| STUB | 12 MP (4240x2832) | total | 4252.1 |
| STUB | 24 MP (6000x4000) | dilate | 143.0 |
| STUB | 24 MP (6000x4000) | crop | 51.8 |
| STUB | 24 MP (6000x4000) | noise_profile | 25.3 |
| STUB | 24 MP (6000x4000) | convert_base | 0.0 |
| STUB | 24 MP (6000x4000) | inpaint | 123.5 |
| STUB | 24 MP (6000x4000) | blend | 373.4 |
| STUB | 24 MP (6000x4000) | paste | 16.7 |
| STUB | 24 MP (6000x4000) | save_tiff8 | 1557.6 |
| STUB | 24 MP (6000x4000) | merge16 | 455.2 |
| STUB | 24 MP (6000x4000) | save_tiff16 | 4666.0 |
| STUB | 24 MP (6000x4000) | total | 7412.4 |
| STUB | 60 MP (9504x6336) | dilate | 348.1 |
| STUB | 60 MP (9504x6336) | crop | 127.4 |
| STUB | 60 MP (9504x6336) | noise_profile | 24.4 |
| STUB | 60 MP (9504x6336) | convert_base | 0.0 |
| STUB | 60 MP (9504x6336) | inpaint | 122.3 |
| STUB | 60 MP (9504x6336) | blend | 375.9 |
| STUB | 60 MP (9504x6336) | paste | 43.3 |
| STUB | 60 MP (9504x6336) | save_tiff8 | 3868.8 |
| STUB | 60 MP (9504x6336) | merge16 | 1101.8 |
| STUB | 60 MP (9504x6336) | save_tiff16 | 11779.7 |
| STUB | 60 MP (9504x6336) | total | 17791.6 |

| engine | size | peak tracemalloc MiB |
|---|---|---:|
| STUB | 12 MP (4240x2832) | 158.6 |
| STUB | 24 MP (6000x4000) | 207.7 |
| STUB | 60 MP (9504x6336) | 455.9 |

## Real LaMa (`python tools/benchmark.py`), after the fixes below

The weights (`big-lama.pt`) were cached on this machine, so real mode was measured on the GPU. The 12 and 24 MP `inpaint` rows look like warmup: the crop is the same size at every resolution, because the mask is the same, yet only the third call ran at steady-state speed (about 224 ms).

| engine | size | stage | ms |
|---|---|---|---:|
| LaMa | 12 MP (4240x2832) | dilate | 68.5 |
| LaMa | 12 MP (4240x2832) | crop | 24.1 |
| LaMa | 12 MP (4240x2832) | noise_profile | 32.5 |
| LaMa | 12 MP (4240x2832) | convert_base | 0.0 |
| LaMa | 12 MP (4240x2832) | inpaint | 2653.1 |
| LaMa | 12 MP (4240x2832) | blend | 371.2 |
| LaMa | 12 MP (4240x2832) | paste | 8.7 |
| LaMa | 12 MP (4240x2832) | save_tiff8 | 982.7 |
| LaMa | 12 MP (4240x2832) | merge16 | 234.7 |
| LaMa | 12 MP (4240x2832) | save_tiff16 | 2358.7 |
| LaMa | 12 MP (4240x2832) | total | 6734.2 |
| LaMa | 24 MP (6000x4000) | dilate | 140.7 |
| LaMa | 24 MP (6000x4000) | crop | 51.3 |
| LaMa | 24 MP (6000x4000) | noise_profile | 25.0 |
| LaMa | 24 MP (6000x4000) | convert_base | 0.0 |
| LaMa | 24 MP (6000x4000) | inpaint | 2108.7 |
| LaMa | 24 MP (6000x4000) | blend | 370.4 |
| LaMa | 24 MP (6000x4000) | paste | 16.4 |
| LaMa | 24 MP (6000x4000) | save_tiff8 | 1559.2 |
| LaMa | 24 MP (6000x4000) | merge16 | 451.7 |
| LaMa | 24 MP (6000x4000) | save_tiff16 | 4668.2 |
| LaMa | 24 MP (6000x4000) | total | 9391.6 |
| LaMa | 60 MP (9504x6336) | dilate | 434.0 |
| LaMa | 60 MP (9504x6336) | crop | 133.5 |
| LaMa | 60 MP (9504x6336) | noise_profile | 26.3 |
| LaMa | 60 MP (9504x6336) | convert_base | 0.0 |
| LaMa | 60 MP (9504x6336) | inpaint | 223.8 |
| LaMa | 60 MP (9504x6336) | blend | 374.0 |
| LaMa | 60 MP (9504x6336) | paste | 42.5 |
| LaMa | 60 MP (9504x6336) | save_tiff8 | 3947.8 |
| LaMa | 60 MP (9504x6336) | merge16 | 1306.7 |
| LaMa | 60 MP (9504x6336) | save_tiff16 | 11853.8 |
| LaMa | 60 MP (9504x6336) | total | 18342.4 |

| engine | size | peak tracemalloc MiB |
|---|---|---:|
| LaMa | 12 MP (4240x2832) | 158.6 |
| LaMa | 24 MP (6000x4000) | 207.7 |
| LaMa | 60 MP (9504x6336) | 456.0 |

## Hotspot fixes

Both fixes come from the stub profile, and both are checked old-vs-new with `np.array_equal` in `tests/test_benchmark_hotspots.py`. Before = `--stub` run on `main` at b38725e. After = the stub table above. Same machine, single runs.

| fix | stage | 12 MP before / after ms | 24 MP before / after ms | 60 MP before / after ms |
|---|---|---|---|---|
| `dilate_mask_for_contact_shadows`: dilate only the mask bbox grown by the kernel radius, not the whole frame; skip `convert("L")` on an L mask | dilate | 156.7 / 70.3 | 309.1 / 143.0 | 820.8 / 348.1 |
| `generate_variations`: use an RGB `image` as `base_rgb` instead of `image.convert("RGB")` (one fewer full-size copy, about 170 MB at 60 MP; each variation still pastes into its own `base_rgb.copy()`) | convert_base | 8.0 / 0.0 | 15.7 / 0.0 | 39.7 / 0.0 |

The 60 MP stub total dropped from 18381.8 ms to 17791.6 ms. Peak tracemalloc did not change (456.0 to 455.9 MiB), because both saved buffers are OpenCV/Pillow native memory that tracemalloc does not see.

Not fixed, for later: the saves dominate (deflate in `write_tiff16` and Pillow's `tiff_deflate`). Speeding them up means changing the compression level or format, which is out of scope here. The remaining `dilate` and `crop` time is full-frame mask decoding (`np.array(mask.convert("L"))`, run several times) and thresholding.
