# tools/finetune: offline LaMa fine-tune

Fine-tune big-lama on your own photos, export it to TorchScript and adopt it only if the eval says ADOPT.
Nothing here is imported by the app (`companion/`), and the base `requirements.txt` is untouched. The
user-facing walk-through is in the main README, section "Fine-tune on your own photos".

| Script | Does | Needs |
| --- | --- | --- |
| `python -m tools.finetune.curate IN OUT` | drop unreadable / tiny / blurry / clipped / near-duplicate photos, split 80/10/10, `manifest.json` | numpy, opencv, Pillow |
| `python -m tools.finetune.masks OUT` | preview the seeded masks (`generate_mask(h, w, seed, regime)`) | numpy, Pillow |
| `python -m tools.finetune.train ... --dry-run` | print the upstream command and Hydra overrides | nothing (no torch) |
| `python -m tools.finetune.train ...` | prepare data, init checkpoint, run upstream `bin/train.py` | training venv, CUDA GPU |
| `python -m tools.finetune.export CKPT OUT` | TorchScript `model.pt` + `meta.json`, verified by reload + forward pass | training venv |
| `python -m tools.finetune.eval ...` | PSNR / SSIM / LPIPS in-mask and whole image, report + contact sheet + verdict | torch, lpips (CPU is fine) |

Every flag and its default is listed by `--help`.

## Pinned upstream (one-time setup, the only network use)

| What | Pin |
| --- | --- |
| Trainer | https://github.com/advimman/lama at commit `786f5936b27fb3dacd2b1ad799e4de968ea697e7` |
| Pretrained checkpoint | https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip (381 428 720 bytes) |
| Checkpoint SHA-256 | `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6` |

```bash
git clone https://github.com/advimman/lama lama && git -C lama checkout 786f5936b27fb3dacd2b1ad799e4de968ea697e7
curl -L -o big-lama.zip https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip
sha256sum big-lama.zip        # Windows: certutil -hashfile big-lama.zip SHA256 ; must match the pin above
unzip big-lama.zip            # -> big-lama/config.yaml, big-lama/models/best.ckpt

# Training venv (upstream targets Python 3.8; use the CUDA build of torch 1.8 that matches your driver)
python3.8 -m venv .venv-finetune && . .venv-finetune/bin/activate
pip install -r tools/finetune/requirements.txt

# Weights upstream loads during training (train.py sets TORCH_HOME to the lama folder)
mkdir -p lama/ade20k/ade20k-resnet50dilated-ppm lama/hub/checkpoints
curl -L -o lama/ade20k/ade20k-resnet50dilated-ppm/encoder_epoch_20.pth \
  http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm/encoder_epoch_20.pth
curl -L -o lama/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth \
  https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth

# LPIPS: the linear layers ship inside the lpips wheel; this caches the AlexNet backbone once.
# Use the same TORCH_HOME you will run eval with. eval refuses to run (instead of downloading) if it is missing.
python -c "import lpips; lpips.LPIPS(net='alex')"
```

After this, nothing in `tools/finetune` uses the network, and there is no telemetry. Your photos never leave
the machine. (Upstream's TensorBoard logs are local files under `WORK/<name>/tb_logs`.)

## Notes per script

- **curate** reads only; originals keep their bytes and mtime. It refuses an OUT_DIR that overlaps INPUT_DIR or is
  not empty. Checks run in order unreadable, tiny, blurry, clipped highlights, clipped shadows, near-duplicate
  (against images already kept, in sorted path order), so duplicates are gone before the split. Split is by image,
  seeded, 80/10/10. 16-bit TIFFs become 8-bit sRGB PNGs in OUT_DIR (embedded ICC profile honoured when Pillow can
  read it). Warns below 200 kept images.
- **masks**: `normal` regime covers 5-40 % with strokes, boxes and blobs; `small` covers 0.2-3 % (spots, wires).
  Masks are 8-bit PNG, 255 = hole, named `<stem>_mask000.png`, which is what upstream's val/visual_test loader reads.
  Upstream generates its own training masks on the fly; train.py uses `generate_mask` for the val masks, eval for the test masks.
- **train** starts from big-lama `best.ckpt` restarted at step 0 with the low LR (`WORK/<name>/init.ckpt`), so
  resuming does not restore big-lama's original LR or epoch count. Defaults: LR 1e-5 (discriminator 1e-6), 3 epochs,
  batch 4, 256 px crops, checkpoint every 500 steps (clamped to one epoch). Re-run the same command to resume from
  `WORK/<name>/models/last.ckpt`. Fails early without CUDA or with less than `--min-vram-gb` (default 12).
- **export** refuses an OUT_DIR that resolves to the base `big-lama.pt` (default: the torch hub cache path the app
  uses) or is not empty. It traces to a temp folder next to OUT_DIR, reloads with `torch.jit.load`, runs one forward
  pass at a different size, writes `meta.json` and only then renames the folder into place.
- **eval** uses `generate_mask(h, w, seed + i, regime)` for image i (sorted by name), so every run sees the same masks.

## Eval verdict rule

ADOPT only if all three hold, else KEEP BASELINE (thresholds are flags):

1. mean in-mask LPIPS improves by at least **2 %** relative (`--min-lpips-gain 0.02`),
2. mean in-mask PSNR drops by at most **0.2 dB** (`--max-psnr-drop 0.2`),
3. the candidate has the lower in-mask LPIPS on **more than half** of the held-out images.

## meta.json

`name` (str), `date` (ISO str), `base_ckpt` (str), `manifest_hash` (SHA-256 of the curate manifest.json), `steps`
(int), `lr` (number), `metrics` (object; paste eval's `summary` here if you like). The app requires all of these keys.
