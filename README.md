# Old Photo Restoration

A Streamlit app for repairing damaged scans of old family photos. You mark the
damaged areas yourself; the app removes them, restores faces, and can
optionally correct colour, sharpen and upscale.

> **Status:** working prototype, tested on a small set of real family photos.
> It is built for *damaged* prints (stains, mould, scratches), not as a general
> image upscaler.

## What it does

| Step | Tool | Notes |
|------|------|-------|
| 1. Remove damage | [LaMa](https://github.com/advimman/lama) via `simple-lama-inpainting` | Only the areas you paint are regenerated |
| 2. Restore faces | [CodeFormer](https://github.com/sczhou/CodeFormer) | Fidelity slider controls how close faces stay to the original |
| 3. Colour correction (optional) | White-patch method, damaged pixels excluded | Strength slider, **off by default** |
| 4. Upscale (optional) | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 1x (off), 1.5x, 2x, 4x; **off by default** |
| 5. Clarity (optional) | Unsharp mask on brightness only | Colours are not touched |

A second **Clean photo** mode skips the damage and face steps and only applies
light sharpening, local contrast and optional upscaling.

## Results

<!-- TODO: add 2-3 before/after pairs you have permission to share, e.g.
![Example 1](examples/example1.jpg)
Put the files in an `examples/` folder (they are un-ignored in .gitignore). -->

## Settings that matter

- **Face fidelity** (0.3 to 1.0): higher values keep faces closer to the
  original. On the portraits I tried, higher values (roughly 0.85 and above)
  kept features more faithful than the 0.7 default.
- **Colour strength**: 0 means no colour change. Raise it only for photos with
  a clear colour cast.
- **Upscale**: leave at 1 unless you need more pixels, and compare faces
  against the original when you do.
- **Brush size**: mark only the real damage. Everything you paint is
  regenerated, so large brushes on undamaged areas change the picture.

## Known limitations (found by testing)

- **Small or heavily degraded faces can change.** Face models invent plausible
  detail when little real information is left, so eyes, eyebrows or lips can
  come out different. Keep fidelity high and always compare with the original.
- **Colour correction can make photos worse.** It treats the brightest pixels
  as white. On a photo with a bright teal wall it pulled the wall's colour out;
  on another it was needed. That is why it is off by default.
- **Upscaling can alter content.** At 4x, Real-ESRGAN changed a face in one
  test and added texture to smooth areas. At 1.5x it lightened a flat wall in
  another.
- **Output may be a few pixels larger than the input.** LaMa pads images to a
  multiple of 8 and the result is not cropped back yet.
- **Sharpening cannot add detail that was never scanned.** Washed-out or
  low-resolution scans will stay that way; a better scan helps more than any
  model.
- **Badly destroyed areas (for example lost emulsion) cannot be recovered.**

## Performance

On my laptop (RTX 2050, 4 GB VRAM) one 1500 px test photo took about 2 minutes
with the GPU. The same pipeline on CPU only took about 15 minutes on a
12-thread CPU, so a free 2-core server would likely be slower. For that reason
there is no hosted live demo.

## Setup

Tested on Windows with Python 3.10.

```powershell
git clone https://github.com/tsananya/old-photo-restoration.git
cd old-photo-restoration

conda create -n restore python=3.10 -y
conda activate restore

# PyTorch: pick the build for your machine.
# Tested: torch 2.1.2 / torchvision 0.16.2 (CPU) and torch 2.5.1 (CUDA 12.1).
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# Third-party repos, cloned next to app.py (not included in this repo)
git clone https://github.com/sczhou/CodeFormer.git
git clone https://github.com/xinntao/Real-ESRGAN.git
```

**Model weights** (not included, too large for git):

```powershell
cd CodeFormer
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py CodeFormer
cd ..

mkdir Real-ESRGAN/weights -Force
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"
```

**Two fixes I needed:**

1. CodeFormer's bundled `basicsr` expects a `version.py`. Create
   `CodeFormer/basicsr/version.py` containing:

   ```python
   __gitsha__ = "unknown"
   __version__ = "1.3.2"
   ```

2. With torchvision 0.17 or newer, `basicsr` fails to import
   `torchvision.transforms.functional_tensor`. In the installed package's
   `basicsr/data/degradations.py`, change it to
   `torchvision.transforms.functional`. (Not needed with torchvision 0.16.2.)

## Run

From the folder that contains `app.py`, `CodeFormer/` and `Real-ESRGAN/`:

```powershell
streamlit run app.py
```

The app calls CodeFormer and Real-ESRGAN with `python`, so activate your
environment first. Working files are written to `app_temp/`.

## Credits

Built on LaMa, CodeFormer and Real-ESRGAN. Check each project's licence before
any commercial use.
