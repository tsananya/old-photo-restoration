import cv2
import numpy as np
from PIL import Image
import subprocess
import os

# ============================================================
# STEP 1: LaMa Inpainting (removes marked damage)
# ============================================================
def inpaint_damage(image_path, mask_path, output_path, max_dim=1500):
    from simple_lama_inpainting import SimpleLama

    lama = SimpleLama()

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    scale = max_dim / max(image.size)
    if scale < 1:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
        mask = mask.resize(new_size, Image.NEAREST)

    result = lama(image, mask)
    result.save(output_path)
    print(f"[1/4] Inpainting done -> {output_path}")
    return output_path


# ============================================================
# STEP 2: CodeFormer Face Restoration
# (calls CodeFormer's script as a subprocess - avoids import conflicts)
# ============================================================
def restore_faces(input_folder, output_folder, fidelity=0.7, codeformer_dir="CodeFormer"):
    cmd = [
    "python", "inference_codeformer.py",
    "-w", str(fidelity),
    "-s", "1",
    "--input_path", input_folder,
    "-o", output_folder
    ]
    subprocess.run(cmd, cwd=codeformer_dir, check=True)
    print(f"[2/4] Face restoration done -> {codeformer_dir}/{output_folder}")


# ============================================================
# STEP 3: Color Correction (white-patch method)
# ============================================================
def correct_color(image_path, output_path, mask_path=None, strength=1.0):
    img = cv2.imread(image_path).astype(np.float32)
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Exclude damaged/masked areas from reference detection, if a mask is given
    valid_area = np.ones(gray.shape, dtype=bool)
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))
        valid_area = mask < 128  # only consider pixels NOT marked as damage

    candidate_gray = gray[valid_area]

    # Try white-patch method on valid (undamaged) pixels only
    threshold = np.percentile(candidate_gray, 95)
    ref_mask = (gray >= threshold) & valid_area

    b, g, r = cv2.split(img)

    # Safety check: need enough reference pixels for a reliable estimate
    if np.sum(ref_mask) > 50:
        avg_b, avg_g, avg_r = np.mean(b[ref_mask]), np.mean(g[ref_mask]), np.mean(r[ref_mask])
    else:
        # Fallback: gray-world on valid pixels only
        avg_b, avg_g, avg_r = np.mean(b[valid_area]), np.mean(g[valid_area]), np.mean(r[valid_area])

    max_avg = max(avg_b, avg_g, avg_r)

    # Clip correction strength to avoid extreme overcorrection (max 1.6x any channel)
    scale_b = np.clip(max_avg / avg_b, 0.7, 1.6)
    scale_g = np.clip(max_avg / avg_g, 0.7, 1.6)
    scale_r = np.clip(max_avg / avg_r, 0.7, 1.6)

    b = np.clip(b * scale_b, 0, 255)
    g = np.clip(g * scale_g, 0, 255)
    r = np.clip(r * scale_r, 0, 255)
    result = cv2.merge([b, g, r]).astype(np.uint8)

    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, bb])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    original = cv2.imread(image_path)
    result = cv2.addWeighted(result, strength, original, 1 - strength, 0)
    cv2.imwrite(output_path, result)
    print(f"[3/4] Color correction done -> {output_path}")
    return output_path


# ============================================================
# STEP 4: Upscale (Real-ESRGAN)
# ============================================================
def upscale_image(input_folder, output_folder, outscale=2, tile=200, esrgan_dir="Real-ESRGAN"):
    cmd = [
        "python", "inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", input_folder,
        "-o", output_folder,
        "--outscale", str(outscale),
        "--tile", str(tile)
    ]
    import torch                          # <- add
    if not torch.cuda.is_available():     # <- add
        cmd.append("--fp32")              # <- add
    subprocess.run(cmd, cwd=esrgan_dir, check=True)
    print(f"[4/4] Upscaling done -> {esrgan_dir}/{output_folder}")


if __name__ == "__main__":
    print("This module defines the restoration pipeline functions.")
    print("Import and call them from another script, or build the Streamlit app on top of this.")

def enhance_clarity(image_path, output_path, sharpen=0.8, contrast=0.0):
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # sharpen brightness only (colour untouched)
    blurred = cv2.GaussianBlur(l, (0, 0), 1.5)
    l_sharp = cv2.addWeighted(l, 1 + sharpen, blurred, -sharpen, 0)
    # gentle local contrast, blended in
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_ctr = clahe.apply(l_sharp)
    l_out = cv2.addWeighted(l_ctr, contrast, l_sharp, 1 - contrast, 0)
    out = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, out)
    return output_path