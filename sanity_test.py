#!/usr/bin/env python3
"""
AIRIS-Net End-to-End Sanity Check.
Tests the full pipeline from raw input to restored output with metric computation.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airis.model import AIRISNet
from data.degradation import RandomDegradationPipeline
from utils.image_utils import load_image, save_image, to_tensor
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips


def run_sanity_check():
    print("=" * 60)
    print("AIRIS-Net Pipeline Sanity Check")
    print("=" * 60)

    # 1. Setup sample clean image
    out_dir = Path("outputs/sanity_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_candidates = list(Path("data/test/clean").glob("*.npy")) + list(Path("data/test/clean").glob("*.png"))
    if test_candidates:
        clean_img = load_image(test_candidates[0], grayscale=True)
        img_name = test_candidates[0].name
        print(f"[1/5] Loaded sample: {img_name} ({clean_img.shape[0]}x{clean_img.shape[1]})")
    else:
        x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
        clean_img = (np.sin(20 * x) * np.cos(20 * y) * 0.4 + 0.5).astype(np.float32)
        clean_img = np.clip(clean_img, 0.0, 1.0)
        img_name = "synthetic_pattern.png"
        print(f"[1/5] Generated sample pattern ({clean_img.shape[0]}x{clean_img.shape[1]})")

    # 2. Apply combined degradation (Gaussian + Speckle)
    pipeline = RandomDegradationPipeline(seed=42)
    degraded_img, meta = pipeline(clean_img, degradation_type="gaussian_speckle")
    psnr_deg = calculate_psnr(degraded_img, clean_img)
    ssim_deg = calculate_ssim(degraded_img, clean_img)
    print(f"[2/5] Applied degradation: {meta['type']} (PSNR: {psnr_deg:.2f} dB, SSIM: {ssim_deg:.4f})")

    # 3. Model inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chk_path = Path("checkpoints/best_airis.pth")
    model = AIRISNet(in_channels=1, base_channels=48, scale_factor=1).to(device)

    if chk_path.exists():
        from utils.checkpoint import load_checkpoint
        load_checkpoint(chk_path, model, device=device)
        print(f"[3/5] Model initialized with checkpoint ({chk_path.name}) on {device}")
    else:
        print(f"[3/5] Model initialized on {device}")

    model.eval()
    tensor_in = to_tensor(degraded_img, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor_in)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    restored_img = outputs["restored"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    mask_img = outputs["mask"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    rel_img = outputs["reliability"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    routing = outputs["routing_weights"].squeeze().cpu().numpy()

    print(f"      Inference completed in {elapsed_ms:.1f} ms | Weights: Local={routing[0]:.2f}, Global={routing[1]:.2f}, Freq={routing[2]:.2f}")

    # 4. Compute metrics
    psnr_res = calculate_psnr(restored_img, clean_img)
    ssim_res = calculate_ssim(restored_img, clean_img)
    lpips_res = calculate_lpips(restored_img, clean_img, device=device)
    lpips_txt = f", LPIPS: {lpips_res:.4f}" if lpips_res is not None else ""
    print(f"[4/5] Restored Quality: PSNR = {psnr_res:.2f} dB, SSIM = {ssim_res:.4f}{lpips_txt}")

    # 5. Save outputs
    save_image(clean_img, out_dir / "01_clean.png")
    save_image(degraded_img, out_dir / "02_degraded.png")
    save_image(restored_img, out_dir / "03_restored.png")
    save_image(mask_img, out_dir / "04_restoration_mask.png")
    save_image(rel_img, out_dir / "05_reliability_map.png")
    print(f"[5/5] Artifacts saved to {out_dir}/")

    # Assertions
    assert restored_img.shape == clean_img.shape, "Output dimension mismatch"
    assert 0.0 <= restored_img.min() and restored_img.max() <= 1.0, "Pixel values out of range"
    assert np.isclose(routing.sum(), 1.0, atol=1e-4), "Routing sum mismatch"

    print("=" * 60)
    print("Sanity check passed.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_sanity_check()
