import os
import csv
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from typing import Optional

from utils.image_utils import load_image
from utils.metrics import calculate_psnr, calculate_ssim
from data.degradation import RandomDegradationPipeline
from baseline_swinir import SwinIRRestorer
from inference import AIRISPredictor


def evaluate_folder(
    clean_folder: str = "data/test/clean",
    model_type: str = "swinir",
    checkpoint_path: str = "checkpoints/best_airis.pth",
    output_csv: str = "results.csv",
    max_images: int = 50,
    degradation_type: str = "noise"
) -> pd.DataFrame:
    """
    Evaluate entire folder of images on synthetic degradation, compute PSNR/SSIM, and save results.csv.
    """
    clean_p = Path(clean_folder)
    extensions = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    files = []
    for ext in extensions:
        files.extend(list(clean_p.glob(ext)))
    files = sorted(list(set(files)))
    files = [f for f in files if not f.name.startswith("._") and "__MACOSX" not in str(f)]

    if not files:
        raise FileNotFoundError(f"No valid clean images found in {clean_p}")

    if max_images and len(files) > max_images:
        files = files[:max_images]

    print(f"\n[Evaluation] Evaluating {len(files)} images with model '{model_type}' on '{degradation_type}' degradation...")

    # Load model
    if model_type.lower() == "swinir":
        restorer = SwinIRRestorer()
        restore_fn = lambda img: restorer.restore(img)
    else:
        predictor = AIRISPredictor(checkpoint_path=checkpoint_path)
        restore_fn = lambda img: predictor.predict(img)["restored"]

    pipeline = RandomDegradationPipeline(seed=42)
    rows = []

    for idx, f in enumerate(files):
        clean_img = load_image(f, grayscale=True)
        degraded_img, meta = pipeline(clean_img, degradation_type=degradation_type)

        restored_img = restore_fn(degraded_img)

        psnr_deg = calculate_psnr(degraded_img, clean_img)
        psnr_res = calculate_psnr(restored_img, clean_img)
        ssim_deg = calculate_ssim(degraded_img, clean_img)
        ssim_res = calculate_ssim(restored_img, clean_img)

        rows.append({
            "image_name": f.name,
            "psnr_degraded": round(psnr_deg, 3),
            "psnr_restored": round(psnr_res, 3),
            "ssim_degraded": round(ssim_deg, 4),
            "ssim_restored": round(ssim_res, 4)
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(files):
            print(f"  Processed {idx + 1}/{len(files)}: PSNR {psnr_deg:.2f} -> {psnr_res:.2f} dB, SSIM {ssim_deg:.4f} -> {ssim_res:.4f}")

    df = pd.DataFrame(rows)
    out_p = Path(output_csv)
    if not out_p.is_absolute():
        out_p = Path.cwd() / output_csv

    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_p), index=False)
    
    # Also save to results/
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir_csv = results_dir / out_p.name
    df.to_csv(str(results_dir_csv), index=False)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY:")
    print(f"  Total Images:        {len(df)}")
    print(f"  Avg PSNR (Degraded): {df['psnr_degraded'].mean():.2f} dB")
    print(f"  Avg PSNR (Restored): {df['psnr_restored'].mean():.2f} dB (+{df['psnr_restored'].mean() - df['psnr_degraded'].mean():.2f} dB)")
    print(f"  Avg SSIM (Degraded): {df['ssim_degraded'].mean():.4f}")
    print(f"  Avg SSIM (Restored): {df['ssim_restored'].mean():.4f} (+{df['ssim_restored'].mean() - df['ssim_degraded'].mean():.4f})")
    print(f"  Saved results to:    {out_p.resolve()} and {results_dir_csv.resolve()}")
    print("=" * 50)

    return df


def evaluate_single_triplet(clean_path: str, degraded_path: str, restored_path: str):
    """
    Evaluate single triplet of clean, degraded, and restored images.
    """
    clean_img = load_image(clean_path, grayscale=True)
    degraded_img = load_image(degraded_path, grayscale=True)
    restored_img = load_image(restored_path, grayscale=True)

    psnr_deg = calculate_psnr(degraded_img, clean_img)
    psnr_res = calculate_psnr(restored_img, clean_img)
    ssim_deg = calculate_ssim(degraded_img, clean_img)
    ssim_res = calculate_ssim(restored_img, clean_img)

    print("\n" + "=" * 50)
    print("SINGLE IMAGE EVALUATION METRICS:")
    print(f"  PSNR (Degraded vs Clean): {psnr_deg:.2f} dB")
    print(f"  PSNR (Restored vs Clean): {psnr_res:.2f} dB")
    print(f"  SSIM (Degraded vs Clean): {ssim_deg:.4f}")
    print(f"  SSIM (Restored vs Clean): {ssim_res:.4f}")
    print("=" * 50)

    return {
        "psnr_degraded": psnr_deg,
        "psnr_restored": psnr_res,
        "ssim_degraded": ssim_deg,
        "ssim_restored": ssim_res
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Restoration Performance on a Dataset Folder or Single Images")
    parser.add_argument("--folder", type=str, default=None, help="Folder of clean ground truth images")
    parser.add_argument("--clean", type=str, default=None, help="Path to clean ground truth image")
    parser.add_argument("--degraded", type=str, default=None, help="Path to degraded image")
    parser.add_argument("--restored", type=str, default=None, help="Path to restored image")
    parser.add_argument("--model", type=str, default="swinir", choices=["swinir", "airis"], help="Model to evaluate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_airis.pth", help="AIRIS checkpoint path")
    parser.add_argument("--output", type=str, default="results.csv", help="Path to save results.csv")
    parser.add_argument("--max_images", type=int, default=20, help="Max images to evaluate")
    parser.add_argument("--degradation", type=str, default="noise", help="Degradation type: noise, blur, contrast, mixed, etc.")
    args = parser.parse_args()

    if args.clean and args.degraded and args.restored:
        evaluate_single_triplet(args.clean, args.degraded, args.restored)
    else:
        folder = args.folder or "data/test/clean"
        evaluate_folder(
            clean_folder=folder,
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            output_csv=args.output,
            max_images=args.max_images,
            degradation_type=args.degradation
        )


if __name__ == "__main__":
    main()
