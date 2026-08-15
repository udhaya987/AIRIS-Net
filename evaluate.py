import os
import sys
import time
from pathlib import Path
import argparse
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import torch

from utils.image_utils import load_image, save_image, to_tensor
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips
from data.degradation import RandomDegradationPipeline
from inference import AIRISPredictor


def get_model_size_mb(checkpoint_path: str) -> float:
    """
    Get file size of model checkpoint in MB.
    """
    p = Path(checkpoint_path)
    if p.exists():
        return p.stat().st_size / (1024.0 * 1024.0)
    return 0.0


def evaluate_dataset(
    clean_folder: str = "data/test/clean",
    degraded_folder: Optional[str] = None,
    checkpoint_path: str = "checkpoints/best_airis.pth",
    output_csv: str = "results/metrics.csv",
    max_images: Optional[int] = 50,
    degradation_type: str = "random",
    device: Optional[str] = None,
    scale: Optional[int] = 1
) -> pd.DataFrame:
    """
    Evaluate AIRIS-Net restoration on a dataset.
    Computes PSNR, SSIM, LPIPS, per-image inference time in ms, parameter count, and size in MB.
    Saves results to results/metrics.csv.
    """
    clean_p = Path(clean_folder)
    extensions = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
    files = []
    if clean_p.exists():
        for ext in extensions:
            files.extend(list(clean_p.glob(ext)))
    files = sorted(list(set(files)))
    files = [f for f in files if not f.name.startswith("._") and "__MACOSX" not in str(f) and not f.name.startswith(".")]

    if not files:
        raise FileNotFoundError(f"No valid clean images found in {clean_p.resolve()}")

    if max_images and len(files) > max_images:
        files = files[:max_images]

    # Model predictor
    predictor = AIRISPredictor(checkpoint_path=checkpoint_path, device=device)
    param_count = sum(p.numel() for p in predictor.model.parameters())
    model_size_mb = get_model_size_mb(checkpoint_path)

    print("=" * 65)
    print("AIRIS-Net Evaluation Suite")
    print(f"Clean Images Folder:  {clean_p.resolve()}")
    print(f"Total Test Images:    {len(files)}")
    print(f"Checkpoint:           {Path(checkpoint_path).resolve()} ({model_size_mb:.2f} MB)")
    print(f"Trainable Parameters: {param_count:,}")
    print(f"Device:               {predictor.device}")
    print(f"Degradation Mode:     {degradation_type}")
    print("=" * 65)

    pipeline = RandomDegradationPipeline(seed=42)
    rows = []
    has_lpips = True

    for idx, f in enumerate(files):
        clean_img = load_image(f, grayscale=True)

        if degraded_folder and Path(degraded_folder).exists():
            deg_file = Path(degraded_folder) / f.name
            if deg_file.exists():
                degraded_img = load_image(deg_file, grayscale=True)
            else:
                degraded_img, _ = pipeline(clean_img, degradation_type=degradation_type)
        else:
            degraded_img, _ = pipeline(clean_img, degradation_type=degradation_type)

        # Run inference and measure time
        t0 = time.perf_counter()
        res = predictor.predict(degraded_img)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        restored_img = res["restored"]

        # Calculate metrics
        psnr_res = calculate_psnr(restored_img, clean_img)
        ssim_res = calculate_ssim(restored_img, clean_img)
        lpips_res = calculate_lpips(restored_img, clean_img, device=predictor.device)

        if lpips_res is None:
            has_lpips = False

        row = {
            "filename": f.name,
            "psnr": round(psnr_res, 3),
            "ssim": round(ssim_res, 4),
            "lpips": round(lpips_res, 4) if lpips_res is not None else "N/A",
            "inference_time_ms": round(t_elapsed_ms, 2)
        }
        rows.append(row)

        lpips_str = f", LPIPS: {lpips_res:.4f}" if lpips_res is not None else ""
        if (idx + 1) % 10 == 0 or (idx + 1) == len(files):
            print(
                f"  [{idx+1:3d}/{len(files):3d}] {f.name} | "
                f"PSNR: {psnr_res:.2f} dB | SSIM: {ssim_res:.4f}{lpips_str} | Time: {t_elapsed_ms:.1f} ms"
            )

    df = pd.DataFrame(rows)

    # Save to output CSV
    out_p = Path(output_csv)
    if not out_p.is_absolute():
        out_p = Path(__file__).resolve().parent / output_csv
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_p), index=False)

    # Compute averages
    avg_psnr = df["psnr"].mean()
    avg_ssim = df["ssim"].mean()
    avg_time = df["inference_time_ms"].mean()
    avg_lpips_val = None
    if has_lpips:
        numeric_lpips = pd.to_numeric(df["lpips"], errors="coerce")
        if not numeric_lpips.isna().all():
            avg_lpips_val = numeric_lpips.mean()

    print("\n" + "=" * 65)
    print("DATASET EVALUATION SUMMARY:")
    print(f"  Total Images Evaluated:  {len(df)}")
    print(f"  Model Parameters:        {param_count:,}")
    print(f"  Model Checkpoint Size:   {model_size_mb:.2f} MB")
    print(f"  Average PSNR:            {avg_psnr:.2f} dB")
    print(f"  Average SSIM:            {avg_ssim:.4f}")
    if avg_lpips_val is not None:
        print(f"  Average LPIPS:           {avg_lpips_val:.4f}")
    print(f"  Average Inference Latency:{avg_time:.2f} ms")
    print(f"  Saved Metrics to:        {out_p.resolve()}")
    print("=" * 65)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AIRIS-Net Image Restoration on Paired Dataset"
    )
    parser.add_argument("--folder", type=str, default="data/test/clean", help="Folder of clean ground truth images")
    parser.add_argument("--degraded_folder", type=str, default=None, help="Optional folder of paired degraded images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_airis.pth", help="Path to trained AIRIS checkpoint")
    parser.add_argument("--output", type=str, default="results/metrics.csv", help="Path to output metrics CSV file")
    parser.add_argument("--max_images", type=int, default=50, help="Max images to evaluate")
    parser.add_argument("--degradation", type=str, default="random", help="Degradation type for synthetic evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu')")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor")
    args = parser.parse_args()

    evaluate_dataset(
        clean_folder=args.folder,
        degraded_folder=args.degraded_folder,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        max_images=args.max_images,
        degradation_type=args.degradation,
        device=args.device,
        scale=args.scale
    )


if __name__ == "__main__":
    main()
