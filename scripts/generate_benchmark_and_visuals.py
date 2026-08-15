#!/usr/bin/env python3
"""
Generate measured test metrics, baseline comparisons, and visual comparison grid.
Evaluates trained AIRIS-Net checkpoint against Ground Truth and Degraded input.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airis.model import AIRISNet
from utils.checkpoint import load_checkpoint
from utils.image_utils import load_image, save_image, to_tensor
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips
from data.degradation import RandomDegradationPipeline


def run_benchmark(
    test_dir: str = "data/test/clean",
    checkpoint_path: str = "checkpoints/best_airis.pth",
    num_samples: int = 25,
    seed: int = 42
):
    print("=" * 65)
    print("Running AIRIS-Net Test Set Evaluation & Comparison Benchmark")
    print("=" * 65)

    test_path = Path(test_dir)
    files = sorted(list(test_path.glob("*.npy")) + list(test_path.glob("*.png")))
    if not files:
        print(f"Error: No images found in {test_dir}")
        return

    files = files[:num_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AIRISNet(in_channels=1, base_channels=48, scale_factor=1).to(device)

    chk_p = Path(checkpoint_path)
    if chk_p.exists():
        load_checkpoint(chk_p, model, device=device)
    model.eval()

    pipeline = RandomDegradationPipeline(seed=seed)
    degradation_types = [
        "gaussian",
        "speckle",
        "gaussian_speckle",
        "gaussian_downsample",
        "mixed"
    ]

    metrics_rows = []
    comparison_visuals = []

    for idx, f in enumerate(files):
        clean_img = load_image(f, grayscale=True)
        deg_type = degradation_types[idx % len(degradation_types)]
        degraded_img, meta = pipeline(clean_img, degradation_type=deg_type)

        # 1. Metrics: Degraded vs Clean
        deg_psnr = calculate_psnr(degraded_img, clean_img)
        deg_ssim = calculate_ssim(degraded_img, clean_img)
        deg_lpips = calculate_lpips(degraded_img, clean_img, device=device)

        # 2. Model Inference
        tensor_in = to_tensor(degraded_img, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tensor_in)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        restored_img = out["restored"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
        mask_img = out["mask"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
        rel_img = out["reliability"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
        routing = out["routing_weights"].squeeze().cpu().numpy()

        # 3. Metrics: Restored vs Clean
        res_psnr = calculate_psnr(restored_img, clean_img)
        res_ssim = calculate_ssim(restored_img, clean_img)
        res_lpips = calculate_lpips(restored_img, clean_img, device=device)

        psnr_delta = res_psnr - deg_psnr
        ssim_delta = res_ssim - deg_ssim

        row = {
            "filename": f.name,
            "degradation_type": meta["type"],
            "degraded_psnr": round(deg_psnr, 3),
            "airis_psnr": round(res_psnr, 3),
            "psnr_improvement": round(psnr_delta, 3),
            "degraded_ssim": round(deg_ssim, 4),
            "airis_ssim": round(res_ssim, 4),
            "ssim_improvement": round(ssim_delta, 4),
            "degraded_lpips": round(deg_lpips, 4) if deg_lpips is not None else "N/A",
            "airis_lpips": round(res_lpips, 4) if res_lpips is not None else "N/A",
            "latency_ms": round(latency_ms, 2),
            "routing_local": round(float(routing[0]), 3),
            "routing_global": round(float(routing[1]), 3),
            "routing_freq": round(float(routing[2]), 3)
        }
        metrics_rows.append(row)

        if idx < 5:
            comparison_visuals.append({
                "name": f.name,
                "deg_type": meta["type"],
                "clean": clean_img,
                "degraded": degraded_img,
                "restored": restored_img,
                "mask": mask_img,
                "reliability": rel_img,
                "deg_psnr": deg_psnr,
                "res_psnr": res_psnr,
                "deg_ssim": deg_ssim,
                "res_ssim": res_ssim
            })

        print(f"  [{idx+1:2d}/{len(files):2d}] {f.name} ({meta['type']}) | "
              f"Deg PSNR: {deg_psnr:.2f} -> AIRIS: {res_psnr:.2f} dB (d: {psnr_delta:+.2f}) | "
              f"SSIM: {deg_ssim:.4f} -> {res_ssim:.4f} (d: {ssim_delta:+.4f}) | {latency_ms:.1f} ms")

    # Save detailed per-image metrics
    df_metrics = pd.DataFrame(metrics_rows)
    out_csv = Path("results/final_test_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved final test metrics to {out_csv}")

    # Generate Baseline Comparison Summary
    avg_deg_psnr = df_metrics["degraded_psnr"].mean()
    avg_airis_psnr = df_metrics["airis_psnr"].mean()
    avg_psnr_gain = df_metrics["psnr_improvement"].mean()

    avg_deg_ssim = df_metrics["degraded_ssim"].mean()
    avg_airis_ssim = df_metrics["airis_ssim"].mean()
    avg_ssim_gain = df_metrics["ssim_improvement"].mean()

    numeric_deg_lpips = pd.to_numeric(df_metrics["degraded_lpips"], errors="coerce")
    numeric_airis_lpips = pd.to_numeric(df_metrics["airis_lpips"], errors="coerce")
    avg_deg_lpips = numeric_deg_lpips.mean() if not numeric_deg_lpips.isna().all() else None
    avg_airis_lpips = numeric_airis_lpips.mean() if not numeric_airis_lpips.isna().all() else None
    avg_lpips_gain = (avg_deg_lpips - avg_airis_lpips) if (avg_deg_lpips and avg_airis_lpips) else None

    baseline_rows = [
        {
            "Method / Model": "Degraded Input (Baseline)",
            "Evaluation Type": "Measured on Test Split",
            "Avg PSNR (dB)": round(avg_deg_psnr, 2),
            "Avg SSIM": round(avg_deg_ssim, 4),
            "Avg LPIPS": round(avg_deg_lpips, 4) if avg_deg_lpips is not None else "N/A",
            "Latency (ms)": "0.0",
            "Status": "Baseline Reference"
        },
        {
            "Method / Model": "AIRIS-Net (Trained Checkpoint)",
            "Evaluation Type": "Measured on Test Split",
            "Avg PSNR (dB)": round(avg_airis_psnr, 2),
            "Avg SSIM": round(avg_airis_ssim, 4),
            "Avg LPIPS": round(avg_airis_lpips, 4) if avg_airis_lpips is not None else "N/A",
            "Latency (ms)": round(df_metrics["latency_ms"].mean(), 2),
            "Status": f"+{avg_psnr_gain:.2f} dB PSNR, +{avg_ssim_gain:.4f} SSIM"
        }
    ]
    df_baseline = pd.DataFrame(baseline_rows)
    baseline_csv = Path("results/baseline_comparison.csv")
    df_baseline.to_csv(baseline_csv, index=False)
    print(f"[OK] Saved baseline comparison table to {baseline_csv}")

    # Generate sample_results/comparison.png visual collage
    fig, axes = plt.subplots(nrows=len(comparison_visuals), ncols=5, figsize=(18, 3.6 * len(comparison_visuals)))
    columns = ["1. Clean Ground Truth", "2. Degraded Input", "3. AIRIS-Net Restored", "4. Restoration Mask (M)", "5. Reliability Map (R)"]

    for col_idx, col_name in enumerate(columns):
        axes[0, col_idx].set_title(col_name, fontsize=12, fontweight="bold", pad=10)

    for row_idx, item in enumerate(comparison_visuals):
        # Clean
        axes[row_idx, 0].imshow(item["clean"], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_ylabel(f"{item['name']}\n({item['deg_type']})", fontsize=9, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Degraded
        axes[row_idx, 1].imshow(item["degraded"], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_xlabel(f"PSNR: {item['deg_psnr']:.1f} dB\nSSIM: {item['deg_ssim']:.3f}", fontsize=8)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Restored
        axes[row_idx, 2].imshow(item["restored"], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_xlabel(f"PSNR: {item['res_psnr']:.1f} dB\nSSIM: {item['res_ssim']:.3f}", fontsize=8, color="green")
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

        # Mask
        axes[row_idx, 3].imshow(item["mask"], cmap="inferno", vmin=0, vmax=1)
        axes[row_idx, 3].set_xlabel(f"Mean M: {item['mask'].mean():.2f}", fontsize=8)
        axes[row_idx, 3].set_xticks([])
        axes[row_idx, 3].set_yticks([])

        # Reliability
        axes[row_idx, 4].imshow(item["reliability"], cmap="viridis", vmin=0, vmax=1)
        axes[row_idx, 4].set_xlabel(f"Mean R: {item['reliability'].mean():.2f}", fontsize=8)
        axes[row_idx, 4].set_xticks([])
        axes[row_idx, 4].set_yticks([])

    plt.tight_layout()
    collage_path = Path("sample_results/comparison.png")
    collage_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(collage_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved visual comparison grid to {collage_path}")

    print("\n" + "=" * 65)
    print("SUMMARY OF MEASURED BENCHMARK ON TEST SPLIT:")
    print(f"  Evaluated Samples:     {len(df_metrics)}")
    print(f"  Avg Degraded PSNR:     {avg_deg_psnr:.2f} dB")
    print(f"  Avg AIRIS Restored:    {avg_airis_psnr:.2f} dB  (Gain: {avg_psnr_gain:+.2f} dB)")
    print(f"  Avg Degraded SSIM:     {avg_deg_ssim:.4f}")
    print(f"  Avg AIRIS SSIM:        {avg_airis_ssim:.4f}  (Gain: {avg_ssim_gain:+.4f})")
    if avg_airis_lpips:
        print(f"  Avg AIRIS LPIPS:       {avg_airis_lpips:.4f} (Degraded: {avg_deg_lpips:.4f})")
    print(f"  Avg Latency (CPU):     {df_metrics['latency_ms'].mean():.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
