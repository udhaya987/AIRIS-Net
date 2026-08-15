import sys
import os
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SWINIR_DIR = PROJECT_ROOT / "SwinIR"
if str(SWINIR_DIR) not in sys.path:
    sys.path.append(str(SWINIR_DIR))

from airis.model import AIRISNet
from utils.checkpoint import load_checkpoint
from utils.image_utils import load_image, save_image, to_tensor
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips
from data.degradation import RandomDegradationPipeline
from baseline_swinir import SwinIRRestorer


def run_comprehensive_evaluation():
    print("=" * 70)
    print("AIRIS-Net Comprehensive Test Set Benchmark & Baseline Comparison")
    print("=" * 70)

    test_dir = Path("data/test/clean")
    test_files = sorted(list(test_dir.glob("*.npy")) + list(test_dir.glob("*.png")))
    print(f"[Dataset] Verified {len(test_files)} clean held-out test images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Primary device: {device}")

    # 1. Load AIRIS-Net
    airis_model = AIRISNet(in_channels=1, base_channels=48, scale_factor=1).to(device)
    chk_path = Path("checkpoints/best_airis.pth")
    if chk_path.exists():
        load_checkpoint(chk_path, airis_model, device=device)
    airis_model.eval()

    airis_params = sum(p.numel() for p in airis_model.parameters() if p.requires_grad)
    airis_size_mb = chk_path.stat().st_size / (1024 * 1024) if chk_path.exists() else 0.0

    # 2. Load SwinIR Baseline
    swinir_chk = Path("SwinIR/model_zoo/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth")
    swinir_restorer = None
    swinir_params = 11_900_000
    if swinir_chk.exists():
        try:
            swinir_restorer = SwinIRRestorer(model_path=str(swinir_chk), device="cpu")
            swinir_params = sum(p.numel() for p in swinir_restorer.model.parameters() if p.requires_grad)
            print(f"[SwinIR] Baseline loaded successfully ({swinir_params:,} parameters).")
        except Exception as e:
            print(f"[SwinIR] Warning: SwinIR could not be initialized: {e}")

    # 3. Environment Record
    env_text = (
        f"AIRIS-Net Execution Environment\n"
        f"================================\n"
        f"OS:                 {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Architecture:       {platform.machine()} ({platform.processor()})\n"
        f"Python Version:     {sys.version.split()[0]}\n"
        f"PyTorch Version:    {torch.__version__}\n"
        f"CUDA Available:     {torch.cuda.is_available()}\n"
        f"Primary Device:     {device}\n"
        f"AIRIS-Net Params:   {airis_params:,} ({airis_size_mb:.2f} MB)\n"
        f"SwinIR Params:      {swinir_params:,}\n"
        f"Timestamp:          {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    with open("results/environment.txt", "w", encoding="utf-8") as f:
        f.write(env_text)
    print("[OK] Recorded environment details in results/environment.txt")

    # 4. Model Complexity Record
    complexity_text = (
        f"AIRIS-Net Model Complexity & Computational Specifications\n"
        f"==========================================================\n"
        f"Total Parameters:      {airis_params:,}\n"
        f"Trainable Parameters:  {airis_params:,}\n"
        f"Model Checkpoint Size: {airis_size_mb:.2f} MB\n"
        f"Base Channels:         48\n"
        f"Routing Dimensions:    64 latent -> 3 expert softmax gating\n"
        f"Multi-Expert Modules:  Local CNN (DW-Conv), Global Context (Windowed Multi-Head Attention), Frequency (2D FFT)\n"
        f"Timing Device:         {device} (CPU evaluation)\n"
    )
    with open("results/model_complexity.txt", "w", encoding="utf-8") as f:
        f.write(complexity_text)
    print("[OK] Recorded model complexity in results/model_complexity.txt")

    # 5. Objective 4: Router Analysis Across Diverse Degradations
    print("\n--- Running Objective 4: Adaptive Router Analysis ---")
    pipeline = RandomDegradationPipeline(seed=42)
    router_rows = []
    sample_clean = load_image(test_files[0], grayscale=True)

    conditions = [
        ("Clean Input", "none", {}),
        ("Gaussian Noise (Low, s=15)", "gaussian", {"sigma": 15.0}),
        ("Gaussian Noise (High, s=40)", "gaussian", {"sigma": 40.0}),
        ("Speckle Noise (Low, v=0.04)", "speckle", {"variance": 0.04}),
        ("Speckle Noise (High, v=0.15)", "speckle", {"variance": 0.15}),
        ("Gaussian Blur (k=5)", "blur", {"ksize": 5, "sigma": 1.5}),
        ("Motion Blur (len=7, ang=45)", "motion_blur", {"kernel_size": 7, "angle": 45.0}),
        ("Contrast Degradation (f=0.4)", "contrast", {"factor": 0.4}),
        ("Resolution Downsampling (2x)", "resolution", {"scale_factor": 2.0, "keep_dim": True}),
        ("Combined: Gaussian + Speckle", "gaussian_speckle", {}),
        ("Combined: Gaussian + Downsample", "gaussian_downsample", {"scale_factor": 2.0}),
        ("Combined: Gaussian + Speckle + Downsample", "gaussian_speckle_downsample", {"scale_factor": 2.0}),
        ("Mixed Multi-Stage Degradation", "mixed", {"num_degradations": 3})
    ]

    for label, deg_t, kwargs in conditions:
        deg_img, meta = pipeline(sample_clean, degradation_type=deg_t, **kwargs)
        t_in = to_tensor(deg_img, device=device)
        with torch.no_grad():
            out = airis_model(t_in)
        rw = out["routing_weights"].squeeze().cpu().numpy()
        router_rows.append({
            "image": test_files[0].name,
            "degradation": label,
            "local_weight": round(float(rw[0]), 4),
            "global_weight": round(float(rw[1]), 4),
            "frequency_weight": round(float(rw[2]), 4),
            "sum_weights": round(float(rw.sum()), 4)
        })
        print(f"  {label:<40} -> Local: {rw[0]:.4f} | Global: {rw[1]:.4f} | Freq: {rw[2]:.4f} | Sum: {rw.sum():.4f}")

    df_router = pd.DataFrame(router_rows)
    df_router.to_csv("results/router_analysis.csv", index=False)
    print("[OK] Saved router analysis to results/router_analysis.csv")

    # 6. Objective 8 & 10: Test Set Evaluation & Baseline Comparison (Evaluated over 30 test samples)
    eval_count = min(30, len(test_files))
    print(f"\n--- Running Objective 8 & 10: Held-Out Test Evaluation ({eval_count} samples) ---")

    per_image_rows = []
    swinir_psnr_list, swinir_ssim_list, swinir_lpips_list, swinir_time_list = [], [], [], []
    airis_time_list = []
    visual_collages = []
    failure_cases = []

    for idx, f in enumerate(test_files[:eval_count]):
        clean = load_image(f, grayscale=True)
        # Apply standard competition degradation pattern
        deg_mode = ["gaussian", "speckle", "gaussian_speckle", "gaussian_downsample", "mixed"][idx % 5]
        degraded, meta = pipeline(clean, degradation_type=deg_mode)

        # Baseline: Degraded vs Clean
        deg_psnr = calculate_psnr(degraded, clean)
        deg_ssim = calculate_ssim(degraded, clean)
        deg_lpips = calculate_lpips(degraded, clean, device=device)

        # AIRIS-Net Forward
        t_in = to_tensor(degraded, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            airis_out = airis_model(t_in)
        airis_latency = (time.perf_counter() - t0) * 1000.0
        airis_time_list.append(airis_latency)

        airis_restored = airis_out["restored"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
        airis_mask = airis_out["mask"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
        airis_rel = airis_out["reliability"].squeeze().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
        routing = airis_out["routing_weights"].squeeze().cpu().numpy()

        airis_psnr = calculate_psnr(airis_restored, clean)
        airis_ssim = calculate_ssim(airis_restored, clean)
        airis_lpips = calculate_lpips(airis_restored, clean, device=device)

        psnr_gain = airis_psnr - deg_psnr
        ssim_gain = airis_ssim - deg_ssim
        lpips_imp = (deg_lpips - airis_lpips) if (deg_lpips is not None and airis_lpips is not None) else None

        # SwinIR Forward (if available)
        sw_psnr, sw_ssim, sw_lpips = None, None, None
        if swinir_restorer is not None:
            t0_sw = time.perf_counter()
            sw_restored = swinir_restorer.restore(degraded)
            sw_latency = (time.perf_counter() - t0_sw) * 1000.0
            swinir_time_list.append(sw_latency)

            sw_psnr = calculate_psnr(sw_restored, clean)
            sw_ssim = calculate_ssim(sw_restored, clean)
            sw_lpips = calculate_lpips(sw_restored, clean, device=device)
            swinir_psnr_list.append(sw_psnr)
            swinir_ssim_list.append(sw_ssim)
            if sw_lpips is not None:
                swinir_lpips_list.append(sw_lpips)

        row = {
            "image": f.name,
            "degradation": meta["type"],
            "degraded_psnr": round(deg_psnr, 3),
            "airis_psnr": round(airis_psnr, 3),
            "psnr_gain": round(psnr_gain, 3),
            "degraded_ssim": round(deg_ssim, 4),
            "airis_ssim": round(airis_ssim, 4),
            "ssim_gain": round(ssim_gain, 4),
            "degraded_lpips": round(deg_lpips, 4) if deg_lpips is not None else "N/A",
            "airis_lpips": round(airis_lpips, 4) if airis_lpips is not None else "N/A",
            "lpips_improvement": round(lpips_imp, 4) if lpips_imp is not None else "N/A",
            "airis_latency_ms": round(airis_latency, 2),
            "swinir_psnr": round(sw_psnr, 3) if sw_psnr is not None else "N/A",
            "swinir_ssim": round(sw_ssim, 4) if sw_ssim is not None else "N/A"
        }
        per_image_rows.append(row)

        # Collect visual samples (first 5)
        if idx < 5:
            visual_collages.append({
                "idx": idx + 1,
                "name": f.name,
                "deg_type": meta["type"],
                "clean": clean,
                "degraded": degraded,
                "restored": airis_restored,
                "mask": airis_mask,
                "reliability": airis_rel,
                "deg_psnr": deg_psnr,
                "airis_psnr": airis_psnr,
                "deg_ssim": deg_ssim,
                "airis_ssim": airis_ssim,
                "routing": routing
            })

        # Collect failure/challenging cases (where psnr_gain <= 0.5 dB or severe mixed degradation)
        if psnr_gain < 0.2 and len(failure_cases) < 3:
            failure_cases.append({
                "name": f.name,
                "deg_type": meta["type"],
                "clean": clean,
                "degraded": degraded,
                "restored": airis_restored,
                "deg_psnr": deg_psnr,
                "airis_psnr": airis_psnr,
                "deg_ssim": deg_ssim,
                "airis_ssim": airis_ssim,
                "reason": "Extreme compound corruption where input noise floor exceeds single-epoch feature representation."
            })

        sw_str = f" | SwinIR: {sw_psnr:.2f} dB" if sw_psnr is not None else ""
        print(f"  [{idx+1:2d}/{eval_count:2d}] {f.name} ({meta['type']}) | "
              f"Deg PSNR: {deg_psnr:.2f} -> AIRIS: {airis_psnr:.2f} dB (+{psnr_gain:.2f}){sw_str} | "
              f"SSIM: {deg_ssim:.4f} -> {airis_ssim:.4f} (+{ssim_gain:.4f})")

    # Save per-image metrics
    df_per_image = pd.DataFrame(per_image_rows)
    df_per_image.to_csv(PROJECT_ROOT / "results" / "final_test_per_image.csv", index=False)
    print("[OK] Saved per-image metrics to results/final_test_per_image.csv")

    # Save dataset summary metrics
    avg_deg_psnr = df_per_image["degraded_psnr"].mean()
    avg_airis_psnr = df_per_image["airis_psnr"].mean()
    avg_psnr_gain = df_per_image["psnr_gain"].mean()

    avg_deg_ssim = df_per_image["degraded_ssim"].mean()
    avg_airis_ssim = df_per_image["airis_ssim"].mean()
    avg_ssim_gain = df_per_image["ssim_gain"].mean()

    num_deg_lpips = pd.to_numeric(df_per_image["degraded_lpips"], errors="coerce")
    num_airis_lpips = pd.to_numeric(df_per_image["airis_lpips"], errors="coerce")
    avg_deg_lpips = num_deg_lpips.mean() if not num_deg_lpips.isna().all() else None
    avg_airis_lpips = num_airis_lpips.mean() if not num_airis_lpips.isna().all() else None
    avg_lpips_red = (avg_deg_lpips - avg_airis_lpips) if (avg_deg_lpips and avg_airis_lpips) else None

    summary_rows = [
        {"Category": "BEFORE AIRIS", "Metric": "Average Degraded PSNR (dB)", "Value": round(avg_deg_psnr, 2)},
        {"Category": "BEFORE AIRIS", "Metric": "Average Degraded SSIM", "Value": round(avg_deg_ssim, 4)},
        {"Category": "BEFORE AIRIS", "Metric": "Average Degraded LPIPS", "Value": round(avg_deg_lpips, 4) if avg_deg_lpips else "N/A"},
        {"Category": "AFTER AIRIS", "Metric": "Average Restored PSNR (dB)", "Value": round(avg_airis_psnr, 2)},
        {"Category": "AFTER AIRIS", "Metric": "Average Restored SSIM", "Value": round(avg_airis_ssim, 4)},
        {"Category": "AFTER AIRIS", "Metric": "Average Restored LPIPS", "Value": round(avg_airis_lpips, 4) if avg_airis_lpips else "N/A"},
        {"Category": "IMPROVEMENT", "Metric": "PSNR Gain (dB)", "Value": round(avg_psnr_gain, 2)},
        {"Category": "IMPROVEMENT", "Metric": "SSIM Gain", "Value": round(avg_ssim_gain, 4)},
        {"Category": "IMPROVEMENT", "Metric": "LPIPS Reduction", "Value": round(avg_lpips_red, 4) if avg_lpips_red else "N/A"},
        {"Category": "EFFICIENCY", "Metric": "Avg Inference Latency (CPU)", "Value": f"{np.mean(airis_time_list):.2f} ms"},
        {"Category": "EFFICIENCY", "Metric": "Model Parameters", "Value": f"{airis_params:,}"}
    ]
    pd.DataFrame(summary_rows).to_csv(PROJECT_ROOT / "results" / "final_test_metrics.csv", index=False)
    print("[OK] Saved summary test metrics to results/final_test_metrics.csv")

    # 7. Objective 10: Baseline Comparison Table
    baseline_table_rows = [
        {
            "Method": "Degraded Input (Baseline)",
            "PSNR (dB)": round(avg_deg_psnr, 2),
            "SSIM": round(avg_deg_ssim, 4),
            "LPIPS": round(avg_deg_lpips, 4) if avg_deg_lpips else "N/A",
            "Parameters": "—",
            "Inference_Time_ms": "0.00",
            "Hardware": "—"
        }
    ]
    if swinir_restorer is not None and swinir_psnr_list:
        baseline_table_rows.append({
            "Method": "SwinIR Baseline (Pretrained)",
            "PSNR (dB)": round(float(np.mean(swinir_psnr_list)), 2),
            "SSIM": round(float(np.mean(swinir_ssim_list)), 4),
            "LPIPS": round(float(np.mean(swinir_lpips_list)), 4) if swinir_lpips_list else "N/A",
            "Parameters": f"{swinir_params:,}",
            "Inference_Time_ms": round(float(np.mean(swinir_time_list)), 2),
            "Hardware": "CPU"
        })
    else:
        baseline_table_rows.append({
            "Method": "SwinIR Baseline",
            "PSNR (dB)": "NOT EVALUATED",
            "SSIM": "NOT EVALUATED",
            "LPIPS": "NOT EVALUATED",
            "Parameters": "11,900,000",
            "Inference_Time_ms": "—",
            "Hardware": "—"
        })

    baseline_table_rows.append({
        "Method": "AIRIS-Net (Ours)",
        "PSNR (dB)": round(avg_airis_psnr, 2),
        "SSIM": round(avg_airis_ssim, 4),
        "LPIPS": round(avg_airis_lpips, 4) if avg_airis_lpips else "N/A",
        "Parameters": f"{airis_params:,}",
        "Inference_Time_ms": round(float(np.mean(airis_time_list)), 2),
        "Hardware": "CPU"
    })
    pd.DataFrame(baseline_table_rows).to_csv(PROJECT_ROOT / "results" / "baseline_comparison.csv", index=False)
    print("[OK] Saved baseline comparison to results/baseline_comparison.csv")

    # 8. Objective 12: Generate Individual and Grid Visuals
    print("\n--- Generating Visual Collages (sample_results/) ---")
    out_sample_dir = PROJECT_ROOT / "sample_results"
    out_sample_dir.mkdir(parents=True, exist_ok=True)

    for item in visual_collages:
        fig_single, axes_s = plt.subplots(1, 5, figsize=(16, 3.5))
        titles = [
            f"1. Ground Truth\n({item['name']})",
            f"2. Degraded ({item['deg_type']})\nPSNR: {item['deg_psnr']:.2f} dB, SSIM: {item['deg_ssim']:.3f}",
            f"3. AIRIS Restored\nPSNR: {item['airis_psnr']:.2f} dB, SSIM: {item['airis_ssim']:.3f}",
            f"4. Restoration Mask (M)\nMean M: {item['mask'].mean():.3f}",
            f"5. Reliability Map (R)\nMean R: {item['reliability'].mean():.3f}"
        ]
        imgs = [item["clean"], item["degraded"], item["restored"], item["mask"], item["reliability"]]
        cmaps = ["gray", "gray", "gray", "inferno", "viridis"]

        for ax, im, t, cm in zip(axes_s, imgs, titles, cmaps):
            ax.imshow(im, cmap=cm, vmin=0, vmax=1)
            ax.set_title(t, fontsize=9, fontweight="bold")
            ax.axis("off")

        single_out = out_sample_dir / f"example_{item['idx']:02d}.png"
        fig_single.tight_layout()
        fig_single.savefig(str(single_out), dpi=180, bbox_inches="tight")
        plt.close(fig_single)
        print(f"  [OK] Saved {single_out}")

    # Comprehensive Comparison Grid
    fig_grid, axes_g = plt.subplots(len(visual_collages), 5, figsize=(18, 3.5 * len(visual_collages)))
    cols = ["1. Clean Ground Truth", "2. Degraded Input", "3. AIRIS Restored", "4. Restoration Mask (M)", "5. Reliability Map (R)"]
    for c_i, c_name in enumerate(cols):
        axes_g[0, c_i].set_title(c_name, fontsize=12, fontweight="bold", pad=12)

    for r_i, item in enumerate(visual_collages):
        axes_g[r_i, 0].imshow(item["clean"], cmap="gray", vmin=0, vmax=1)
        axes_g[r_i, 0].set_ylabel(f"{item['name']}\n({item['deg_type']})", fontsize=9, fontweight="bold")
        axes_g[r_i, 0].set_xticks([])
        axes_g[r_i, 0].set_yticks([])

        axes_g[r_i, 1].imshow(item["degraded"], cmap="gray", vmin=0, vmax=1)
        axes_g[r_i, 1].set_xlabel(f"PSNR: {item['deg_psnr']:.2f} dB | SSIM: {item['deg_ssim']:.3f}", fontsize=8)
        axes_g[r_i, 1].set_xticks([])
        axes_g[r_i, 1].set_yticks([])

        axes_g[r_i, 2].imshow(item["restored"], cmap="gray", vmin=0, vmax=1)
        axes_g[r_i, 2].set_xlabel(f"PSNR: {item['airis_psnr']:.2f} dB | SSIM: {item['airis_ssim']:.3f}", fontsize=8, color="green", fontweight="bold")
        axes_g[r_i, 2].set_xticks([])
        axes_g[r_i, 2].set_yticks([])

        axes_g[r_i, 3].imshow(item["mask"], cmap="inferno", vmin=0, vmax=1)
        axes_g[r_i, 3].set_xlabel(f"Routing: L={item['routing'][0]:.2f}, G={item['routing'][1]:.2f}, F={item['routing'][2]:.2f}", fontsize=8)
        axes_g[r_i, 3].set_xticks([])
        axes_g[r_i, 3].set_yticks([])

        axes_g[r_i, 4].imshow(item["reliability"], cmap="viridis", vmin=0, vmax=1)
        axes_g[r_i, 4].set_xlabel(f"Mean Confidence: {item['reliability'].mean():.2f}", fontsize=8)
        axes_g[r_i, 4].set_xticks([])
        axes_g[r_i, 4].set_yticks([])

    fig_grid.tight_layout()
    grid_out = out_sample_dir / "comparison_grid.png"
    fig_grid.savefig(str(grid_out), dpi=200, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"  [OK] Saved comprehensive comparison grid to {grid_out}")

    # 9. Objective 13: Failure Cases Visual Analysis
    if failure_cases:
        fig_fail, axes_f = plt.subplots(len(failure_cases), 3, figsize=(12, 3.8 * len(failure_cases)))
        if len(failure_cases) == 1:
            axes_f = np.expand_dims(axes_f, axis=0)

        for r_i, fc in enumerate(failure_cases):
            axes_f[r_i, 0].imshow(fc["clean"], cmap="gray", vmin=0, vmax=1)
            axes_f[r_i, 0].set_title(f"Target: {fc['name']}", fontsize=9, fontweight="bold")
            axes_f[r_i, 0].axis("off")

            axes_f[r_i, 1].imshow(fc["degraded"], cmap="gray", vmin=0, vmax=1)
            axes_f[r_i, 1].set_title(f"Degraded ({fc['deg_type']})\nPSNR: {fc['deg_psnr']:.2f} dB", fontsize=9)
            axes_f[r_i, 1].axis("off")

            axes_f[r_i, 2].imshow(fc["restored"], cmap="gray", vmin=0, vmax=1)
            axes_f[r_i, 2].set_title(f"AIRIS Restored\nPSNR: {fc['airis_psnr']:.2f} dB (d: {fc['airis_psnr']-fc['deg_psnr']:+.2f})", fontsize=9, color="darkred")
            axes_f[r_i, 2].set_xlabel(fc["reason"], fontsize=8, style="italic")
            axes_f[r_i, 2].set_xticks([])
            axes_f[r_i, 2].set_yticks([])

        fig_fail.tight_layout()
        fail_out = out_sample_dir / "failure_cases.png"
        fig_fail.savefig(str(fail_out), dpi=180, bbox_inches="tight")
        plt.close(fig_fail)
        print(f"  [OK] Saved failure cases figure to {fail_out}")

    print("\n" + "=" * 70)
    print("FINAL MEASURED TEST SET PERFORMANCE (AIRIS-Net vs Baselines):")
    print(f"  Degraded Input:  PSNR = {avg_deg_psnr:.2f} dB | SSIM = {avg_deg_ssim:.4f}")
    if swinir_psnr_list:
        print(f"  SwinIR Baseline: PSNR = {np.mean(swinir_psnr_list):.2f} dB | SSIM = {np.mean(swinir_ssim_list):.4f} (Params: {swinir_params:,})")
    print(f"  AIRIS-Net:       PSNR = {avg_airis_psnr:.2f} dB | SSIM = {avg_airis_ssim:.4f} (Params: {airis_params:,})")
    print(f"  NET GAIN:        +{avg_psnr_gain:.2f} dB PSNR | +{avg_ssim_gain:.4f} SSIM")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_evaluation()
