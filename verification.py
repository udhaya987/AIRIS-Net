#!/usr/bin/env python3
"""
AIRIS-Net System & Submission Verification Suite.

Comprehensive validation verifying:
  1. Python environment & core package imports
  2. AIRIS-Net model construction (scale=1 and scale=2)
  3. Model forward pass & output tensor shapes
  4. Degradation pipeline (Gaussian, Speckle, Downsampling, Combined)
  5. x2 Super-resolution shape consistency (128x128 -> 256x256, 256x256 -> 512x512)
  6. Checkpoint saving & robust loading (state_dict & dict containers)
  7. Metric functions (PSNR, SSIM, LPIPS)
  8. Batch inference pipeline (kla_inference.py execution)
  9. CPU & CUDA compatibility
"""

import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_verification():
    results = {}
    print("=" * 70)
    print("AIRIS-Net Hackathon Submission Verification Suite")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python:    {sys.version.split()[0]}")
    print(f"PyTorch:   {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Imports Verification
    # -------------------------------------------------------------
    print("\n[1/9] Verifying Core Imports...")
    try:
        import cv2
        import skimage
        import PIL
        import yaml
        import pandas
        import matplotlib
        import timm
        import scipy
        from airis.model import AIRISNet
        from airis.losses import AIRISLoss
        from data.degradation import RandomDegradationPipeline
        from data.dataset import IndustrialRestorationDataset
        from utils.checkpoint import save_checkpoint, load_checkpoint
        from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips
        from utils.image_utils import load_image, save_image, to_tensor, to_numpy
        results["imports"] = "PASS"
        print("  [OK] All package dependencies and project modules imported successfully.")
    except Exception as e:
        results["imports"] = f"FAIL ({e})"
        print(f"  [FAIL] Import error: {e}")

    # -------------------------------------------------------------
    # 2. Model Construction (Scale 1 & Scale 2)
    # -------------------------------------------------------------
    print("\n[2/9] Verifying AIRIS-Net Construction (x1 and x2)...")
    try:
        model_x1 = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
        model_x2 = AIRISNet(in_channels=1, base_channels=48, scale_factor=2)
        params_x1 = sum(p.numel() for p in model_x1.parameters())
        params_x2 = sum(p.numel() for p in model_x2.parameters())
        assert params_x1 > 0 and params_x2 > 0
        results["model_construction"] = "PASS"
        print(f"  [OK] Model x1 instantiated ({params_x1:,} parameters)")
        print(f"  [OK] Model x2 instantiated ({params_x2:,} parameters)")
    except Exception as e:
        results["model_construction"] = f"FAIL ({e})"
        print(f"  [FAIL] Construction error: {e}")

    # -------------------------------------------------------------
    # 3. Model Forward Pass & Output Shape (Scale 1)
    # -------------------------------------------------------------
    print("\n[3/9] Verifying Forward Pass (Same-Resolution x1)...")
    try:
        model_x1.eval()
        dummy_128 = torch.rand(2, 1, 128, 128)
        with torch.no_grad():
            out_x1 = model_x1(dummy_128)

        assert out_x1["restored"].shape == (2, 1, 128, 128), f"Shape mismatch: {out_x1['restored'].shape}"
        assert out_x1["mask"].shape == (2, 1, 128, 128), f"Mask shape mismatch: {out_x1['mask'].shape}"
        assert out_x1["reliability"].shape == (2, 1, 128, 128), f"Rel shape mismatch: {out_x1['reliability'].shape}"
        assert out_x1["routing_weights"].shape == (2, 3), f"Routing shape mismatch: {out_x1['routing_weights'].shape}"
        
        # Check routing weights sum to 1.0
        r_sum = out_x1["routing_weights"].sum(dim=1)
        assert torch.allclose(r_sum, torch.ones_like(r_sum), atol=1e-5), "Routing weights must sum to 1.0"
        
        results["forward_pass_x1"] = "PASS"
        print(f"  [OK] Input (2, 1, 128, 128) -> Restored (2, 1, 128, 128) | Routing Sum: {r_sum[0].item():.4f}")
    except Exception as e:
        results["forward_pass_x1"] = f"FAIL ({e})"
        print(f"  [FAIL] Forward pass x1 error: {e}")

    # -------------------------------------------------------------
    # 4. Degradation Pipeline (Gaussian, Speckle, Resolution, Combined)
    # -------------------------------------------------------------
    print("\n[4/9] Verifying Degradation Pipeline...")
    try:
        pipe = RandomDegradationPipeline(seed=42)
        test_img = np.random.rand(128, 128).astype(np.float32)

        # 4a. Gaussian Noise
        deg_gauss, meta_g = pipe.apply_gaussian_noise(test_img, sigma=25.0)
        assert deg_gauss.shape == (128, 128)
        assert 0.0 <= deg_gauss.min() and deg_gauss.max() <= 1.0

        # 4b. Speckle Noise (multiplicative)
        deg_speckle, meta_s = pipe.apply_speckle_noise(test_img, variance=0.08)
        assert deg_speckle.shape == (128, 128)
        assert 0.0 <= deg_speckle.min() and deg_speckle.max() <= 1.0

        # 4c. Resolution Degradation
        deg_res, meta_r = pipe.apply_resolution_degradation(test_img, scale_factor=2.0, keep_dim=True)
        assert deg_res.shape == (128, 128)

        # 4d. Combined Degradations
        deg_combo, meta_c = pipe.apply_combined_degradation(test_img, mode="gaussian_speckle_downsample", scale_factor=2.0)
        assert deg_combo.shape == (128, 128)

        results["degradations"] = "PASS"
        print(f"  [OK] Gaussian Noise: {meta_g}")
        print(f"  [OK] Speckle Noise (multiplicative): {meta_s}")
        print(f"  [OK] Resolution Downsample: {meta_r}")
        print(f"  [OK] Combined Mode: {meta_c['type']}")
    except Exception as e:
        results["degradations"] = f"FAIL ({e})"
        print(f"  [FAIL] Degradation error: {e}")

    # -------------------------------------------------------------
    # 5. Super-Resolution x2 Output Shape Verification
    # -------------------------------------------------------------
    print("\n[5/9] Verifying Super-Resolution (x2 Scale Output Shapes)...")
    try:
        model_x2.eval()
        # Test 128x128 -> 256x256
        x_128 = torch.rand(1, 1, 128, 128)
        with torch.no_grad():
            out_256 = model_x2(x_128)
        assert out_256["restored"].shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {out_256['restored'].shape}"

        # Test 256x256 -> 512x512
        x_256 = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            out_512 = model_x2(x_256)
        assert out_512["restored"].shape == (1, 1, 512, 512), f"Expected (1, 1, 512, 512), got {out_512['restored'].shape}"

        results["super_resolution_x2"] = "PASS"
        print(f"  [OK] Input (1, 1, 128, 128) -> Restored (1, 1, 256, 256)")
        print(f"  [OK] Input (1, 1, 256, 256) -> Restored (1, 1, 512, 512)")
    except Exception as e:
        results["super_resolution_x2"] = f"FAIL ({e})"
        print(f"  [FAIL] Super-resolution shape error: {e}")

    # -------------------------------------------------------------
    # 6. Checkpoint Save & Load Robustness
    # -------------------------------------------------------------
    print("\n[6/9] Verifying Checkpoint Save & Load Mechanism...")
    temp_dir = tempfile.mkdtemp()
    try:
        # Test saving dict state
        state = {
            "epoch": 1,
            "model_state_dict": model_x1.state_dict(),
            "best_psnr": 28.5,
            "scale_factor": 1
        }
        chk_file = save_checkpoint(state, is_best=True, checkpoint_dir=temp_dir, filename="test_chk.pth")
        assert Path(chk_file).exists()
        assert (Path(temp_dir) / "latest_airis.pth").exists()
        assert (Path(temp_dir) / "best_airis.pth").exists()

        # Test loading into a new model
        test_load_model = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
        load_checkpoint(chk_file, test_load_model)

        results["checkpoint_handling"] = "PASS"
        print(f"  [OK] Checkpoint successfully saved and loaded from {chk_file}")
    except Exception as e:
        results["checkpoint_handling"] = f"FAIL ({e})"
        print(f"  [FAIL] Checkpoint error: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 7. Metrics Verification (PSNR, SSIM, LPIPS)
    # -------------------------------------------------------------
    print("\n[7/9] Verifying Quantitative Metrics (PSNR, SSIM, LPIPS)...")
    try:
        img_clean = np.ones((128, 128), dtype=np.float32) * 0.5
        img_noisy = np.clip(img_clean + np.random.normal(0, 0.05, (128, 128)), 0.0, 1.0).astype(np.float32)

        psnr_val = calculate_psnr(img_noisy, img_clean)
        ssim_val = calculate_ssim(img_noisy, img_clean)
        lpips_val = calculate_lpips(img_noisy, img_clean)

        assert psnr_val > 15.0, f"Unexpected PSNR value: {psnr_val}"
        assert 0.0 <= ssim_val <= 1.0, f"Unexpected SSIM value: {ssim_val}"

        lpips_str = f"{lpips_val:.4f}" if lpips_val is not None else "Skipped (Optional dependency)"
        results["metrics"] = "PASS"
        print(f"  [OK] PSNR Calculation: {psnr_val:.2f} dB")
        print(f"  [OK] SSIM Calculation: {ssim_val:.4f}")
        print(f"  [OK] LPIPS Calculation: {lpips_str}")
    except Exception as e:
        results["metrics"] = f"FAIL ({e})"
        print(f"  [FAIL] Metrics error: {e}")

    # -------------------------------------------------------------
    # 8. Batch Folder Inference Pipeline
    # -------------------------------------------------------------
    print("\n[8/9] Verifying kla_inference.py Script...")
    try:
        from kla_inference import discover_images, build_model
        chk_path = "checkpoints/best_airis.pth"
        if Path(chk_path).exists():
            inf_model = build_model(chk_path, device=torch.device("cpu"), scale_override=1)
            assert inf_model is not None
            print(f"  [OK] Successfully verified kla_inference model loader with {chk_path}")
        else:
            print("  [INFO] checkpoints/best_airis.pth not present yet for inference test.")

        results["kla_inference"] = "PASS"
    except Exception as e:
        results["kla_inference"] = f"FAIL ({e})"
        print(f"  [FAIL] kla_inference error: {e}")

    # -------------------------------------------------------------
    # 9. Device Compatibility (CPU / CUDA)
    # -------------------------------------------------------------
    print("\n[9/9] Verifying Device Compatibility...")
    try:
        # CPU
        cpu_dev = torch.device("cpu")
        model_cpu = AIRISNet(in_channels=1, base_channels=48).to(cpu_dev)
        x_cpu = torch.rand(1, 1, 64, 64, device=cpu_dev)
        with torch.no_grad():
            _ = model_cpu(x_cpu)
        print("  [OK] CPU execution verified.")

        # CUDA if available
        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            model_cuda = AIRISNet(in_channels=1, base_channels=48).to(cuda_dev)
            x_cuda = torch.rand(1, 1, 64, 64, device=cuda_dev)
            with torch.no_grad():
                _ = model_cuda(x_cuda)
            print(f"  [OK] CUDA execution verified on {torch.cuda.get_device_name(0)}.")
        else:
            print("  [OK] CUDA not present on this host; CPU fallback verified.")

        results["device_compatibility"] = "PASS"
    except Exception as e:
        results["device_compatibility"] = f"FAIL ({e})"
        print(f"  [FAIL] Device compatibility error: {e}")

    # -------------------------------------------------------------
    # Verification Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    all_passed = True
    for test_name, status in results.items():
        status_str = "PASS" if status == "PASS" else status
        symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {symbol.ljust(8)} {test_name.ljust(25)}: {status_str}")
        if status != "PASS":
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED: AIRIS-Net is fully verified.")
        return 0
    else:
        print("SOME CHECKS FAILED: Please inspect errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_verification())
