import os
import sys
from pathlib import Path
import argparse
import time
import numpy as np
import cv2
import torch

# Ensure SwinIR is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SWINIR_DIR = PROJECT_ROOT / "SwinIR"
if str(SWINIR_DIR) not in sys.path:
    sys.path.append(str(SWINIR_DIR))

from models.network_swinir import SwinIR
from utils.metrics import calculate_psnr, calculate_ssim
from utils.image_utils import load_image, save_image, to_tensor, to_numpy, compute_change_map
from utils.edge_utils import compute_sobel_edges_np, edge_consistency_score
from utils.frequency_utils import compute_fft_magnitude_np, frequency_consistency_score


class SwinIRRestorer:
    """
    Wrapper for Pretrained SwinIR Grayscale Denoising / Restoration model.
    """
    def __init__(
        self,
        model_path: str = "SwinIR/model_zoo/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth",
        device: str = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"[SwinIR] Initializing SwinIR Restorer on device: {self.device}")
        
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"SwinIR model checkpoint not found at: {self.model_path}")
            
        self.window_size = 8
        self.model = SwinIR(
            upscale=1,
            in_chans=1,
            img_size=128,
            window_size=self.window_size,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv'
        )
        
        print(f"[SwinIR] Loading weights from {self.model_path}")
        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        param_key = 'params' if 'params' in checkpoint else 'state_dict'
        state_dict = checkpoint[param_key] if param_key in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)
        print("[SwinIR] Model loaded successfully.")

    @torch.no_grad()
    def restore(self, img: np.ndarray) -> np.ndarray:
        """
        Restore a degraded image using SwinIR.
        Input: (H, W) or (H, W, 1) numpy array in [0.0, 1.0].
        Output: (H, W) restored numpy array in [0.0, 1.0].
        """
        img_in = np.squeeze(img).astype(np.float32)
        h_orig, w_orig = img_in.shape[:2]
        
        # Convert to tensor (1, 1, H, W)
        tensor_in = torch.from_numpy(img_in).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Pad to multiple of window size
        h_pad = (h_orig // self.window_size + 1) * self.window_size - h_orig if h_orig % self.window_size != 0 else 0
        w_pad = (w_orig // self.window_size + 1) * self.window_size - w_orig if w_orig % self.window_size != 0 else 0
        
        if h_pad > 0 or w_pad > 0:
            tensor_pad = torch.nn.functional.pad(tensor_in, (0, w_pad, 0, h_pad), mode='reflect')
        else:
            tensor_pad = tensor_in

        # Run model inference
        tensor_out = self.model(tensor_pad)
        
        # Crop back to original dimensions
        tensor_out = tensor_out[:, :, :h_orig, :w_orig]
        
        # Squeeze and clamp
        restored = tensor_out.squeeze().cpu().clamp(0.0, 1.0).numpy()
        return restored


def run_swinir_pipeline(
    input_path: str,
    clean_path: str = None,
    output_dir: str = "outputs"
) -> dict:
    """
    Run complete Level-1 POC pipeline on an image.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load degraded image
    degraded = load_image(input_path, grayscale=True)
    print(f"[Pipeline] Loaded degraded image: shape={degraded.shape}, min={degraded.min():.4f}, max={degraded.max():.4f}")
    
    # 2. Run SwinIR Restoration
    restorer = SwinIRRestorer()
    t0 = time.time()
    restored = restorer.restore(degraded)
    elapsed = time.time() - t0
    print(f"[Pipeline] Restoration complete in {elapsed:.3f}s")
    
    # 3. Compute analysis maps
    change_map = compute_change_map(degraded, restored)
    edge_degraded = compute_sobel_edges_np(degraded)
    edge_restored = compute_sobel_edges_np(restored)
    fft_degraded = compute_fft_magnitude_np(degraded)
    fft_restored = compute_fft_magnitude_np(restored)
    
    # 4. Save results
    base_name = Path(input_path).stem
    save_image(degraded, out_path / f"{base_name}_degraded.png")
    save_image(restored, out_path / f"{base_name}_swinir_restored.png")
    save_image(change_map, out_path / f"{base_name}_change_map.png")
    save_image(edge_restored, out_path / f"{base_name}_edge_map.png")
    
    metrics = {
        "model": "SwinIR Baseline",
        "elapsed_seconds": elapsed,
        "input_shape": degraded.shape,
    }
    
    # 5. Evaluate if clean ground truth is available
    if clean_path and Path(clean_path).exists():
        clean = load_image(clean_path, grayscale=True)
        # Resize clean if needed to match degraded resolution
        if clean.shape != degraded.shape:
            clean = cv2.resize(clean, (degraded.shape[1], degraded.shape[0]))
        
        psnr_deg = calculate_psnr(degraded, clean)
        psnr_res = calculate_psnr(restored, clean)
        ssim_deg = calculate_ssim(degraded, clean)
        ssim_res = calculate_ssim(restored, clean)
        edge_score = edge_consistency_score(restored, clean)
        freq_score = frequency_consistency_score(restored, clean)
        
        metrics.update({
            "psnr_degraded": psnr_deg,
            "psnr_restored": psnr_res,
            "psnr_improvement": psnr_res - psnr_deg,
            "ssim_degraded": ssim_deg,
            "ssim_restored": ssim_res,
            "ssim_improvement": ssim_res - ssim_deg,
            "edge_consistency": edge_score,
            "frequency_consistency": freq_score
        })
        print(f"[Metrics] PSNR: {psnr_deg:.2f} dB -> {psnr_res:.2f} dB (+{psnr_res - psnr_deg:.2f} dB)")
        print(f"[Metrics] SSIM: {ssim_deg:.4f} -> {ssim_res:.4f} (+{ssim_res - ssim_deg:.4f})")
        print(f"[Metrics] Edge Consistency: {edge_score:.4f}, Frequency Consistency: {freq_score:.4f}")
    
    print(f"[Pipeline] Saved results to {out_path.resolve()}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIRIS-Net SwinIR Baseline Restoration")
    parser.add_argument("--input", type=str, required=True, help="Path to degraded image (.npy, .png, etc.)")
    parser.add_argument("--clean", type=str, default=None, help="Path to clean ground truth image if available")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()
    
    run_swinir_pipeline(args.input, args.clean, args.output)
