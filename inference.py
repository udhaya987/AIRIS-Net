import os
import sys
from pathlib import Path
import argparse
import time
import torch
import numpy as np

from airis.model import AIRISNet
from utils.image_utils import load_image, save_image, to_tensor, to_numpy, compute_change_map
from utils.metrics import calculate_psnr, calculate_ssim
from utils.edge_utils import compute_sobel_edges_np
from utils.frequency_utils import compute_fft_magnitude_np


class AIRISPredictor:
    """
    Inference helper for trained AIRIS-Net models.
    """
    def __init__(self, checkpoint_path: str = "checkpoints/best_airis.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        chk_path = Path(checkpoint_path)
        if not chk_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {chk_path}")

        print(f"[AIRIS-Net] Loading checkpoint from {chk_path} on {self.device}")
        checkpoint = torch.load(str(chk_path), map_location=self.device)

        cfg = checkpoint.get("config", {}).get("model", {})
        self.model = AIRISNet(
            in_channels=cfg.get("in_channels", 1),
            base_channels=cfg.get("base_channels", 48),
            degradation_dim=cfg.get("degradation_dim", 64),
            use_local_expert=cfg.get("use_local_expert", True),
            use_global_expert=cfg.get("use_global_expert", True),
            use_frequency_expert=cfg.get("use_frequency_expert", True),
            use_adaptive_routing=cfg.get("use_adaptive_routing", True),
            use_integrity_mask=cfg.get("use_integrity_mask", True)
        )

        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        print("[AIRIS-Net] Model loaded successfully.")

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> dict:
        """
        Run inference on single image array.
        Input: (H, W) or (H, W, C) numpy array in [0, 1].
        """
        img_np = np.squeeze(img).astype(np.float32)
        h, w = img_np.shape[:2]

        tensor_in = to_tensor(img_np, device=self.device)
        t0 = time.time()
        outputs = self.model(tensor_in)
        elapsed = time.time() - t0

        restored_np = outputs["restored"].squeeze().cpu().clamp(0.0, 1.0).numpy()
        mask_np = outputs["mask"].squeeze().cpu().clamp(0.0, 1.0).numpy()
        reliability_np = outputs["reliability"].squeeze().cpu().clamp(0.0, 1.0).numpy()
        weights = outputs["routing_weights"].squeeze().cpu().numpy()
        diag_scores = outputs["diagnostic_scores"].squeeze().cpu().numpy()

        return {
            "restored": restored_np,
            "mask": mask_np,
            "reliability": reliability_np,
            "routing_weights": weights,
            "diagnostic_scores": diag_scores,
            "elapsed_seconds": elapsed
        }


def main():
    parser = argparse.ArgumentParser(description="AIRIS-Net Single Image Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image (.npy, .png, etc.)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_airis.pth", help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load input image
    img = load_image(args.input, grayscale=True)

    # 2. Run AIRIS-Net inference
    predictor = AIRISPredictor(checkpoint_path=args.checkpoint)
    res = predictor.predict(img)

    # 3. Print Routing Weights
    w = res["routing_weights"]
    print("\n" + "=" * 40)
    print("AIRIS-Net Expert Routing Weights:")
    print(f"  Local Expert:     {w[0]:.4f} ({w[0]*100:.1f}%)")
    print(f"  Global Expert:    {w[1]:.4f} ({w[1]*100:.1f}%)")
    print(f"  Frequency Expert: {w[2]:.4f} ({w[2]*100:.1f}%)")
    print("=" * 40)

    # 4. Save Outputs
    base_name = Path(args.input).stem
    save_image(res["restored"], out_dir / "restored.png")
    save_image(res["mask"], out_dir / "restoration_mask.png")
    save_image(res["reliability"], out_dir / "reliability_map.png")

    weights_txt = (
        f"Local Expert: {w[0]:.4f}\n"
        f"Global Expert: {w[1]:.4f}\n"
        f"Frequency Expert: {w[2]:.4f}\n"
    )
    with open(out_dir / "routing_weights.txt", "w") as f:
        f.write(weights_txt)

    print(f"\n[AIRIS-Net] Outputs saved to {out_dir.resolve()}:")
    print(f"  - {out_dir / 'restored.png'}")
    print(f"  - {out_dir / 'restoration_mask.png'}")
    print(f"  - {out_dir / 'reliability_map.png'}")
    print(f"  - {out_dir / 'routing_weights.txt'}")


if __name__ == "__main__":
    main()
