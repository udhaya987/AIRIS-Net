#!/usr/bin/env python3
"""
KLA / SEMICON Hackathon Inference Script for AIRIS-Net.

Usage:
    python kla_inference.py --input_dir PATH --output_dir PATH --checkpoint PATH [--device auto] [--scale 1]

Windows PowerShell Example:
    python kla_inference.py --input_dir data/test/clean --output_dir outputs/restored --checkpoint checkpoints/best_airis.pth
"""

import os
import sys
import time
from pathlib import Path
import argparse
from typing import List, Optional
import numpy as np
import cv2
import torch

from airis.model import AIRISNet
from utils.checkpoint import load_checkpoint
from utils.image_utils import load_image, save_image, to_tensor, to_numpy


def parse_args():
    parser = argparse.ArgumentParser(
        description="AIRIS-Net Hackathon Batch Inference Script for Industrial/Semiconductor Restoration"
    )
    parser.add_argument(
        "--input_dir", "--input-dir", "-i",
        type=str,
        required=True,
        help="Path to directory containing degraded test images (.npy, .png, .jpg, .bmp, .tiff)"
    )
    parser.add_argument(
        "--output_dir", "--output-dir", "-o",
        type=str,
        required=True,
        help="Path to output directory where restored images will be saved"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="checkpoints/best_airis.pth",
        help="Path to trained AIRIS-Net checkpoint (.pth)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computation device ('auto', 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=None,
        help="Super-resolution scale factor (1 for same-resolution denoising, 2 for 2x super-resolution). Defaults to checkpoint setting or 1."
    )
    return parser.parse_args()


def discover_images(input_dir: Path) -> List[Path]:
    """
    Discover all supported image files in input directory.
    """
    extensions = ("*.npy", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(input_dir.glob(ext)))
        image_paths.extend(list(input_dir.glob(ext.upper())))

    # Deduplicate and sort
    image_paths = sorted(list(set(image_paths)))
    
    # Filter out hidden files and macOS artifacts
    image_paths = [
        p for p in image_paths
        if not p.name.startswith("._") and "__MACOSX" not in str(p) and not p.name.startswith(".")
    ]
    return image_paths


def build_model(
    checkpoint_path: str,
    device: torch.device,
    scale_override: Optional[int] = None
) -> AIRISNet:
    """
    Instantiate AIRISNet and load weights from checkpoint.
    """
    chk_path = Path(checkpoint_path)
    if not chk_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{chk_path.resolve()}'.\n"
            f"Please download or train the model before running inference."
        )

    # First load checkpoint metadata to inspect model configuration
    checkpoint_data = torch.load(str(chk_path), map_location="cpu")
    
    cfg = {}
    if isinstance(checkpoint_data, dict):
        cfg = checkpoint_data.get("config", {}).get("model", {})
        chk_scale = checkpoint_data.get("scale_factor", cfg.get("scale_factor", 1))
    else:
        chk_scale = 1

    scale_factor = scale_override if scale_override is not None else chk_scale

    in_channels = cfg.get("in_channels", 1)
    base_channels = cfg.get("base_channels", 48)
    degradation_dim = cfg.get("degradation_dim", 64)
    use_local_expert = cfg.get("use_local_expert", True)
    use_global_expert = cfg.get("use_global_expert", True)
    use_frequency_expert = cfg.get("use_frequency_expert", True)
    use_adaptive_routing = cfg.get("use_adaptive_routing", True)
    use_integrity_mask = cfg.get("use_integrity_mask", True)

    model = AIRISNet(
        in_channels=in_channels,
        base_channels=base_channels,
        degradation_dim=degradation_dim,
        scale_factor=scale_factor,
        use_local_expert=use_local_expert,
        use_global_expert=use_global_expert,
        use_frequency_expert=use_frequency_expert,
        use_adaptive_routing=use_adaptive_routing,
        use_integrity_mask=use_integrity_mask
    )

    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    # 1. Resolve Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print("=" * 60)
    print("AIRIS-Net Industrial Image Restoration - KLA Inference")
    print(f"Device:             {device.type.upper()} ({device_name})")
    print(f"Input Directory:    {Path(args.input_dir).resolve()}")
    print(f"Output Directory:   {Path(args.output_dir).resolve()}")
    print(f"Checkpoint:         {Path(args.checkpoint).resolve()}")
    print("=" * 60)

    # 2. Discover Input Images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir.resolve()}")

    image_paths = discover_images(input_dir)
    if not image_paths:
        print(f"[Warning] No supported image files (.npy, .png, .jpg, .bmp, .tiff) found in {input_dir.resolve()}")
        return

    print(f"Found {len(image_paths)} images to restore.")

    # 3. Create Output Directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Load Model
    model = build_model(args.checkpoint, device=device, scale_override=args.scale)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded AIRIS-Net ({total_params:,} parameters, scale={model.scale_factor}x). Processing...")

    # 5. Process Images
    total_time = 0.0
    processed_count = 0

    for idx, img_path in enumerate(image_paths):
        # Load image normalized to [0, 1]
        try:
            is_npy = img_path.suffix.lower() == ".npy"
            # Keep grayscale default if 1-channel or npy
            img = load_image(img_path, grayscale=True)

            # Convert to tensor (1, C, H, W)
            tensor_in = to_tensor(img, device=device)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(tensor_in)
            t_elapsed = time.perf_counter() - t0
            total_time += t_elapsed

            # Extract restored image and clamp to [0.0, 1.0]
            restored_tensor = outputs["restored"].squeeze().cpu().clamp(0.0, 1.0)
            restored_np = restored_tensor.numpy().astype(np.float32)

            # Preserve exact filename and extension
            out_file = output_dir / img_path.name
            save_image(restored_np, out_file)
            processed_count += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(image_paths):
                print(
                    f"  [{idx+1:4d}/{len(image_paths):4d}] Processed {img_path.name} "
                    f"in {t_elapsed*1000.0:.1f} ms"
                )

        except Exception as e:
            print(f"  [Error] Failed to process {img_path.name}: {e}")

    # 6. Summary Report
    avg_latency_ms = (total_time / max(1, processed_count)) * 1000.0
    avg_latency_sec = total_time / max(1, processed_count)

    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY:")
    print(f"  Total Images Processed: {processed_count} / {len(image_paths)}")
    print(f"  Total Elapsed Time:     {total_time:.2f} s")
    print(f"  Average Latency/Image:  {avg_latency_ms:.2f} ms ({avg_latency_sec:.4f} s)")
    print(f"  Device Used:            {device.type.upper()} ({device_name})")
    print(f"  Output Directory:       {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
