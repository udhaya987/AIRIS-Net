import os
import json
from pathlib import Path
import argparse
import numpy as np

from utils.image_utils import load_image
from utils.metrics import calculate_psnr, calculate_ssim
from utils.edge_utils import edge_consistency_score, compute_sobel_edges_np
from utils.frequency_utils import frequency_consistency_score, compute_fft_magnitude_np


def verify_restoration(
    restored_path: str,
    input_path: str,
    clean_path: str = None,
    output_json: str = None
) -> dict:
    """
    Perform post-restoration structural and frequency consistency verification.
    """
    restored = load_image(restored_path, grayscale=True)
    degraded = load_image(input_path, grayscale=True)

    # Edge consistency
    edge_consist_input = edge_consistency_score(restored, degraded)
    freq_consist_input = frequency_consistency_score(restored, degraded)

    report = {
        "edge_consistency_vs_input": edge_consist_input,
        "frequency_consistency_vs_input": freq_consist_input,
    }

    if clean_path and Path(clean_path).exists():
        clean = load_image(clean_path, grayscale=True)
        psnr_val = calculate_psnr(restored, clean)
        ssim_val = calculate_ssim(restored, clean)
        edge_consist_clean = edge_consistency_score(restored, clean)
        freq_consist_clean = frequency_consistency_score(restored, clean)

        report.update({
            "psnr": psnr_val,
            "ssim": ssim_val,
            "edge_consistency": edge_consist_clean,
            "frequency_consistency": freq_consist_clean,
        })
    else:
        report.update({
            "edge_consistency": edge_consist_input,
            "frequency_consistency": freq_consist_input,
            "ssim": float(calculate_ssim(restored, degraded))
        })

    report["disclaimer"] = "Metrics are mathematical consistency indicators and do not guarantee zero hallucination."

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Verification] Report saved to {out_p}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Consistency Verification for Restored Images")
    parser.add_argument("--restored", type=str, required=True, help="Path to restored image")
    parser.add_argument("--input", type=str, required=True, help="Path to degraded input image")
    parser.add_argument("--clean", type=str, default=None, help="Path to clean ground truth if available")
    parser.add_argument("--output", type=str, default="outputs/verification_report.json", help="Path to output JSON")
    args = parser.parse_args()

    report = verify_restoration(args.restored, args.input, args.clean, args.output)
    print("\n" + "=" * 50)
    print("AIRIS Consistency Verification Report:")
    print(json.dumps(report, indent=2))
    print("=" * 50)


if __name__ == "__main__":
    main()
