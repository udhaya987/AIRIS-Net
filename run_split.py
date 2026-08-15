import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import prepare_dataset_splits, prepare_real_paired_splits


def main():
    parser = argparse.ArgumentParser(description="Prepare zero-leakage train/val/test splits for AIRIS-Net")
    parser.add_argument("--mode", type=str, default="paired", choices=["paired", "clean"], help="Dataset split mode ('paired' for real NoisyLR/GT pairs, 'clean' for single GT clean directory)")
    parser.add_argument("--noisy_dir", type=str, default="train/train/NoisyLR", help="Source directory containing real NoisyLR files")
    parser.add_argument("--source_dir", "--gt_dir", dest="source_dir", type=str, default="train/train/GT", help="Source directory containing GT files")
    parser.add_argument("--base_data_dir", type=str, default="data", help="Base target data directory")
    parser.add_argument("--train_ratio", type=float, default=0.80, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument("--force", action="store_true", help="Force re-partitioning to eliminate data leakage")
    args = parser.parse_args()

    if args.mode == "paired":
        (tr_noisy, tr_gt), (va_noisy, va_gt), (te_noisy, te_gt) = prepare_real_paired_splits(
            noisy_source_dir=args.noisy_dir,
            gt_source_dir=args.source_dir,
            base_data_dir=args.base_data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force=args.force
        )
        print(f"Real Paired Split Completed Successfully (Zero Leakage Verified):")
        print(f"  Train: {tr_noisy} & {tr_gt}")
        print(f"  Val:   {va_noisy} & {va_gt}")
        print(f"  Test:  {te_noisy} & {te_gt}")
    else:
        train_p, val_p, test_p = prepare_dataset_splits(
            source_dir=args.source_dir,
            base_data_dir=args.base_data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force=args.force
        )
        print(f"Clean Dataset Split Completed:\n  Train: {train_p}\n  Val:   {val_p}\n  Test:  {test_p}")


if __name__ == "__main__":
    main()
