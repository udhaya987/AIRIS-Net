import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import prepare_dataset_splits


def main():
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits from raw ground truth images")
    parser.add_argument("--source_dir", type=str, default="train/train/GT", help="Source directory containing raw GT files")
    parser.add_argument("--base_data_dir", type=str, default="data", help="Base target data directory")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    args = parser.parse_args()

    train_p, val_p, test_p = prepare_dataset_splits(
        source_dir=args.source_dir,
        base_data_dir=args.base_data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    print(f"Data split completed:\n  Train: {train_p}\n  Val:   {val_p}\n  Test:  {test_p}")


if __name__ == "__main__":
    main()
