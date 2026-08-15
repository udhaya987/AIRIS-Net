import pytest
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import IndustrialRestorationDataset, create_dataloader


def test_dataset_loading(tmp_path):
    """Verify IndustrialRestorationDataset loads and yields paired tensors."""
    data_dir = tmp_path / "clean"
    data_dir.mkdir(parents=True)
    for i in range(4):
        arr = np.random.rand(128, 128).astype(np.float32)
        np.save(str(data_dir / f"img_{i:03d}.npy"), arr)

    dataset = IndustrialRestorationDataset(
        clean_dir=data_dir,
        patch_size=64,
        is_train=True,
        scale_factor=1,
        degradation_mode="random",
        seed=42
    )

    assert len(dataset) == 4
    sample = dataset[0]
    assert "clean" in sample
    assert "degraded" in sample
    assert sample["clean"].shape == (1, 64, 64)
    assert sample["degraded"].shape == (1, 64, 64)


def test_dataloader_batching(tmp_path):
    """Verify DataLoader produces batches with proper tensor dimensions."""
    data_dir = tmp_path / "clean"
    data_dir.mkdir(parents=True)
    for i in range(4):
        arr = np.random.rand(128, 128).astype(np.float32)
        np.save(str(data_dir / f"img_{i:03d}.npy"), arr)

    loader = create_dataloader(
        clean_dir=data_dir,
        batch_size=2,
        patch_size=64,
        is_train=False,
        scale_factor=1
    )

    batch = next(iter(loader))
    assert batch["clean"].shape == (2, 1, 64, 64)
    assert batch["degraded"].shape == (2, 1, 64, 64)
