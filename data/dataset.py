import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from data.degradation import RandomDegradationPipeline
from utils.image_utils import load_image


class IndustrialRestorationDataset(Dataset):
    """
    Paired Industrial/Semiconductor Restoration Dataset.
    Loads clean images, crops/augments, and generates synthetic degradations on the fly.
    """
    def __init__(
        self,
        clean_dir: Union[str, Path],
        patch_size: int = 128,
        is_train: bool = True,
        grayscale: bool = True,
        degradation_mode: str = "random",
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.is_train = is_train
        self.grayscale = grayscale
        self.degradation_mode = degradation_mode
        self.degradation_pipeline = RandomDegradationPipeline()

        # Collect all image and npy files
        extensions = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        self.image_paths: List[Path] = []
        if self.clean_dir.exists():
            for ext in extensions:
                self.image_paths.extend(list(self.clean_dir.glob(ext)))
                self.image_paths.extend(list(self.clean_dir.glob(f"**/{ext}")))

        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        # Filter out macOS artifacts if any
        self.image_paths = [p for p in self.image_paths if not p.name.startswith("._") and "__MACOSX" not in str(p)]

        if max_samples and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]

        print(f"[Dataset] Loaded {len(self.image_paths)} images from {self.clean_dir} (is_train={is_train})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            # Resize if smaller than patch size
            target_h = max(h, self.patch_size)
            target_w = max(w, self.patch_size)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        if not self.is_train:
            # Center crop for validation
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2
        else:
            # Random crop for training
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)

        return img[top:top + self.patch_size, left:left + self.patch_size]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return img

        # Random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img)

        # Random vertical flip
        if random.random() < 0.5:
            img = np.flipud(img)

        # Random 90, 180, 270 degree rotation
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            img = np.rot90(img, rot_k)

        return np.ascontiguousarray(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        clean_img = load_image(path, grayscale=self.grayscale)

        # 1. Random / center crop
        clean_patch = self._random_crop(clean_img)

        # 2. Spatial augmentations
        clean_patch = self._augment(clean_patch)

        # 3. Dynamic synthetic degradation
        degraded_patch, meta = self.degradation_pipeline(clean_patch, degradation_type=self.degradation_mode)

        # 4. Convert to PyTorch tensors (C, H, W)
        if clean_patch.ndim == 2:
            clean_tensor = torch.from_numpy(clean_patch).unsqueeze(0).float()
            degraded_tensor = torch.from_numpy(degraded_patch).unsqueeze(0).float()
        else:
            clean_tensor = torch.from_numpy(np.transpose(clean_patch, (2, 0, 1))).float()
            degraded_tensor = torch.from_numpy(np.transpose(degraded_patch, (2, 0, 1))).float()

        return {
            "degraded": degraded_tensor,
            "clean": clean_tensor,
            "metadata": meta,
            "filename": path.name
        }


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    degraded = torch.stack([b["degraded"] for b in batch], dim=0)
    clean = torch.stack([b["clean"] for b in batch], dim=0)
    metadata = [b["metadata"] for b in batch]
    filenames = [b["filename"] for b in batch]
    return {
        "degraded": degraded,
        "clean": clean,
        "metadata": metadata,
        "filename": filenames
    }


def create_dataloader(
    clean_dir: Union[str, Path],
    batch_size: int = 8,
    patch_size: int = 128,
    is_train: bool = True,
    grayscale: bool = True,
    num_workers: int = 0,
    degradation_mode: str = "random",
    max_samples: Optional[int] = None
) -> DataLoader:
    """
    Factory function to build DataLoader.
    """
    dataset = IndustrialRestorationDataset(
        clean_dir=clean_dir,
        patch_size=patch_size,
        is_train=is_train,
        grayscale=grayscale,
        degradation_mode=degradation_mode,
        max_samples=max_samples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=is_train,
        collate_fn=custom_collate_fn
    )


def prepare_dataset_splits(
    source_dir: Union[str, Path] = "train/train/GT",
    base_data_dir: Union[str, Path] = "data",
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[Path, Path, Path]:
    """
    Create standard data/train/clean, data/val/clean, data/test/clean directories
    and populate with train/train/GT semiconductor files.
    """
    source_path = Path(source_dir)
    base_path = Path(base_data_dir)
    
    train_clean = base_path / "train" / "clean"
    val_clean = base_path / "val" / "clean"
    test_clean = base_path / "test" / "clean"

    train_clean.mkdir(parents=True, exist_ok=True)
    val_clean.mkdir(parents=True, exist_ok=True)
    test_clean.mkdir(parents=True, exist_ok=True)

    # Check if already populated
    if len(list(train_clean.glob("*.npy"))) > 50:
        print("[Data Split] Dataset splits already populated.")
        return train_clean, val_clean, test_clean

    files = sorted(list(source_path.glob("*.npy")))
    if not files:
        print(f"[Data Split] Warning: No files found in {source_path}")
        return train_clean, val_clean, test_clean

    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    import shutil
    print(f"[Data Split] Copying/Linking {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images...")

    for f in train_files:
        dest = train_clean / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    for f in val_files:
        dest = val_clean / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    for f in test_files:
        dest = test_clean / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    print("[Data Split] Finished organizing data directory.")
    return train_clean, val_clean, test_clean
