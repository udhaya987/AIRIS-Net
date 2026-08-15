import os
import random
import shutil
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
    Synthetic Industrial/Semiconductor Restoration Dataset.
    Loads clean images, crops/augments, and generates synthetic degradations on the fly.
    Supports both same-resolution (scale_factor=1) and super-resolution (scale_factor=2).
    """
    def __init__(
        self,
        clean_dir: Union[str, Path],
        patch_size: Optional[int] = 128,
        scale_factor: int = 1,
        is_train: bool = True,
        grayscale: bool = True,
        degradation_mode: str = "random",
        max_samples: Optional[int] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.grayscale = grayscale
        self.degradation_mode = degradation_mode
        self.degradation_pipeline = RandomDegradationPipeline(seed=seed)

        # Collect all image and npy files
        extensions = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        self.image_paths: List[Path] = []
        if self.clean_dir.exists():
            for ext in extensions:
                self.image_paths.extend(list(self.clean_dir.glob(ext)))
                self.image_paths.extend(list(self.clean_dir.glob(f"**/{ext}")))

        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        # Filter out macOS and hidden artifacts
        self.image_paths = [
            p for p in self.image_paths
            if not p.name.startswith("._") and "__MACOSX" not in str(p) and not p.name.startswith(".")
        ]

        if max_samples and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]

        # Fast In-Memory RAM Caching
        self.cached_images: List[np.ndarray] = []
        for p in self.image_paths:
            try:
                self.cached_images.append(load_image(p, grayscale=self.grayscale))
            except Exception as e:
                print(f"[Dataset] Warning: Failed to load {p}: {e}")

        print(f"[SyntheticDataset] Loaded {len(self.cached_images)} images from {self.clean_dir} (is_train={is_train}, scale={scale_factor})")

    def __len__(self) -> int:
        return len(self.cached_images)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        if self.patch_size is None:
            return img
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            target_h = max(h, self.patch_size)
            target_w = max(w, self.patch_size)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        if not self.is_train:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2
        else:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)

        return img[top:top + self.patch_size, left:left + self.patch_size]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return img

        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            img = np.flipud(img)

        rot_k = random.randint(0, 3)
        if rot_k > 0:
            img = np.rot90(img, rot_k)

        return np.ascontiguousarray(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        clean_img = self.cached_images[idx]

        # 1. Random / center crop clean target
        clean_patch = self._random_crop(clean_img)

        # 2. Spatial augmentations
        clean_patch = self._augment(clean_patch)

        # 3. Dynamic synthetic degradation
        degraded_patch, meta = self.degradation_pipeline(
            clean_patch,
            degradation_type=self.degradation_mode
        )

        # 4. Handle super-resolution downsampling if scale_factor > 1
        if self.scale_factor > 1:
            h, w = degraded_patch.shape[:2]
            low_h = max(4, h // self.scale_factor)
            low_w = max(4, w // self.scale_factor)
            degraded_patch = cv2.resize(degraded_patch, (low_w, low_h), interpolation=cv2.INTER_AREA)

        # 5. Convert to PyTorch tensors (C, H, W)
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


class RealPairedRestorationDataset(Dataset):
    """
    Real Paired Semiconductor Restoration Dataset.
    Loads real degraded Low-Resolution images (e.g. 128x128) and real clean Ground-Truth images (e.g. 256x256).
    Supports joint synchronized augmentation (flips, 90-deg rotations) and cropping.
    """
    def __init__(
        self,
        noisy_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        patch_size: Optional[int] = None,
        scale_factor: int = 2,
        is_train: bool = True,
        grayscale: bool = True,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.grayscale = grayscale
        if seed is not None:
            random.seed(seed)

        # Discover matching pairs
        extensions = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        noisy_map: Dict[str, Path] = {}
        gt_map: Dict[str, Path] = {}

        if self.noisy_dir.exists():
            for ext in extensions:
                for p in self.noisy_dir.glob(ext):
                    if not p.name.startswith("._") and "__MACOSX" not in str(p) and not p.name.startswith("."):
                        noisy_map[p.name] = p
                for p in self.noisy_dir.glob(f"**/{ext}"):
                    if not p.name.startswith("._") and "__MACOSX" not in str(p) and not p.name.startswith("."):
                        noisy_map[p.name] = p

        if self.gt_dir.exists():
            for ext in extensions:
                for p in self.gt_dir.glob(ext):
                    if not p.name.startswith("._") and "__MACOSX" not in str(p) and not p.name.startswith("."):
                        gt_map[p.name] = p
                for p in self.gt_dir.glob(f"**/{ext}"):
                    if not p.name.startswith("._") and "__MACOSX" not in str(p) and not p.name.startswith("."):
                        gt_map[p.name] = p

        common_keys = sorted(list(set(noisy_map.keys()) & set(gt_map.keys())))
        if max_samples and max_samples < len(common_keys):
            common_keys = common_keys[:max_samples]

        self.paired_paths: List[Tuple[Path, Path]] = [
            (noisy_map[k], gt_map[k]) for k in common_keys
        ]

        print(f"[RealPairedDataset] Loaded {len(self.paired_paths)} verified pairs from {self.noisy_dir} and {self.gt_dir} (is_train={is_train}, scale={scale_factor})")

    def __len__(self) -> int:
        return len(self.paired_paths)

    def _joint_augment(self, noisy: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_train:
            return noisy, gt

        if random.random() < 0.5:
            noisy = np.fliplr(noisy)
            gt = np.fliplr(gt)
        if random.random() < 0.5:
            noisy = np.flipud(noisy)
            gt = np.flipud(gt)

        rot_k = random.randint(0, 3)
        if rot_k > 0:
            noisy = np.rot90(noisy, rot_k)
            gt = np.rot90(gt, rot_k)

        return np.ascontiguousarray(noisy), np.ascontiguousarray(gt)

    def _joint_crop(self, noisy: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return noisy, gt

        h_noisy, w_noisy = noisy.shape[:2]
        if h_noisy <= self.patch_size or w_noisy <= self.patch_size:
            return noisy, gt

        target_gt_h = self.patch_size * self.scale_factor
        target_gt_w = self.patch_size * self.scale_factor

        if not self.is_train:
            top_n = (h_noisy - self.patch_size) // 2
            left_n = (w_noisy - self.patch_size) // 2
        else:
            top_n = random.randint(0, h_noisy - self.patch_size)
            left_n = random.randint(0, w_noisy - self.patch_size)

        top_gt = top_n * self.scale_factor
        left_gt = left_n * self.scale_factor

        crop_noisy = noisy[top_n:top_n + self.patch_size, left_n:left_n + self.patch_size]
        crop_gt = gt[top_gt:top_gt + target_gt_h, left_gt:left_gt + target_gt_w]

        return crop_noisy, crop_gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        noisy_path, gt_path = self.paired_paths[idx]

        noisy_img = load_image(noisy_path, grayscale=self.grayscale)
        gt_img = load_image(gt_path, grayscale=self.grayscale)

        # Joint spatial crop if configured
        noisy_patch, gt_patch = self._joint_crop(noisy_img, gt_img)

        # Joint augmentation
        noisy_patch, gt_patch = self._joint_augment(noisy_patch, gt_patch)

        # Convert to PyTorch tensors (C, H, W)
        if noisy_patch.ndim == 2:
            noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0).float()
            gt_tensor = torch.from_numpy(gt_patch).unsqueeze(0).float()
        else:
            noisy_tensor = torch.from_numpy(np.transpose(noisy_patch, (2, 0, 1))).float()
            gt_tensor = torch.from_numpy(np.transpose(gt_patch, (2, 0, 1))).float()

        # Ensure range [0, 1]
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        gt_tensor = torch.clamp(gt_tensor, 0.0, 1.0)

        meta = {
            "type": "real_semiconductor_noisy_lr",
            "scale": self.scale_factor,
            "filename": noisy_path.name
        }

        return {
            "degraded": noisy_tensor,
            "clean": gt_tensor,
            "metadata": meta,
            "filename": noisy_path.name
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
    clean_dir: Optional[Union[str, Path]] = None,
    noisy_dir: Optional[Union[str, Path]] = None,
    gt_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    patch_size: Optional[int] = 128,
    scale_factor: int = 1,
    is_train: bool = True,
    grayscale: bool = True,
    num_workers: int = 0,
    degradation_mode: str = "random",
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    is_paired: bool = False
) -> DataLoader:
    """
    Factory function to build DataLoader for either Real Paired dataset or Synthetic degradation dataset.
    """
    if is_paired or (noisy_dir is not None and gt_dir is not None):
        dataset = RealPairedRestorationDataset(
            noisy_dir=noisy_dir or clean_dir,
            gt_dir=gt_dir,
            patch_size=patch_size,
            scale_factor=scale_factor,
            is_train=is_train,
            grayscale=grayscale,
            max_samples=max_samples,
            seed=seed
        )
    else:
        dataset = IndustrialRestorationDataset(
            clean_dir=clean_dir or gt_dir,
            patch_size=patch_size or 128,
            scale_factor=scale_factor,
            is_train=is_train,
            grayscale=grayscale,
            degradation_mode=degradation_mode,
            max_samples=max_samples,
            seed=seed
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=is_train and len(dataset) >= batch_size,
        collate_fn=custom_collate_fn
    )


def prepare_real_paired_splits(
    noisy_source_dir: Union[str, Path] = "train/train/NoisyLR",
    gt_source_dir: Union[str, Path] = "train/train/GT",
    base_data_dir: Union[str, Path] = "data",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    force: bool = False
) -> Tuple[Tuple[Path, Path], Tuple[Path, Path], Tuple[Path, Path]]:
    """
    Create standard data/real_train, data/real_val, data/real_test paired splits
    with 0 leakage across splits.
    """
    noisy_path = Path(noisy_source_dir)
    gt_path = Path(gt_source_dir)
    base_path = Path(base_data_dir)

    tr_noisy = base_path / "real_train" / "NoisyLR"
    tr_gt = base_path / "real_train" / "GT"
    va_noisy = base_path / "real_val" / "NoisyLR"
    va_gt = base_path / "real_val" / "GT"
    te_noisy = base_path / "real_test" / "NoisyLR"
    te_gt = base_path / "real_test" / "GT"

    for d in [tr_noisy, tr_gt, va_noisy, va_gt, te_noisy, te_gt]:
        d.mkdir(parents=True, exist_ok=True)

    tr_existing = list(tr_noisy.glob("*.npy"))
    if len(tr_existing) > 50 and not force:
        va_existing = list(va_noisy.glob("*.npy"))
        te_existing = list(te_noisy.glob("*.npy"))
        set_tr = set(p.name for p in tr_existing)
        set_va = set(p.name for p in va_existing)
        set_te = set(p.name for p in te_existing)
        if len(set_tr & set_va) == 0 and len(set_tr & set_te) == 0 and len(set_va & set_te) == 0:
            print(f"[RealPairedSplit] Using existing verified zero-leakage splits ({len(set_tr)} train, {len(set_va)} val, {len(set_te)} test)")
            return (tr_noisy, tr_gt), (va_noisy, va_gt), (te_noisy, te_gt)

    # Discover source files
    noisy_files = {p.name: p for p in noisy_path.glob("*.npy")}
    gt_files = {p.name: p for p in gt_path.glob("*.npy")}

    common_names = sorted(list(set(noisy_files.keys()) & set(gt_files.keys())))
    if not common_names:
        print(f"[Warning] No paired files found between {noisy_path} and {gt_path}")
        return (tr_noisy, tr_gt), (va_noisy, va_gt), (te_noisy, te_gt)

    # Clean existing
    for d in [tr_noisy, tr_gt, va_noisy, va_gt, te_noisy, te_gt]:
        for f in d.glob("*.*"):
            if f.is_file() and f.name != ".gitkeep":
                f.unlink()

    # Deterministic shuffle
    random.seed(seed)
    shuffled_names = common_names.copy()
    random.shuffle(shuffled_names)

    n_total = len(shuffled_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_names = shuffled_names[:n_train]
    val_names = shuffled_names[n_train:n_train + n_val]
    test_names = shuffled_names[n_train + n_val:]

    for name in train_names:
        shutil.copy2(noisy_files[name], tr_noisy / name)
        shutil.copy2(gt_files[name], tr_gt / name)

    for name in val_names:
        shutil.copy2(noisy_files[name], va_noisy / name)
        shutil.copy2(gt_files[name], va_gt / name)

    for name in test_names:
        shutil.copy2(noisy_files[name], te_noisy / name)
        shutil.copy2(gt_files[name], te_gt / name)

    print(f"[RealPairedSplit] Partitioned {n_total} pairs into:\n  Train: {len(train_names)} | Val: {len(val_names)} | Test: {len(test_names)}")
    return (tr_noisy, tr_gt), (va_noisy, va_gt), (te_noisy, te_gt)


def prepare_dataset_splits(
    source_dir: Union[str, Path] = "train/train/GT",
    base_data_dir: Union[str, Path] = "data",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    force: bool = False
) -> Tuple[Path, Path, Path]:
    """
    Create standard data/train/clean, data/val/clean, data/test/clean directories
    and populate with semiconductor GT files without train/val/test leakage.
    """
    source_path = Path(source_dir)
    base_path = Path(base_data_dir)
    
    train_clean = base_path / "train" / "clean"
    val_clean = base_path / "val" / "clean"
    test_clean = base_path / "test" / "clean"

    train_clean.mkdir(parents=True, exist_ok=True)
    val_clean.mkdir(parents=True, exist_ok=True)
    test_clean.mkdir(parents=True, exist_ok=True)

    # If already partitioned and not forcing, check for leakage
    train_existing = list(train_clean.glob("*.npy")) + list(train_clean.glob("*.png"))
    if len(train_existing) > 10 and not force:
        val_existing = list(val_clean.glob("*.npy")) + list(val_clean.glob("*.png"))
        test_existing = list(test_clean.glob("*.npy")) + list(test_clean.glob("*.png"))
        set_tr = set(p.name for p in train_existing)
        set_va = set(p.name for p in val_existing)
        set_te = set(p.name for p in test_existing)
        if len(set_tr & set_va) == 0 and len(set_tr & set_te) == 0 and len(set_va & set_te) == 0:
            return train_clean, val_clean, test_clean

    # Look for GT files in potential source locations
    possible_sources = [
        source_path,
        Path("train/GT"),
        Path("train/train/GT"),
        Path("GT")
    ]
    files = []
    for s in possible_sources:
        if s.exists():
            files = sorted(list(s.glob("*.npy")) + list(s.glob("*.png")))
            if files:
                break

    if not files:
        return train_clean, val_clean, test_clean

    # Clean existing destination directories to prevent leakage
    for d in [train_clean, val_clean, test_clean]:
        for existing_file in d.glob("*.*"):
            if existing_file.is_file() and existing_file.name != ".gitkeep":
                existing_file.unlink()

    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    for f in train_files:
        shutil.copy2(f, train_clean / f.name)

    for f in val_files:
        shutil.copy2(f, val_clean / f.name)

    for f in test_files:
        shutil.copy2(f, test_clean / f.name)

    return train_clean, val_clean, test_clean
