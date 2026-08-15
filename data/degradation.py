import random
import math
import io
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, List, Optional, Union


class RandomDegradationPipeline:
    """
    Synthetic degradation generator for industrial/semiconductor image restoration.
    All inputs and outputs are numpy arrays in range [0.0, 1.0], float32.
    Supports individual and combined degradations:
      - Gaussian Noise (additive)
      - Speckle Noise (multiplicative)
      - Defocus & Motion Blur
      - Contrast & Brightness Variations
      - Uneven Illumination Fields
      - JPEG Compression
      - Spatial Resolution Reduction / Super-Resolution Downsampling
      - Compound & Combined Degradations
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def apply_gaussian_noise(
        self,
        img: np.ndarray,
        sigma: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Add additive Gaussian noise.
        sigma: on [0, 1] scale (or > 1 on 0-255 scale). Default: 5-50/255 (0.02 - 0.20).
        """
        if sigma is None:
            sigma = random.uniform(5.0, 50.0) / 255.0
        else:
            sigma = sigma / 255.0 if sigma > 1.0 else sigma

        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        degraded = np.clip(img + noise, 0.0, 1.0)
        return degraded, {"type": "gaussian_noise", "sigma": float(sigma * 255.0)}

    def apply_speckle_noise(
        self,
        img: np.ndarray,
        variance: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply multiplicative speckle noise:
          degraded = clean + clean * noise
        where noise is drawn from Normal(0, variance).
        variance: default [0.02, 0.20].
        """
        if variance is None:
            variance = random.uniform(0.02, 0.20)

        std = math.sqrt(max(1e-6, variance))
        noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
        degraded = img + img * noise
        degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)
        return degraded, {"type": "speckle_noise", "variance": float(variance)}

    def apply_gaussian_blur(
        self,
        img: np.ndarray,
        ksize: Optional[int] = None,
        sigma: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Gaussian blur with kernel size in {3, 5, 7, 9} and sigma in [0.5, 3.0].
        """
        if ksize is None:
            ksize = random.choice([3, 5, 7, 9])
        if sigma is None:
            sigma = random.uniform(0.5, 3.0)

        degraded = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), {
            "type": "gaussian_blur", "ksize": ksize, "sigma": float(sigma)
        }

    def apply_motion_blur(
        self,
        img: np.ndarray,
        kernel_size: Optional[int] = None,
        angle: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Motion blur with random kernel length (3 to 15) and random angle (0 to 360 degrees).
        """
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7, 9, 11, 13, 15])
        if angle is None:
            angle = random.uniform(0, 360)

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        rad = np.deg2rad(angle)
        cos_val = np.cos(rad)
        sin_val = np.sin(rad)

        for i in range(kernel_size):
            offset = i - center
            r = int(round(center + offset * sin_val))
            c = int(round(center + offset * cos_val))
            if 0 <= r < kernel_size and 0 <= c < kernel_size:
                kernel[r, c] = 1.0

        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[center, center] = 1.0

        degraded = cv2.filter2D(img, -1, kernel)
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), {
            "type": "motion_blur", "ksize": kernel_size, "angle": float(angle)
        }

    def apply_contrast_degradation(
        self,
        img: np.ndarray,
        factor: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adjust contrast by scaling around mean value. Factor in [0.4, 0.9].
        """
        if factor is None:
            factor = random.uniform(0.4, 0.9)

        mean = np.mean(img)
        degraded = mean + factor * (img - mean)
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), {
            "type": "contrast", "factor": float(factor)
        }

    def apply_brightness_variation(
        self,
        img: np.ndarray,
        factor: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adjust overall brightness by scaling factor in [0.6, 1.4].
        """
        if factor is None:
            factor = random.uniform(0.6, 1.4)

        degraded = img * factor
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), {
            "type": "brightness", "factor": float(factor)
        }

    def apply_uneven_illumination(
        self,
        img: np.ndarray,
        strength: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply a smooth low-frequency 2D illumination gradient across the image.
        """
        if strength is None:
            strength = random.uniform(0.3, 0.7)

        h, w = img.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        a, b, c = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        gradient = a * xx + b * yy + c * (xx ** 2 + yy ** 2)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
        illumination = 1.0 + strength * (gradient - 0.5)

        if img.ndim == 3 and img.shape[2] == 3:
            illumination = illumination[:, :, np.newaxis]

        degraded = img * illumination
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), {
            "type": "uneven_illumination", "strength": float(strength)
        }

    def apply_jpeg_compression(
        self,
        img: np.ndarray,
        quality: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply JPEG compression artifacts with quality in [30, 95].
        """
        if quality is None:
            quality = random.randint(30, 95)

        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        is_gray = (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1))

        if is_gray:
            pil_img = Image.fromarray(np.squeeze(img_uint8), mode='L')
        else:
            pil_img = Image.fromarray(img_uint8, mode='RGB')

        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_pil = Image.open(buffer)
        degraded = np.array(compressed_pil, dtype=np.float32) / 255.0

        if is_gray and img.ndim == 3:
            degraded = degraded[:, :, np.newaxis]

        return np.clip(degraded, 0.0, 1.0).astype(np.float32), {
            "type": "jpeg_compression", "quality": quality
        }

    def apply_resolution_degradation(
        self,
        img: np.ndarray,
        scale_factor: Optional[float] = 2.0,
        keep_dim: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Downsample image by scale_factor.
        If keep_dim is True: downsamples and then bicubically upsamples back to original resolution.
        If keep_dim is False: returns the low-resolution image (H/scale, W/scale).
        """
        if scale_factor is None:
            scale_factor = 2.0

        h, w = img.shape[:2]
        low_h = max(4, int(round(h / scale_factor)))
        low_w = max(4, int(round(w / scale_factor)))

        down = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)

        if keep_dim:
            degraded = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            degraded = down

        if img.ndim == 3 and degraded.ndim == 2:
            degraded = degraded[:, :, np.newaxis]

        return np.clip(degraded, 0.0, 1.0).astype(np.float32), {
            "type": "resolution_degradation",
            "scale_factor": float(scale_factor),
            "keep_dim": keep_dim
        }

    def apply_combined_degradation(
        self,
        img: np.ndarray,
        mode: str = "gaussian_downsample",
        scale_factor: float = 2.0,
        keep_dim: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply designated combination degradations:
          - 'gaussian_downsample': Gaussian noise + downsampling
          - 'speckle_downsample': Speckle noise + downsampling
          - 'gaussian_speckle': Gaussian noise + Speckle noise
          - 'gaussian_speckle_downsample': Gaussian noise + Speckle noise + downsampling
        """
        curr = img.copy()
        meta_stages = []

        if mode == "gaussian_downsample":
            curr, m1 = self.apply_gaussian_noise(curr)
            curr, m2 = self.apply_resolution_degradation(curr, scale_factor=scale_factor, keep_dim=keep_dim)
            meta_stages = [m1, m2]
        elif mode == "speckle_downsample":
            curr, m1 = self.apply_speckle_noise(curr)
            curr, m2 = self.apply_resolution_degradation(curr, scale_factor=scale_factor, keep_dim=keep_dim)
            meta_stages = [m1, m2]
        elif mode == "gaussian_speckle":
            curr, m1 = self.apply_gaussian_noise(curr)
            curr, m2 = self.apply_speckle_noise(curr)
            meta_stages = [m1, m2]
        elif mode == "gaussian_speckle_downsample":
            curr, m1 = self.apply_gaussian_noise(curr)
            curr, m2 = self.apply_speckle_noise(curr)
            curr, m3 = self.apply_resolution_degradation(curr, scale_factor=scale_factor, keep_dim=keep_dim)
            meta_stages = [m1, m2, m3]
        else:
            return self.apply_mixed_degradation(img)

        return curr, {"type": f"combined_{mode}", "stages": meta_stages}

    def apply_mixed_degradation(
        self,
        img: np.ndarray,
        num_degradations: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Randomly apply 1 to 4 degradation types sequentially.
        """
        if num_degradations is None:
            num_degradations = random.randint(1, 4)

        available_ops = [
            self.apply_gaussian_noise,
            self.apply_speckle_noise,
            self.apply_gaussian_blur,
            self.apply_motion_blur,
            self.apply_contrast_degradation,
            self.apply_brightness_variation,
            self.apply_uneven_illumination,
            self.apply_jpeg_compression,
            self.apply_resolution_degradation,
        ]

        selected_ops = random.sample(available_ops, k=min(num_degradations, len(available_ops)))
        curr_img = img.copy()
        applied_metadata = []

        for op in selected_ops:
            curr_img, meta = op(curr_img)
            applied_metadata.append(meta)

        return curr_img, {"type": "mixed_degradation", "stages": applied_metadata}

    def __call__(
        self,
        clean_img: np.ndarray,
        degradation_type: str = "random",
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply requested degradation to clean image.
        Options:
          'none', 'noise', 'gaussian', 'speckle', 'blur', 'motion_blur',
          'contrast', 'brightness', 'illumination', 'jpeg', 'resolution',
          'gaussian_downsample', 'speckle_downsample', 'gaussian_speckle',
          'gaussian_speckle_downsample', 'mixed', 'random'.
        """
        clean_img = np.clip(clean_img.astype(np.float32), 0.0, 1.0)
        mode = degradation_type.lower().strip()

        if mode in ("none", "clean"):
            return clean_img.copy(), {"type": "none"}
        elif mode in ("noise", "gaussian", "gaussian_noise"):
            return self.apply_gaussian_noise(clean_img, sigma=kwargs.get("sigma"))
        elif mode in ("speckle", "speckle_noise"):
            return self.apply_speckle_noise(clean_img, variance=kwargs.get("variance"))
        elif mode in ("blur", "gaussian_blur"):
            return self.apply_gaussian_blur(clean_img, ksize=kwargs.get("ksize"), sigma=kwargs.get("sigma"))
        elif mode == "motion_blur":
            return self.apply_motion_blur(clean_img, kernel_size=kwargs.get("kernel_size"), angle=kwargs.get("angle"))
        elif mode == "contrast":
            return self.apply_contrast_degradation(clean_img, factor=kwargs.get("factor"))
        elif mode == "brightness":
            return self.apply_brightness_variation(clean_img, factor=kwargs.get("factor"))
        elif mode == "illumination":
            return self.apply_uneven_illumination(clean_img, strength=kwargs.get("strength"))
        elif mode in ("jpeg", "compression"):
            return self.apply_jpeg_compression(clean_img, quality=kwargs.get("quality"))
        elif mode in ("resolution", "downsample", "sr_x2"):
            return self.apply_resolution_degradation(
                clean_img,
                scale_factor=kwargs.get("scale_factor", 2.0),
                keep_dim=kwargs.get("keep_dim", True)
            )
        elif mode in ("gaussian_downsample", "speckle_downsample", "gaussian_speckle", "gaussian_speckle_downsample"):
            return self.apply_combined_degradation(
                clean_img,
                mode=mode,
                scale_factor=kwargs.get("scale_factor", 2.0),
                keep_dim=kwargs.get("keep_dim", True)
            )
        elif mode == "mixed":
            return self.apply_mixed_degradation(clean_img, num_degradations=kwargs.get("num_degradations"))
        elif mode == "random":
            choices = [
                "noise", "speckle", "blur", "motion_blur", "contrast",
                "illumination", "jpeg", "resolution", "gaussian_downsample",
                "speckle_downsample", "gaussian_speckle", "mixed"
            ]
            chosen = random.choice(choices)
            return self(clean_img, degradation_type=chosen, **kwargs)
        else:
            return self.apply_mixed_degradation(clean_img)
