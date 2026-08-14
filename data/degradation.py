import random
import math
import io
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, List, Optional


class RandomDegradationPipeline:
    """
    Synthetic degradation generator for industrial/semiconductor image restoration.
    All inputs and outputs are numpy arrays in range [0.0, 1.0], float32.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def apply_gaussian_noise(self, img: np.ndarray, sigma: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Add Gaussian noise with sigma in range 5-50 (on 0-255 scale) -> 0.02 - 0.2 (on 0-1 scale).
        """
        if sigma is None:
            sigma = random.uniform(5.0, 50.0) / 255.0
        else:
            sigma = sigma / 255.0 if sigma > 1.0 else sigma

        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        degraded = np.clip(img + noise, 0.0, 1.0)
        return degraded, {"type": "gaussian_noise", "sigma": float(sigma * 255.0)}

    def apply_gaussian_blur(self, img: np.ndarray, ksize: Optional[int] = None, sigma: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Gaussian blur with kernel size in {3, 5, 7, 9} and sigma in [0.5, 3.0].
        """
        if ksize is None:
            ksize = random.choice([3, 5, 7, 9])
        if sigma is None:
            sigma = random.uniform(0.5, 3.0)

        degraded = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return np.clip(degraded, 0.0, 1.0), {"type": "gaussian_blur", "ksize": ksize, "sigma": float(sigma)}

    def apply_motion_blur(self, img: np.ndarray, kernel_size: Optional[int] = None, angle: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Motion blur with random kernel length (3 to 15) and random angle (0 to 360 degrees).
        """
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7, 9, 11, 13, 15])
        if angle is None:
            angle = random.uniform(0, 360)

        # Create motion blur kernel
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
        return np.clip(degraded, 0.0, 1.0), {"type": "motion_blur", "ksize": kernel_size, "angle": float(angle)}

    def apply_contrast_degradation(self, img: np.ndarray, factor: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adjust contrast by scaling around mean value. Factor in [0.4, 1.0].
        """
        if factor is None:
            factor = random.uniform(0.4, 0.9)

        mean = np.mean(img)
        degraded = mean + factor * (img - mean)
        return np.clip(degraded, 0.0, 1.0), {"type": "contrast", "factor": float(factor)}

    def apply_brightness_variation(self, img: np.ndarray, factor: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adjust overall brightness by scaling factor in [0.6, 1.4].
        """
        if factor is None:
            factor = random.uniform(0.6, 1.4)

        degraded = img * factor
        return np.clip(degraded, 0.0, 1.0), {"type": "brightness", "factor": float(factor)}

    def apply_uneven_illumination(self, img: np.ndarray, strength: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply a smooth low-frequency 2D illumination gradient / field across the image.
        """
        if strength is None:
            strength = random.uniform(0.3, 0.7)

        h, w = img.shape[:2]
        # Generate smooth 2D gradient
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Random polynomial illumination field
        a, b, c = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        gradient = a * xx + b * yy + c * (xx**2 + yy**2)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
        # Shift to illumination multiplier (e.g. 0.5 to 1.5)
        illumination = 1.0 + strength * (gradient - 0.5)

        if img.ndim == 3 and img.shape[2] == 3:
            illumination = illumination[:, :, np.newaxis]

        degraded = img * illumination
        return np.clip(degraded, 0.0, 1.0), {"type": "uneven_illumination", "strength": float(strength)}

    def apply_jpeg_compression(self, img: np.ndarray, quality: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        return np.clip(degraded, 0.0, 1.0), {"type": "jpeg_compression", "quality": quality}

    def apply_resolution_degradation(self, img: np.ndarray, scale_factor: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Downsample image then upsample back to original resolution. Scale factor in [2.0, 4.0].
        """
        if scale_factor is None:
            scale_factor = random.uniform(2.0, 4.0)

        h, w = img.shape[:2]
        low_h, low_w = max(4, int(h / scale_factor)), max(4, int(w / scale_factor))
        down = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

        if img.ndim == 3 and degraded.ndim == 2:
            degraded = degraded[:, :, np.newaxis]

        return np.clip(degraded, 0.0, 1.0), {"type": "resolution_degradation", "scale_factor": float(scale_factor)}

    def apply_mixed_degradation(self, img: np.ndarray, num_degradations: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Randomly apply 1 to 4 degradation types sequentially.
        """
        if num_degradations is None:
            num_degradations = random.randint(1, 4)

        available_ops = [
            self.apply_gaussian_noise,
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

    def __call__(self, clean_img: np.ndarray, degradation_type: str = "random") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply requested degradation to clean image.
        Options: 'none', 'noise', 'blur', 'motion_blur', 'contrast', 'brightness', 'illumination', 'jpeg', 'resolution', 'mixed', 'random'.
        """
        clean_img = np.clip(clean_img.astype(np.float32), 0.0, 1.0)

        if degradation_type == "none":
            return clean_img.copy(), {"type": "none"}
        elif degradation_type == "noise":
            return self.apply_gaussian_noise(clean_img)
        elif degradation_type == "blur":
            return self.apply_gaussian_blur(clean_img)
        elif degradation_type == "motion_blur":
            return self.apply_motion_blur(clean_img)
        elif degradation_type == "contrast":
            return self.apply_contrast_degradation(clean_img)
        elif degradation_type == "brightness":
            return self.apply_brightness_variation(clean_img)
        elif degradation_type == "illumination":
            return self.apply_uneven_illumination(clean_img)
        elif degradation_type == "jpeg":
            return self.apply_jpeg_compression(clean_img)
        elif degradation_type == "resolution":
            return self.apply_resolution_degradation(clean_img)
        elif degradation_type == "mixed":
            return self.apply_mixed_degradation(clean_img)
        elif degradation_type == "random":
            choices = ["noise", "blur", "motion_blur", "contrast", "illumination", "jpeg", "resolution", "mixed"]
            chosen = random.choice(choices)
            return self(clean_img, degradation_type=chosen)
        else:
            return self.apply_mixed_degradation(clean_img)
