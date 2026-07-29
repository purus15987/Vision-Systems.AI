"""
Spatial and Intensity Transformations Module.
Includes spatial scaling, rotation, flipping, brightness/contrast adjustments,
negative, log/inverse log, gamma correction, piecewise linear, and thresholding.
"""

from typing import Tuple, List, Optional
import numpy as np
import cv2


# ==========================================
# 1. Spatial Transformations
# ==========================================

def resize_image(img: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, interpolation: str = "bilinear") -> np.ndarray:
    """Resizes image keeping aspect ratio if only width or height is provided."""
    h, w = img.shape[:2]
    if width is None and height is None:
        return img.copy()

    if width is None:
        aspect = height / float(h)
        width = int(w * aspect)
    elif height is None:
        aspect = width / float(w)
        height = int(h * aspect)

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    flag = interp_map.get(interpolation.lower(), cv2.INTER_LINEAR)
    return cv2.resize(img, (width, height), interpolation=flag)


def rotate_image(img: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
    """Rotates image by specified angle (degrees) around its center."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    if expand:
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]
        w, h = new_w, new_h

    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def flip_image(img: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """Flips image horizontally, vertically, or both."""
    if mode == "horizontal":
        return cv2.flip(img, 1)
    elif mode == "vertical":
        return cv2.flip(img, 0)
    elif mode == "both":
        return cv2.flip(img, -1)
    return img.copy()


# ==========================================
# 2. Intensity Transformations
# ==========================================

def adjust_brightness_contrast(img: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """
    Adjusts brightness (-100 to +100) and contrast (0.1 to 3.0).
    Formulation: output = contrast * input + brightness
    """
    img_float = img.astype(np.float32)
    adjusted = contrast * img_float + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def negative_transform(img: np.ndarray) -> np.ndarray:
    """Computes negative image: s = L - 1 - r (where L = 256)."""
    return 255 - img


def log_transform(img: np.ndarray, c: Optional[float] = None) -> np.ndarray:
    """
    Logarithmic intensity transformation: s = c * log(1 + r).
    Dynamic range expansion for low intensity values.
    """
    img_float = img.astype(np.float32)
    if c is None:
        c = 255.0 / np.log(1.0 + np.max(img_float) + 1e-6)
    log_img = c * np.log(1.0 + img_float)
    return np.clip(log_img, 0, 255).astype(np.uint8)


def inverse_log_transform(img: np.ndarray, c: Optional[float] = None) -> np.ndarray:
    """
    Inverse log transformation: s = exp(r / c) - 1.
    Dynamic range compression.
    """
    img_float = img.astype(np.float32)
    if c is None:
        c = 255.0 / np.log(1.0 + 255.0)
    inv_log = np.exp(img_float / c) - 1.0
    return np.clip(inv_log, 0, 255).astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    Power-law (Gamma) transformation: s = c * (r / 255)^gamma * 255.
    Gamma < 1 expands dark values; Gamma > 1 expands highlight values.
    """
    if gamma <= 0:
        gamma = 0.01
    norm = img.astype(np.float32) / 255.0
    gamma_img = c * np.power(norm, gamma) * 255.0
    return np.clip(gamma_img, 0, 255).astype(np.uint8)


def piecewise_linear_transform(
    img: np.ndarray,
    r1: int = 70, s1: int = 30,
    r2: int = 180, s2: int = 220
) -> np.ndarray:
    """
    Contrast stretching using piecewise linear function defined by (r1, s1) and (r2, s2).
    """
    r1, s1 = max(0, r1), max(0, s1)
    r2, s2 = min(255, max(r1 + 1, r2)), min(255, s2)

    # Pre-calculate LUT (Look Up Table) for speed
    lut = np.zeros(256, dtype=np.uint8)
    for r in range(256):
        if r < r1:
            slope = s1 / float(r1) if r1 > 0 else 0
            val = slope * r
        elif r <= r2:
            slope = (s2 - s1) / float(r2 - r1) if r2 > r1 else 0
            val = s1 + slope * (r - r1)
        else:
            slope = (255 - s2) / float(255 - r2) if r2 < 255 else 0
            val = s2 + slope * (r - r2)
        lut[r] = np.clip(val, 0, 255)

    return cv2.LUT(img, lut)


def apply_threshold_slice(img: np.ndarray, t_low: int = 100, t_high: int = 200, retain_background: bool = True) -> np.ndarray:
    """
    Intensity-level slicing: Highlights a specific range of intensities [t_low, t_high].
    """
    mask = (img >= t_low) & (img <= t_high)
    result = img.copy()
    if retain_background:
        result[mask] = 255
    else:
        result[mask] = 255
        result[~mask] = 0
    return result
