"""
Color Image Processing Module.
Supports conversions across 6 color spaces (RGB, HSV, LAB, XYZ, YCbCr, HSL),
White Balancing (Gray World & White Patch), Tone Mapping, and Color Enhancements.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import cv2


def convert_color_space(img: np.ndarray, target_space: str = "HSV") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Converts RGB image to target color space (HSV, LAB, XYZ, YCbCr, HSL).
    Returns (converted_image, channel_dict).
    """
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    target = target_space.upper()

    if target == "HSV":
        converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        channels = {
            "Hue (H)": converted[:, :, 0],
            "Saturation (S)": converted[:, :, 1],
            "Value (V)": converted[:, :, 2]
        }
    elif target == "LAB":
        converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        channels = {
            "Lightness (L)": converted[:, :, 0],
            "a (Green-Red)": converted[:, :, 1],
            "b (Blue-Yellow)": converted[:, :, 2]
        }
    elif target == "XYZ":
        converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
        channels = {
            "X": converted[:, :, 0],
            "Y (Luminance)": converted[:, :, 1],
            "Z": converted[:, :, 2]
        }
    elif target == "YCBCR":
        converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        channels = {
            "Y (Luma)": converted[:, :, 0],
            "Cb (Blue-Difference)": converted[:, :, 2],
            "Cr (Red-Difference)": converted[:, :, 1]
        }
    elif target == "HSL":
        converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        channels = {
            "Hue (H)": converted[:, :, 0],
            "Lightness (L)": converted[:, :, 1],
            "Saturation (S)": converted[:, :, 2]
        }
    else:  # Default RGB
        converted = img_rgb.copy()
        channels = {
            "Red (R)": converted[:, :, 0],
            "Green (G)": converted[:, :, 1],
            "Blue (B)": converted[:, :, 2]
        }

    return converted, channels


def white_balance_gray_world(img: np.ndarray) -> np.ndarray:
    """
    White Balance using Gray World Hypothesis:
    Assumes average channel intensity across image is neutral gray.
    """
    if img.ndim == 2:
        return img.copy()

    img_float = img.astype(np.float32)
    r_mean = np.mean(img_float[:, :, 0])
    g_mean = np.mean(img_float[:, :, 1])
    b_mean = np.mean(img_float[:, :, 2])

    gray_mean = (r_mean + g_mean + b_mean) / 3.0

    r_scale = gray_mean / (r_mean + 1e-6)
    g_scale = gray_mean / (g_mean + 1e-6)
    b_scale = gray_mean / (b_mean + 1e-6)

    balanced = np.zeros_like(img_float)
    balanced[:, :, 0] = img_float[:, :, 0] * r_scale
    balanced[:, :, 1] = img_float[:, :, 1] * g_scale
    balanced[:, :, 2] = img_float[:, :, 2] * b_scale

    return np.clip(balanced, 0, 255).astype(np.uint8)


def white_balance_white_patch(img: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    White Balance using White Patch (Max RGB) Hypothesis:
    Scales maximum channel intensities (at specified percentile) to 255.
    """
    if img.ndim == 2:
        return img.copy()

    img_float = img.astype(np.float32)
    r_max = np.percentile(img_float[:, :, 0], percentile)
    g_max = np.percentile(img_float[:, :, 1], percentile)
    b_max = np.percentile(img_float[:, :, 2], percentile)

    balanced = np.zeros_like(img_float)
    balanced[:, :, 0] = (img_float[:, :, 0] / (r_max + 1e-6)) * 255.0
    balanced[:, :, 1] = (img_float[:, :, 1] / (g_max + 1e-6)) * 255.0
    balanced[:, :, 2] = (img_float[:, :, 2] / (b_max + 1e-6)) * 255.0

    return np.clip(balanced, 0, 255).astype(np.uint8)


def enhance_saturation(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Multiplies Saturation channel in HSV space by specified factor."""
    if img.ndim == 2:
        return img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def reinhard_tone_mapping(img: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Applies Reinhard Tone Mapping Operator: output = input / (1 + input / intensity).
    Compresses high dynamic range values.
    """
    norm = img.astype(np.float32) / 255.0
    mapped = norm / (1.0 + norm / intensity)
    return (mapped * 255.0).clip(0, 255).astype(np.uint8)
