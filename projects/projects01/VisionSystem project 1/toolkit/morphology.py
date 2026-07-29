"""
Morphological Image Processing Module.
Implements binary morphology (Erosion, Dilation, Opening, Closing)
and advanced operations (Morphological Gradient, Top Hat, Black Hat, Skeletonization, Thinning, Hit-or-Miss).
"""

from typing import Tuple
import numpy as np
import cv2
from skimage.morphology import skeletonize, thin


def get_structuring_element(shape_str: str = "rect", size: int = 5) -> np.ndarray:
    """Creates a 2D morphological structuring element (kernel)."""
    k_size = max(1, size)
    if shape_str.lower() == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    elif shape_str.lower() == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
    else:  # rect
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))


def apply_binary_morphology(
    img: np.ndarray,
    operation: str = "erosion",
    kernel_shape: str = "rect",
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    """
    Applies fundamental morphological operation:
    'erosion', 'dilation', 'opening', 'closing'
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    op = operation.lower()

    if op == "erosion":
        return cv2.erode(img, kernel, iterations=iterations)
    elif op == "dilation":
        return cv2.dilate(img, kernel, iterations=iterations)
    elif op == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        return img.copy()


def apply_advanced_morphology(
    img: np.ndarray,
    operation: str = "gradient",
    kernel_shape: str = "rect",
    kernel_size: int = 5
) -> np.ndarray:
    """
    Applies advanced morphological operation:
    'gradient', 'tophat', 'blackhat', 'skeletonization', 'thinning', 'hit_or_miss'
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    op = operation.lower()

    # Binarize if grayscale/color for skeletonization/thinning
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    if op == "gradient":
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif op == "tophat":
        return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif op == "blackhat":
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    elif op == "skeletonization":
        binary = (gray > 127).astype(np.uint8)
        skel = skeletonize(binary)
        return (skel * 255).astype(np.uint8)
    elif op == "thinning":
        binary = (gray > 127).astype(np.uint8)
        thinned = thin(binary)
        return (thinned * 255).astype(np.uint8)
    elif op == "hit_or_miss":
        # Hit-or-miss transform using cross kernel
        return cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel)
    else:
        return gray
