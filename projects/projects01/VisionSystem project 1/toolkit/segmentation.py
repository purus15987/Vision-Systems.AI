"""
Image Segmentation Module.
Includes Thresholding (Global, Adaptive, Otsu), Edge Detection (Sobel, Prewitt, Roberts, Laplacian, Canny),
Marker-based Watershed, and K-Means Clustering segmentation.
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2


# ==========================================
# 1. Thresholding Operations
# ==========================================

def threshold_global(img: np.ndarray, thresh_val: int = 128, max_val: int = 255) -> np.ndarray:
    """Applies global binary thresholding."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
    return binary


def threshold_adaptive(img: np.ndarray, method: str = "gaussian", block_size: int = 11, C: float = 2.0) -> np.ndarray:
    """Applies Adaptive Thresholding (Mean or Gaussian)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    block_size = max(3, block_size | 1)  # Odd number >= 3
    method_flag = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(gray, 255, method_flag, cv2.THRESH_BINARY, block_size, C)


def threshold_otsu(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Applies Otsu's binarization. Returns (binary_image, optimal_threshold_value)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(val)


# ==========================================
# 2. Edge Detection Operators
# ==========================================

def detect_edges(img: np.ndarray, operator: str = "canny", low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """
    Edge Detection using 1 of 5 operators:
    'canny', 'sobel', 'prewitt', 'roberts', 'laplacian'
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    op = operator.lower()

    if op == "canny":
        return cv2.Canny(gray, low_thresh, high_thresh)

    elif op == "sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif op == "prewitt":
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        px = cv2.filter2D(gray, cv2.CV_64F, kernelx)
        py = cv2.filter2D(gray, cv2.CV_64F, kernely)
        magnitude = cv2.magnitude(px, py)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif op == "roberts":
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        rx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
        ry = cv2.filter2D(gray, cv2.CV_64F, kernely)
        magnitude = cv2.magnitude(rx, ry)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif op == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


# ==========================================
# 3. Region & Clustering Segmentation
# ==========================================

def watershed_segmentation(img: np.ndarray) -> np.ndarray:
    """Applies Marker-based Watershed Segmentation algorithm."""
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area using Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)

    # Draw boundary outlines on output image
    result = img_rgb.copy()
    result[markers == -1] = [255, 0, 0]  # Red watershed boundaries
    return result


def kmeans_segmentation(img: np.ndarray, K: int = 4) -> np.ndarray:
    """Applies K-Means Clustering segmentation in RGB color space."""
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    return segmented_data.reshape(img_rgb.shape)
