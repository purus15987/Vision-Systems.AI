"""
Image Quality Assessment (IQA) Metrics Module.
Computes objective fidelity and perceptual image quality metrics:
MSE, RMSE, PSNR, and SSIM with structural difference visualization maps.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_fn


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE) between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(compute_mse(img1, img2)))


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    mse = compute_mse(img1, img2)
    if mse == 0:
        return 100.0  # Infinite PSNR (identical images)
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index Measure (SSIM) between 0.0 and 1.0."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if img1.ndim == 3:
        val, _ = ssim_fn(img1, img2, full=True, channel_axis=-1)
    else:
        val, _ = ssim_fn(img1, img2, full=True)

    return float(val)


def evaluate_quality_metrics(img_orig: np.ndarray, img_proc: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluates full suite of IQA metrics: MSE, RMSE, PSNR, SSIM.
    Returns (metrics_dict, ssim_difference_heatmap_rgb).
    """
    if img_orig.shape != img_proc.shape:
        img_proc = cv2.resize(img_proc, (img_orig.shape[1], img_orig.shape[0]))

    mse = compute_mse(img_orig, img_proc)
    rmse = compute_rmse(img_orig, img_proc)
    psnr = compute_psnr(img_orig, img_proc)

    if img_orig.ndim == 3:
        ssim_val, ssim_map = ssim_fn(img_orig, img_proc, full=True, channel_axis=-1)
        ssim_map_gray = np.mean(ssim_map, axis=-1)
    else:
        ssim_val, ssim_map_gray = ssim_fn(img_orig, img_proc, full=True)

    # Difference map visualization
    diff_map = (1.0 - ssim_map_gray) * 255.0
    heatmap_bgr = cv2.applyColorMap(diff_map.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    metrics = {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "PSNR (dB)": round(psnr, 2),
        "SSIM": round(ssim_val, 4)
    }

    return metrics, heatmap_rgb
