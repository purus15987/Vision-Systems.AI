"""
Image Compression and Trade-off Analysis Module.
Simulates JPEG & PNG compression, computes compression ratios,
file size savings, and quality degeneration curves (PSNR/SSIM vs Quality Factor).
"""

from typing import Dict, Any, Tuple, List, Optional
import io
import numpy as np
import cv2
from PIL import Image

from toolkit.quality import compute_psnr, compute_ssim


def simulate_jpeg_compression(img: np.ndarray, quality: int = 85) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates JPEG lossy compression with quality factor (1 to 100).
    Returns (compressed_decompressed_img, metrics_dict).
    """
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = Image.fromarray(img, mode="RGB")

    # Original uncompressed size (raw byte array)
    orig_bytes = img.nbytes

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    compressed_bytes_data = buf.getvalue()
    compressed_size = len(compressed_bytes_data)

    # Decompress back to numpy array
    decompressed_pil = Image.open(buf)
    decompressed_img = np.array(decompressed_pil)

    ratio = orig_bytes / float(compressed_size) if compressed_size > 0 else 1.0
    space_saved = (1.0 - compressed_size / float(orig_bytes)) * 100.0 if orig_bytes > 0 else 0.0

    psnr_val = compute_psnr(img, decompressed_img)
    ssim_val = compute_ssim(img, decompressed_img)

    metrics = {
        "format": "JPEG",
        "quality_factor": quality,
        "original_size_kb": round(orig_bytes / 1024.0, 2),
        "compressed_size_kb": round(compressed_size / 1024.0, 2),
        "compression_ratio": round(ratio, 2),
        "space_saved_percent": round(space_saved, 2),
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4)
    }

    return decompressed_img, metrics


def analyze_compression_quality_tradeoff(img: np.ndarray, quality_levels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Evaluates JPEG compression performance across multiple quality factors.
    Returns list of metric dictionaries for plotting rate-distortion curves.
    """
    if quality_levels is None:
        quality_levels = [10, 25, 50, 75, 85, 95]

    results = []
    for q in quality_levels:
        _, m = simulate_jpeg_compression(img, quality=q)
        results.append(m)

    return results
