"""
Histogram Processing and Analysis Module.
Includes histogram computation, CDF calculation, contrast stretching,
global histogram equalization, CLAHE, histogram matching, and plotting routines.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import exposure


def compute_histogram(img: np.ndarray, bins: int = 256) -> Dict[str, np.ndarray]:
    """Computes histogram counts for Grayscale or RGB channels."""
    if img.ndim == 2:
        hist, bin_edges = np.histogram(img, bins=bins, range=(0, 256))
        return {"Gray": hist, "bins": bin_edges[:-1]}
    else:
        hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 256))
        hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 256))
        hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 256))
        bin_edges = np.linspace(0, 255, bins)
        return {"Red": hist_r, "Green": hist_g, "Blue": hist_b, "bins": bin_edges}


def compute_cdf(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes normalized Cumulative Distribution Function (CDF)."""
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return cdf_normalized, bins[:-1]


def histogram_stretching(img: np.ndarray, min_out: int = 0, max_out: int = 255) -> np.ndarray:
    """Performs Min-Max contrast stretching to map pixel intensities across [min_out, max_out]."""
    if img.ndim == 3:
        stretched = np.zeros_like(img)
        for i in range(3):
            c = img[:, :, i].astype(np.float32)
            c_min, c_max = np.min(c), np.max(c)
            if c_max > c_min:
                stretched[:, :, i] = (c - c_min) / (c_max - c_min) * (max_out - min_out) + min_out
            else:
                stretched[:, :, i] = c
        return np.clip(stretched, 0, 255).astype(np.uint8)
    else:
        c = img.astype(np.float32)
        c_min, c_max = np.min(c), np.max(c)
        if c_max > c_min:
            stretched = (c - c_min) / (c_max - c_min) * (max_out - min_out) + min_out
        else:
            stretched = c
        return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Performs Global Histogram Equalization (Grayscale or Y component of YCrCb)."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)

    # For color image, equalize Y channel in YCrCb space to preserve color balance
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Performs Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if img.ndim == 2:
        return clahe.apply(img)

    # Apply CLAHE on LAB luminance (L) channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def histogram_matching(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    """Matches the histogram distribution of source_img to match reference_img."""
    matched = exposure.match_histograms(source_img, reference_img, channel_axis=-1 if source_img.ndim == 3 else None)
    return np.clip(matched, 0, 255).astype(np.uint8)


def plot_histogram(img: np.ndarray, title: str = "Histogram Plot") -> go.Figure:
    """Generates an interactive Plotly histogram figure with CDF curve."""
    fig = go.Figure()
    bins = np.arange(256)

    if img.ndim == 2:
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        fig.add_trace(go.Bar(x=bins, y=hist, name="Intensity", marker_color="#94A3B8"))
    else:
        hist_r, _ = np.histogram(img[:, :, 0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(img[:, :, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(img[:, :, 2], bins=256, range=(0, 256))

        fig.add_trace(go.Bar(x=bins, y=hist_r, name="Red", marker_color="#EF4444", opacity=0.7))
        fig.add_trace(go.Bar(x=bins, y=hist_g, name="Green", marker_color="#10B981", opacity=0.7))
        fig.add_trace(go.Bar(x=bins, y=hist_b, name="Blue", marker_color="#3B82F6", opacity=0.7))

    cdf_norm, _ = compute_cdf(img)
    fig.add_trace(go.Scatter(x=bins, y=cdf_norm, name="CDF", yaxis="y2", line=dict(color="#F59E0B", width=2)))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F8FAFC")),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        barmode='overlay',
        xaxis=dict(title="Pixel Intensity (0-255)", color="#94A3B8", gridcolor="#334155"),
        yaxis=dict(title="Frequency", color="#94A3B8", gridcolor="#334155"),
        yaxis2=dict(title="CDF Scale", color="#F59E0B", overlaying="y", side="right"),
        legend=dict(font=dict(color="#F8FAFC"), bgcolor="rgba(15, 23, 42, 0.5)"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
