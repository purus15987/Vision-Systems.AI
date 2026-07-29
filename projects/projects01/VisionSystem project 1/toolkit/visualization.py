"""
Image Visualization, Channel Inspection, Pixel Analysis, and Render Utilities Module.
Provides functions for RGB/RGBA/Grayscale/Binary rendering, channel splitting/merging,
pixel probe inspection, and interactive comparison plots.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts RGB/RGBA image to 8-bit Grayscale."""
    if img.ndim == 2:
        return img.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def convert_to_binary(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Converts image to Binary using a specified threshold value (0-255)."""
    gray = convert_to_grayscale(img)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def split_channels(img: np.ndarray) -> Dict[str, np.ndarray]:
    """Splits an RGB image into R, G, B single-channel matrices and colored visualizations."""
    if img.ndim == 2:
        return {"Gray": img}

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Create channel isolated RGB representations
    zeros = np.zeros_like(r)
    r_colored = np.stack([r, zeros, zeros], axis=-1)
    g_colored = np.stack([zeros, g, zeros], axis=-1)
    b_colored = np.stack([zeros, zeros, b], axis=-1)

    return {
        "Red_Channel": r,
        "Green_Channel": g,
        "Blue_Channel": b,
        "Red_Visual": r_colored,
        "Green_Visual": g_colored,
        "Blue_Visual": b_colored
    }


def merge_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merges single channel R, G, B matrices back into an RGB image."""
    return np.stack([r, g, b], axis=-1)


def inspect_pixel(img: np.ndarray, x: int, y: int) -> Dict[str, Any]:
    """
    Inspects pixel values and color information at coordinate (x, y).
    Returns coordinate, RGB values, Grayscale value, and HSV values.
    """
    h, w = img.shape[:2]
    x_clamped = int(np.clip(x, 0, w - 1))
    y_clamped = int(np.clip(y, 0, h - 1))

    if img.ndim == 2:
        val = int(img[y_clamped, x_clamped])
        return {
            "coordinate": (x_clamped, y_clamped),
            "grayscale": val,
            "RGB": (val, val, val),
            "HSV": (0, 0, val)
        }

    r, g, b = [int(c) for c in img[y_clamped, x_clamped, :3]]
    # Convert pixel to HSV
    pixel_bgr = np.uint8([[[b, g, r]]])
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0, 0]

    return {
        "coordinate": (x_clamped, y_clamped),
        "RGB": (r, g, b),
        "Hex": f"#{r:02x}{g:02x}{b:02x}",
        "Grayscale": int(0.299 * r + 0.587 * g + 0.114 * b),
        "HSV": (int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2]))
    }


def plot_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    title1: str = "Original",
    title2: str = "Processed",
    cmap1: Optional[str] = None,
    cmap2: Optional[str] = None
) -> plt.Figure:
    """Generates a side-by-side comparison Matplotlib Figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="#0F172A")

    for ax, img, title, cmap in zip(axes, [img1, img2], [title1, title2], [cmap1, cmap2]):
        ax.set_facecolor("#1E293B")
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(img)
        ax.set_title(title, color="#F8FAFC", fontsize=14, pad=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_3d_color_space(img: np.ndarray, max_points: int = 2500) -> go.Figure:
    """Generates an interactive 3D RGB Color Space Scatter Plot using Plotly."""
    h, w = img.shape[:2]
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img[:, :, :3]

    # Subsample pixels to prevent performance lag
    total_pixels = h * w
    step = max(1, int(np.sqrt(total_pixels / max_points)))

    sampled = img_rgb[::step, ::step, :]
    r = sampled[:, :, 0].flatten()
    g = sampled[:, :, 1].flatten()
    b = sampled[:, :, 2].flatten()

    colors = [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]

    fig = go.Figure(data=[go.Scatter3d(
        x=r, y=g, z=b,
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.7
        )
    )])

    fig.update_layout(
        title=dict(text="3D RGB Color Cube Distribution", font=dict(color="#F8FAFC")),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        scene=dict(
            xaxis=dict(title="Red", range=[0, 255], backgroundcolor="#1E293B", gridcolor="#334155"),
            yaxis=dict(title="Green", range=[0, 255], backgroundcolor="#1E293B", gridcolor="#334155"),
            zaxis=dict(title="Blue", range=[0, 255], backgroundcolor="#1E293B", gridcolor="#334155")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig
