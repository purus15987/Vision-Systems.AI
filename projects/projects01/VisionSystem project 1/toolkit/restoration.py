"""
Image Restoration and Deblurring Module.
Includes blur simulation (Motion blur, Gaussian, Defocus blur)
and restoration algorithms (Inverse filtering, Wiener filtering, Richardson-Lucy deconvolution).
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from skimage import restoration


# ==========================================
# 1. Blur Kernels & Simulation
# ==========================================

def create_motion_blur_kernel(kernel_size: int = 15, angle: float = 45.0) -> np.ndarray:
    """Creates a 2D motion blur Point Spread Function (PSF) kernel."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    angle_rad = np.deg2rad(angle)

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    sum_k = np.sum(kernel)
    if sum_k > 0:
        kernel /= sum_k
    else:
        kernel[center, center] = 1.0
    return kernel


def create_defocus_blur_kernel(radius: int = 5) -> np.ndarray:
    """Creates a circular disk defocus blur PSF kernel."""
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = radius
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = (x**2 + y**2) <= (radius**2)
    kernel[mask] = 1.0
    sum_k = np.sum(kernel)
    if sum_k > 0:
        kernel /= sum_k
    return kernel


def simulate_blur(img: np.ndarray, blur_type: str = "motion", size: int = 15, angle: float = 45.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates image blurring with a specified PSF.
    Returns (blurred_image, psf_kernel).
    """
    if blur_type == "motion":
        psf = create_motion_blur_kernel(kernel_size=size, angle=angle)
    elif blur_type == "defocus":
        psf = create_defocus_blur_kernel(radius=size // 2)
    else:  # Gaussian
        k = max(1, size | 1)
        return cv2.GaussianBlur(img, (k, k), 0), np.ones((k, k), dtype=np.float32) / (k * k)

    if img.ndim == 2:
        blurred = cv2.filter2D(img, -1, psf)
    else:
        blurred = np.zeros_like(img)
        for i in range(3):
            blurred[:, :, i] = cv2.filter2D(img[:, :, i], -1, psf)
    return blurred, psf


# ==========================================
# 2. Image Restoration Algorithms
# ==========================================

def inverse_filter(blurred_img: np.ndarray, psf: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Inverse Filtering in Frequency Domain: F_hat(u, v) = G(u, v) / H(u, v).
    Uses cutoff threshold on |H(u, v)| to limit noise amplification.
    """
    def _inverse_2d(channel):
        h, w = channel.shape
        kh, kw = psf.shape

        # Pad PSF to image size
        psf_padded = np.zeros((h, w), dtype=np.float32)
        psf_padded[:kh, :kw] = psf
        psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
        psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

        G = np.fft.fft2(channel.astype(np.float32))
        H = np.fft.fft2(psf_padded)

        # Truncated inverse filtering
        H_abs = np.abs(H)
        H_inv = np.zeros_like(H, dtype=np.complex64)
        mask = H_abs >= threshold
        H_inv[mask] = 1.0 / H[mask]

        F_hat = G * H_inv
        restored = np.real(np.fft.ifft2(F_hat))
        return np.clip(restored, 0, 255).astype(np.uint8)

    if blurred_img.ndim == 2:
        return _inverse_2d(blurred_img)
    else:
        channels = [_inverse_2d(blurred_img[:, :, i]) for i in range(3)]
        return np.stack(channels, axis=-1)


def wiener_filter(blurred_img: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """
    Wiener Deconvolution: Minimizes Mean Squared Error in presence of noise.
    F_hat(u, v) = [ |H(u, v)|^2 / (|H(u, v)|^2 + K) ] * [ G(u, v) / H(u, v) ]
    """
    def _wiener_2d(channel):
        norm_channel = channel.astype(np.float32) / 255.0
        deconv = restoration.wiener(norm_channel, psf, balance=K, clip=True)
        return (deconv * 255.0).astype(np.uint8)

    if blurred_img.ndim == 2:
        return _wiener_2d(blurred_img)
    else:
        channels = [_wiener_2d(blurred_img[:, :, i]) for i in range(3)]
        return np.stack(channels, axis=-1)


def richardson_lucy_deconvolution(blurred_img: np.ndarray, psf: np.ndarray, num_iter: int = 30) -> np.ndarray:
    """
    Iterative Richardson-Lucy Deconvolution algorithm for non-blind restoration.
    """
    def _rl_2d(channel):
        norm_channel = channel.astype(np.float32) / 255.0
        deconv = restoration.richardson_lucy(norm_channel, psf, num_iter=num_iter, clip=True)
        return (deconv * 255.0).astype(np.uint8)

    if blurred_img.ndim == 2:
        return _rl_2d(blurred_img)
    else:
        channels = [_rl_2d(blurred_img[:, :, i]) for i in range(3)]
        return np.stack(channels, axis=-1)
