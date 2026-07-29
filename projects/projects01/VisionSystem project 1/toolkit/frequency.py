"""
Frequency Domain Processing Module.
Includes 2D FFT, Magnitude/Phase spectrum analysis, and Frequency Domain Filters:
Ideal LP/HP, Gaussian LP/HP, Butterworth LP/HP, Band Pass, and Band Stop.
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2


def compute_fft(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 2D Discrete Fourier Transform (DFT).
    Returns (f_shift, magnitude_spectrum, phase_spectrum).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    f = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f)

    # Magnitude Spectrum (log scaled)
    magnitude = np.log(1.0 + np.abs(f_shift))
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Phase Spectrum
    phase = np.angle(f_shift)
    phase_norm = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return f_shift, magnitude_norm, phase_norm


def inverse_fft(f_shift: np.ndarray) -> np.ndarray:
    """Computes Inverse 2D FFT to reconstruct spatial domain image."""
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_real = np.abs(img_back)
    return np.clip(img_real, 0, 255).astype(np.uint8)


def generate_frequency_filter_mask(
    shape: Tuple[int, int],
    filter_type: str = "gaussian_lowpass",
    cutoff: float = 50.0,
    cutoff_high: float = 100.0,
    order: int = 2
) -> np.ndarray:
    """
    Generates 2D frequency domain transfer function H(u, v):
    - 'ideal_lowpass', 'ideal_highpass'
    - 'gaussian_lowpass', 'gaussian_highpass'
    - 'butterworth_lowpass', 'butterworth_highpass'
    - 'bandpass', 'bandstop'
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx)**2 + (y - cy)**2)
    d = np.maximum(d, 1e-5)  # Avoid division by zero

    ftype = filter_type.lower()

    if ftype == "ideal_lowpass":
        H = (d <= cutoff).astype(np.float32)

    elif ftype == "ideal_highpass":
        H = (d > cutoff).astype(np.float32)

    elif ftype == "gaussian_lowpass":
        H = np.exp(-(d**2) / (2.0 * (cutoff**2))).astype(np.float32)

    elif ftype == "gaussian_highpass":
        H = (1.0 - np.exp(-(d**2) / (2.0 * (cutoff**2)))).astype(np.float32)

    elif ftype == "butterworth_lowpass":
        H = (1.0 / (1.0 + (d / cutoff)**(2 * order))).astype(np.float32)

    elif ftype == "butterworth_highpass":
        H = (1.0 / (1.0 + (cutoff / d)**(2 * order))).astype(np.float32)

    elif ftype == "bandpass":
        # Pass frequencies between cutoff and cutoff_high
        center = (cutoff + cutoff_high) / 2.0
        bandwidth = cutoff_high - cutoff
        H = np.exp(-(((d**2 - center**2) / (d * bandwidth + 1e-5))**2)).astype(np.float32)

    elif ftype == "bandstop":
        center = (cutoff + cutoff_high) / 2.0
        bandwidth = cutoff_high - cutoff
        H = (1.0 - np.exp(-(((d**2 - center**2) / (d * bandwidth + 1e-5))**2))).astype(np.float32)

    else:
        H = np.ones((h, w), dtype=np.float32)

    return H


def apply_frequency_filter(
    img: np.ndarray,
    filter_type: str = "gaussian_lowpass",
    cutoff: float = 50.0,
    cutoff_high: float = 100.0,
    order: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies frequency domain filter to an image.
    Returns (filtered_spatial_img, filter_mask_visualization).
    """
    f_shift, _, _ = compute_fft(img)
    mask = generate_frequency_filter_mask(
        shape=f_shift.shape,
        filter_type=filter_type,
        cutoff=cutoff,
        cutoff_high=cutoff_high,
        order=order
    )

    filtered_f = f_shift * mask
    filtered_img = inverse_fft(filtered_f)
    mask_vis = (mask * 255).astype(np.uint8)

    # Preserve color if input was RGB
    if img.ndim == 3:
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

    return filtered_img, mask_vis
