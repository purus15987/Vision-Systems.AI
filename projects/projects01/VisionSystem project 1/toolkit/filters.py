"""
Spatial Filtering and Noise Injection Module.
Implements spatial smoothing (Mean, Gaussian, Median, Bilateral),
sharpening (Laplacian, Unsharp Mask, High-Boost), and noise injection routines.
"""

from typing import Tuple
import numpy as np
import cv2


# ==========================================
# 1. Smoothing Filters
# ==========================================

def mean_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies Mean (Average) spatial smoothing filter."""
    k = max(1, kernel_size | 1)  # Ensure odd integer
    return cv2.blur(img, (k, k))


def gaussian_filter(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Applies 2D Gaussian smoothing filter with specified kernel size and sigma."""
    k = max(1, kernel_size | 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def median_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies Non-linear Median filter (highly effective for Salt & Pepper noise)."""
    k = max(1, kernel_size | 1)
    return cv2.medianBlur(img, k)


def bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Applies Edge-preserving Bilateral Filter."""
    if img.ndim == 2:
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    else:
        # Convert to BGR for opencv bilateral filter
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filtered_bgr = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
        return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)


# ==========================================
# 2. Sharpening Filters
# ==========================================

def laplacian_sharpen(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Applies 2D Laplacian second-derivative sharpening operator."""
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + alpha, -1],
                       [0, -1, 0]], dtype=np.float32)
    img_float = img.astype(np.float32)
    sharpened = cv2.filter2D(img_float, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_masking(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    """
    Applies Unsharp Masking: High-pass = Original - Blurred.
    Sharpened = Original + amount * High-pass.
    """
    blurred = gaussian_filter(img, kernel_size=kernel_size, sigma=sigma)
    img_float = img.astype(np.float32)
    blurred_float = blurred.astype(np.float32)

    high_pass = img_float - blurred_float
    sharpened = img_float + amount * high_pass
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def high_boost_filter(img: np.ndarray, A: float = 2.0, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    High-Boost Filtering: High-Boost = A * Original - Low-Pass.
    Equals (A - 1) * Original + Unsharp-Mask.
    """
    blurred = gaussian_filter(img, kernel_size=kernel_size, sigma=sigma)
    img_float = img.astype(np.float32)
    blurred_float = blurred.astype(np.float32)

    boosted = A * img_float - blurred_float
    return np.clip(boosted, 0, 255).astype(np.uint8)


# ==========================================
# 3. Noise Simulation
# ==========================================

def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, std: float = 25.0) -> np.ndarray:
    """Adds zero-mean additive Gaussian noise to image."""
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray:
    """Adds impulse (Salt & Pepper) noise with specified probability ratios."""
    noisy = img.copy()
    num_salt = np.ceil(salt_prob * img.size)
    num_pepper = np.ceil(pepper_prob * img.size)

    # Salt (255)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    if img.ndim == 2:
        noisy[coords[0], coords[1]] = 255
    else:
        noisy[coords[0], coords[1], :] = 255

    # Pepper (0)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
    if img.ndim == 2:
        noisy[coords[0], coords[1]] = 0
    else:
        noisy[coords[0], coords[1], :] = 0

    return noisy


def add_speckle_noise(img: np.ndarray, variance: float = 0.05) -> np.ndarray:
    """Adds multiplicative Speckle noise: output = input + input * noise."""
    noise = np.random.normal(0, np.sqrt(variance), img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + img.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
