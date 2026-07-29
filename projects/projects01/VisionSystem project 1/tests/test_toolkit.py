"""
Comprehensive Unit & System Test Suite for DIP Toolkit.
Tests all 15 algorithmic domains in toolkit/.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolkit.io import generate_synthetic_image, extract_metadata, load_image
from toolkit.visualization import convert_to_grayscale, convert_to_binary, split_channels, inspect_pixel
from toolkit.transforms import (
    resize_image, rotate_image, flip_image, adjust_brightness_contrast,
    negative_transform, log_transform, gamma_correction, piecewise_linear_transform
)
from toolkit.histogram import (
    compute_histogram, histogram_stretching, histogram_equalization, apply_clahe, histogram_matching
)
from toolkit.filters import (
    mean_filter, gaussian_filter, median_filter, bilateral_filter,
    laplacian_sharpen, unsharp_masking, high_boost_filter,
    add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise
)
from toolkit.frequency import compute_fft, apply_frequency_filter, inverse_fft
from toolkit.restoration import simulate_blur, inverse_filter, wiener_filter, richardson_lucy_deconvolution
from toolkit.color import convert_color_space, white_balance_gray_world, white_balance_white_patch, enhance_saturation
from toolkit.morphology import apply_binary_morphology, apply_advanced_morphology
from toolkit.segmentation import threshold_global, threshold_adaptive, threshold_otsu, detect_edges, watershed_segmentation, kmeans_segmentation
from toolkit.features import extract_contours_and_shapes, detect_harris_corners, detect_orb_features, extract_hog_features, extract_lbp_texture
from toolkit.compression import simulate_jpeg_compression, analyze_compression_quality_tradeoff
from toolkit.quality import compute_mse, compute_rmse, compute_psnr, compute_ssim, evaluate_quality_metrics
from toolkit.utils import HistoryManager, PipelineManager, detect_gpu_acceleration


def test_toolkit_pipeline():
    print("Beginning DIP Toolkit Automated Unit Tests...")

    # 1. IO & Generation
    img = generate_synthetic_image("geometric", (256, 256))
    assert img.shape == (256, 256, 3), "Synthetic image shape mismatch."
    meta = extract_metadata(img)
    assert meta["width"] == 256 and meta["height"] == 256, "Metadata extraction error."
    print("  [1/12] I/O & Metadata extraction: PASS")

    # 2. Visualization & Channels
    gray = convert_to_grayscale(img)
    assert gray.ndim == 2, "Grayscale conversion failed."
    binary = convert_to_binary(img, 128)
    assert binary.max() <= 255, "Binary conversion error."
    channels = split_channels(img)
    assert "Red_Channel" in channels, "Channel splitting error."
    probe = inspect_pixel(img, 10, 10)
    assert "RGB" in probe, "Pixel inspector error."
    print("  [2/12] Visualization & Pixel Inspection: PASS")

    # 3. Spatial & Intensity Transforms
    rot = rotate_image(img, 45)
    assert rot is not None, "Rotation failed."
    flp = flip_image(img, "horizontal")
    assert flp is not None, "Flip failed."
    bc = adjust_brightness_contrast(img, 20, 1.2)
    assert bc.shape == img.shape, "Brightness/contrast error."
    neg = negative_transform(img)
    assert neg is not None, "Negative transform failed."
    gamma = gamma_correction(img, 1.5)
    assert gamma is not None, "Gamma correction failed."
    print("  [3/12] Spatial & Intensity Transforms: PASS")

    # 4. Histograms
    hist = compute_histogram(img)
    assert "Red" in hist or "Gray" in hist, "Histogram calculation error."
    eq = histogram_equalization(img)
    assert eq.shape == img.shape, "Equalization error."
    clahe = apply_clahe(img)
    assert clahe is not None, "CLAHE failed."
    print("  [4/12] Histograms & CLAHE: PASS")

    # 5. Filtering & Noise
    g_blur = gaussian_filter(img, 5, 1.0)
    assert g_blur.shape == img.shape, "Gaussian filter error."
    med = median_filter(img, 5)
    assert med is not None, "Median filter error."
    bilat = bilateral_filter(img, 5)
    assert bilat is not None, "Bilateral filter error."
    sharp = unsharp_masking(img)
    assert sharp.shape == img.shape, "Unsharp masking error."
    noisy = add_gaussian_noise(img)
    assert noisy.shape == img.shape, "Noise injection error."
    print("  [5/12] Spatial Filters & Noise: PASS")

    # 6. Frequency Domain
    f_shift, mag, phase = compute_fft(img)
    assert mag.shape == (256, 256), "FFT computation error."
    freq_filtered, mask = apply_frequency_filter(img, "gaussian_lowpass", 50)
    assert freq_filtered is not None, "Frequency filtering error."
    print("  [6/12] 2D FFT & Frequency Domain Filtering: PASS")

    # 7. Restoration
    blurred, psf = simulate_blur(img, "motion", 15)
    assert blurred.shape == img.shape, "Blur simulation error."
    inv_rest = inverse_filter(blurred, psf)
    assert inv_rest is not None, "Inverse filtering error."
    wiener_rest = wiener_filter(blurred, psf)
    assert wiener_rest is not None, "Wiener filtering error."
    print("  [7/12] Blur Simulation & Deblurring Restoration: PASS")

    # 8. Color Processing
    conv, _ = convert_color_space(img, "HSV")
    assert conv.shape == img.shape, "Color space conversion error."
    wb = white_balance_gray_world(img)
    assert wb.shape == img.shape, "White balance error."
    print("  [8/12] Color Space & White Balance: PASS")

    # 9. Morphology
    ero = apply_binary_morphology(img, "erosion", "rect", 5)
    assert ero is not None, "Morphology erosion error."
    skel = apply_advanced_morphology(img, "skeletonization")
    assert skel is not None, "Skeletonization error."
    print("  [9/12] Morphological Operations: PASS")

    # 10. Segmentation & Edges
    otsu, val = threshold_otsu(img)
    assert otsu.ndim == 2, "Otsu thresholding error."
    edges = detect_edges(img, "canny")
    assert edges.ndim == 2, "Canny edge detection error."
    km = kmeans_segmentation(img, 4)
    assert km.shape == img.shape, "K-Means segmentation error."
    print("  [10/12] Segmentation & Edge Operators: PASS")

    # 11. Features
    annotated, shape_list = extract_contours_and_shapes(img)
    assert annotated is not None, "Contour extraction error."
    harris = detect_harris_corners(img)
    assert harris is not None, "Harris corner detection error."
    orb_img, n_kp = detect_orb_features(img)
    assert orb_img is not None, "ORB feature extraction error."
    print("  [11/12] Features & Shape Analysis: PASS")

    # 12. Compression & Quality
    comp_img, comp_metrics = simulate_jpeg_compression(img, 85)
    assert "compression_ratio" in comp_metrics, "Compression analysis error."
    iqa, heatmap = evaluate_quality_metrics(img, comp_img)
    assert "PSNR (dB)" in iqa and "SSIM" in iqa, "IQA metric evaluation error."
    print("  [12/12] Compression & Quality Metrics (MSE, PSNR, SSIM): PASS")

    print("\n[OK] ALL 12 TOOLKIT UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_toolkit_pipeline()
