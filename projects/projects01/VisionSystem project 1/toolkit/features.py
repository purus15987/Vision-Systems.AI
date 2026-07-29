"""
Feature Extraction and Texture Analysis Module.
Includes Contour Analysis (Area, Perimeter, Centroid, Convex Hull),
Harris Corner Detection, ORB Features, HOG Descriptor, and LBP Texture analysis.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2

from skimage.feature import hog, local_binary_pattern


def extract_contours_and_shapes(img: np.ndarray, min_area: float = 100.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts contours and computes shape metrics for each detected region:
    Area, Perimeter, Centroid, Bounding Box, Convex Hull, Solidity, Circularity.
    Returns (annotated_image, metrics_list).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        annotated = img.copy()
    else:
        gray = img.copy()
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    metrics_list: List[Dict[str, Any]] = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))

        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0.0
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Draw contour & bounding box on image
        cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(annotated, [hull], -1, (255, 165, 0), 1)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        metrics_list.append({
            "id": idx + 1,
            "area": round(area, 2),
            "perimeter": round(perimeter, 2),
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "solidity": round(solidity, 4),
            "circularity": round(circularity, 4)
        })

    return annotated, metrics_list


def detect_harris_corners(img: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04, threshold_factor: float = 0.01) -> np.ndarray:
    """Detects corner interest points using Harris Corner Detection."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        annotated = img.copy()
    else:
        gray = img.copy()
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, block_size, ksize, k)
    dst = cv2.dilate(dst, None)

    # Highlight corners in bright red
    annotated[dst > threshold_factor * dst.max()] = [255, 0, 0]
    return annotated


def detect_orb_features(img: np.ndarray, max_keypoints: int = 500) -> Tuple[np.ndarray, int]:
    """
    Detects and draws ORB (Oriented FAST and Rotated BRIEF) keypoints.
    Returns (annotated_image, num_keypoints).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        color_img = img.copy()
    else:
        gray = img.copy()
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    annotated = cv2.drawKeypoints(color_img, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return annotated, len(keypoints) if keypoints else 0


def extract_hog_features(img: np.ndarray, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts Histogram of Oriented Gradients (HOG) features.
    Returns (hog_feature_vector, hog_visualization_image).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    features, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None
    )
    hog_image_rescaled = (hog_image / (hog_image.max() + 1e-6) * 255).astype(np.uint8)
    return features, cv2.cvtColor(hog_image_rescaled, cv2.COLOR_GRAY2RGB)


def extract_lbp_texture(img: np.ndarray, P: int = 8, R: float = 1.0, method: str = "uniform") -> np.ndarray:
    """Extracts Local Binary Pattern (LBP) texture descriptor image."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    lbp = local_binary_pattern(gray, P, R, method=method)
    lbp_norm = (lbp / (lbp.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.cvtColor(lbp_norm, cv2.COLOR_GRAY2RGB)
