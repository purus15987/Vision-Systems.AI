"""
Image I/O, Metadata Extraction, and Synthetic Sample Generation Module.
Handles image reading, writing, format conversions, EXIF/properties extraction,
and generation of standard test patterns (checkerboard, gradient, color chart, shapes).
"""

import os
import io
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image, ImageOps, ExifTags


def load_image(source: Union[str, bytes, io.BytesIO, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads an image from a filepath, byte stream, PIL Image, or numpy array.
    Returns an RGB image as a float32 or uint8 numpy array in shape (H, W, C) or (H, W).
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        elif img.ndim == 2:
            return img
        elif img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    if isinstance(source, Image.Image):
        img_pil = source.convert("RGB")
        return np.array(img_pil)

    if isinstance(source, (bytes, io.BytesIO)):
        if isinstance(source, io.BytesIO):
            source = source.getvalue()
        file_bytes = np.frombuffer(source, np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise ValueError("Failed to decode image from bytes.")
        if img_bgr.ndim == 2:
            return img_bgr
        elif img_bgr.shape[2] == 4:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        else:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image path not found: {source}")
        img_pil = Image.open(source)
        # Handle EXIF orientation if present
        img_pil = ImageOps.exif_transpose(img_pil)
        img_pil = img_pil.convert("RGB")
        return np.array(img_pil)

    raise TypeError(f"Unsupported image source type: {type(source)}")


def extract_metadata(source: Union[str, bytes, Image.Image, np.ndarray], filename: str = "image.png") -> Dict[str, Any]:
    """
    Extracts comprehensive metadata including dimensions, channels, bit depth, 
    data type, file size, aspect ratio, color mode, and EXIF attributes.
    """
    metadata: Dict[str, Any] = {
        "filename": filename,
        "width": 0,
        "height": 0,
        "channels": 1,
        "bit_depth": 8,
        "dtype": "uint8",
        "file_size_kb": 0.0,
        "aspect_ratio": 1.0,
        "color_mode": "Grayscale",
        "exif": {}
    }

    if isinstance(source, str) and os.path.exists(source):
        metadata["file_size_kb"] = round(os.path.getsize(source) / 1024.0, 2)
        metadata["filename"] = os.path.basename(source)
        try:
            with Image.open(source) as img:
                metadata["width"], metadata["height"] = img.size
                metadata["color_mode"] = img.mode
                
                # Extract EXIF if available
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if isinstance(value, (int, float, str)):
                            metadata["exif"][tag_name] = value
        except Exception:
            pass

    elif isinstance(source, bytes):
        metadata["file_size_kb"] = round(len(source) / 1024.0, 2)

    img_arr = load_image(source)
    if img_arr is not None:
        metadata["height"] = img_arr.shape[0]
        metadata["width"] = img_arr.shape[1]
        metadata["dtype"] = str(img_arr.dtype)

        if img_arr.dtype == np.uint8:
            metadata["bit_depth"] = 8
        elif img_arr.dtype == np.uint16:
            metadata["bit_depth"] = 16
        elif img_arr.dtype in (np.float32, np.float64):
            metadata["bit_depth"] = 32

        if img_arr.ndim == 2:
            metadata["channels"] = 1
            metadata["color_mode"] = "Grayscale"
        else:
            metadata["channels"] = img_arr.shape[2]
            metadata["color_mode"] = "RGB" if img_arr.shape[2] == 3 else "RGBA"

        if metadata["height"] > 0:
            metadata["aspect_ratio"] = round(metadata["width"] / metadata["height"], 2)

    return metadata


def generate_synthetic_image(pattern_type: str = "checkerboard", size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Generates high quality synthetic test images for algorithm testing:
    - 'checkerboard': Classic checkerboard matrix
    - 'gradient': Smooth multi-color and grayscale gradients
    - 'geometric': Circles, rectangles, triangles, and lines for edge/contour testing
    - 'color_chart': Multi-swatch color chart (similar to Macbeth chart)
    - 'texture': Fine frequency textures (zone plate, sine waves)
    """
    h, w = size
    if pattern_type == "checkerboard":
        grid_size = 64
        y, x = np.ogrid[:h, :w]
        board = ((x // grid_size) + (y // grid_size)) % 2
        img = (board * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif pattern_type == "gradient":
        x = np.linspace(0, 1, w, endpoint=True)
        y = np.linspace(0, 1, h, endpoint=True)
        xx, yy = np.meshgrid(x, y)

        r = np.clip(xx * 255, 0, 255).astype(np.uint8)
        g = np.clip(yy * 255, 0, 255).astype(np.uint8)
        b = np.clip((1 - xx) * 255, 0, 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    elif pattern_type == "geometric":
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        # Circle
        cv2.circle(img, (w // 4, h // 4), min(h, w) // 6, (230, 50, 50), -1)
        # Rectangle
        cv2.rectangle(img, (w // 2, h // 8), (w * 7 // 8, h // 3), (50, 200, 50), -1)
        # Triangle
        pts = np.array([[w // 4, h * 7 // 8], [w // 8, h // 2], [w * 3 // 8, h // 2]], np.int32)
        cv2.fillPoly(img, [pts], (50, 100, 230))
        # Concentric circles
        cv2.circle(img, (w * 3 // 4, h * 3 // 4), min(h, w) // 5, (230, 230, 50), 4)
        cv2.circle(img, (w * 3 // 4, h * 3 // 4), min(h, w) // 10, (230, 50, 230), -1)
        return img

    elif pattern_type == "color_chart":
        img = np.zeros((h, w, 3), dtype=np.uint8)
        rows, cols = 4, 6
        colors = [
            (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
            (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
            (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161),
            (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52)
        ]
        cell_h, cell_w = h // rows, w // cols
        idx = 0
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                color = colors[idx % len(colors)]
                img[y1:y2, x1:x2] = color
                idx += 1
        return img

    elif pattern_type == "texture":
        # Zone plate / Circular sine grid
        x = np.linspace(-10, 10, w)
        y = np.linspace(-10, 10, h)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        z = np.sin(r**2)
        z_norm = ((z + 1) / 2.0 * 255).astype(np.uint8)
        return cv2.cvtColor(z_norm, cv2.COLOR_GRAY2RGB)

    else:
        return generate_synthetic_image("checkerboard", size)


def encode_image_to_bytes(img: np.ndarray, format_str: str = "PNG", quality: int = 95) -> bytes:
    """Encodes a numpy image array to PNG/JPEG bytes."""
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = Image.fromarray(img, mode="RGB")
        
    buf = io.BytesIO()
    if format_str.upper() in ("JPG", "JPEG"):
        pil_img.save(buf, format="JPEG", quality=quality)
    else:
        pil_img.save(buf, format="PNG")
    return buf.getvalue()
