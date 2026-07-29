"""
Generate sample test images in samples/ directory for instant toolkit testing.
"""

import os
from pathlib import Path
import numpy as np
import cv2

from toolkit.io import generate_synthetic_image
from config import SAMPLES_DIR

def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    patterns = ["checkerboard", "gradient", "geometric", "color_chart", "texture"]
    for pattern in patterns:
        img = generate_synthetic_image(pattern, size=(512, 512))
        out_path = SAMPLES_DIR / f"sample_{pattern}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Generated sample: {out_path}")

if __name__ == "__main__":
    main()
