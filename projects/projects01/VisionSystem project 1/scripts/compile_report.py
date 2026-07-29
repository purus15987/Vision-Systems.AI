"""
Script to generate pre-bundled sample images and compile technical_report.pdf
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLES_DIR, REPORTS_DIR, BASE_DIR
from toolkit.io import generate_synthetic_image, load_image
from toolkit.export import generate_pdf_report
from toolkit.filters import gaussian_filter, laplacian_sharpen
from toolkit.quality import evaluate_quality_metrics
import cv2

def main():
    print("Generating synthetic sample images...")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    patterns = ["checkerboard", "gradient", "geometric", "color_chart", "texture"]
    sample_imgs = {}

    for pattern in patterns:
        img = generate_synthetic_image(pattern, size=(512, 512))
        out_path = SAMPLES_DIR / f"sample_{pattern}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sample_imgs[pattern] = img
        print(f"  [OK] Saved: {out_path}")

    # Generate sample processing operation for report demonstration
    orig = sample_imgs["geometric"]
    proc = laplacian_sharpen(orig)

    metrics_dict, _ = evaluate_quality_metrics(orig, proc)

    pdf_out = str(BASE_DIR / "technical_report.pdf")
    print(f"\nCompiling Technical Report PDF at: {pdf_out}...")

    generate_pdf_report(
        output_pdf_path=pdf_out,
        title="Digital Image Processing Technical Report",
        author="DIP-Vision.AI Project Team",
        original_img=orig,
        processed_img=proc,
        pipeline_steps=[
            {"operation": "Grayscale & Representation", "params": {"color_mode": "RGB"}},
            {"operation": "Laplacian Sharpening Filter", "params": {"alpha": 1.0, "kernel": "3x3"}}
        ],
        metrics=metrics_dict,
        additional_notes="Report compiled programmatically via DIP Toolkit PDF Export Engine."
    )

    print(f"  [OK] PDF generated successfully: {pdf_out}")

if __name__ == "__main__":
    main()
