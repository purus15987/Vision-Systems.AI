# 🖼️ Digital Image Processing Toolkit (DIP Toolkit)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/GUI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, production-grade **Digital Image Processing (DIP) Toolkit** written in Python with an interactive **Streamlit Web GUI**. 

Designed for students, researchers, and computer vision engineers, this toolkit allows users to execute, analyze, compare, and export 15 core digital image processing operations, generate PDF technical reports, and execute batch processing pipelines.

---

## 🎯 Features Checklist

### 1. Image Fundamentals & Representation
- [x] Multi-format Loading (PNG, JPG, TIFF, BMP, Bytes, PIL, NumPy)
- [x] EXIF Metadata & Image Properties Extraction (Dimensions, Bit Depth, File Size, Aspect Ratio)
- [x] RGB & RGBA Channel Splitting, Isolation, and Merging
- [x] High-Precision Pixel Coordinate & Color Inspector Probe (RGB, Hex, HSV, Grayscale)
- [x] Spatial Transforms: Arbitrary Rotation, Horizontal/Vertical Flip, Aspect-Preserving Scaling

### 2. Histogram Processing & Intensity Transformations
- [x] Interactive RGB & Grayscale Histogram Computation with Cumulative Distribution Function (CDF)
- [x] Min-Max Contrast Stretching to [0, 255]
- [x] Global Histogram Equalization & Y-Channel Color Preserving Equalization
- [x] Contrast Limited Adaptive Histogram Equalization (CLAHE)
- [x] Histogram Matching & Histogram Specification
- [x] Brightness (-100 to +100) & Contrast (0.1 to 3.0) Tuning
- [x] Power-Law (Gamma Correction), Log, Inverse Log, and Negative Transformations
- [x] Piecewise Linear Contrast Stretching & Intensity-Level Slicing

### 3. Spatial Filtering & Noise Simulation
- [x] **Smoothing:** Mean (Average), Gaussian, Median (Impulse Removal), Bilateral (Edge-Preserving)
- [x] **Sharpening:** Laplacian Operator, Unsharp Masking, High-Boost Filter (Tunable factor A)
- [x] **Noise Simulation:** Additive Zero-Mean Gaussian Noise, Impulse (Salt & Pepper) Noise, Multiplicative Speckle Noise

### 4. Frequency Domain Processing (2D FFT)
- [x] 2D Fast Fourier Transform (FFT) & Shifted Centered Spectrum Computation
- [x] Logarithmic Magnitude Spectrum $|F(u,v)|$ and Phase Spectrum $\angle F(u,v)$ Visualization
- [x] **Frequency Domain Filters:** Ideal Low-Pass & High-Pass (ILPF/IHPF), Gaussian Low-Pass & High-Pass (GLPF/GHPF), Butterworth Low-Pass & High-Pass (BLPF/BHPF), Band-Pass & Band-Stop

### 5. Image Restoration & Deblurring
- [x] **Blur Simulation:** Motion Blur Point Spread Function (PSF) generator with angle/length control, Circular Defocus Blur, Gaussian Blur
- [x] **Non-Blind Restoration:** Truncated Frequency Inverse Filtering, Wiener Deconvolution (MMSE Noise-to-Signal ratio $K$), Richardson-Lucy Iterative Deconvolution

### 6. Color Image Processing
- [x] Conversions across 6 Color Spaces: **RGB, HSV, LAB, XYZ, YCbCr, HSL**
- [x] Interactive 3D RGB Color Cube Scatter Plot (Plotly)
- [x] **White Balance:** Gray World Hypothesis & White Patch (Max RGB) Algorithms
- [x] **Enhancement & Tone Mapping:** Saturation Boosting, Reinhard Tone Mapping

### 7. Morphological Image Processing
- [x] **Binary Morphology:** Erosion, Dilation, Morphological Opening, Morphological Closing
- [x] **Structuring Elements:** Rectangular, Elliptical, Cross-shaped Kernels
- [x] **Advanced Morphology:** Morphological Gradient, Top-Hat, Black-Hat, Medial Axis Skeletonization, Topological Thinning, Hit-or-Miss Transform

### 8. Image Segmentation & Edge Detection
- [x] **Thresholding:** Global Manual Threshold, Adaptive Mean/Gaussian, Otsu's Automated Variance Minimization
- [x] **Edge Detection:** Sobel, Prewitt, Roberts Cross, Laplacian of Gaussian, Canny Dual-Threshold Edge Detector
- [x] **Region & Clustering:** Marker-Based Watershed Segmentation, K-Means Color/Spatial Clustering

### 9. Feature Extraction & Texture Analysis
- [x] Contour Detection, Bounding Box Annotation, Convex Hull Calculation
- [x] Shape Metrics Table: Area, Perimeter, Centroid, Solidity, Circularity
- [x] Harris Corner Interest Point Detection
- [x] ORB (Oriented FAST and Rotated BRIEF) Feature Keypoint & Descriptor Visualizer
- [x] Histogram of Oriented Gradients (HOG) Descriptor Visualization
- [x] Local Binary Patterns (LBP) Texture Analysis

### 10. Compression & Image Quality Assessment (IQA)
- [x] JPEG Compression Simulation with Quality Factor (1-100) & Compression Ratio Calculation
- [x] Rate-Distortion Curves (PSNR & SSIM vs Quality Factor)
- [x] **Fidelity Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Peak Signal-to-Noise Ratio (PSNR in dB), Structural Similarity Index Measure (SSIM)
- [x] Structural Difference Heatmap Visualizer

### 11. Batch Processing & Automation
- [x] Multi-Image Upload & Batch Job Processing Engine
- [x] Automatic ZIP Output Packaging

### 12. Automated PDF Technical Report & Exports
- [x] Programmatically Compiled PDF Technical Report (ReportLab) with Tables, Formulations, Visual Figures
- [x] CSV Quantitative Metrics Exporter
- [x] JSON Processing Pipeline Serializer (Save & Load Pipeline States)
- [x] Interactive Undo / Redo History Stack

---

## 🗂️ Project Structure

```text
VisionSystem/
├── app.py                     # Main Streamlit Web Application
├── config.py                  # Project Configuration & Styling Tokens
├── requirements.txt           # Python Package Dependencies
├── README.md                  # Comprehensive Documentation
├── technical_report.pdf       # Generated 5-10 Page PDF Technical Report
│
├── toolkit/                   # Modular DIP Package
│   ├── __init__.py
│   ├── io.py                  # Image I/O, Metadata & Synthetic Pattern Generator
│   ├── visualization.py       # Render Routines, Pixel Inspector, 3D Color Cube Plotly
│   ├── transforms.py          # Spatial & Intensity Transformations
│   ├── histogram.py           # Histograms, CLAHE, Matching & Equalization
│   ├── filters.py             # Smoothing, Sharpening & Noise Generators
│   ├── frequency.py           # 2D FFT, Magnitude/Phase & Frequency Filters
│   ├── restoration.py         # Motion Blur, Inverse Filter, Wiener, Richardson-Lucy
│   ├── color.py               # 6 Color Spaces, White Balance, Tone Mapping
│   ├── morphology.py          # Binary & Advanced Morphological Operators
│   ├── segmentation.py        # Otsu, Canny, Watershed, K-Means Clustering
│   ├── features.py             # Contours, Shape Analysis, Harris, ORB, HOG, LBP
│   ├── compression.py         # Compression Ratio & Rate-Distortion Curves
│   ├── quality.py             # MSE, RMSE, PSNR, SSIM & Heatmaps
│   ├── export.py              # ReportLab PDF Report, CSV, ZIP Archiving
│   └── utils.py               # Undo/Redo Stack, JSON Pipeline, CUDA Detection
│
├── samples/                   # Pre-bundled Test Target Images
└── outputs/                   # Generated Results (Processed, Reports, Screenshots)
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/purus15987/vision-systems.ai.git
cd vision-systems.ai/projects/project01/<Team-Name>
pip install -r requirements.txt
```

### 2. Generate Pre-bundled Test Samples

```bash
python scripts/generate_samples.py
```

### 3. Launch Streamlit Web GUI

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧮 Mathematical Foundations

### 1. 2D Discrete Fourier Transform (FFT)
$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

### 2. Wiener Deconvolution Filter
$$\hat{F}(u, v) = \left[ \frac{|H(u, v)|^2}{|H(u, v)|^2 + K} \right] \frac{G(u, v)}{H(u, v)}$$

### 3. Structural Similarity Index (SSIM)
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

---

## 🌐 Deployment Instructions

### A. Streamlit Community Cloud
1. Push repository to GitHub.
2. Login to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect GitHub repository, set `Main file path` to `app.py`.
4. Click **Deploy**.

### B. Hugging Face Spaces
1. Create a Space on Hugging Face (SDK: Streamlit).
2. Upload `app.py`, `config.py`, `toolkit/`, `requirements.txt`.
3. Space will automatically build and host your application.

### C. Render Web Service
1. Create a Web Service on Render.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `streamlit run app.py --server.port $PORT`

---

## 👥 Authors & Team
- **Course:** DIP-Vision.AI
- **Release Date:** July 04, 2026
- **Deadline Date:** July 12, 2026
