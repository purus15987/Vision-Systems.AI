"""
Digital Image Processing Toolkit (DIP Toolkit) - Streamlit Web Application.
Main entry point for interactive image processing, analysis, visualization, and export.
"""

import os
import sys
import io
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import Toolkit Modules
from config import APP_TITLE, APP_SUBTITLE, SAMPLES_DIR, OUTPUTS_DIR, PROCESSED_DIR, REPORTS_DIR, THEME
from toolkit.io import (
    load_image, extract_metadata, generate_synthetic_image, encode_image_to_bytes
)
from toolkit.visualization import (
    convert_to_grayscale, convert_to_binary, split_channels, merge_channels,
    inspect_pixel, plot_side_by_side, plot_3d_color_space
)
from toolkit.transforms import (
    resize_image, rotate_image, flip_image, adjust_brightness_contrast,
    negative_transform, log_transform, inverse_log_transform, gamma_correction,
    piecewise_linear_transform, apply_threshold_slice
)
from toolkit.histogram import (
    compute_histogram, compute_cdf, histogram_stretching, histogram_equalization,
    apply_clahe, histogram_matching, plot_histogram
)
from toolkit.filters import (
    mean_filter, gaussian_filter, median_filter, bilateral_filter,
    laplacian_sharpen, unsharp_masking, high_boost_filter,
    add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise
)
from toolkit.frequency import (
    compute_fft, apply_frequency_filter, generate_frequency_filter_mask
)
from toolkit.restoration import (
    simulate_blur, inverse_filter, wiener_filter, richardson_lucy_deconvolution
)
from toolkit.color import (
    convert_color_space, white_balance_gray_world, white_balance_white_patch,
    enhance_saturation, reinhard_tone_mapping
)
from toolkit.morphology import (
    apply_binary_morphology, apply_advanced_morphology
)
from toolkit.segmentation import (
    threshold_global, threshold_adaptive, threshold_otsu,
    detect_edges, watershed_segmentation, kmeans_segmentation
)
from toolkit.features import (
    extract_contours_and_shapes, detect_harris_corners, detect_orb_features,
    extract_hog_features, extract_lbp_texture
)
from toolkit.compression import (
    simulate_jpeg_compression, analyze_compression_quality_tradeoff
)
from toolkit.quality import (
    evaluate_quality_metrics
)
from toolkit.export import (
    generate_pdf_report, export_metrics_to_csv, create_zip_archive
)
from toolkit.utils import (
    HistoryManager, PipelineManager, detect_gpu_acceleration
)

# Optional image comparison slider
try:
    from streamlit_image_comparison import image_comparison
    HAS_IMAGE_COMPARISON = True
except ImportError:
    HAS_IMAGE_COMPARISON = False


# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="DIP Toolkit | Vision.AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Slate Dark Aesthetic
st.markdown("""
<style>
    /* Dark Glassmorphism Styling */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .css-1d3 Sterling {
        background-color: #1E293B;
    }
    div[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    .stButton button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton button:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #10B981;
    }
    .metric-label {
        font-size: 13px;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Session State Initialization
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = HistoryManager()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = PipelineManager()

if "current_image" not in st.session_state:
    # Default synthetic checkerboard image
    st.session_state.original_image = generate_synthetic_image("checkerboard", size=(512, 512))
    st.session_state.current_image = st.session_state.original_image.copy()
    st.session_state.history.reset(st.session_state.original_image)


def update_current_image(new_img: np.ndarray, step_name: str, params: dict):
    """Updates active image and pushes state to history stack."""
    st.session_state.history.push(st.session_state.current_image)
    st.session_state.current_image = new_img.copy()
    st.session_state.pipeline.add_step(step_name, params)


# ==========================================
# Header & Navigation Toolbar
# ==========================================
col_h1, col_h2 = st.columns([3, 1])

with col_h1:
    st.title("🖼️ DIP Toolkit — Digital Image Processing Studio")
    st.caption("Modular Python Computer Vision & Image Processing Engine | DIP-Vision.AI")

with col_h2:
    gpu_info = detect_gpu_acceleration()
    if gpu_info["cuda_available"]:
        st.success(f"⚡ GPU Accelerated (CUDA Active - {gpu_info['cuda_devices']} Device)")
    elif gpu_info["opencl_available"]:
        st.info("⚡ OpenCL Acceleration Active")
    else:
        st.caption("💻 CPU Execution Engine Active")

# Action Bar (Undo, Redo, Reset, Save/Load Pipeline)
act_c1, act_c2, act_c3, act_c4, act_c5, act_c6 = st.columns(6)

with act_c1:
    if st.button("↩️ Undo"):
        img, ok = st.session_state.history.undo(st.session_state.current_image)
        if ok:
            st.session_state.current_image = img
            st.rerun()

with act_c2:
    if st.button("↪️ Redo"):
        img, ok = st.session_state.history.redo(st.session_state.current_image)
        if ok:
            st.session_state.current_image = img
            st.rerun()

with act_c3:
    if st.button("🔄 Reset Original"):
        st.session_state.current_image = st.session_state.original_image.copy()
        st.session_state.history.reset(st.session_state.original_image)
        st.rerun()

with act_c4:
    pipe_json = json.dumps({"pipeline": st.session_state.pipeline.steps}, indent=2)
    st.download_button(
        "💾 Export Pipeline",
        data=pipe_json,
        file_name="processing_pipeline.json",
        mime="application/json"
    )

with act_c5:
    uploaded_pipe = st.file_uploader("📂 Load Pipeline JSON", type=["json"], label_visibility="collapsed")
    if uploaded_pipe is not None:
        try:
            data = json.load(uploaded_pipe)
            st.session_state.pipeline.steps = data.get("pipeline", [])
            st.success("Pipeline loaded!")
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")

with act_c6:
    curr_bytes = encode_image_to_bytes(st.session_state.current_image, "PNG")
    st.download_button(
        "📥 Download Image",
        data=curr_bytes,
        file_name="processed_image.png",
        mime="image/png"
    )

st.markdown("---")

# ==========================================
# Sidebar: Image Source Selection
# ==========================================
st.sidebar.header("📁 Image Source Input")
source_type = st.sidebar.radio(
    "Select Source",
    ["Upload File", "Pre-bundled Sample", "Synthetic Generator", "Webcam Capture"]
)

if source_type == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Image (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    if uploaded_file is not None:
        try:
            loaded_img = load_image(uploaded_file.getvalue())
            if not np.array_equal(loaded_img, st.session_state.original_image):
                st.session_state.original_image = loaded_img
                st.session_state.current_image = loaded_img.copy()
                st.session_state.history.reset(loaded_img)
        except Exception as e:
            st.sidebar.error(f"Error loading image: {e}")

elif source_type == "Pre-bundled Sample":
    sample_files = list(SAMPLES_DIR.glob("*.png")) + list(SAMPLES_DIR.glob("*.jpg"))
    sample_names = [f.name for f in sample_files] if sample_files else ["sample_checkerboard.png"]
    selected_sample = st.sidebar.selectbox("Choose Sample Image", sample_names)
    sample_path = SAMPLES_DIR / selected_sample
    if sample_path.exists():
        loaded_img = load_image(str(sample_path))
        if not np.array_equal(loaded_img, st.session_state.original_image):
            st.session_state.original_image = loaded_img
            st.session_state.current_image = loaded_img.copy()
            st.session_state.history.reset(loaded_img)

elif source_type == "Synthetic Generator":
    pattern = st.sidebar.selectbox("Select Pattern", ["checkerboard", "gradient", "geometric", "color_chart", "texture"])
    size_px = st.sidebar.slider("Image Dimension (px)", 256, 1024, 512, step=128)
    if st.sidebar.button("Generate Pattern"):
        synth_img = generate_synthetic_image(pattern, size=(size_px, size_px))
        st.session_state.original_image = synth_img
        st.session_state.current_image = synth_img.copy()
        st.session_state.history.reset(synth_img)
        st.rerun()

elif source_type == "Webcam Capture":
    webcam_img = st.sidebar.camera_input("Take a photo")
    if webcam_img is not None:
        loaded_img = load_image(webcam_img.getvalue())
        st.session_state.original_image = loaded_img
        st.session_state.current_image = loaded_img.copy()
        st.session_state.history.reset(loaded_img)

# Display Metadata Summary Card in Sidebar
meta = extract_metadata(st.session_state.current_image)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Active Image Metadata")
st.sidebar.write(f"**Dimensions:** {meta['width']} × {meta['height']} px")
st.sidebar.write(f"**Channels:** {meta['channels']} ({meta['color_mode']})")
st.sidebar.write(f"**Bit Depth:** {meta['bit_depth']}-bit ({meta['dtype']})")
st.sidebar.write(f"**Aspect Ratio:** {meta['aspect_ratio']}")
st.sidebar.write(f"**Raw Size:** ~{meta['file_size_kb']} KB")


# ==========================================
# Main Visual Preview Area
# ==========================================
st.subheader("👁️ Visual Output Comparison")
view_mode = st.radio("Display Mode", ["Side-by-Side", "Interactive Slider", "Processed Only"], horizontal=True)

if view_mode == "Side-by-Side":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Image**")
        st.image(st.session_state.original_image, use_container_width=True)
    with c2:
        st.markdown("**Processed Image**")
        st.image(st.session_state.current_image, use_container_width=True)

elif view_mode == "Interactive Slider":
    if HAS_IMAGE_COMPARISON:
        image_comparison(
            img1=st.session_state.original_image,
            img2=st.session_state.current_image,
            label1="Original",
            label2="Processed"
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.image(st.session_state.original_image, caption="Original Image", use_container_width=True)
        with c2:
            st.image(st.session_state.current_image, caption="Processed Image", use_container_width=True)

else:
    st.image(st.session_state.current_image, use_container_width=True)

st.markdown("---")

# ==========================================
# Categorized Operations Tabs (13 Tabs)
# ==========================================
tabs = st.tabs([
    "🖼️ Fundamentals",
    "📊 Histograms",
    "🧹 Filtering",
    "🌊 Frequency",
    "🩹 Restoration",
    "🎨 Color Space",
    "🔍 Morphology",
    "✂️ Segmentation",
    "📐 Features",
    "🗜️ Compression",
    "⚡ Batch Processing",
    "📄 Report Export",
    "⚙️ Pipeline Trace"
])

# ------------------------------------------
# Tab 1: Image Fundamentals & Inspector
# ------------------------------------------
with tabs[0]:
    st.header("🖼️ Image Fundamentals & Representation")

    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["Channel Breakdown", "Pixel Inspector", "Spatial Transforms", "Grayscale/Binary"])

    with sub_t1:
        st.subheader("RGB Channel Splitting & Merging")
        channels = split_channels(st.session_state.current_image)
        if "Red_Channel" in channels:
            ch_col1, ch_col2, ch_col3 = st.columns(3)
            with ch_col1:
                st.image(channels["Red_Visual"], caption="Red Channel", use_container_width=True)
            with ch_col2:
                st.image(channels["Green_Visual"], caption="Green Channel", use_container_width=True)
            with ch_col3:
                st.image(channels["Blue_Visual"], caption="Blue Channel", use_container_width=True)

    with sub_t2:
        st.subheader("Pixel Value Inspection")
        h, w = st.session_state.current_image.shape[:2]
        px_c1, px_c2 = st.columns(2)
        with px_c1:
            x_coord = st.number_input("X Coordinate", 0, w - 1, w // 2)
            y_coord = st.number_input("Y Coordinate", 0, h - 1, h // 2)
        with px_c2:
            px_info = inspect_pixel(st.session_state.current_image, x_coord, y_coord)
            st.json(px_info)

    with sub_t3:
        st.subheader("Spatial Transformations")
        st_c1, st_c2, st_c3 = st.columns(3)
        with st_c1:
            rot_angle = st.slider("Rotate Angle (°)", -180, 180, 0)
            if st.button("Apply Rotation"):
                res = rotate_image(st.session_state.current_image, rot_angle)
                update_current_image(res, "rotate", {"angle": rot_angle})
                st.rerun()

        with st_c2:
            flip_mode = st.selectbox("Flip Axis", ["horizontal", "vertical", "both"])
            if st.button("Apply Flip"):
                res = flip_image(st.session_state.current_image, flip_mode)
                update_current_image(res, "flip", {"mode": flip_mode})
                st.rerun()

        with st_c3:
            scale_pct = st.slider("Resize Scale (%)", 10, 200, 100)
            if st.button("Apply Resize"):
                new_w = int(w * scale_pct / 100.0)
                res = resize_image(st.session_state.current_image, width=new_w)
                update_current_image(res, "resize", {"scale_percent": scale_pct})
                st.rerun()

    with sub_t4:
        st.subheader("Representation Conversions")
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            if st.button("Convert to Grayscale"):
                gray = convert_to_grayscale(st.session_state.current_image)
                update_current_image(gray, "grayscale", {})
                st.rerun()
        with rep_col2:
            thresh = st.slider("Binarization Threshold", 0, 255, 128)
            if st.button("Convert to Binary"):
                binary = convert_to_binary(st.session_state.current_image, thresh)
                update_current_image(binary, "binary", {"threshold": thresh})
                st.rerun()

# ------------------------------------------
# Tab 2: Histograms & Intensity Transformations
# ------------------------------------------
with tabs[1]:
    st.header("📊 Histogram Processing & Intensity Transformations")

    hist_c1, hist_c2 = st.columns([1, 1])

    with hist_c1:
        st.subheader("Interactive Histogram & CDF")
        fig_hist = plot_histogram(st.session_state.current_image, "Active Image Histogram")
        st.plotly_chart(fig_hist, use_container_width=True)

    with hist_c2:
        st.subheader("Histogram Enhancements")
        if st.button("Apply Contrast Stretching"):
            res = histogram_stretching(st.session_state.current_image)
            update_current_image(res, "histogram_stretch", {})
            st.rerun()

        if st.button("Apply Global Histogram Equalization"):
            res = histogram_equalization(st.session_state.current_image)
            update_current_image(res, "histogram_equalize", {})
            st.rerun()

        clip = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, step=0.5)
        grid = st.slider("CLAHE Tile Grid Size", 2, 16, 8, step=2)
        if st.button("Apply CLAHE"):
            res = apply_clahe(st.session_state.current_image, clip, (grid, grid))
            update_current_image(res, "clahe", {"clip_limit": clip, "grid_size": grid})
            st.rerun()

    st.markdown("---")
    st.subheader("Intensity Transformations")
    it_c1, it_c2, it_c3 = st.columns(3)

    with it_c1:
        bright = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", 0.1, 3.0, 1.0, step=0.1)
        if st.button("Apply Brightness/Contrast"):
            res = adjust_brightness_contrast(st.session_state.current_image, bright, contrast)
            update_current_image(res, "brightness_contrast", {"brightness": bright, "contrast": contrast})
            st.rerun()

    with it_c2:
        gamma = st.slider("Gamma Correction (γ)", 0.1, 5.0, 1.0, step=0.1)
        if st.button("Apply Gamma Correction"):
            res = gamma_correction(st.session_state.current_image, gamma)
            update_current_image(res, "gamma", {"gamma": gamma})
            st.rerun()

        if st.button("Apply Negative Transform"):
            res = negative_transform(st.session_state.current_image)
            update_current_image(res, "negative", {})
            st.rerun()

    with it_c3:
        if st.button("Apply Log Transform"):
            res = log_transform(st.session_state.current_image)
            update_current_image(res, "log_transform", {})
            st.rerun()

        if st.button("Apply Inverse Log Transform"):
            res = inverse_log_transform(st.session_state.current_image)
            update_current_image(res, "inverse_log_transform", {})
            st.rerun()

# ------------------------------------------
# Tab 3: Filtering & Sharpening
# ------------------------------------------
with tabs[2]:
    st.header("🧹 Spatial Filtering, Sharpening & Noise Simulation")

    flt_c1, flt_c2, flt_c3 = st.columns(3)

    with flt_c1:
        st.subheader("Smoothing Filters")
        s_filter = st.selectbox("Smoothing Filter", ["Mean", "Gaussian", "Median", "Bilateral"])
        k_size = st.slider("Kernel Size", 3, 21, 5, step=2)
        sigma = st.slider("Gaussian Sigma", 0.1, 10.0, 1.0)

        if st.button("Apply Smoothing Filter"):
            if s_filter == "Mean":
                res = mean_filter(st.session_state.current_image, k_size)
            elif s_filter == "Gaussian":
                res = gaussian_filter(st.session_state.current_image, k_size, sigma)
            elif s_filter == "Median":
                res = median_filter(st.session_state.current_image, k_size)
            else:
                res = bilateral_filter(st.session_state.current_image, d=k_size)
            update_current_image(res, "smoothing", {"filter": s_filter, "kernel_size": k_size})
            st.rerun()

    with flt_c2:
        st.subheader("Sharpening Filters")
        sh_filter = st.selectbox("Sharpening Method", ["Laplacian", "Unsharp Masking", "High-Boost"])
        boost_A = st.slider("High-Boost Factor (A)", 1.0, 5.0, 2.0, step=0.1)

        if st.button("Apply Sharpening"):
            if sh_filter == "Laplacian":
                res = laplacian_sharpen(st.session_state.current_image)
            elif sh_filter == "Unsharp Masking":
                res = unsharp_masking(st.session_state.current_image, k_size, sigma)
            else:
                res = high_boost_filter(st.session_state.current_image, A=boost_A, kernel_size=k_size, sigma=sigma)
            update_current_image(res, "sharpening", {"method": sh_filter})
            st.rerun()

    with flt_c3:
        st.subheader("Noise Simulation")
        noise_type = st.selectbox("Noise Type", ["Gaussian Noise", "Salt & Pepper", "Speckle Noise"])
        noise_param = st.slider("Noise Intensity", 0.01, 0.2, 0.05, step=0.01)

        if st.button("Inject Noise"):
            if noise_type == "Gaussian Noise":
                res = add_gaussian_noise(st.session_state.current_image, std=noise_param * 100.0)
            elif noise_type == "Salt & Pepper":
                res = add_salt_and_pepper_noise(st.session_state.current_image, salt_prob=noise_param, pepper_prob=noise_param)
            else:
                res = add_speckle_noise(st.session_state.current_image, variance=noise_param)
            update_current_image(res, "noise_injection", {"type": noise_type, "intensity": noise_param})
            st.rerun()

# ------------------------------------------
# Tab 4: Frequency Domain Processing
# ------------------------------------------
with tabs[3]:
    st.header("🌊 Frequency Domain Processing (2D FFT)")

    freq_c1, freq_c2 = st.columns([1, 1])

    with freq_c1:
        st.subheader("2D FFT Spectrum Analysis")
        if st.button("Compute Magnitude & Phase Spectrum"):
            _, mag, phase = compute_fft(st.session_state.current_image)
            sp_c1, sp_c2 = st.columns(2)
            with sp_c1:
                st.image(mag, caption="Magnitude Spectrum |F(u,v)|", use_container_width=True)
            with sp_c2:
                st.image(phase, caption="Phase Spectrum ∠F(u,v)", use_container_width=True)

    with freq_c2:
        st.subheader("Frequency Domain Filtering")
        filter_type = st.selectbox(
            "Filter Transfer Function",
            ["gaussian_lowpass", "gaussian_highpass", "ideal_lowpass", "ideal_highpass",
             "butterworth_lowpass", "butterworth_highpass", "bandpass", "bandstop"]
        )
        cutoff = st.slider("Cutoff Frequency (D0)", 5, 200, 50)
        cutoff_high = st.slider("High Cutoff (Band Pass/Stop)", cutoff + 5, 250, cutoff + 50)
        order = st.slider("Butterworth Order (n)", 1, 10, 2)

        if st.button("Apply Frequency Filter"):
            res, mask_vis = apply_frequency_filter(
                st.session_state.current_image,
                filter_type=filter_type,
                cutoff=cutoff,
                cutoff_high=cutoff_high,
                order=order
            )
            st.image(mask_vis, caption="Filter Transfer Function Mask H(u,v)", width=200)
            update_current_image(res, "frequency_filter", {"type": filter_type, "cutoff": cutoff})
            st.rerun()

# ------------------------------------------
# Tab 5: Image Restoration & Deblurring
# ------------------------------------------
with tabs[4]:
    st.header("🩹 Image Restoration & Deblurring")

    rest_c1, rest_c2 = st.columns(2)

    with rest_c1:
        st.subheader("1. Blur Degradation Simulation")
        blur_mode = st.selectbox("Blur Type", ["motion", "defocus", "gaussian"])
        blur_size = st.slider("PSF Kernel Size", 3, 31, 15, step=2)
        motion_angle = st.slider("Motion Angle (°)", 0.0, 180.0, 45.0)

        if st.button("Simulate Blur Degradation"):
            blurred, psf = simulate_blur(st.session_state.current_image, blur_type=blur_mode, size=blur_size, angle=motion_angle)
            st.session_state.current_psf = psf
            update_current_image(blurred, "simulate_blur", {"mode": blur_mode, "size": blur_size})
            st.rerun()

    with rest_c2:
        st.subheader("2. Restoration & Deblurring")
        algo = st.selectbox("Restoration Algorithm", ["Inverse Filtering", "Wiener Filtering", "Richardson-Lucy Deconvolution"])
        k_param = st.slider("Wiener K (Noise Factor)", 0.001, 0.1, 0.01, step=0.001)
        rl_iters = st.slider("Richardson-Lucy Iterations", 5, 100, 30, step=5)

        if st.button("Apply Restoration"):
            psf = getattr(st.session_state, "current_psf", None)
            if psf is None:
                psf = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)).astype(np.float32)
                psf /= np.sum(psf)

            if algo == "Inverse Filtering":
                restored = inverse_filter(st.session_state.current_image, psf)
            elif algo == "Wiener Filtering":
                restored = wiener_filter(st.session_state.current_image, psf, K=k_param)
            else:
                restored = richardson_lucy_deconvolution(st.session_state.current_image, psf, num_iter=rl_iters)

            update_current_image(restored, "restoration", {"algorithm": algo})
            st.rerun()

# ------------------------------------------
# Tab 6: Color Image Processing
# ------------------------------------------
with tabs[5]:
    st.header("🎨 Color Image Processing & Color Spaces")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Color Space Conversion")
        target_cs = st.selectbox("Select Color Space", ["HSV", "LAB", "XYZ", "YCbCr", "HSL"])
        if st.button("Convert & Inspect Channels"):
            converted, ch_dict = convert_color_space(st.session_state.current_image, target_cs)
            st.write(f"Converted to {target_cs} space.")
            cc_cols = st.columns(len(ch_dict))
            for i, (k, v) in enumerate(ch_dict.items()):
                with cc_cols[i]:
                    st.image(v, caption=k, use_container_width=True)

            # Option to display 3D RGB plot
            fig_3d = plot_3d_color_space(st.session_state.current_image)
            st.plotly_chart(fig_3d, use_container_width=True)

    with col_c2:
        st.subheader("White Balance & Color Enhancement")
        if st.button("Apply Gray World White Balance"):
            res = white_balance_gray_world(st.session_state.current_image)
            update_current_image(res, "gray_world_wb", {})
            st.rerun()

        if st.button("Apply White Patch White Balance"):
            res = white_balance_white_patch(st.session_state.current_image)
            update_current_image(res, "white_patch_wb", {})
            st.rerun()

        sat_factor = st.slider("Saturation Boost Factor", 0.5, 3.0, 1.5, step=0.1)
        if st.button("Enhance Saturation"):
            res = enhance_saturation(st.session_state.current_image, sat_factor)
            update_current_image(res, "enhance_saturation", {"factor": sat_factor})
            st.rerun()

        if st.button("Apply Reinhard Tone Mapping"):
            res = reinhard_tone_mapping(st.session_state.current_image)
            update_current_image(res, "reinhard_tone_mapping", {})
            st.rerun()

# ------------------------------------------
# Tab 7: Morphological Processing
# ------------------------------------------
with tabs[6]:
    st.header("🔍 Morphological Processing")

    morph_c1, morph_c2 = st.columns(2)

    with morph_c1:
        st.subheader("Binary Morphology")
        b_op = st.selectbox("Operation", ["erosion", "dilation", "opening", "closing"])
        k_shape = st.selectbox("Kernel Shape", ["rect", "ellipse", "cross"])
        k_sz = st.slider("Structuring Element Size", 3, 21, 5, step=2)

        if st.button("Apply Binary Morphology"):
            res = apply_binary_morphology(st.session_state.current_image, operation=b_op, kernel_shape=k_shape, kernel_size=k_sz)
            update_current_image(res, "morphology_binary", {"op": b_op, "shape": k_shape, "size": k_sz})
            st.rerun()

    with morph_c2:
        st.subheader("Advanced Morphological Ops")
        adv_op = st.selectbox("Advanced Op", ["gradient", "tophat", "blackhat", "skeletonization", "thinning", "hit_or_miss"])

        if st.button("Apply Advanced Morphology"):
            res = apply_advanced_morphology(st.session_state.current_image, operation=adv_op, kernel_shape=k_shape, kernel_size=k_sz)
            update_current_image(res, "morphology_advanced", {"op": adv_op})
            st.rerun()

# ------------------------------------------
# Tab 8: Image Segmentation
# ------------------------------------------
with tabs[7]:
    st.header("✂️ Image Segmentation")

    seg_c1, seg_c2, seg_c3 = st.columns(3)

    with seg_c1:
        st.subheader("Thresholding")
        t_mode = st.selectbox("Threshold Technique", ["Global", "Adaptive Mean", "Adaptive Gaussian", "Otsu"])
        g_val = st.slider("Global Threshold Value", 0, 255, 128)

        if st.button("Apply Threshold Segmentation"):
            if t_mode == "Global":
                res = threshold_global(st.session_state.current_image, g_val)
            elif t_mode == "Adaptive Mean":
                res = threshold_adaptive(st.session_state.current_image, method="mean")
            elif t_mode == "Adaptive Gaussian":
                res = threshold_adaptive(st.session_state.current_image, method="gaussian")
            else:
                res, otsu_val = threshold_otsu(st.session_state.current_image)
                st.info(f"Otsu Optimal Threshold Calculated: {otsu_val:.1f}")
            update_current_image(res, "segmentation_threshold", {"mode": t_mode})
            st.rerun()

    with seg_c2:
        st.subheader("Edge Detection")
        edge_op = st.selectbox("Edge Operator", ["canny", "sobel", "prewitt", "roberts", "laplacian"])
        c_low = st.slider("Canny Low Threshold", 10, 150, 50)
        c_high = st.slider("Canny High Threshold", 50, 300, 150)

        if st.button("Detect Edges"):
            res = detect_edges(st.session_state.current_image, operator=edge_op, low_thresh=c_low, high_thresh=c_high)
            update_current_image(res, "edge_detection", {"operator": edge_op})
            st.rerun()

    with seg_c3:
        st.subheader("Region & Clustering")
        clusters_k = st.slider("K-Means Clusters (K)", 2, 10, 4)

        if st.button("Apply Watershed Segmentation"):
            res = watershed_segmentation(st.session_state.current_image)
            update_current_image(res, "watershed", {})
            st.rerun()

        if st.button("Apply K-Means Clustering"):
            res = kmeans_segmentation(st.session_state.current_image, K=clusters_k)
            update_current_image(res, "kmeans_clustering", {"K": clusters_k})
            st.rerun()

# ------------------------------------------
# Tab 9: Feature Extraction
# ------------------------------------------
with tabs[8]:
    st.header("📐 Feature Extraction & Shape Analysis")

    feat_c1, feat_c2 = st.columns(2)

    with feat_c1:
        st.subheader("Contours & Shape Analysis")
        min_contour_area = st.number_input("Minimum Contour Area (px²)", 10, 5000, 100)
        if st.button("Extract Contours & Calculate Shape Metrics"):
            annotated, metrics_list = extract_contours_and_shapes(st.session_state.current_image, min_area=min_contour_area)
            st.image(annotated, caption="Contours (Green), Hulls (Orange), Bounding Boxes (Blue)", use_container_width=True)
            if metrics_list:
                df_metrics = pd.DataFrame(metrics_list)
                st.dataframe(df_metrics, use_container_width=True)

    with feat_c2:
        st.subheader("Interest Points & Descriptors")
        feat_type = st.selectbox("Feature Extractor", ["Harris Corners", "ORB Keypoints", "HOG Descriptor", "LBP Texture"])

        if st.button("Extract Features"):
            if feat_type == "Harris Corners":
                res = detect_harris_corners(st.session_state.current_image)
            elif feat_type == "ORB Keypoints":
                res, n_kp = detect_orb_features(st.session_state.current_image)
                st.info(f"Detected {n_kp} ORB keypoint descriptors.")
            elif feat_type == "HOG Descriptor":
                _, res = extract_hog_features(st.session_state.current_image)
            else:
                res = extract_lbp_texture(st.session_state.current_image)
            st.image(res, caption=f"Extracted {feat_type}", use_container_width=True)

# ------------------------------------------
# Tab 10: Compression & Quality Assessment
# ------------------------------------------
with tabs[9]:
    st.header("🗜️ Image Compression & Quality Assessment (IQA)")

    comp_c1, comp_c2 = st.columns(2)

    with comp_c1:
        st.subheader("JPEG Compression Trade-off")
        jpeg_q = st.slider("JPEG Quality Factor (1-100)", 1, 100, 85)
        if st.button("Simulate JPEG Compression"):
            decomp, comp_meta = simulate_jpeg_compression(st.session_state.current_image, quality=jpeg_q)
            st.json(comp_meta)

            # Analyze rate-distortion curve
            tradeoff_data = analyze_compression_quality_tradeoff(st.session_state.current_image)
            df_td = pd.DataFrame(tradeoff_data)
            st.line_chart(df_td.set_index("quality_factor")[["psnr_db", "ssim"]])

    with comp_c2:
        st.subheader("Objective Image Quality Assessment")
        if st.button("Compute Quality Metrics vs Original Image"):
            iqa_metrics, heatmap = evaluate_quality_metrics(st.session_state.original_image, st.session_state.current_image)
            st.write("### Quantitative Metrics Table")
            df_iqa = pd.DataFrame([iqa_metrics])
            st.table(df_iqa)

            st.image(heatmap, caption="SSIM Structural Difference Heatmap (Red = High Error)", use_container_width=True)

# ------------------------------------------
# Tab 11: Batch Processing Engine
# ------------------------------------------
with tabs[10]:
    st.header("⚡ Batch Processing Engine")
    st.caption("Apply a processing pipeline across multiple images simultaneously.")

    batch_files = st.file_uploader("Upload Multiple Images for Batch Job", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    batch_op = st.selectbox("Select Batch Pipeline Operation", ["Grayscale Conversion", "Histogram Equalization", "Gaussian Blur", "Canny Edge Detection"])

    if st.button("Run Batch Processing Job") and batch_files:
        batch_out_files = []
        batch_dir = PROCESSED_DIR / "batch_job"
        batch_dir.mkdir(parents=True, exist_ok=True)

        for b_file in batch_files:
            b_img = load_image(b_file.getvalue())

            if batch_op == "Grayscale Conversion":
                proc = convert_to_grayscale(b_img)
            elif batch_op == "Histogram Equalization":
                proc = histogram_equalization(b_img)
            elif batch_op == "Gaussian Blur":
                proc = gaussian_filter(b_img, 5)
            else:
                proc = detect_edges(b_img, "canny")

            out_p = batch_dir / f"proc_{b_file.name}"
            Image.fromarray(proc).save(out_p)
            batch_out_files.append(str(out_p))

        zip_p = str(PROCESSED_DIR / "batch_results.zip")
        create_zip_archive(batch_out_files, zip_p)

        with open(zip_p, "rb") as f:
            st.download_button(
                "📥 Download Batch Results (ZIP)",
                data=f.read(),
                file_name="batch_results.zip",
                mime="application/zip"
            )
        st.success(f"Successfully batch processed {len(batch_files)} images!")

# ------------------------------------------
# Tab 12: Report Generator & Exports
# ------------------------------------------
with tabs[11]:
    st.header("📄 Technical Report & Data Export Center")

    rep_c1, rep_c2 = st.columns(2)

    with rep_c1:
        st.subheader("Generate PDF Technical Report")
        author_name = st.text_input("Author / Team Name", "DIP-Vision.AI Student Team")
        report_title = st.text_input("Report Title", "Digital Image Processing Project Technical Report")

        if st.button("Generate & Compile PDF Report"):
            pdf_path = str(REPORTS_DIR / "technical_report.pdf")
            metrics_dict, _ = evaluate_quality_metrics(st.session_state.original_image, st.session_state.current_image)
            generate_pdf_report(
                output_pdf_path=pdf_path,
                title=report_title,
                author=author_name,
                original_img=st.session_state.original_image,
                processed_img=st.session_state.current_image,
                pipeline_steps=st.session_state.pipeline.steps,
                metrics=metrics_dict
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download Technical Report (PDF)",
                    data=f.read(),
                    file_name="technical_report.pdf",
                    mime="application/pdf"
                )
            st.success("PDF Technical Report generated!")

    with rep_c2:
        st.subheader("Export CSV Data & ZIP Archives")
        if st.button("Export Active Metrics to CSV"):
            csv_path = str(REPORTS_DIR / "metrics_report.csv")
            metrics_dict, _ = evaluate_quality_metrics(st.session_state.original_image, st.session_state.current_image)
            export_metrics_to_csv([metrics_dict], csv_path)

            with open(csv_path, "rb") as f:
                st.download_button(
                    "📥 Download CSV Metrics",
                    data=f.read(),
                    file_name="metrics_report.csv",
                    mime="text/csv"
                )

# ------------------------------------------
# Tab 13: Pipeline Trace
# ------------------------------------------
with tabs[12]:
    st.header("⚙️ Applied Processing Pipeline JSON Trace")
    st.json({"pipeline": st.session_state.pipeline.steps})
