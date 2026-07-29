"""
Configuration file for Digital Image Processing Toolkit (DIP Toolkit).
Contains global settings, default paths, color themes, and constants.
"""

import os
from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"
DOCS_DIR = BASE_DIR / "docs"

# Create necessary directories
for folder in [SAMPLES_DIR, OUTPUTS_DIR, PROCESSED_DIR, REPORTS_DIR, SCREENSHOTS_DIR, DOCS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# App Title & Branding
APP_TITLE = "DIP Toolkit - Digital Image Processing Studio"
APP_SUBTITLE = "Modular Python Toolkit for Computer Vision & Image Processing"
COURSE_NAME = "DIP-Vision.AI"
RELEASE_DATE = "July 04, 2026"
DEADLINE_DATE = "July 12, 2026"

# UI Aesthetic Theme Settings (Modern Dark / Slate Glass)
THEME = {
    "primary": "#4F46E5",       # Indigo / Electric Blue
    "secondary": "#10B981",     # Emerald Green
    "accent": "#F59E0B",        # Amber
    "bg_dark": "#0F172A",       # Slate 900
    "bg_card": "#1E293B",       # Slate 800
    "text_main": "#F8FAFC",     # Slate 50
    "text_muted": "#94A3B8"     # Slate 400
}

# Image Processing Defaults
MAX_IMAGE_SIZE = (2048, 2048)
DEFAULT_JPEG_QUALITY = 85
DEFAULT_HISTOGRAM_BINS = 256
