# 📝 Task 3: CodeSignal Streamlit Certification

> **Deadline: July 04, 2026**  
> **Module: DIP-Vision.AI / Deploy-Vision.AI**  
> **Topics: Making a Frontend for your AI Researcher with Streamlit**

---

## 🎯 Objective

Complete the **CodeSignal Streamlit course** to learn how to build interactive web frontends for machine learning models — a critical deployment skill.

---

## 📋 Task Description

Streamlit transforms Python scripts into shareable web apps with minimal code. In this course you will learn:

- Setting up a Streamlit app
- Building interactive widgets (sliders, buttons, file uploaders)
- Displaying images, videos, and charts
- Adding session state for multi-step interactions
- Deploying a Streamlit app

In this course, you will build a small **AI Researcher frontend** — a Streamlit app that allows a user to upload an image and get instant visual feedback (e.g., display image properties, histogram, etc.).

---

## ✅ Steps to Complete

1. **Register & Sign in** to CodeSignal: [codesignal.com](https://codesignal.com/)

2. **Complete:** [Making a Frontend for our Researcher with Streamlit](https://codesignal.com/learn/courses/making-a-frontend-for-our-researcher-with-streamlit)

3. **Build** your mini Streamlit app (see Bonus below).

4. **Submit** via GitHub Issue while **Project 01 Submission**:
   👉 [Vision-Systems.AI Issues](https://github.com/purus15987/vision-systems.ai/issues)

---

## 📤 Submission Format

Create a new GitHub Issue with:
- **Title**: `[Task 3] Your Name – Streamlit Certification`
- **Attach**: Screenshot of your completion certificate
- **Include**: Roll Number and Section

---

## 🌟 Bonus Task (Highly Recommended)

Build a simple **Image Analysis App** using Streamlit:

```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Analyzer")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded:
    img = np.array(Image.open(uploaded))
    st.image(img, caption="Uploaded Image")
    st.write(f"Shape: {img.shape}")
    st.write(f"Dtype: {img.dtype}")
    # Add histogram...
```

Submit your app code in the GitHub Issue for bonus marks.

---

> [!NOTE]
> Streamlit skills learned here will be directly used in the Deploy-Vision.AI module (Week 25).

---

[← Back to Homepage](../../readme.md)
