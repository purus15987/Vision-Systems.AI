# 🧰 Installation & Tool Setup – Vision-Systems.AI 2026

This guide walks you through setting up your development environment for the course.

---

## ✅ Prerequisites

- Python 3.10+ (recommended: 3.11)
- pip or conda package manager
- Git
- A modern browser (Chrome recommended)
- Google account (for Colab)

---

## 🐍 Step 1: Python Environment

### Option A – Conda (Recommended)

```bash
conda create -n vision-ai python=3.11
conda activate vision-ai
```

### Option B – venv

```bash
python -m venv vision-env
source vision-env/bin/activate       # macOS/Linux
vision-env\Scripts\activate          # Windows
```

---

## 📦 Step 2: Install Core Libraries

```bash
pip install numpy matplotlib scikit-image pillow
pip install opencv-python-headless
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow
pip install ultralytics
pip install transformers datasets accelerate
pip install mediapipe
pip install wandb
pip install streamlit
pip install fastapi uvicorn
pip install onnx onnxruntime
```

---

## 🐳 Step 3: Docker (Optional, for Deployment Module)

Install Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

Verify:
```bash
docker --version
```

---

## 🔑 Step 4: API Keys & Accounts

| Service | Link | Required For |
|---|---|---|
| Hugging Face | [huggingface.co](https://huggingface.co) | Foundation models |
| Weights & Biases | [wandb.ai](https://wandb.ai) | Experiment tracking |
| Kaggle | [kaggle.com](https://kaggle.com) | Datasets & competitions |
| Google Colab | [colab.research.google.com](https://colab.research.google.com) | GPU labs |
| Roboflow | [roboflow.com](https://roboflow.com) | Dataset management |

---

## 🧪 Step 5: Verify Installation

```python
import cv2
import numpy as np
import torch
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
```

---

## 🆘 Troubleshooting

If you run into issues, check the [FAQs](faqs.md) or open a discussion on [GitHub](https://github.com/purus15987/vision-systems.ai/discussions).
