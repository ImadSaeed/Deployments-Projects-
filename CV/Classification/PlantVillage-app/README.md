# 🌿 Plant Disease Classifier

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nzzuuuhpyzpaiybgyqvtbg.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen?style=flat)
![Classes](https://img.shields.io/badge/Classes-38-blue?style=flat)
![Dataset](https://img.shields.io/badge/Dataset-54%2C305%20images-orange?style=flat)

> **Upload a leaf image → Get instant plant disease diagnosis with confidence score.**

---

## 🚀 Live Demo

👉 **[plantvillage-app.streamlit.app](https://nzzuuuhpyzpaiybgyqvtbg.streamlit.app/)**

---

## 📖 Overview

This app identifies plant diseases from leaf photos using a **Tiny Neural Network** trained on top of **EfficientNetV2B0 features**, enhanced with **CLAHE preprocessing** for improved image clarity.

The pipeline is lightweight (model is only **9 MB**) yet achieves **98% accuracy** across **38 disease/healthy classes**, making it fast and deployable on free-tier hosting.

---

## 🧠 How It Works

```
Input Leaf Image
       │
       ▼
┌─────────────────────────────┐
│  CLAHE Preprocessing        │  ← Contrast enhancement in LAB color space
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  EfficientNetV2B0           │  ← Frozen backbone (ImageNet weights)
│  Feature Extractor          │  ← Outputs 1280-dim feature vector
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Tiny Neural Network        │  ← 512 → 256 → 38 (BatchNorm + Dropout)
│  Classifier                 │
└─────────────────────────────┘
       │
       ▼
 Disease Label + Confidence Score
```

---

## ⚙️ Model Architecture

| Component | Details |
|-----------|---------|
| **Preprocessing** | CLAHE in LAB color space |
| **Feature Extractor** | EfficientNetV2B0 (frozen, ImageNet weights) |
| **Classifier** | Tiny NN — `512 → 256 → 38` with BatchNorm & Dropout |
| **Accuracy** | **98%** on test set |
| **Classes** | 38 (diseased + healthy across multiple crops) |
| **Model Size** | ~9 MB (`tiny_nn_final.h5`) |
| **Dataset** | PlantVillage — 54,305 images |

---

## 📁 Project Structure

```
plantvillage-app/
├── app.py                 # Streamlit application
├── tiny_nn_final.h5       # Trained classifier model (9 MB)
├── requirements.txt       # Python dependencies
├── python_version.txt     # Python version pinning
└── README.md              # This file
```

---

## 🛠️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ImadSaeed/plantvillage-app.git
cd plantvillage-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for full list):

```
streamlit
tensorflow
opencv-python
numpy
Pillow
```

---

## 🔗 Related Resources

| Resource | Link |
|----------|------|
| 🧪 Training Code & EDA | [CV-Projects / PlantVillage](https://github.com/ImadSaeed/CV-Projects/tree/main/Classification/PlantVillage) |
| 📊 Dataset | [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — 54,305 images, 38 classes |
| 🌐 Live App | [Streamlit Cloud Deployment](https://nzzuuuhpyzpaiybgyqvtbg.streamlit.app/) |

---

## 👤 Author

**Imad Saeed**
- GitHub: [@ImadSaeed](https://github.com/ImadSaeed)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ and 🌿 for smarter agriculture
</p>
