# 🧠 Brain Tumor Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  A professional, CNN-powered web application that classifies brain MRI scans into <strong>four categories</strong> — Glioma, Meningioma, No Tumor, and Pituitary — with real-time confidence scores and an intuitive medical UI.
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Deep Learning** | Convolutional Neural Network trained on brain MRI data |
| 📊 **Confidence Scores** | Visual probability bars for all 4 classes |
| 🎨 **Medical UI** | Dark-themed, professional interface with color-coded results |
| 📱 **Responsive** | Works on desktop & mobile browsers |
| ☁️ **Cloud-Ready** | One-click deploy to Streamlit Community Cloud |

## 🏗️ Model Architecture

```
Input (150x150x3)
 ├── Data Augmentation (Sequential)
 ├── Rescaling (1/255)
 ├── Conv2D(32) + BatchNorm + MaxPool
 ├── Conv2D(64) + BatchNorm + MaxPool
 ├── Conv2D(128) + BatchNorm + MaxPool
 ├── Conv2D(128) + BatchNorm + MaxPool
 ├── GlobalAveragePooling2D
 ├── Dense(128) + Dropout
 └── Dense(4, softmax) → [Glioma, Meningioma, No Tumor, Pituitary]
```

**Total Parameters:** 776K (trainable: 258K)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/brain-tumor-classifier.git
cd brain-tumor-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

### ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo.
4. Set **Main file path** to `app.py`.
5. Click **Deploy** — done! 🎉

> **Note:** The model file is ~3 MB — no Git LFS required.

---

## 📁 Project Structure

```
brain-tumor-classifier/
├── app.py                      # Main Streamlit application
├── utils.py                    # Model loading, preprocessing, prediction
├── brain_tumor_model.keras     # Trained CNN model (Git LFS)
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (Streamlit Cloud)
├── runtime.txt                 # Python version for Streamlit Cloud
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── .gitignore
├── .gitattributes              # Git LFS tracking rules
└── README.md
```

---

## 🖼️ Usage

1. Open the app in your browser.
2. Upload a brain MRI scan (JPG or PNG).
3. View the predicted tumor type and confidence scores.
4. Read the description card for clinical context.

---

## ⚠️ Disclaimer

> This application is intended for **educational and research purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
