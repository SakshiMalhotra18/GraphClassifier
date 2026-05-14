# 📊 Graph Quality Classifier AI

A high-end, production-ready system for classifying forecasting graphs into four categories: **Good**, **Passable**, **Bad**, and **None**. Using Deep Learning (MobileNetV2) and a modern Next.js dashboard.

## ✨ Features

- **Deep Learning Core**: Convolutional Neural Network trained on synthetic forecasting data.
- **Interactive Dashboard**: Real-time graph generation and classification in the browser.
- **High-End UI**: Premium glassmorphic design, dark mode, and smooth animations.
- **Browser Inference**: Uses TensorFlow.js for lightning-fast, client-side classification.

## 📁 Project Structure

```
├── data/               # Dataset storage
├── src/
│   └── ml/
│       ├── data_gen.py # Synthetic data generation logic
│       ├── train.py    # Model training script
│       ├── model.py    # CNN architecture definition
│       └── convert.py  # TF.js conversion utility
├── web/                # Next.js Web Dashboard
├── models/             # Saved trained models (.h5)
├── requirements.txt    # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Train Model
```bash
# Generate synthetic dataset
python src/ml/data_gen.py

# Train the CNN
python src/ml/train.py --epochs 10

# Convert for Web
python src/ml/convert.py
```

### 3. Launch Dashboard
```bash
cd web
npm install
npm run dev
```

## 🛠 Technology Stack

- **ML Backend**: TensorFlow, Keras, NumPy, Matplotlib
- **Web Frontend**: Next.js 14, React, Framer Motion, Recharts
- **Web ML**: TensorFlow.js
- **Design**: Vanilla CSS (Premium Custom Design)

## 📈 Optimization Details

- **Refactored Codebase**: Clean, modular Python structure with type hinting and CLI support.
- **Enhanced Dataset**: Improved noise generation logic and diverse distractor types.
- **Performance**: MobileNetV2 backbone ensures high accuracy with minimal computational overhead.
- **Aesthetics**: Fully custom CSS design system optimized for first-glance "WOW" factor.

---
Designed with ❤️ for SakshiMalhotra18.
