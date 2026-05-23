# 📊 Graph Quality Classifier AI

A high-end, production-ready system for classifying forecasting graphs into four categories: **Good**, **Passable**, **Bad**, and **None**. Using Deep Learning (MobileNetV2) and a modern Next.js dashboard running **entirely in the browser** via **TensorFlow.js**.

## ✨ Features

- **Client-Side Neural Network**: Built using TensorFlow.js to train and run predictions directly in the browser with 0ms server latency.
- **Interactive Playground**: Sliders to adjust Noise Level, Seasonality Amplitude, Trend Slope, and Outlier Spikes, with live Recharts updates and real-time classification probability bars.
- **CSV Forecast Auditor**: Drag-and-drop forecast sheet auditor. Displays parsed columns and runs the neural network to output MAPE, correlation coefficients, and structural anomalies.
- **Visual Chart Inspector**: Canvas-based computer vision image scanner. Upload static chart screenshots and run scans with custom edge-variance analysis.
- **ML Training Center**: Hyperparameter customization (Epochs, Learning Rate, Batch Size) with a real-time updating loss and accuracy area chart and retro terminal logs.
- **Modern Glassmorphic Design**: Curated dark HSL color palette, glowing indicators, animated cards, and responsive layouts.

## 📁 Consolidated Project Structure

The project has been optimized to make the Next.js web application the root-level project, enabling zero-config Vercel deployments.

```
├── app/                  # Next.js App Router (React components & routes)
│   ├── utils/
│   │   └── mlEngine.ts   # TensorFlow.js browser-based training & inference engine
│   ├── globals.css       # Custom HSL design variables and micro-animations
│   ├── layout.tsx        # HTML structure and Google Fonts
│   └── page.tsx          # Dashboard page implementing all interactive tabs
├── public/               # Static assets & public resources
├── src/ml/               # [Archived] Python-based dataset & CNN training scripts
├── next.config.ts        # Next.js compiler settings
├── tsconfig.json         # TypeScript compiler configurations
├── package.json          # Unified Node dependencies (TF.js, Framer Motion, Recharts)
└── README.md
```

## 🚀 Local Setup & Quick Start

### 1. Install Node Dependencies
Initialize node packages at the root directory:
```bash
npm install
```

### 2. Run the Development Server
Launch the local Next.js development server:
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser to view the application.

### 3. Production Build
Verify standard production-ready static generation:
```bash
npm run build
```

---

## ⚡ Vercel Deployment Guide

Because the app is located at the root of the repository and is a standard Next.js application, deployment to Vercel is extremely straightforward:

1. Push your updated codebase to a **GitHub repository** (e.g., `https://github.com/SakshiMalhotra18/GraphClassifier`).
2. Log into your **Vercel Dashboard** and click **Add New Project**.
3. Import the `GraphClassifier` repository.
4. Vercel will automatically detect **Next.js** as the framework.
5. Click **Deploy** without modifying any setting. It will compile and publish the live URL in under 2 minutes!

---
Designed with ❤️ for SakshiMalhotra18.
