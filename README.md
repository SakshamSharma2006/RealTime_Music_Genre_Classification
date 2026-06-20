# 🎧 Real-Time Music Genre Classification using CNN and MFCC

[![Paper](https://img.shields.io/badge/Published-IJAIR%20Vol.13%20Issue%201%20(XIII)-blue)](https://iaraedu.com/about-journal/ijair-volume-13-issue-1-xiii-january-march-2026.php)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20354001.svg)](https://doi.org/10.5281/zenodo.20354001)
[![ISSN](https://img.shields.io/badge/ISSN-2394--7780-orange)](https://iaraedu.com/about-journal/ijair-volume-13-issue-1-xiii-january-march-2026.php)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-95.0%25-success)]()

A lightweight, deployment-ready **MFCC + CNN** framework for classifying music genres in **real time**, built for interactive systems where both accuracy and low latency matter. This repository accompanies the peer-reviewed research paper *"Real-Time Music Genre Classification using CNN and MFCC for Interactive Systems"*, published in the **International Journal of Advance and Innovative Research (IJAIR)**, Volume 13, Issue 1 (XIII), January–March 2026.

📄 **Published Paper:** [IJAIR Volume 13, Issue 1 (XIII)](https://iaraedu.com/about-journal/ijair-volume-13-issue-1-xiii-january-march-2026.php)

---

## 📌 Overview

Most deep learning approaches to music genre classification optimize for offline accuracy, making them too heavy for real-time, interactive deployment. This project addresses that gap by combining:

- **Mel-Frequency Cepstral Coefficients (MFCCs)** — a compact, perceptually grounded audio representation
- **A lightweight, regularized 2D CNN** — optimized for low inference latency

The result is a model that achieves **95.0% accuracy** on the GTZAN benchmark while running end-to-end inference in just **~100 ms**, well within the 200 ms threshold required for seamless interactive systems — and it's deployed live in a **Streamlit web app** supporting both file uploads and live microphone input.

---

## 🏗️ Architecture

**Pipeline:** `Audio Input → MFCC Extraction → CNN → Softmax → Genre Prediction`

```
Input: (129, 40, 1) MFCC feature map
├── Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
├── Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
├── Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
├── Flatten → Dense(128) → ReLU → Dropout(0.4)
└── Dense(10) → Softmax
```

- **Loss:** Categorical Cross-Entropy | **Optimizer:** Adam
- **Regularization:** Batch normalization, dropout (0.3–0.5), early stopping
- **Data Augmentation:** Pitch shifting, time stretching, additive noise injection (training set only)
- **Feature Extraction:** STFT → Mel-filterbank mapping → cepstral coefficients (via `librosa`)

---

## 📊 Results

### Overall Performance (GTZAN Dataset)

| Metric    | Value (%) |
|-----------|-----------|
| Accuracy  | 95.0      |
| Precision | 94.6      |
| Recall    | 94.2      |
| F1-Score  | 94.4      |

### Real-Time Inference Latency

| Stage                  | Avg. Time (ms) |
|-------------------------|----------------|
| Audio buffering          | 18             |
| MFCC extraction          | 42             |
| Feature normalization    | 11             |
| CNN forward pass         | 29             |
| **Total latency**        | **~100 ms**    |

The system maintains responsiveness (≤176 ms average) even under 20 concurrent requests, well within the 200 ms real-time threshold.

### Feature Representation Comparison

| Representation   | Dimensionality | Accuracy (%) | Efficiency |
|-------------------|-----------------|---------------|------------|
| MFCC-based         | Low             | 95.0          | High       |
| Mel-spectrogram    | Medium          | 95.8          | Medium     |
| Raw waveform       | Very High       | 96.1          | Low        |

MFCCs offer the best accuracy-to-efficiency trade-off for real-time, interactive deployment.

---

## 📁 Repository Structure

```
RealTime_Music_Genre_Classification/
├── app/
│   └── app.py                  # Streamlit web app (upload or live mic → genre prediction)
├── training/
│   ├── 01_feature_extraction.ipynb   # MFCC extraction & preprocessing pipeline
│   ├── 02_cnn_training.ipynb         # CNN model training & evaluation
│   └── Transformer.ipynb             # Experimental attention/Transformer-based approach
├── model/
│   ├── cnn_genre_model.h5      # Trained CNN model
│   ├── best_model.h5           # Best checkpoint (early stopping)
│   ├── scaler.npy              # StandardScaler mean
│   └── scaler_scale.npy        # StandardScaler scale
├── features/
│   ├── data.npy                # Extracted MFCC feature vectors
│   └── labels.npy              # Corresponding genre labels
├── dataset/
│   └── gtzan/genres_original/  # GTZAN dataset (10 genres × 100 tracks)
├── Documentation/
│   └── Citation/                # Reference papers cited in the research
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SakshamSharma2006/RealTime_Music_Genre_Classification.git
cd RealTime_Music_Genre_Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
cd app
streamlit run app.py
```

The app supports two modes:
- **Upload Audio File** — classify a `.wav` or `.mp3` file
- **Live Microphone Recording** — record 5 seconds of live audio and classify it instantly

### 4. (Optional) Retrain the model
Run the notebooks in order:
1. `training/01_feature_extraction.ipynb` — extracts MFCC features from the GTZAN dataset
2. `training/02_cnn_training.ipynb` — trains and evaluates the CNN

---

## 🧠 Tech Stack

- **Audio Processing:** `librosa`
- **Deep Learning:** `TensorFlow` / `Keras`
- **Web App:** `Streamlit`, `sounddevice`
- **Data Handling:** `NumPy`, `scikit-learn`, `SciPy`, `Matplotlib`

---

## 🎯 Dataset

[GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) — 1,000 audio clips (30s each), evenly split across 10 genres: *blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock*.

---

## 📖 Citation

If you use this work, please cite:

> Sharma, S., Tripathi, S., & Mahendra, K. (2026). Real-Time Music Genre Classification using CNN and MFCC for Interactive Systems. *International Journal of Advance and Innovative Research (IJAIR)*, 13(1-XIII). ISSN: 2394-7780.

```bibtex
@article{sharma2026realtime,
  title   = {Real-Time Music Genre Classification using CNN and MFCC for Interactive Systems},
  author  = {Sharma, Saksham and Tripathi, Sumitkumar and Mahendra, Kanojia},
  journal = {International Journal of Advance and Innovative Research (IJAIR)},
  volume  = {13},
  number  = {1 (XIII)},
  year    = {2026},
  issn    = {2394-7780},
  url     = {https://iaraedu.com/about-journal/ijair-volume-13-issue-1-xiii-january-march-2026.php}
}
```

---

## 🔭 Future Work

- Model compression (pruning, quantization) for edge/mobile deployment
- Attention/Transformer-based temporal modeling for long-range pattern capture
- Expanding beyond GTZAN to diverse, contemporary genres
- Multi-task learning (genre + mood detection)
- Continuous stream processing with adaptive buffering
- Online learning from user feedback

---

## 👥 Authors

- **Saksham Sharma** — Information Technology, Sheth L.U. Jhaveri and Sir M.V. College, India — `bscit.saksham@gmail.com`
- **Sumitkumar Tripathi** — Information Technology, Sheth L.U. Jhaveri and Sir M.V. College, India
- **Kanojia Mahendra** — Department of Computer Science, Sheth L.U. Jhaveri and Sir M.V. College, India

## 🙏 Acknowledgement

The authors thank the Department of Information Technology, Sheth L.U. Jhaveri College, Mumbai, for the academic environment and infrastructural support that made this research possible.

## 📄 License

This project is intended for academic and research purposes. Please contact the corresponding author for reuse of trained models or processed data.
