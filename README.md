# Audio ML Classifier: Speech Command Recognition Pipeline

An end-to-end audio classification pipeline built with classical Digital Signal Processing (DSP) feature extraction (**STFT**, **MFCC**) and Machine Learning classifiers (**Random Forest**, **Support Vector Machines**, and **PyTorch MLP**) on the **Google Speech Commands** dataset.

Designed with modular software engineering practices, reproducible workflows, and structured for edge-AI and embedded acoustic classification applications (such as Voice Activity Detection and Keyword Spotting).

---

## 📌 Architecture & DSP Pipeline

```text
Raw Audio (.wav)
       │
       ▼
[1] Load & Preprocess (src/preprocess.py)
       ├── Load with Librosa @ 16,000 Hz (mono)
       ├── Trim leading & trailing silence (librosa.effects.trim, top_db=20)
       └── Pad / truncate to fixed length: 1.0s (16,000 samples)
       │
       ▼
[2] Feature Extraction & DSP (src/features.py)
       ├── Short-Time Fourier Transform (STFT): n_fft=512, hop_length=256
       ├── Mel-Frequency Cepstral Coefficients (MFCC): n_mfcc=13
       └── Statistical Aggregation: mean + std across time frames
           └── Compact 26-dimensional feature vector per clip
       │
       ▼
[3] Dataset Assembly & Partitioning (src/train.py)
       ├── Construct feature matrix X (500 samples × 26 features)
       ├── Export tabular dataset to data/processed/features.csv
       └── Stratified 80/20 train/test split (400 train, 100 test)
       │
       ▼
[4] Model Training (src/train.py & src/train_dl.py)
       ├── Baseline: RandomForestClassifier (200 trees, max_depth=15)
       ├── Classical Benchmark: Support Vector Machine (RBF kernel, C=5.0)
       └── Deep Learning Stretch: PyTorch MLP (Linear → BatchNorm → ReLU → Dropout)
       │
       ▼
[5] Evaluation & Diagnostics (src/evaluate.py)
       ├── Metrics: Accuracy, Precision, Recall, Weighted F1-Score
       ├── Class-wise classification reports
       └── Visualizations: Confusion matrix & Model performance comparisons
```

---

## 🔬 Digital Signal Processing (DSP) Visualizations

The acoustic pipeline captures the temporal and spectral signature of spoken commands (`"go"`, `"no"`, `"stop"`, `"up"`, `"yes"`):

| Waveform (Time Domain) | STFT Spectrogram (Time-Frequency) |
| :---: | :---: |
| ![Waveform](outputs/sample_waveform.png) | ![STFT Spectrogram](outputs/sample_spectrogram.png) |

| Mel-Frequency Cepstral Coefficients (MFCC) | Confusion Matrix (Random Forest) |
| :---: | :---: |
| ![MFCC Heatmap](outputs/sample_mfcc.png) | ![Confusion Matrix](outputs/confusion_matrix.png) |

---

## 📊 Benchmark Results

All models were evaluated on the exact same stratified 20% test partition (100 audio clips, 20 clips per class):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 56.00% | 55.72% | 56.00% | 55.29% |
| **Support Vector Machine (RBF)** | 60.00% | 63.07% | 60.00% | 59.93% |
| **PyTorch MLP (Stretch Goal)** | **66.00%** | **69.84%** | **66.00%** | **65.86%** |

*Note: Random guess baseline on 5 balanced classes is 20.00%. The PyTorch MLP achieved 88.89% precision and 84.21% F1 on the `"yes"` class.*

### Benchmark Comparison Plot
![Model Comparison](outputs/model_comparison.png)

---

## 📁 Repository Structure

```text
audio-ml-classifier/
├── README.md                          # Comprehensive documentation & results
├── requirements.txt                   # Pinned dependency manifest
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT License
├── data/
│   ├── raw/                           # Downloaded 16kHz speech clips by class
│   │   ├── go/
│   │   ├── no/
│   │   ├── stop/
│   │   ├── up/
│   │   └── yes/
│   └── processed/
│       ├── features.csv               # Extracted 26-dim DSP feature matrix
│       └── test_split.npz             # Serialized test partition
├── notebooks/
│   └── 01_explore_and_prototype.ipynb # Interactive EDA, DSP & training notebook
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── data_loader.py                 # Streaming dataset acquisition & indexing
│   ├── preprocess.py                  # Resampling, silence trimming & length fixing
│   ├── features.py                    # STFT, MFCC & statistical aggregation
│   ├── visualize.py                   # Plotting utilities for waveforms/spectrograms
│   ├── train.py                       # Feature pipeline & classical model training
│   ├── train_dl.py                    # PyTorch MLP deep learning training
│   └── evaluate.py                    # Metrics computation & benchmark reporting
├── models/
│   ├── rf_classifier.pkl              # Serialized Random Forest model
│   ├── svm_classifier.pkl             # Serialized SVM model
│   ├── label_encoder.pkl              # Fitted class label encoder
│   ├── scaler.pkl                     # Fitted standard scaler
│   └── mlp_classifier.pt              # PyTorch MLP checkpoint
└── outputs/
    ├── sample_waveform.png            # 16 kHz time-domain waveform
    ├── sample_spectrogram.png         # STFT magnitude spectrogram in dB
    ├── sample_mfcc.png                # Mel-Frequency Cepstral Coefficients
    ├── confusion_matrix.png           # Annotated confusion matrix heatmap
    └── model_comparison.png           # Bar chart comparing RF, SVM, and MLP
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd "c:/Projects/Audio_ML project"

# Create a virtual environment with Python 3.10+ (Python 3.12 recommended)
python -m venv venv

# Activate the virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Acquisition
Streams and extracts 100 clips per target class (`yes`, `no`, `stop`, `go`, `up`) from Google Speech Commands:
```bash
python -m src.data_loader
```

### 3. Generate DSP Visualizations
Preprocesses sample audio and generates waveform, STFT spectrogram, and MFCC plots:
```bash
python -m src.visualize
```

### 4. Train Classical Models (Random Forest & SVM)
Extracts DSP features, exports `data/processed/features.csv`, and trains classifiers:
```bash
python -m src.train
```

### 5. Train Deep Learning Classifier (PyTorch MLP)
Trains the multi-layer perceptron on normalized MFCC statistical vectors:
```bash
python -m src.train_dl
```

### 6. Evaluate & Benchmark All Models
Runs comprehensive evaluation, prints classification reports, and saves comparison plots:
```bash
python -m src.evaluate
```

### 7. Interactive Exploration Notebook
Launch the Jupyter notebook:
```bash
jupyter notebook notebooks/01_explore_and_prototype.ipynb
```

---

## 💡 Engineering Reflections & Edge-AI Roadmap

In edge devices (e.g., True Wireless Stereo / TWS earbuds, smart speakers, hearing wearables), low latency and memory efficiency are paramount:

1. **2D Spectrograms + Lightweight CNNs:**
   - Replacing statistical time-aggregation with raw 2D Log-Mel Spectrograms feeding a MobileNetV3 or small 1D/2D CNN preserves intra-word formant trajectory timing, pushing accuracy toward 90%+.
2. **Acoustic Data Augmentation:**
   - Adding background noise (ambient street/cafe noise), room impulse response (reverberation), and pitch shifting improves model robustness in varying acoustic environments.
3. **Real-time Streaming Inference:**
   - Utilizing a circular ring buffer with a 250ms hop size enables continuous wake-word and command listening with sub-50ms inference latency.
4. **Edge Quantization & Microcontroller Deployment:**
   - Exporting the trained PyTorch / scikit-learn models to **ONNX** and quantizing to **INT8** (via TensorFlow Lite Micro or ONNX Runtime) shrinks the model memory footprint to < 50 KB, suitable for Cortex-M or embedded DSP execution.

---

## 📄 Resume Bullet Point (Tailored for Audio & Edge-AI)

> **Audio ML Classifier (Speech Command Recognition)**
> Built an end-to-end audio classification pipeline: performed preprocessing (silence trimming, resampling to 16 kHz, fixed-length padding) and extracted DSP features (STFT spectrograms, MFCCs) using Librosa and SciPy; trained and evaluated ML/DL classifiers (Random Forest, SVM, PyTorch MLP) achieving up to 66% test accuracy on Google Speech Commands; analyzed performance via confusion matrices and per-class F1-scores. Directly applied signal processing and machine learning fundamentals relevant to speech/audio and low-power edge-AI applications.
