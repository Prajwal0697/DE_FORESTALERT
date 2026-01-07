# Illegal Sound Detection System

A machine learning system for detecting illegal sounds (chainsaw, gun shots, jackhammer, etc.) in audio recordings using the UrbanSound8K dataset.

---

## 🎯 Project Overview

This project addresses the challenge of detecting illegal activities through audio analysis, specifically focusing on:
- **Chainsaw detection** for forest protection
- **Gun shot detection** for security applications
- **Construction noise detection** for compliance monitoring
- **Engine idling detection** for environmental monitoring

---

## 📁 Project Structure

```
major/
├── code/                          # Main code directory
│   ├── app.py                     # FastAPI backend for predictions
│   ├── train_chainsaw_model.ipynb # Model training notebook
│   ├── test_audio_files.py        # Feature extraction & prediction logic
│   ├── check_performance.py       # Quick model performance check
│   └── ...                        # Other scripts
├── dataset_forest/
│   ├── UrbanSound8K.csv           # Metadata (8,732 samples)
│   └── audio/                     # Audio files (EXCLUDED from repo)
├── frontend/                      # React frontend (user interface)
├── merged_dataset.csv             # Merged/processed dataset
└── .gitignore                     # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install librosa numpy pandas scikit-learn matplotlib seaborn fastapi uvicorn joblib
```

### 2. Download Dataset
- Download [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/)
- Extract to `dataset_forest/audio/` directory
- Ensure folder structure: `dataset_forest/audio/fold1/`, `fold2/`, etc.

### 3. Train the Model
```bash
cd code
jupyter notebook train_chainsaw_model.ipynb
```

### 4. Run the Backend API
```bash
cd code
  ```

### 5. Use the Frontend
- The React frontend (in `/frontend`) lets you upload audio and view predictions.

---

## 📊 Latest Model Performance

**(Based on `check_performance.py` and recent training runs)**

| Metric                | Value      |
|-----------------------|------------|
| **Overall Accuracy**  | 94.91%     |
| **Illegal Precision** | 98%        |
| **Illegal Recall**    | 94%        |
| **Illegal F1-Score**  | 96%        |
| **Natural Precision** | 89%        |
| **Natural Recall**    | 97%        |
| **Natural F1-Score**  | 93%        |

**Confusion Matrix:**
```
[[1077   70]  # Natural: 1077 correct, 70 misclassified as illegal
 [  19  581]] # Illegal: 581 correct, 19 misclassified as natural
```

- **Interpretation:**  
  - ✅ Excellent performance for security applications
  - ✅ High precision (98%) means few false alarms
  - ✅ High recall (94%) means few illegal sounds missed
  - ✅ Balanced performance across both classes

---

## 🔑 Key Features

- **End-to-end pipeline:** Data ingestion, feature extraction, model training, evaluation, and prediction API.
- **Audio feature extraction:** MFCCs, chroma, spectral contrast, zero-crossing rate, RMS, mel-spectrogram.
- **Robust SVM classifier:** Tuned for imbalanced data, high accuracy.
- **REST API:** FastAPI backend for real-time predictions.
- **User-friendly frontend:** React app for easy audio upload and result visualization.
- **Extensible:** Modular code for easy updates and improvements.

---

## 🛠️ Customization & Extending

- **Add new illegal sound classes:** Edit `illegal_classes` in training scripts.
- **Tune augmentation:** Adjust parameters in augmentation scripts for better balance.
- **Retrain with new data:** Add new audio samples and retrain for improved accuracy.

---

## 📚 Dependencies

- `librosa`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn`, `joblib`

---

## ⚠️ Notes

- **Audio files are excluded** from the repository for size and copyright reasons.
- To reproduce results, download the UrbanSound8K dataset and place it in the correct directory.

---

## 📈 Goal

**Improve illegal sound detection for real-world security and environmental monitoring, achieving high recall and precision for actionable alerts.**
