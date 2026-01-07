# Sound Classification Backend

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST `/api/classify-audio`

Classify an audio file as legal or illegal.

- **Input**: Multipart form-data with an audio file
- **Output**: JSON with classification results
  ```json
  {
    "classification": "illegal",
    "confidence": 0.95,
    "confidence_scores": {
      "illegal": 0.95,
      "natural": 0.05
    }
  }
  ```

## Model Details

- Trained on a dataset of sound samples
- Binary classification: illegal vs. natural sounds
- Uses SVM with RBF kernel
- Feature extraction includes MFCC, Chroma, Spectral Contrast, ZCR, RMS, and Mel spectrogram
