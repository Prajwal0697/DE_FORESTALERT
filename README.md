# DEFORESTALERT

An acoustic-based real-time system to detect illegal logging using machine learning and IoT. The project includes model training & evaluation, a FastAPI backend, and a React frontend for live audio capture and classification.

## Technologies
- Python (librosa, scikit-learn, FastAPI)
- JavaScript / React (frontend)
- ESP32 + LoRa (optional IoT integration)
- Firebase (optional alerting)
- Docker (optional)

## Features
- Real-time sound detection
- ML-based binary classification (illegal vs natural)
- REST API for audio upload and inference
- Simple React UI for live recording and visualization
- Example integration points for ESP32/LoRa alerting

## Quickstart

Prerequisites:
- Python 3.8+ and pip
- Node 16+ and npm
- Git

Backend:
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# API endpoints:
# GET  /test
# POST /api/classify-audio  (multipart file)
```

Frontend:
```bash
cd frontend
npm install
npm start
# Open http://localhost:3000
```

Model & data:
- Model artifact: `code/chainsaw_detection_model.pkl`
- Dataset CSV: `dataset_forest/merged_dataset.csv`
- Audio files are excluded from the repo; download UrbanSound8K and place under `dataset_forest/audio/`

## Run a local test (example)
Use `curl` to POST an audio file to the API:
```bash
curl -F "audio=@/path/to/sample.wav" http://localhost:8000/api/classify-audio
```

## Contributing
- Add new labeled audio samples to the dataset and retrain using the scripts in `code/`.
- Improve frontend UI under `frontend/src/components`.
- Open issues or PRs for bugs and enhancements.

## License
Add your license here (e.g., MIT).

## Contact
Prajwal — link or email
