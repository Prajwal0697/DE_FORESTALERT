from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from typing import Dict, Any

import numpy as np
import librosa
import joblib

from test_audio_files import extract_features, load_model, predict_audio

app = FastAPI()

# Allow CORS for local React dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "chainsaw_detection_model.pkl")
model_data = None

@app.on_event("startup")
def load_model_on_startup():
    global model_data
    model_data = load_model(MODEL_PATH)
    if model_data is None:
        raise RuntimeError("Failed to load model.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join("/tmp" if os.name != "nt" else os.getcwd(), temp_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        predicted_label, confidence, confidence_scores = predict_audio(model_data, temp_path)
        if predicted_label == "ERROR":
            return JSONResponse(status_code=400, content={"error": "Failed to process audio file."})
        return {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "confidence_scores": {k: float(v) for k, v in confidence_scores.items()}
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
