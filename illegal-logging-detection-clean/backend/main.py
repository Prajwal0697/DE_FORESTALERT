import os
import io
import sys
import traceback
import numpy as np
import librosa
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_classification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'code', 'chainsaw_detection_model.pkl')

try:
    model_bundle = joblib.load(MODEL_PATH)
    clf = model_bundle['model']
    le = model_bundle['label_encoder']
    ILLEGAL_CLASSES = set(model_bundle.get('illegal_classes', []))
    NATURAL_CLASSES = set(model_bundle.get('natural_classes', []))
    logger.info(f"Model loaded successfully. Classes: {le.classes_}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())
    raise

def extract_features(audio_data: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract robust features from audio data."""
    # Ensure finite values
    audio_data = np.nan_to_num(audio_data)
    
    # Features
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    spec_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)

    zcr = librosa.feature.zero_crossing_rate(audio_data)
    zcr_mean = np.mean(zcr, axis=1)

    rms = librosa.feature.rms(y=audio_data)
    rms_mean = np.mean(rms, axis=1)

    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)

    features = np.concatenate([
        mfcc_mean,
        chroma_mean,
        spec_contrast_mean,
        zcr_mean,
        rms_mean,
        mel_mean,
    ], axis=0)
    
    return np.nan_to_num(features)

# Create FastAPI app
app = FastAPI(title="Sound Classification API")

# Add CORS middleware to allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.get("/test")
async def test_endpoint():
    return {"message": "API is working!"}
@app.post("/api/classify-audio")
async def classify_audio(audio: UploadFile = File(...)):
    """
    Classify an uploaded audio file as legal or illegal.
    
    :param audio: Uploaded audio file
    :return: Classification result
    """
    try:
        # Log file details
        logger.info(f"Received file: {audio.filename}, content type: {audio.content_type}")
        
        # Read the uploaded file
        contents = await audio.read()
        
        # Log file size
        logger.info(f"File size: {len(contents)} bytes")
        
        # Convert to numpy array
        audio_bytes = io.BytesIO(contents)
        
        # Try different methods to load the audio
        try:
            # First try librosa (supports more formats)
            y, sr = librosa.load(audio_bytes, sr=22050, mono=True)
            logger.info(f"Audio loaded with librosa. Shape: {y.shape}, Sample rate: {sr}")
        except Exception as librosa_err:
            logger.warning(f"Librosa loading failed: {librosa_err}")
            try:
                # Fallback to soundfile if librosa fails
                import soundfile as sf
                y, sr = sf.read(audio_bytes)
                logger.info(f"Audio loaded with soundfile. Shape: {y.shape}, Sample rate: {sr}")
                
                # Resample if needed
                if sr != 22050:
                    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                    logger.info(f"Resampled audio. New shape: {y.shape}")
                
                # Ensure mono
                if y.ndim > 1:
                    y = y.mean(axis=1)
                    logger.info(f"Converted to mono. Final shape: {y.shape}")
            except Exception as sf_err:
                logger.error(f"SoundFile loading failed: {sf_err}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=400, detail=f"Could not read audio file: {str(sf_err)}")
        
        # Check if audio is empty
        if y is None or y.size == 0:
            logger.error("Empty audio file")
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Extract features
        features = extract_features(y, sr)
        logger.info(f"Features extracted. Shape: {features.shape}")
        
        # Reshape for prediction (add batch dimension)
        features_reshaped = features.reshape(1, -1)
        
        # Predict
        prediction = clf.predict(features_reshaped)[0]
        proba = clf.predict_proba(features_reshaped)[0]
        
        # Get class names
        class_names = le.classes_
        
        # Log prediction details
        logger.info(f"Prediction: {class_names[prediction]}")
        logger.info(f"Probabilities: {dict(zip(class_names, proba))}")
        
        # Prepare response
        response = {
            "classification": class_names[prediction],
            "confidence": float(max(proba)),
            "confidence_scores": {
                class_name: float(prob) 
                for class_name, prob in zip(class_names, proba)
            }
        }
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
