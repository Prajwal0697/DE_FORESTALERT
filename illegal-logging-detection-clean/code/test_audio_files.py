#!/usr/bin/env python3
"""
Test the trained illegal sound detection model with audio files
"""

import os
import sys
import warnings
import numpy as np
import librosa
import joblib
from typing import Dict, Any, Tuple

warnings.filterwarnings("ignore")

def extract_features(filepath: str, target_sr: int = 22050) -> np.ndarray:
    """Extract features from an audio file using the same method as training"""
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if y is None or y.size == 0:
            raise ValueError("Empty audio file")
        
        # Ensure finite values
        y = np.nan_to_num(y)
        
        # Extract features (same as training)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr, axis=1)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms, axis=1)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
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
    except Exception as e:
        print(f"Error extracting features from {filepath}: {e}")
        return None

def load_model(model_path: str) -> Dict[str, Any]:
    """Load the trained model"""
    try:
        model_data = joblib.load(model_path)
        print("✅ Model loaded successfully")
        return model_data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_audio(model_data: Dict[str, Any], audio_path: str) -> Tuple[str, float, Dict[str, float]]:
    """Predict whether an audio file contains illegal sounds"""
    try:
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            return "ERROR", 0.0, {}
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Get model components
        model = model_data["model"]
        label_encoder = model_data["label_encoder"]
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Convert prediction back to label
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores
        confidence_scores = {}
        for i, label in enumerate(label_encoder.classes_):
            confidence_scores[label] = probabilities[i]
        
        # Get confidence for predicted class
        confidence = probabilities[prediction]
        
        return predicted_label, confidence, confidence_scores
        
    except Exception as e:
        print(f"❌ Error predicting {audio_path}: {e}")
        return "ERROR", 0.0, {}

def analyze_audio_file(audio_path: str) -> Dict[str, Any]:
    """Analyze audio file properties"""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Basic properties
        duration = len(y) / sr
        rms_energy = np.sqrt(np.mean(y**2))
        
        # Spectral properties
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Temporal properties
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "rms_energy": rms_energy,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "zero_crossing_rate": zero_crossing_rate
        }
    except Exception as e:
        print(f"❌ Error analyzing {audio_path}: {e}")
        return {}

def main():
    print("🔍 Testing Illegal Sound Detection Model")
    print("=" * 50)
    
    # Load model
    model_path = "chainsaw_detection_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    model_data = load_model(model_path)
    if model_data is None:
        sys.exit(1)
    
    # Test files
    test_files = ["test3.mp3", "test8.mp3","test5.wav","test7.wav","test2.wav","test10.mp3"]
    
    print(f"\n📊 Model Information:")
    print(f"   - Illegal classes: {model_data.get('illegal_classes', 'N/A')}")
    print(f"   - Natural classes: {model_data.get('natural_classes', 'N/A')}")
    print(f"   - Feature names: {model_data.get('feature_names', 'N/A')}")
    
    print(f"\n🧪 Testing {len(test_files)} audio files:")
    print("=" * 50)
    
    results = []
    
    for audio_file in test_files:
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
            continue
        
        print(f"\n📁 Testing: {audio_file}")
        print("-" * 30)
        
        # Analyze audio properties
        audio_info = analyze_audio_file(audio_file)
        if audio_info:
            print(f"   Duration: {audio_info['duration']:.2f} seconds")
            print(f"   Sample rate: {audio_info['sample_rate']} Hz")
            print(f"   RMS Energy: {audio_info['rms_energy']:.4f}")
            print(f"   Spectral Centroid: {audio_info['spectral_centroid']:.2f}")
        
        # Make prediction
        predicted_label, confidence, confidence_scores = predict_audio(model_data, audio_file)
        
        if predicted_label != "ERROR":
            print(f"   🎯 Prediction: {predicted_label.upper()}")
            print(f"   📊 Confidence: {confidence:.2%}")
            
            # Show confidence for both classes
            for label, prob in confidence_scores.items():
                print(f"      {label}: {prob:.2%}")
            
            # Color-coded result
            if predicted_label == "illegal":
                print(f"   🚨 RESULT: ILLEGAL SOUND DETECTED!")
            else:
                print(f"   ✅ RESULT: NATURAL SOUND")
            
            results.append({
                "file": audio_file,
                "prediction": predicted_label,
                "confidence": confidence,
                "confidence_scores": confidence_scores,
                "audio_info": audio_info
            })
        else:
            print(f"   ❌ Failed to analyze file")
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 50)
    
    if results:
        illegal_count = sum(1 for r in results if r["prediction"] == "illegal")
        natural_count = sum(1 for r in results if r["prediction"] == "natural")
        
        print(f"   Total files tested: {len(results)}")
        print(f"   Illegal sounds detected: {illegal_count}")
        print(f"   Natural sounds detected: {natural_count}")
        
        print(f"\n📊 Detailed Results:")
        for result in results:
            status = "🚨 ILLEGAL" if result["prediction"] == "illegal" else "✅ NATURAL"
            print(f"   {result['file']}: {status} ({result['confidence']:.1%} confidence)")
    else:
        print("   No files were successfully analyzed")

if __name__ == "__main__":
    main()
