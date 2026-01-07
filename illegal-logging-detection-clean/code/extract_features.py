# ===============================================
# CHAINSAW DETECTION - FEATURE EXTRACTION MODULE
# ===============================================

import os
import librosa
import numpy as np
import joblib
from typing import Union, List

class ChainSawFeatureExtractor:
    """
    Feature extraction class for chainsaw/illegal sound detection.
    Extracts MFCC features from audio files for model inference.
    """
    
    def __init__(self, model_path: str = "chainsaw_detection_model.pkl"):
        """
        Initialize the feature extractor and load the trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained SVM model from pickle file.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ Model loaded successfully from {self.model_path}")
            else:
                print(f"❌ Model file not found: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def extract_features(self, file_path: str) -> np.ndarray:
        """
        Extract MFCC features from an audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Extracted MFCC features (13 features)
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=22050, mono=True)
            
            # Extract MFCCs (13 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Take mean across time axis to get fixed-size feature vector
            mfcc_mean = np.mean(mfcc, axis=1)
            
            return mfcc_mean
            
        except Exception as e:
            print(f"❌ Error extracting features from {file_path}: {e}")
            return np.zeros(13)  # Return zero vector on error
    
    def extract_features_batch(self, file_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple audio files.
        
        Args:
            file_paths (List[str]): List of audio file paths
            
        Returns:
            np.ndarray: Feature matrix (n_files, 13)
        """
        features = []
        
        for file_path in file_paths:
            feature = self.extract_features(file_path)
            features.append(feature)
            
        return np.array(features)
    
    def predict_single(self, file_path: str) -> dict:
        """
        Predict whether a single audio file contains illegal sounds.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            dict: Prediction results with probability and label
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Extract features
        features = self.extract_features(file_path).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Interpret results
        label = "Illegal Sound" if prediction == 1 else "Natural Sound"
        confidence = max(probability) * 100
        
        return {
            "file": os.path.basename(file_path),
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "natural": round(probability[0] * 100, 2),
                "illegal": round(probability[1] * 100, 2)
            }
        }
    
    def predict_batch(self, file_paths: List[str]) -> List[dict]:
        """
        Predict for multiple audio files.
        
        Args:
            file_paths (List[str]): List of audio file paths
            
        Returns:
            List[dict]: List of prediction results
        """
        results = []
        
        for file_path in file_paths:
            result = self.predict_single(file_path)
            results.append(result)
            
        return results


# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def process_directory(directory_path: str, extractor: ChainSawFeatureExtractor) -> List[dict]:
    """
    Process all audio files in a directory.
    
    Args:
        directory_path (str): Path to directory containing audio files
        extractor (ChainSawFeatureExtractor): Feature extractor instance
        
    Returns:
        List[dict]: Prediction results for all files
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    # Find all audio files in directory
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(directory_path, file))
    
    if not audio_files:
        print(f"❌ No audio files found in {directory_path}")
        return []
    
    print(f"🔍 Found {len(audio_files)} audio files. Processing...")
    
    # Process all files
    results = extractor.predict_batch(audio_files)
    
    return results


def print_results(results: List[dict]):
    """
    Print prediction results in a formatted way.
    
    Args:
        results (List[dict]): Prediction results
    """
    print("\n" + "="*60)
    print("🎯 CHAINSAW DETECTION RESULTS")
    print("="*60)
    
    illegal_count = 0
    natural_count = 0
    
    for result in results:
        if "error" in result:
            print(f"❌ {result['file']}: {result['error']}")
            continue
            
        icon = "🚨" if result['prediction'] == 1 else "🌿"
        print(f"{icon} {result['file']}:")
        print(f"   Label: {result['label']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Natural: {result['probabilities']['natural']}% | Illegal: {result['probabilities']['illegal']}%")
        print()
        
        if result['prediction'] == 1:
            illegal_count += 1
        else:
            natural_count += 1
    
    print(f"📊 Summary: {natural_count} Natural sounds, {illegal_count} Illegal sounds")
    print("="*60)


# ===============================================
# MAIN EXECUTION (FOR TESTING)
# ===============================================

if __name__ == "__main__":
    # Example usage
    extractor = ChainSawFeatureExtractor()
    
    # Test with a single file (update path as needed)
    # result = extractor.predict_single("path/to/your/audio/file.wav")
    # print(result)
    
    # Test with a directory (update path as needed)
    # results = process_directory("path/to/your/audio/directory", extractor)
    # print_results(results)
    
    print("\n🔧 Feature extractor ready! Use the following methods:")
    print("   - extractor.predict_single('file.wav') for single file")
    print("   - process_directory('folder/', extractor) for batch processing")