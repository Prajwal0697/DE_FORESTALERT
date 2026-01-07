#!/usr/bin/env python3
"""
Train model using only the files that can be successfully mapped
"""

import os
import sys
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

warnings.filterwarnings("ignore")

# Configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_CSV_PATH = os.path.join(REPO_ROOT, "merged_dataset.csv")

def load_mapped_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load dataset using only files that can be mapped to actual audio files"""
    
    # Load the mapping
    mapping = {}
    if os.path.exists('file_mapping.txt'):
        with open('file_mapping.txt', 'r') as f:
            for line in f:
                csv_file, audio_file = line.strip().split('\t')
                mapping[csv_file] = audio_file
    
    if not mapping:
        print("❌ No file mapping found. Run create_file_mapping.py first.")
        sys.exit(1)
    
    # Load CSV
    df = pd.read_csv(MERGED_CSV_PATH)
    
    # Filter to only mapped files
    mapped_df = df[df['file'].isin(mapping.keys())].copy()
    mapped_df['audio_path'] = mapped_df['file'].map(mapping)
    
    print(f"📊 Using {len(mapped_df)} mapped files out of {len(df)} total files")
    
    # Extract features
    X_list = []
    y_list = []
    failed_count = 0
    
    print("🔧 Extracting features...")
    for i, (_, row) in enumerate(mapped_df.iterrows()):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(mapped_df)} features extracted")
        
        audio_path = row['audio_path']
        label = row['label']
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            if y is None or y.size == 0:
                failed_count += 1
                continue
            
            # Ensure finite values
            y = np.nan_to_num(y)
            
            # Extract features
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
            
            features = np.nan_to_num(features)
            
            X_list.append(features)
            y_list.append(str(label).strip())
            
        except Exception as e:
            failed_count += 1
            continue
    
    if failed_count > 0:
        print(f"⚠️ Failed to extract features from {failed_count} files")
    
    if len(X_list) == 0:
        print("❌ No features extracted successfully")
        sys.exit(1)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"✅ Successfully extracted features from {len(X_list)} files")
    return X, y, [mapping[f] for f in mapped_df['file'].tolist()]

def main():
    print("📦 Loading mapped dataset...")
    X, y, paths = load_mapped_dataset()
    
    # Define illegal classes (class 1) and natural classes (class 0)
    ILLEGAL_CLASSES = {
        'axe', 'chainsaw', 'fire', 'firework', 'generator', 'gunshot', 
        'handsaw', 'helicopter', 'VehicleEngine', 'car_horn', 'drilling', 
        'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
    }
    
    NATURAL_CLASSES = {
        'BirdChirping', 'Clapping', 'Footsteps', 'Frog', 'Insect', 'Lion', 
        'Rain', 'Silence', 'Speaking', 'Squirrel', 'Thunderstorm', 'TreeFalling', 
        'WaterDrops', 'Whistling', 'Wind', 'WingFlaping', 'WolfHowl', 'WoodChop',
        'air_conditioner', 'children_playing', 'dog_bark'
    }
    
    # Convert multi-class labels to binary (illegal vs natural)
    print("🔄 Converting multi-class labels to binary (illegal vs natural)...")
    y_binary = []
    unknown_classes = set()
    
    for label in y:
        if label in ILLEGAL_CLASSES:
            y_binary.append('illegal')
        elif label in NATURAL_CLASSES:
            y_binary.append('natural')
        else:
            unknown_classes.add(label)
            y_binary.append('natural')  # Treat unknown as natural
    
    if unknown_classes:
        print(f"⚠️ Unknown classes found: {unknown_classes}")
    
    y = np.array(y_binary)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 Class distribution:")
    for class_name, count in zip(unique, counts):
        print(f"   {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Check if we have both classes
    if len(unique) < 2:
        print("❌ Need at least 2 classes for binary classification")
        print(f"Available classes: {unique}")
        sys.exit(1)
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    
    print(f"📊 Class weights: {class_weight_dict}")
    
    # Pipeline: scale + SVM
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=3.0,
                    gamma="scale",
                    class_weight=class_weight_dict,
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    
    print("🚀 Training model...")
    clf.fit(X_train, y_train)
    
    print("🧪 Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save model bundle
    model_path = os.path.join(REPO_ROOT, "code", "chainsaw_detection_model.pkl")
    model_data = {
        "model": clf,
        "label_encoder": le,
        "feature_names": ["mfcc", "chroma", "spectral_contrast", "zcr", "rms", "mel"],
        "illegal_classes": list(ILLEGAL_CLASSES),
        "natural_classes": list(NATURAL_CLASSES),
    }
    
    joblib.dump(model_data, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Print model info
    print(f"\n📋 Model Information:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Feature dimension: {X.shape[1]}")
    print(f"   - Classes: {le.classes_}")
    print(f"   - Class weights: {class_weight_dict}")

if __name__ == "__main__":
    main()
