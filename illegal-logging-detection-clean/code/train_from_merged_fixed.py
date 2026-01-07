#!/usr/bin/env python3
"""
Train model using merged_dataset.csv with improved file path resolution
"""

import os
import sys
import json
import warnings
from typing import List, Tuple, Optional
import glob

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
MERGED_CSV_PATH = os.path.join(REPO_ROOT, "dataset_forest", "merged_dataset.csv")
AUDIO_ROOTS = [
    os.path.join(REPO_ROOT, "Audios"),
    os.path.join(REPO_ROOT, "dataset_forest", "audio"),
    os.path.join(REPO_ROOT, "dataset_forest", "Audios"),
]

def find_audio_file(filename: str) -> Optional[str]:
    """
    Find audio file with flexible matching:
    1. Exact match in any audio root
    2. Basename match (ignoring path)
    3. Similar name match (ignoring extensions)
    4. Recursive search in dataset_forest
    """
    # Skip if filename looks like a directory
    if filename.endswith('/') or filename.endswith('\\'):
        return None
    
    # Try exact match first
    for root in AUDIO_ROOTS:
        if os.path.exists(root):
            exact_path = os.path.join(root, filename)
            if os.path.exists(exact_path) and os.path.isfile(exact_path) and exact_path.lower().endswith('.wav'):
                return exact_path
    
    # Try basename match
    basename = os.path.basename(filename)
    for root in AUDIO_ROOTS:
        if os.path.exists(root):
            basename_path = os.path.join(root, basename)
            if os.path.exists(basename_path) and os.path.isfile(basename_path) and basename_path.lower().endswith('.wav'):
                return basename_path
    
    # Try similar name match (ignore extensions)
    base_no_ext = os.path.splitext(basename)[0]
    for root in AUDIO_ROOTS:
        if os.path.exists(root):
            # Search for files with similar names
            for file in os.listdir(root):
                if (file.startswith(base_no_ext) or base_no_ext in file) and file.lower().endswith('.wav'):
                    full_path = os.path.join(root, file)
                    if os.path.isfile(full_path):
                        return full_path
    
    # Recursive search in dataset_forest
    dataset_forest_path = os.path.join(REPO_ROOT, "dataset_forest")
    for root, _, files in os.walk(dataset_forest_path):
        for file in files:
            if file.lower().endswith('.wav'):
                if file == basename or base_no_ext in file:
                    return os.path.join(root, file)
    
    return None

def extract_features(filepath: str, target_sr: int = 22050) -> Optional[np.ndarray]:
    """Extract robust features from an audio file. Returns 1D feature vector or None on failure."""
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if y is None or y.size == 0:
            return None

        # Ensure finite values
        y = np.nan_to_num(y)

        # Features
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

def load_dataset(merged_csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not os.path.exists(merged_csv_path):
        print(f"❌ merged_dataset.csv not found at: {merged_csv_path}")
        sys.exit(1)

    df = pd.read_csv(merged_csv_path)
    if not {"file", "label"}.issubset(df.columns):
        print("❌ merged_dataset.csv must contain columns: file,label")
        sys.exit(1)

    file_col = df["file"].astype(str).tolist()
    labels_raw = df["label"].tolist()

    resolved_paths: List[str] = []
    y_labels: List[str] = []
    found_count = 0
    missing_count = 0

    print(f"🔍 Resolving {len(file_col)} audio files...")
    
    for i, (rel, label) in enumerate(zip(file_col, labels_raw)):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(file_col)} files processed")
        
        path = find_audio_file(rel)
        if path is not None and os.path.exists(path):
            resolved_paths.append(path)
            y_labels.append(label)
            found_count += 1
        else:
            missing_count += 1

    print(f"✅ Found {found_count} files, missing {missing_count} files")

    if len(resolved_paths) == 0:
        print("❌ No audio files found matching entries in merged_dataset.csv")
        sys.exit(1)

    # Normalize labels
    labels_norm: List[str] = []
    for l in y_labels:
        if isinstance(l, (int, float)):
            labels_norm.append(str(int(l)))
        else:
            labels_norm.append(str(l).strip())

    # Extract features
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    feature_failures = 0

    print("🔧 Extracting features...")
    for i, (path, lbl) in enumerate(zip(resolved_paths, labels_norm)):
        if i % 500 == 0:
            print(f"   Progress: {i}/{len(resolved_paths)} features extracted")
        
        feats = extract_features(path)
        if feats is None:
            feature_failures += 1
            continue
        X_list.append(feats)
        y_list.append(lbl)

    if len(X_list) == 0:
        print("❌ Feature extraction failed for all files")
        sys.exit(1)

    if feature_failures:
        print(f"⚠️ Skipped {feature_failures} files due to feature extraction errors")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, resolved_paths

def main():
    print("📦 Loading dataset from merged_dataset.csv ...")
    X, y, paths = load_dataset(MERGED_CSV_PATH)

    # Define illegal classes (class 1) and natural classes (class 0)
    ILLEGAL_CLASSES = {
        'axe', 'chainsaw', 'fire', 'firework', 'generator', 'gunshot', 
        'handsaw', 'helicopter', 'VehicleEngine', 'car_horn', 'drilling', 
        'engine_idling', 'gun_shot', 'jackhammer', 'siren'
    }
    
    NATURAL_CLASSES = {
        'BirdChirping', 'Clapping', 'Footsteps', 'Frog', 'Insect', 'Lion', 
        'Rain', 'Silence', 'Speaking', 'Squirrel', 'Thunderstorm', 'TreeFalling', 
        'WaterDrops', 'Whistling', 'Wind', 'WingFlaping', 'WolfHowl', 'WoodChop',
        'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music',
        'air_conditioner', 'children_playing', 'dog_bark'
    }

    # Convert multi-class labels to binary (illegal vs natural)
    print("🔄 Converting multi-class labels to binary (illegal vs natural)...")
    y_binary = []
    for label in y:
        if label in ILLEGAL_CLASSES:
            y_binary.append('illegal')
        elif label in NATURAL_CLASSES:
            y_binary.append('natural')
        else:
            print(f"⚠️ Unknown class: {label}, treating as natural")
            y_binary.append('natural')
    
    y = np.array(y_binary)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 Class distribution:")
    for class_name, count in zip(unique, counts):
        print(f"   {class_name}: {count} samples ({count/len(y)*100:.1f}%)")

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

    print("🚀 Training model ...")
    clf.fit(X_train, y_train)

    print("🧪 Evaluating ...")
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
