#!/usr/bin/env python3
"""
Train model using merged_dataset.csv, combining audio from both Audios/ and dataset_forest/audio/

- Expects merged_dataset.csv at repo root with columns: file,label
- Resolves file paths flexibly (absolute/relative, Audios/, dataset_forest/audio/)
- Extracts robust features with librosa (MFCC, chroma, spectral, ZCR, RMS, mel)
- Trains an SVM with class_weight=balanced
- Prints report and saves model to code/chainsaw_detection_model.pkl
"""

import os
import sys
import json
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MERGED_CSV_PATH = os.path.join(REPO_ROOT, "dataset_forest", "merged_dataset.csv")
AUDIO_ROOTS = [
    os.path.join(REPO_ROOT, "dataset_forest", "Audios"),
    os.path.join(REPO_ROOT, "dataset_forest", "audio"),
]
MODEL_OUT_PATH = os.path.join(REPO_ROOT, "code", "chainsaw_detection_model.pkl")


def resolve_audio_path(rel_or_name: str) -> Optional[str]:
    """Resolve audio file path with multiple flexible strategies"""
    print(f"Resolving: {rel_or_name}")
    
    # Normalize filename (remove any leading paths)
    filename = os.path.basename(rel_or_name)
    
    # Search paths
    search_paths = [
        REPO_ROOT,
        os.path.join(REPO_ROOT, "dataset_forest"),
        os.path.join(REPO_ROOT, "dataset_forest", "Audios"),
        os.path.join(REPO_ROOT, "dataset_forest", "audio"),
    ]
    
    # Additional search paths for nested directories
    for root, dirs, _ in os.walk(os.path.join(REPO_ROOT, "dataset_forest")):
        search_paths.append(root)
    
    # Print all search paths for debugging
    print("Search paths:", search_paths)
    
    # Matching strategies
    def match_filename(file_path):
        # Exact match
        if os.path.basename(file_path) == filename:
            return True
        
        # Remove special characters and compare
        clean_filename = ''.join(c for c in filename if c.isalnum())
        clean_file = ''.join(c for c in os.path.basename(file_path) if c.isalnum())
        
        return (clean_filename.lower() == clean_file.lower() or 
                clean_filename in clean_file or 
                clean_file in clean_filename)
    
    # Search through all paths
    for search_path in search_paths:
        try:
            # Direct file check
            direct_path = os.path.join(search_path, filename)
            if os.path.exists(direct_path):
                print(f"✅ Found direct match: {direct_path}")
                return direct_path
            
            # Recursive search in directory
            for root, _, files in os.walk(search_path):
                matching_files = [
                    os.path.join(root, f) 
                    for f in files 
                    if f.endswith('.wav') and match_filename(f)
                ]
                
                if matching_files:
                    print(f"✅ Found matching file: {matching_files[0]}")
                    return matching_files[0]
                
                # If no match, print files in the current directory for debugging
                print(f"Files in {root}:", files)
        
        except Exception as e:
            print(f"Error searching in {search_path}: {e}")
    
    print(f"❌ Could not resolve path for: {rel_or_name}")
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
    except Exception:
        return None


def load_dataset(merged_csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not os.path.exists(merged_csv_path):
        print(f"❌ merged_dataset.csv not found at: {merged_csv_path}")
        sys.exit(1)

    print(f"🔍 Resolving audio files from CSV...")
    print(f"CSV Path: {merged_csv_path}")
    print(f"Audio Roots: {AUDIO_ROOTS}")
    
    # Print first few rows of the CSV
    df = pd.read_csv(merged_csv_path)
    print("\nFirst 5 rows of CSV:")
    print(df.head())
    
    print("\nColumn names:", list(df.columns))

    # Expect columns: file,label
    if not {"file", "label"}.issubset(df.columns):
        print("❌ merged_dataset.csv must contain columns: file,label")
        sys.exit(1)

    file_col = df["file"].astype(str).tolist()
    labels_raw = df["label"].tolist()

    resolved_paths: List[str] = []
    y_labels: List[str] = []

    for rel in file_col:
        path = resolve_audio_path(rel)
        if path is None:
            # Also try basename match under both roots
            base = os.path.basename(rel)
            found = None
            for root in AUDIO_ROOTS:
                candidate = os.path.join(root, base)
                if os.path.exists(candidate):
                    found = candidate
                    break
            path = found

        if path is not None and os.path.exists(path):
            resolved_paths.append(path)
        else:
            resolved_paths.append("")

    # Filter rows with missing files
    keep_mask = [p != "" for p in resolved_paths]
    kept_paths = [p for p, k in zip(resolved_paths, keep_mask) if k]
    kept_labels = [l for l, k in zip(labels_raw, keep_mask) if k]

    if len(kept_paths) == 0:
        print("❌ No audio files found matching entries in merged_dataset.csv")
        sys.exit(1)

    # Normalize labels to string classes '0'/'1' or names
    labels_norm: List[str] = []
    for l in kept_labels:
        if isinstance(l, (int, float)):
            labels_norm.append(str(int(l)))
        else:
            labels_norm.append(str(l).strip())

    print(f"✅ Found {len(kept_paths)} audio files referenced in merged_dataset.csv")

    # Extract features
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    missing_count = 0

    for path, lbl in zip(kept_paths, labels_norm):
        feats = extract_features(path)
        if feats is None:
            missing_count += 1
            continue
        X_list.append(feats)
        y_list.append(lbl)

    if len(X_list) == 0:
        print("❌ Feature extraction failed for all files")
        sys.exit(1)

    if missing_count:
        print(f"⚠️ Skipped {missing_count} files due to read/feature errors")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, kept_paths


def main():
    print("📦 Loading dataset from merged_dataset.csv ...")
    X, y, paths = load_dataset(MERGED_CSV_PATH)

    # Define illegal classes (class 1) and natural classes (class 0)
    # Based on your merged dataset classification
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
    bundle = {
        "pipeline": clf,
        "label_encoder_classes_": le.classes_.tolist(),
        "feature_version": 1,
        "notes": "Trained on merged_dataset.csv with paths from Audios/ and dataset_forest/audio/",
    }
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_OUT_PATH)
    print(f"💾 Saved model to: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
