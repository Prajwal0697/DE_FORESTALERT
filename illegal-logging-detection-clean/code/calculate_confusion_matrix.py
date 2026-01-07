#!/usr/bin/env python3
"""
Calculate and display updated confusion matrix for the trained model
This script loads the model, recreates the test split, and calculates fresh metrics
"""

import os
import sys
import warnings
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# Configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "code", "chainsaw_detection_model.pkl")
MERGED_CSV_PATH = os.path.join(REPO_ROOT, "dataset_forest", "merged_dataset.csv")
AUDIO_ROOTS = [
    os.path.join(REPO_ROOT, "Audios"),
    os.path.join(REPO_ROOT, "dataset_forest", "audio"),
    os.path.join(REPO_ROOT, "dataset_forest", "Audios"),
]

def find_audio_file(filename: str) -> Optional[str]:
    """
    Find audio file with flexible matching (same as training script)
    """
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
    """Extract robust features from an audio file (same as training script)"""
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if y is None or y.size == 0:
            return None

        y = np.nan_to_num(y)

        # Features (same as training)
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
    """Load dataset and extract features (same as training script)"""
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
    print("=" * 70)
    print("📊 CALCULATING UPDATED CONFUSION MATRIX FOR TRAINED MODEL")
    print("=" * 70)
    print()
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    print("📦 Loading trained model...")
    try:
        model_data = joblib.load(MODEL_PATH)
        clf = model_data['model']
        le = model_data['label_encoder']
        ILLEGAL_CLASSES = set(model_data.get('illegal_classes', []))
        NATURAL_CLASSES = set(model_data.get('natural_classes', []))
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {le.classes_}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load dataset
    print("📦 Loading dataset from merged_dataset.csv...")
    X, y, paths = load_dataset(MERGED_CSV_PATH)
    
    # Convert multi-class labels to binary (same as training)
    print("🔄 Converting multi-class labels to binary (illegal vs natural)...")
    y_binary = []
    for label in y:
        if label in ILLEGAL_CLASSES:
            y_binary.append('illegal')
        elif label in NATURAL_CLASSES:
            y_binary.append('natural')
        else:
            y_binary.append('natural')
    
    y = np.array(y_binary)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 Class distribution:")
    for class_name, count in zip(unique, counts):
        print(f"   {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    print()
    
    # Encode labels (must match training)
    le_new = LabelEncoder()
    y_enc = le_new.fit_transform(y)
    
    # Recreate the exact same train/test split (same random_state=42)
    print("🔄 Creating train/test split (random_state=42, test_size=0.2)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print()
    
    # Make predictions on test set
    print("🔮 Making predictions on test set...")
    y_pred = clf.predict(X_test)
    print("✅ Predictions completed")
    print()
    
    # Calculate metrics
    print("📊 CALCULATING METRICS")
    print("=" * 70)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Overall Accuracy: {acc*100:.2f}%")
    print()
    
    # Calculate per-class metrics
    class_names = le.classes_
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    print("📈 Per-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"   {class_name.upper()}:")
        print(f"      Precision: {precision[i]*100:.2f}%")
        print(f"      Recall: {recall[i]*100:.2f}%")
        print(f"      F1-Score: {f1[i]*100:.2f}%")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("📊 CONFUSION MATRIX")
    print("=" * 70)
    print()
    print("Raw matrix:")
    print(cm)
    print()
    
    # Pretty print confusion matrix
    print("Formatted confusion matrix:")
    print(f"      Predicted: {class_names[0]:<12} {class_names[1]:<12}")
    print(f"Actual {class_names[0]:<12} {cm[0,0]:<12} {cm[0,1]:<12}")
    print(f"      {class_names[1]:<12} {cm[1,0]:<12} {cm[1,1]:<12}")
    print()
    
    # Interpretation
    if len(class_names) == 2:
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        print("📋 Interpretation:")
        print(f"   True Negatives ({class_names[0]} correctly identified): {tn}")
        print(f"   False Positives ({class_names[0]} misclassified as {class_names[1]}): {fp}")
        print(f"   False Negatives ({class_names[1]} misclassified as {class_names[0]}): {fn}")
        print(f"   True Positives ({class_names[1]} correctly identified): {tp}")
    print()
    
    # Classification report
    print("📋 CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=class_names))
    print()
    
    # Create visualization
    print("📊 Creating confusion matrix visualization...")
    try:
        plt.figure(figsize=(10, 8))
        
        # Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Updated Results', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(REPO_ROOT, "code", "confusion_matrix_updated.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {output_path}")
        plt.close()
        
        # Create detailed performance visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Actual', fontsize=11)
        axes[0].set_xlabel('Predicted', fontsize=11)
        
        # Metrics Bar Chart
        metrics_data = {
            'Metric': ['Accuracy', f'Precision\n({class_names[0]})', f'Recall\n({class_names[0]})', 
                      f'F1-Score\n({class_names[0]})', f'Precision\n({class_names[1]})', 
                      f'Recall\n({class_names[1]})', f'F1-Score\n({class_names[1]})'],
            'Score': [acc*100, precision[0]*100, recall[0]*100, f1[0]*100,
                     precision[1]*100, recall[1]*100, f1[1]*100]
        }
        
        bars = axes[1].bar(metrics_data['Metric'], metrics_data['Score'], 
                           color=['#2E8B57', '#4ECDC4', '#4ECDC4', '#4ECDC4', 
                                  '#FF6B6B', '#FF6B6B', '#FF6B6B'])
        axes[1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score (%)', fontsize=11)
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, metrics_data['Score']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        detailed_output = os.path.join(REPO_ROOT, "code", "model_performance_updated.png")
        plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed performance visualization saved to: {detailed_output}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
        print("   (Continuing without visualization)")
    
    # Save metrics to file
    metrics_output = os.path.join(REPO_ROOT, "code", "model_metrics_updated.txt")
    with open(metrics_output, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("UPDATED MODEL PERFORMANCE METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n")
        f.write("Per-Class Metrics:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  {class_name.upper()}:\n")
            f.write(f"    Precision: {precision[i]*100:.2f}%\n")
            f.write(f"    Recall: {recall[i]*100:.2f}%\n")
            f.write(f"    F1-Score: {f1[i]*100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    print(f"✅ Metrics saved to: {metrics_output}")
    print()
    print("=" * 70)
    print("✅ CONFUSION MATRIX CALCULATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()

