#!/usr/bin/env python3
import os
import sys
import csv
import zipfile
import numpy as np
from pathlib import Path

# Make sure the starter_kits folder is on the Python path
sys.path.append(os.path.abspath("starter_kits"))

# Import plotting (if needed) with a non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import required modules from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import auxiliary functions from the provided starter kit files
from data import get_challenge_points
from metrics import get_tpr_at_fpr

# Define the data directory for ClavaDDPM white box challenge
CLAVADDPM_DATA_DIR = "clavaddpm_white_box"


##########################################
# 1. Feature Extraction
##########################################

def extract_features(cp):
    """
    Given a single challenge point (assumed to be a torch.Tensor),
    compute and return an 11-dimensional feature vector.
    Features computed:
      - mean, standard deviation, min, max, log(mean+1), median,
      - skewness, kurtosis, 25th percentile, 75th percentile, and IQR.
    """
    cp_np = cp.numpy()
    mean_val = np.mean(cp_np)
    std_val = np.std(cp_np)
    min_val = np.min(cp_np)
    max_val = np.max(cp_np)
    log_mean = np.log1p(mean_val)
    median_val = np.median(cp_np)
    # Use scipy.stats functions for skew and kurtosis
    # (Assuming you have scipy installed)
    from scipy.stats import skew, kurtosis
    skew_val = skew(cp_np)
    kurt_val = kurtosis(cp_np)
    pct25 = np.percentile(cp_np, 25)
    pct75 = np.percentile(cp_np, 75)
    iqr = pct75 - pct25
    return [mean_val, std_val, min_val, max_val, log_mean,
            median_val, skew_val, kurt_val, pct25, pct75, iqr]


def load_training_data(model_path):
    """
    For a given training model folder, load its challenge points
    and corresponding ground truth labels and compute features.
    Returns:
      X: (num_points, 11) numpy array of features.
      y: (num_points,) numpy array of labels.
    """
    challenge_points = get_challenge_points(model_path)
    label_file = os.path.join(model_path, "challenge_label.csv")
    # Skip header row if present (adjust skiprows if needed)
    labels = np.loadtxt(label_file, delimiter=',', skiprows=1)
    
    features = [extract_features(cp) for cp in challenge_points]
    return np.array(features), labels


##########################################
# 2. Train a Basic Logistic Regression Attack Model
##########################################

def train_attack_model():
    """
    Load all training challenge points from each model folder,
    extract features, and train a basic logistic regression model.
    The model uses a pipeline with StandardScaler followed by LogisticRegression.
    """
    X_list, y_list = [], []
    train_dir = os.path.join(CLAVADDPM_DATA_DIR, "train")
    
    # Get all training model folders and sort them by (assumed) numeric suffix
    train_model_folders = [
        os.path.join(train_dir, d)
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    train_model_folders.sort(key=lambda d: int(d.split('_')[-1]))
    
    # Loop over all model folders to build our training set
    for model_path in train_model_folders:
        X, y = load_training_data(model_path)
        X_list.append(X)
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    # Split for local evaluation (optional)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    
    # Build a simple pipeline: scale the features then use Logistic Regression.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=500,
            solver='liblinear',
            class_weight='balanced'
        ))
    ])
    
    print("Training basic Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation data using the provided TPR @ 10% FPR metric
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_tpr = get_tpr_at_fpr(y_val, val_probs)
    print(f"Local Validation TPR at 10% FPR: {val_tpr:.4f}")
    
    return pipeline


##########################################
# 3. Generate Predictions on Dev/Final Sets
##########################################

def generate_predictions(model, model_path):
    """
    For a given model folder (either dev or final), load its challenge points,
    extract features, and return the predicted membership probabilities.
    """
    challenge_points = get_challenge_points(model_path)
    features = [extract_features(cp) for cp in challenge_points]
    X = np.array(features)
    probs = model.predict_proba(X)[:, 1]
    return probs


def process_phase(phase, model):
    """
    For each model folder in a given phase (dev/final),
    generate predictions and write them to a 'prediction.csv' file.
    """
    base_path = os.path.join(CLAVADDPM_DATA_DIR, phase)
    # List only directories
    model_folders = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    # Sort folders (assumes folder names end with an underscore and number)
    model_folders.sort(key=lambda d: int(d.split('_')[-1]))
    
    for folder in model_folders:
        model_path = os.path.join(base_path, folder)
        predictions = generate_predictions(model, model_path)
        prediction_file = os.path.join(model_path, "prediction.csv")
        with open(prediction_file, "w", newline="") as f:
            writer = csv.writer(f)
            for p in predictions:
                writer.writerow([p])
        print(f"Written predictions to {prediction_file}")


##########################################
# 4. Package the Submission
##########################################

def package_submission():
    """
    Package all 'prediction.csv' files under CLAVADDPM_DATA_DIR into a zip file.
    The structure of the archive will reflect the folder hierarchy.
    """
    submission_file = "white_box_multi_table_submission.zip"
    with zipfile.ZipFile(submission_file, "w") as zipf:
        for root, dirs, files in os.walk(CLAVADDPM_DATA_DIR):
            for file in files:
                if file == "prediction.csv":
                    filepath = os.path.join(root, file)
                    # Use a relative path (arcname) for the zip file structure
                    arcname = os.path.relpath(filepath, os.getcwd())
                    zipf.write(filepath, arcname)
    print("Submission packaged in:", submission_file)


##########################################
# 5. (Optional) Evaluate on a Labeled Phase
##########################################

def evaluate_phase(model, phase):
    """
    Evaluate the model on a phase for which labels are available (e.g., train).
    This function aggregates the challenge points and labels from all folders
    and computes the TPR @ 10% FPR.
    """
    base_path = os.path.join(CLAVADDPM_DATA_DIR, phase)
    model_folders = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    model_folders.sort(key=lambda d: int(d.split('_')[-1]))
    
    X_list, y_list = [], []
    for folder in model_folders:
        try:
            X, y = load_training_data(folder)
            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            print(f"Could not load labels from {folder}: {e}")
    if X_list and y_list:
        X_phase = np.concatenate(X_list, axis=0)
        y_phase = np.concatenate(y_list, axis=0)
        tpr = get_tpr_at_fpr(y_phase, model.predict_proba(X_phase)[:, 1])
        print(f"{phase.capitalize()} TPR at 10% FPR: {tpr:.4f}")
    else:
        print(f"No labeled data available for phase: {phase}")


##########################################
# Main Entry Point
##########################################

if __name__ == "__main__":
    # Train the attack model (using basic logistic regression)
    attack_model = train_attack_model()
    
    # Optional evaluation on the 'train' phase
    print("\n=== Evaluating on Train Phase ===")
    evaluate_phase(attack_model, "train")
    
    # Process the dev and final phases to generate predictions
    for phase in ["dev", "final"]:
        print(f"\nProcessing phase: {phase}")
        process_phase(phase, attack_model)
    
    # Package everything into a submission zip file
    package_submission()