"""
Model Training Script
Trains ML models with support for multiple iterations
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier
import joblib
import json
import os
import argparse
from datetime import datetime

# Directories
DATA_DIR = "processed_data"
MODELS_DIR = "models"

def load_processed_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

    # XGBoost requires labels starting from 0, so remap 1,2,3 to 0,1,2
    y_train = y_train - 1
    y_test = y_test - 1

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)} (remapped from 1,2,3)")
    return X_train, X_test, y_train, y_test

def train_iteration_1(X_train, X_test, y_train, y_test):
    """
    Iteration 1: Baseline Model
    Simple Random Forest with default parameters
    """
    print("\n" + "="*70)
    print("TRAINING ITERATION 1: Baseline Random Forest")
    print("="*70)

    # Simple Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'iteration': 1,
        'model_type': 'RandomForest',
        'timestamp': datetime.now().isoformat(),
        'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'train_f1': float(f1_score(y_train, y_train_pred, average='weighted')),
        'test_f1': float(f1_score(y_test, y_test_pred, average='weighted')),
        'test_precision': float(precision_score(y_test, y_test_pred, average='weighted')),
        'test_recall': float(recall_score(y_test, y_test_pred, average='weighted')),
        'parameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    }

    print("\nIteration 1 Results:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Test F1 Score:  {metrics['test_f1']:.4f}")

    # Save model and metrics
    iteration_dir = f"{MODELS_DIR}/iteration_1"
    os.makedirs(iteration_dir, exist_ok=True)

    joblib.dump(model, f"{iteration_dir}/model.pkl")
    with open(f"{iteration_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {iteration_dir}/")

    return model, metrics

def train_iteration_2(X_train, X_test, y_train, y_test):
    """
    Iteration 2: Improved Model
    XGBoost with tuned hyperparameters
    """
    print("\n" + "="*70)
    print("TRAINING ITERATION 2: Improved XGBoost")
    print("="*70)

    # XGBoost with tuned parameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'iteration': 2,
        'model_type': 'XGBoost',
        'timestamp': datetime.now().isoformat(),
        'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'train_f1': float(f1_score(y_train, y_train_pred, average='weighted')),
        'test_f1': float(f1_score(y_test, y_test_pred, average='weighted')),
        'test_precision': float(precision_score(y_test, y_test_pred, average='weighted')),
        'test_recall': float(recall_score(y_test, y_test_pred, average='weighted')),
        'parameters': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    }

    print("\nIteration 2 Results:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Test F1 Score:  {metrics['test_f1']:.4f}")

    # Save model and metrics
    iteration_dir = f"{MODELS_DIR}/iteration_2"
    os.makedirs(iteration_dir, exist_ok=True)

    joblib.dump(model, f"{iteration_dir}/model.pkl")
    with open(f"{iteration_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {iteration_dir}/")

    return model, metrics

def main(iteration=None):
    """Main training function"""
    print("="*70)
    print("MODEL TRAINING PIPELINE")
    print("="*70)

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Train requested iteration(s)
    if iteration == 1:
        train_iteration_1(X_train, X_test, y_train, y_test)
    elif iteration == 2:
        train_iteration_2(X_train, X_test, y_train, y_test)
    else:
        # Train both iterations
        train_iteration_1(X_train, X_test, y_train, y_test)
        train_iteration_2(X_train, X_test, y_train, y_test)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--iteration', type=int, choices=[1, 2], default=None,
                        help='Specific iteration to train (1 or 2), or both if not specified')
    args = parser.parse_args()

    main(iteration=args.iteration)
