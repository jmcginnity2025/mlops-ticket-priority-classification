"""
Training Script for Azure ML
Trains both iterations and logs metrics with MLflow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import argparse
import os
import pickle
import tempfile

# Try to import MLflow (optional for local testing)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLflow not available - running in local test mode (no logging)")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
args = parser.parse_args()

print("="*70)
print("AZURE ML TRAINING - Support Ticket Priority Classification")
print("="*70)

# Verify data path received
print(f"\n1. Loading dataset from: {args.data_path}")
if not os.path.exists(args.data_path):
    print(f"ERROR: Dataset file not found at: {args.data_path}")
    exit(1)

df = pd.read_csv(args.data_path)
print(f"   Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Feature selection
numeric_features = [
    'org_users', 'past_30d_tickets', 'customers_affected',
    'error_rate_pct', 'downtime_min', 'description_length',
    'resolution_time_hours', 'customer_satisfaction_score',
    'revenue_dollars', 'api_calls_per_day', 'team_size',
    'satisfaction_score_0_10'
]

categorical_features = [
    'day_of_week_num', 'company_size_cat', 'industry_cat',
    'customer_tier_cat', 'region_cat', 'product_area_cat',
    'booking_channel_cat', 'reported_by_role_cat',
    'customer_sentiment_cat'
]

binary_features = ['payment_impact_flag', 'data_loss_flag', 'has_runbook']
target = 'priority_cat'

# Select available features
all_features = numeric_features + categorical_features + binary_features
features = [f for f in all_features if f in df.columns]

print(f"   Selected {len(features)} features")

# Clean data
print("\n2. Preprocessing...")
df = df.drop_duplicates()

for col in features:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
        df[col].fillna(mode_val, inplace=True)

df = df[df[target].notna()]

# Prepare X and y
X = df[features]
y = df[target].values

# Remap labels from 1,2,3 to 0,1,2 for XGBoost
y = y - 1

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# ============================================================================
# ITERATION 1: Random Forest (Baseline)
# ============================================================================
print("\n" + "="*70)
print("ITERATION 1: Baseline Random Forest")
print("="*70)

# Azure ML automatically manages the run, so we don't need start_run()
# Log parameters directly
if MLFLOW_AVAILABLE:
    mlflow.log_param("iteration_1_model_type", "RandomForest")
    mlflow.log_param("iteration_1_n_estimators", 100)
    mlflow.log_param("iteration_1_max_depth", 10)

# Train
print("Training...")
model1 = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model1.fit(X_train, y_train)

# Evaluate
y_train_pred = model1.predict(X_train)
y_test_pred = model1.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

# Log metrics for iteration 1
if MLFLOW_AVAILABLE:
    mlflow.log_metric("iteration_1_train_accuracy", train_acc)
    mlflow.log_metric("iteration_1_test_accuracy", test_acc)
    mlflow.log_metric("iteration_1_train_f1", train_f1)
    mlflow.log_metric("iteration_1_test_f1", test_f1)
    mlflow.log_metric("iteration_1_test_precision", test_precision)
    mlflow.log_metric("iteration_1_test_recall", test_recall)

    # Save model as artifact (avoids MLflow logged-models API issue)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "iteration_1_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model1, f)
        mlflow.log_artifact(model_path, artifact_path="iteration_1_model")
        print("   Model saved as artifact")

print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Test Accuracy:  {test_acc:.4f}")
print(f"   Test F1 Score:  {test_f1:.4f}")

# ============================================================================
# ITERATION 2: XGBoost (Improved)
# ============================================================================
print("\n" + "="*70)
print("ITERATION 2: Improved XGBoost")
print("="*70)

# Azure ML automatically manages the run, so we don't need start_run()
# Log parameters directly
if MLFLOW_AVAILABLE:
    mlflow.log_param("iteration_2_model_type", "XGBoost")
    mlflow.log_param("iteration_2_n_estimators", 200)
    mlflow.log_param("iteration_2_max_depth", 6)
    mlflow.log_param("iteration_2_learning_rate", 0.1)

# Train
print("Training...")
model2 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)
model2.fit(X_train, y_train)

# Evaluate
y_train_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

# Log metrics for iteration 2
if MLFLOW_AVAILABLE:
    mlflow.log_metric("iteration_2_train_accuracy", train_acc)
    mlflow.log_metric("iteration_2_test_accuracy", test_acc)
    mlflow.log_metric("iteration_2_train_f1", train_f1)
    mlflow.log_metric("iteration_2_test_f1", test_f1)
    mlflow.log_metric("iteration_2_test_precision", test_precision)
    mlflow.log_metric("iteration_2_test_recall", test_recall)

    # Save model as artifact (avoids MLflow logged-models API issue)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "iteration_2_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model2, f)
        mlflow.log_artifact(model_path, artifact_path="iteration_2_model")
        print("   Model saved as artifact")

print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Test Accuracy:  {test_acc:.4f}")
print(f"   Test F1 Score:  {test_f1:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nBoth iterations trained successfully!")
print("Metrics logged to MLflow")
