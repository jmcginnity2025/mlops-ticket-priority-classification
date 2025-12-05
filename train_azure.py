"""
Training Script for Azure ML
Trains both iterations and logs metrics to Azure ML Studio (native logging, not MLflow)
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
import json

# Import both Azure ML native logging and MLflow (minimal usage)
try:
    from azureml.core import Run
    run = Run.get_context()
    AZURE_ML_LOGGING = True
    print("INFO: Azure ML native logging enabled")
except:
    AZURE_ML_LOGGING = False
    print("INFO: Running locally - Azure ML logging not available")

# Try to import MLflow for simple metric logging only (not model logging)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    print("INFO: MLflow available for metric logging")
except:
    MLFLOW_AVAILABLE = False
    print("INFO: MLflow not available")

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

# No MLflow logging - we'll save everything to files

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

# Save metrics and model to files
os.makedirs("outputs", exist_ok=True)

# Save iteration 1 metrics
metrics_1 = {
    "iteration": 1,
    "model_type": "RandomForest",
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "train_f1": float(train_f1),
    "test_f1": float(test_f1),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall)
}
with open("outputs/iteration_1_metrics.json", "w") as f:
    json.dump(metrics_1, f, indent=2)
print("   Metrics saved to outputs/iteration_1_metrics.json")

# Log to Azure ML Studio (will appear in Metrics tab)
if AZURE_ML_LOGGING:
    run.log("iteration_1_train_accuracy", float(train_acc))
    run.log("iteration_1_test_accuracy", float(test_acc))
    run.log("iteration_1_train_f1", float(train_f1))
    run.log("iteration_1_test_f1", float(test_f1))
    run.log("iteration_1_test_precision", float(test_precision))
    run.log("iteration_1_test_recall", float(test_recall))
    print("   Metrics logged to Azure ML Studio")

# Also log to MLflow (simple metrics only, no model artifacts)
if MLFLOW_AVAILABLE:
    mlflow.log_metric("iteration_1_train_accuracy", float(train_acc))
    mlflow.log_metric("iteration_1_test_accuracy", float(test_acc))
    mlflow.log_metric("iteration_1_train_f1", float(train_f1))
    mlflow.log_metric("iteration_1_test_f1", float(test_f1))
    mlflow.log_param("iteration_1_model_type", "RandomForest")
    mlflow.log_param("iteration_1_n_estimators", 100)
    mlflow.log_param("iteration_1_max_depth", 10)
    print("   Metrics logged to MLflow")

# Save iteration 1 model
with open("outputs/iteration_1_model.pkl", "wb") as f:
    pickle.dump(model1, f)
print("   Model saved to outputs/iteration_1_model.pkl")

print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Test Accuracy:  {test_acc:.4f}")
print(f"   Test F1 Score:  {test_f1:.4f}")

# ============================================================================
# ITERATION 2: XGBoost (Improved)
# ============================================================================
print("\n" + "="*70)
print("ITERATION 2: Improved XGBoost")
print("="*70)

# No MLflow logging - we'll save everything to files

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

# Save iteration 2 metrics
metrics_2 = {
    "iteration": 2,
    "model_type": "XGBoost",
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "train_f1": float(train_f1),
    "test_f1": float(test_f1),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall)
}
with open("outputs/iteration_2_metrics.json", "w") as f:
    json.dump(metrics_2, f, indent=2)
print("   Metrics saved to outputs/iteration_2_metrics.json")

# Log to Azure ML Studio (will appear in Metrics tab)
if AZURE_ML_LOGGING:
    run.log("iteration_2_train_accuracy", float(train_acc))
    run.log("iteration_2_test_accuracy", float(test_acc))
    run.log("iteration_2_train_f1", float(train_f1))
    run.log("iteration_2_test_f1", float(test_f1))
    run.log("iteration_2_test_precision", float(test_precision))
    run.log("iteration_2_test_recall", float(test_recall))

    # Log comparison metrics
    improvement = (metrics_2['test_accuracy'] - metrics_1['test_accuracy']) * 100
    run.log("accuracy_improvement_percent", float(improvement))
    print("   Metrics logged to Azure ML Studio")

# Also log to MLflow (simple metrics only, no model artifacts)
if MLFLOW_AVAILABLE:
    mlflow.log_metric("iteration_2_train_accuracy", float(train_acc))
    mlflow.log_metric("iteration_2_test_accuracy", float(test_acc))
    mlflow.log_metric("iteration_2_train_f1", float(train_f1))
    mlflow.log_metric("iteration_2_test_f1", float(test_f1))
    mlflow.log_param("iteration_2_model_type", "XGBoost")
    mlflow.log_param("iteration_2_n_estimators", 200)
    mlflow.log_param("iteration_2_max_depth", 6)
    mlflow.log_param("iteration_2_learning_rate", 0.1)

    # Log comparison
    improvement = (metrics_2['test_accuracy'] - metrics_1['test_accuracy']) * 100
    mlflow.log_metric("accuracy_improvement_percent", float(improvement))
    print("   Metrics logged to MLflow")

# Save iteration 2 model
with open("outputs/iteration_2_model.pkl", "wb") as f:
    pickle.dump(model2, f)
print("   Model saved to outputs/iteration_2_model.pkl")

print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Test Accuracy:  {test_acc:.4f}")
print(f"   Test F1 Score:  {test_f1:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nBoth iterations trained successfully!")
print("Models and metrics saved to outputs/ directory")
print("\nSummary:")
print(f"  Iteration 1 (Random Forest): {metrics_1['test_accuracy']:.4f} accuracy")
print(f"  Iteration 2 (XGBoost):       {metrics_2['test_accuracy']:.4f} accuracy")
print(f"  Improvement: {(metrics_2['test_accuracy'] - metrics_1['test_accuracy'])*100:.2f}%")
