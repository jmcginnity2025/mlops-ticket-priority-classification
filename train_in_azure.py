"""
Standalone training script to run in Azure ML Notebooks
This avoids the Serverless/Spark quota issues
Run this as a regular Python script in Azure ML Studio
"""

print("="*70)
print("MLOps PIPELINE - Azure ML Training")
print("="*70)

# Step 1: Import libraries
print("\n1. Importing libraries...")
from azureml.core import Workspace, Dataset, Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Step 2: Get workspace context
print("\n2. Connecting to workspace...")
try:
    # Try to get the current run context (if running in Azure ML)
    run = Run.get_context()
    ws = run.experiment.workspace
    print(f"✓ Running in Azure ML - Workspace: {ws.name}")
except:
    # Fallback to config file
    ws = Workspace.from_config()
    print(f"✓ Connected to workspace: {ws.name}")

# Step 3: Load dataset
print("\n3. Loading dataset...")
dataset = Dataset.get_by_name(ws, name="support_tickets")
df = dataset.to_pandas_dataframe()
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 4: Data Preprocessing
print("\n4. Preprocessing data...")

# Feature configuration
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
target_col = 'priority_cat'

# Clean data
df_clean = df.drop_duplicates()

# Handle missing values
for col in numeric_features:
    if col in df_clean.columns:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

for col in categorical_features:
    if col in df_clean.columns:
        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
        df_clean[col].fillna(mode_val, inplace=True)

for col in binary_features:
    if col in df_clean.columns:
        df_clean[col].fillna(0, inplace=True)

# Encode categorical
label_encoders = {}
for col in categorical_features:
    if col in df_clean.columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

# Prepare features and target
feature_cols = [col for col in numeric_features + categorical_features + binary_features if col in df_clean.columns]
X = df_clean[feature_cols]
y = df_clean[target_col]

# Remap target to 0-indexed (XGBoost requirement)
# priority_cat has values 1, 2, 3 but XGBoost expects 0, 1, 2
y = y - 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

print(f"✓ Preprocessing complete - Train: {X_train.shape}, Test: {X_test.shape}")

# Step 5: Train Iteration 1
print("\n5. Training Model - Iteration 1...")
print("   (RandomizedSearchCV with 5-fold CV - optimized for speed)")

param_distributions = {
    'max_depth': [4, 5, 6, 7, 8],
    'n_estimators': [100, 150, 200, 250, 300, 350],
    'learning_rate': [0.05, 0.08, 0.1, 0.12, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

base_model = XGBClassifier(random_state=42, eval_metric='mlogloss')

random_search = RandomizedSearchCV(
    base_model, param_distributions, n_iter=20, cv=5,
    scoring='f1_weighted', verbose=1, n_jobs=-1, random_state=42
)

random_search.fit(X_train, y_train)
model_iter1 = random_search.best_estimator_

print(f"\n   CV Best Score: {random_search.best_score_:.4f}")

# Evaluate
y_train_pred = model_iter1.predict(X_train)
y_test_pred = model_iter1.predict(X_test)

metrics_iter1 = {
    'train_accuracy': accuracy_score(y_train, y_train_pred),
    'test_accuracy': accuracy_score(y_test, y_test_pred),
    'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
    'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
    'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
    'test_recall': recall_score(y_test, y_test_pred, average='weighted')
}

print(f"\n   Best params: {random_search.best_params_}")
print(f"   Test Accuracy: {metrics_iter1['test_accuracy']:.4f}")
print(f"   Test F1: {metrics_iter1['test_f1']:.4f}")

# Step 6: Train Iteration 2
print("\n6. Training Model - Iteration 2...")
print("   (RandomizedSearchCV with 5-fold CV - optimized for speed)")

param_distributions = {
    'max_depth': [6, 7, 8, 9, 10],
    'n_estimators': [200, 250, 300, 350, 400],
    'learning_rate': [0.08, 0.1, 0.12, 0.14, 0.16],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

base_model = XGBClassifier(random_state=42, eval_metric='mlogloss')

random_search = RandomizedSearchCV(
    base_model, param_distributions, n_iter=20, cv=5,
    scoring='f1_weighted', verbose=1, n_jobs=-1, random_state=42
)

random_search.fit(X_train, y_train)
model_iter2 = random_search.best_estimator_

print(f"\n   CV Best Score: {random_search.best_score_:.4f}")

# Evaluate
y_train_pred = model_iter2.predict(X_train)
y_test_pred = model_iter2.predict(X_test)

metrics_iter2 = {
    'train_accuracy': accuracy_score(y_train, y_train_pred),
    'test_accuracy': accuracy_score(y_test, y_test_pred),
    'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
    'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
    'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
    'test_recall': recall_score(y_test, y_test_pred, average='weighted')
}

print(f"\n   Best params: {random_search.best_params_}")
print(f"   Test Accuracy: {metrics_iter2['test_accuracy']:.4f}")
print(f"   Test F1: {metrics_iter2['test_f1']:.4f}")

# Step 7: Compare & Test
print("\n7. Model Comparison & Regression Tests...")

comparison = pd.DataFrame([
    {'Iteration': 1, **metrics_iter1},
    {'Iteration': 2, **metrics_iter2}
])
print("\n", comparison.to_string(index=False))

# Regression tests
MIN_ACCURACY = 0.75
MIN_F1_SCORE = 0.70
all_passed = True

for i, metrics in [(1, metrics_iter1), (2, metrics_iter2)]:
    print(f"\n   Iteration {i}:")
    if metrics['test_accuracy'] >= MIN_ACCURACY:
        print(f"     ✓ Accuracy {metrics['test_accuracy']:.4f} >= {MIN_ACCURACY}")
    else:
        print(f"     ✗ Accuracy {metrics['test_accuracy']:.4f} < {MIN_ACCURACY}")
        all_passed = False

    if metrics['test_f1'] >= MIN_F1_SCORE:
        print(f"     ✓ F1 Score {metrics['test_f1']:.4f} >= {MIN_F1_SCORE}")
    else:
        print(f"     ✗ F1 Score {metrics['test_f1']:.4f} < {MIN_F1_SCORE}")
        all_passed = False

# Step 8: Register Best Model
print("\n8. Registering best model...")

best_iteration = 1 if metrics_iter1['test_f1'] > metrics_iter2['test_f1'] else 2
best_model = model_iter1 if best_iteration == 1 else model_iter2
best_metrics = metrics_iter1 if best_iteration == 1 else metrics_iter2

import joblib
from azureml.core import Model

model_path = "ticket_priority_classifier.pkl"
joblib.dump(best_model, model_path)

registered_model = Model.register(
    workspace=ws,
    model_path=model_path,
    model_name="ticket-priority-classifier",
    description="XGBoost classifier for support ticket priority",
    tags={
        'iteration': str(best_iteration),
        'accuracy': f"{best_metrics['test_accuracy']:.4f}",
        'f1_score': f"{best_metrics['test_f1']:.4f}",
        'framework': 'xgboost',
        'type': 'classification'
    }
)

print(f"✓ Model registered: {registered_model.name} v{registered_model.version}")

# Final Summary
print("\n" + "="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
print(f"\n🏆 Best Model: Iteration {best_iteration}")
print(f"   Accuracy: {best_metrics['test_accuracy']:.4f}")
print(f"   F1 Score: {best_metrics['test_f1']:.4f}")
print(f"   Precision: {best_metrics['test_precision']:.4f}")
print(f"   Recall: {best_metrics['test_recall']:.4f}")

if all_passed:
    print("\n✅ All regression tests PASSED")
else:
    print("\n⚠️  Some regression tests FAILED")

print("\n📍 View results in Azure ML Studio:")
print("   - Experiments → ticket-priority-classification")
print("   - Models → ticket-priority-classifier")
print("="*70)
