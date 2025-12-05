"""
Local test script for training - simpler version without Azure ML
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LOCAL TRAINING TEST - Ticket Priority Classification")
print("="*70)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('data/cleaned_support_tickets - with context.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

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
print("\n2. Preprocessing data...")
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

print(f"✓ Preprocessing complete - Train: {X_train.shape}, Test: {X_test.shape}")

# Train Iteration 1 (quick version - fewer combinations)
print("\n3. Training Model - Iteration 1...")
print("   (RandomizedSearchCV with 3-fold CV - 10 combinations for speed)")

param_distributions = {
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

base_model = XGBClassifier(random_state=42, eval_metric='mlogloss')

random_search = RandomizedSearchCV(
    base_model, param_distributions, n_iter=10, cv=3,
    scoring='f1_weighted', verbose=1, n_jobs=-1, random_state=42
)

random_search.fit(X_train, y_train)
model_iter1 = random_search.best_estimator_

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

# Regression tests
print("\n4. Regression Tests...")
MIN_ACCURACY = 0.75
MIN_F1_SCORE = 0.70

if metrics_iter1['test_accuracy'] >= MIN_ACCURACY:
    print(f"   ✓ Accuracy {metrics_iter1['test_accuracy']:.4f} >= {MIN_ACCURACY}")
else:
    print(f"   ✗ Accuracy {metrics_iter1['test_accuracy']:.4f} < {MIN_ACCURACY}")

if metrics_iter1['test_f1'] >= MIN_F1_SCORE:
    print(f"   ✓ F1 Score {metrics_iter1['test_f1']:.4f} >= {MIN_F1_SCORE}")
else:
    print(f"   ✗ F1 Score {metrics_iter1['test_f1']:.4f} < {MIN_F1_SCORE}")

print("\n" + "="*70)
print("✅ LOCAL TRAINING TEST COMPLETE!")
print("="*70)
print("\nIf this succeeded, the training code works fine.")
print("The issue is likely with Azure ML environment/configuration.")
