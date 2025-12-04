"""
Data Preprocessing Script
Cleans and prepares the support ticket dataset for model training
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime

# Dataset path
DATA_PATH = "data/cleaned_support_tickets - with context.csv"
OUTPUT_DIR = "processed_data"

def load_data():
    """Load the dataset"""
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def select_features(df):
    """
    Select relevant features for the model
    Focus on key business metrics and ticket characteristics
    """
    # Numeric features - business metrics
    numeric_features = [
        'org_users', 'past_30d_tickets', 'customers_affected',
        'error_rate_pct', 'downtime_min', 'description_length',
        'resolution_time_hours', 'customer_satisfaction_score',
        'revenue_dollars', 'api_calls_per_day', 'team_size',
        'satisfaction_score_0_10'
    ]

    # Categorical features (encoded as numeric)
    categorical_features = [
        'day_of_week_num', 'company_size_cat', 'industry_cat',
        'customer_tier_cat', 'region_cat', 'product_area_cat',
        'booking_channel_cat', 'reported_by_role_cat',
        'customer_sentiment_cat'
    ]

    # Binary features
    binary_features = ['payment_impact_flag', 'data_loss_flag', 'has_runbook']

    # Target
    target = 'priority_cat'

    # Select only features that exist in the dataframe
    all_features = numeric_features + categorical_features + binary_features
    available_features = [f for f in all_features if f in df.columns]

    print(f"\nSelected {len(available_features)} features:")
    print(f"  - Numeric: {len([f for f in numeric_features if f in df.columns])}")
    print(f"  - Categorical: {len([f for f in categorical_features if f in df.columns])}")
    print(f"  - Binary: {len([f for f in binary_features if f in df.columns])}")

    return available_features, target

def clean_data(df, features, target):
    """Clean and handle missing values"""
    print("\nCleaning data...")

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in features:
        if col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Fill numeric with median
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Fill categorical with mode
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col].fillna(mode_val, inplace=True)

    # Ensure target has no missing values
    df = df[df[target].notna()]

    print(f"After cleaning: {len(df)} rows")

    return df

def preprocess_pipeline():
    """Main preprocessing pipeline"""
    print("="*70)
    print("PREPROCESSING PIPELINE - Support Ticket Priority Classification")
    print("="*70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_data()

    # Select features
    features, target = select_features(df)

    # Clean data
    df_clean = clean_data(df, features, target)

    # Prepare X and y
    X = df_clean[features]
    y = df_clean[target]

    # Split into train and test
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

    # Save processed data
    print("\nSaving processed data...")
    X_train_scaled.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    # Save scaler for later use
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(df_clean),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(features),
        'features': features,
        'target': target,
        'class_distribution': y.value_counts().to_dict()
    }

    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nClass distribution:")
    print(y.value_counts().sort_index())
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess_pipeline()
