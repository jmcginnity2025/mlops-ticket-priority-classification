"""
Azure ML Training Script - Random Forest & XGBoost
==================================================

This script trains both Random Forest and XGBoost models on the support ticket
classification dataset and logs comprehensive metrics to Azure ML.

Dataset: cleaned_support_tickets - with context.csv (48,388 records)
Models: Random Forest Classifier + XGBoost Classifier
Task: Multi-class classification (4 priority classes: Low, Medium, High, Critical)
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
from pathlib import Path

# Azure ML imports
try:
    import azureml.core
    from azureml.core import Run
    run = Run.get_context()
    AZURE_ML = True
except:
    AZURE_ML = False
    print("Not running in Azure ML context")


def load_and_preprocess_data(data_path):
    """
    Load and preprocess the support tickets dataset

    Args:
        data_path: Path to the CSV file

    Returns:
        X_train, X_test, y_train, y_test, vectorizer, label_encoder
    """
    print("="*80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"✅ Dataset loaded successfully")
    print(f"   - Total records: {len(df):,}")
    print(f"   - Total features: {len(df.columns)}")
    print(f"   - Columns: {list(df.columns)}")

    # Identify text and target columns
    # Assuming there's a 'text' or 'description' column and a 'priority' column
    text_columns = [col for col in df.columns if 'text' in col.lower() or 'description' in col.lower() or 'title' in col.lower()]
    target_columns = [col for col in df.columns if 'priority' in col.lower() or 'label' in col.lower() or 'class' in col.lower()]

    if not text_columns:
        text_col = df.columns[0]  # Use first column if no obvious text column
    else:
        text_col = text_columns[0]

    if not target_columns:
        target_col = df.columns[-1]  # Use last column if no obvious target
    else:
        target_col = target_columns[0]

    print(f"\n📊 Using columns:")
    print(f"   - Text column: '{text_col}'")
    print(f"   - Target column: '{target_col}'")

    # Handle missing values
    df = df.dropna(subset=[text_col, target_col])
    print(f"\n✅ After removing missing values: {len(df):,} records")

    # Class distribution
    print(f"\n📊 Class Distribution:")
    class_dist = df[target_col].value_counts()
    for label, count in class_dist.items():
        print(f"   - {label}: {count:,} ({count/len(df)*100:.1f}%)")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])

    print(f"\n✅ Encoded classes:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"   - {idx}: {label}")

    # TF-IDF Vectorization
    print(f"\n🔄 Performing TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X = vectorizer.fit_transform(df[text_col].astype(str))
    print(f"✅ TF-IDF completed")
    print(f"   - Features extracted: {X.shape[1]:,}")
    print(f"   - Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✅ Train-Test Split:")
    print(f"   - Training samples: {X_train.shape[0]:,}")
    print(f"   - Test samples: {X_test.shape[0]:,}")
    print(f"   - Features: {X_train.shape[1]:,}")

    return X_train, X_test, y_train, y_test, vectorizer, label_encoder


def train_random_forest(X_train, X_test, y_train, y_test, label_encoder):
    """
    Train Random Forest Classifier

    Training Process:
    - Uses ensemble of decision trees (100 trees)
    - Each tree trained on bootstrap sample
    - Final prediction via majority voting
    - Accuracy calculated as: (Correct Predictions / Total Predictions) * 100
    """
    print("\n" + "="*80)
    print("STEP 2: TRAINING RANDOM FOREST MODEL")
    print("="*80)

    print("\n🌲 Random Forest Configuration:")
    print("   - Algorithm: Random Forest Classifier")
    print("   - Number of trees: 100")
    print("   - Max depth: 20")
    print("   - Min samples split: 5")
    print("   - Bootstrap: True (sampling with replacement)")
    print("   - Criterion: Gini impurity")

    # Initialize model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("\n⏳ Training Random Forest...")
    rf_model.fit(X_train.toarray(), y_train)
    print("✅ Training completed!")

    # Predictions
    y_train_pred = rf_model.predict(X_train.toarray())
    y_test_pred = rf_model.predict(X_test.toarray())

    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted')
    }

    print("\n📊 Random Forest Results:")
    print(f"   Training Accuracy:   {metrics['train_accuracy']*100:.2f}%")
    print(f"   Test Accuracy:       {metrics['test_accuracy']*100:.2f}%")
    print(f"   Test F1 Score:       {metrics['test_f1']*100:.2f}%")
    print(f"   Test Precision:      {metrics['test_precision']*100:.2f}%")
    print(f"   Test Recall:         {metrics['test_recall']*100:.2f}%")

    print("\n📈 How Accuracy is Calculated:")
    print("   Accuracy = (Correct Predictions / Total Predictions) × 100")
    print(f"   Accuracy = ({int(metrics['test_accuracy']*len(y_test))}/{len(y_test)}) × 100")
    print(f"   Accuracy = {metrics['test_accuracy']*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n📊 Confusion Matrix:")
    print(cm)

    # Classification Report
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    # Log to Azure ML / MLflow
    if AZURE_ML:
        for metric_name, value in metrics.items():
            run.log(f"rf_{metric_name}", float(value))
        run.log("rf_confusion_matrix", cm.tolist())

    mlflow.log_params({
        "rf_n_estimators": 100,
        "rf_max_depth": 20,
        "rf_algorithm": "Random Forest"
    })
    mlflow.log_metrics({f"rf_{k}": v for k, v in metrics.items()})

    return rf_model, metrics


def train_xgboost(X_train, X_test, y_train, y_test, label_encoder):
    """
    Train XGBoost Classifier

    Training Process:
    - Uses gradient boosting framework
    - Builds trees sequentially, each correcting errors of previous
    - Regularization prevents overfitting
    - Accuracy calculated as: (Correct Predictions / Total Predictions) * 100
    """
    print("\n" + "="*80)
    print("STEP 3: TRAINING XGBOOST MODEL")
    print("="*80)

    print("\n🚀 XGBoost Configuration:")
    print("   - Algorithm: XGBoost Gradient Boosting")
    print("   - Number of boosting rounds: 100")
    print("   - Max depth: 6")
    print("   - Learning rate: 0.1")
    print("   - Objective: multi:softprob (multiclass probability)")
    print("   - Boosting type: gbtree (tree-based)")

    # Initialize model
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )

    print("\n⏳ Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    print("✅ Training completed!")

    # Predictions
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)

    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted')
    }

    print("\n📊 XGBoost Results:")
    print(f"   Training Accuracy:   {metrics['train_accuracy']*100:.2f}%")
    print(f"   Test Accuracy:       {metrics['test_accuracy']*100:.2f}%")
    print(f"   Test F1 Score:       {metrics['test_f1']*100:.2f}%")
    print(f"   Test Precision:      {metrics['test_precision']*100:.2f}%")
    print(f"   Test Recall:         {metrics['test_recall']*100:.2f}%")

    print("\n📈 How Accuracy is Calculated:")
    print("   Accuracy = (Correct Predictions / Total Predictions) × 100")
    print(f"   Accuracy = ({int(metrics['test_accuracy']*len(y_test))}/{len(y_test)}) × 100")
    print(f"   Accuracy = {metrics['test_accuracy']*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n📊 Confusion Matrix:")
    print(cm)

    # Classification Report
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    # Log to Azure ML / MLflow
    if AZURE_ML:
        for metric_name, value in metrics.items():
            run.log(f"xgb_{metric_name}", float(value))
        run.log("xgb_confusion_matrix", cm.tolist())

    mlflow.log_params({
        "xgb_n_estimators": 100,
        "xgb_max_depth": 6,
        "xgb_learning_rate": 0.1,
        "xgb_algorithm": "XGBoost"
    })
    mlflow.log_metrics({f"xgb_{k}": v for k, v in metrics.items()})

    return xgb_model, metrics


def save_models_and_artifacts(rf_model, xgb_model, vectorizer, label_encoder, output_dir):
    """Save trained models and preprocessing artifacts"""
    print("\n" + "="*80)
    print("STEP 4: SAVING MODELS AND ARTIFACTS")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save models
    print("\n💾 Saving models...")
    joblib.dump(rf_model, output_path / "random_forest_model.pkl")
    joblib.dump(xgb_model, output_path / "xgboost_model.pkl")
    joblib.dump(vectorizer, output_path / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, output_path / "label_encoder.pkl")

    print(f"✅ Models saved to: {output_path}")
    print(f"   - random_forest_model.pkl")
    print(f"   - xgboost_model.pkl")
    print(f"   - tfidf_vectorizer.pkl")
    print(f"   - label_encoder.pkl")

    # Log models to Azure ML / MLflow
    mlflow.sklearn.log_model(rf_model, "random_forest")
    mlflow.xgboost.log_model(xgb_model, "xgboost")

    if AZURE_ML:
        run.upload_file("random_forest_model.pkl", str(output_path / "random_forest_model.pkl"))
        run.upload_file("xgboost_model.pkl", str(output_path / "xgboost_model.pkl"))


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Random Forest and XGBoost models on Azure ML')
    parser.add_argument('--data_path', type=str, default='data/cleaned_support_tickets - with context.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model outputs')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("AZURE ML TRAINING JOB - RANDOM FOREST & XGBOOST")
    print("="*80)
    print(f"\n📁 Data path: {args.data_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Azure ML context: {AZURE_ML}")

    # Start MLflow run
    mlflow.start_run()

    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, vectorizer, label_encoder = load_and_preprocess_data(args.data_path)

        # Log dataset info to Azure ML
        if AZURE_ML:
            run.log("dataset_name", "cleaned_support_tickets - with context.csv")
            run.log("total_samples", X_train.shape[0] + X_test.shape[0])
            run.log("train_samples", X_train.shape[0])
            run.log("test_samples", X_test.shape[0])
            run.log("num_features", X_train.shape[1])
            run.log("num_classes", len(label_encoder.classes_))

        mlflow.log_params({
            "dataset": "cleaned_support_tickets - with context.csv",
            "total_samples": X_train.shape[0] + X_test.shape[0],
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "num_features": X_train.shape[1]
        })

        # Train Random Forest
        rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, label_encoder)

        # Train XGBoost
        xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test, label_encoder)

        # Model Comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"\n{'Metric':<25} {'Random Forest':<20} {'XGBoost':<20}")
        print("-" * 65)
        print(f"{'Test Accuracy':<25} {rf_metrics['test_accuracy']*100:>18.2f}% {xgb_metrics['test_accuracy']*100:>18.2f}%")
        print(f"{'Test F1 Score':<25} {rf_metrics['test_f1']*100:>18.2f}% {xgb_metrics['test_f1']*100:>18.2f}%")
        print(f"{'Test Precision':<25} {rf_metrics['test_precision']*100:>18.2f}% {xgb_metrics['test_precision']*100:>18.2f}%")
        print(f"{'Test Recall':<25} {rf_metrics['test_recall']*100:>18.2f}% {xgb_metrics['test_recall']*100:>18.2f}%")

        # Determine best model
        best_model = "Random Forest" if rf_metrics['test_accuracy'] > xgb_metrics['test_accuracy'] else "XGBoost"
        print(f"\n🏆 Best Model: {best_model}")

        if AZURE_ML:
            run.log("best_model", best_model)
        mlflow.log_param("best_model", best_model)

        # Save models
        save_models_and_artifacts(rf_model, xgb_model, vectorizer, label_encoder, args.output_dir)

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
