# Model Training Documentation

## Overview
This document explains how the support ticket priority classification models are trained and tested using Azure Machine Learning.

## Dataset
- **Name**: cleaned_support_tickets - with context.csv
- **Total Records**: 48,388
- **Features**: Text-based support ticket descriptions
- **Target**: Priority classification (Low, Medium, High, Critical)

## Models

### 1. Random Forest Classifier

#### Training Process
The Random Forest model uses an ensemble learning approach:

1. **Ensemble Construction**: Creates 100 decision trees
2. **Bootstrap Sampling**: Each tree is trained on a random subset of the data (with replacement)
3. **Feature Randomness**: At each split, only a random subset of features is considered
4. **Prediction**: Final prediction is made by majority voting across all trees

#### Hyperparameters
```python
n_estimators=100      # Number of trees in the forest
max_depth=20          # Maximum depth of each tree
min_samples_split=5   # Minimum samples required to split a node
random_state=42       # For reproducibility
n_jobs=-1            # Use all CPU cores
```

#### How It Works
- **Training**: Each tree learns to classify tickets based on TF-IDF features extracted from text
- **Prediction**: When a new ticket arrives, all 100 trees vote on the priority class
- **Robustness**: Random sampling and feature selection reduce overfitting

### 2. XGBoost Classifier

#### Training Process
XGBoost uses gradient boosting, a sequential ensemble method:

1. **Sequential Learning**: Trees are built one after another
2. **Error Correction**: Each new tree focuses on correcting mistakes made by previous trees
3. **Gradient Descent**: Uses gradients to minimize classification errors
4. **Regularization**: Includes L1 and L2 penalties to prevent overfitting

#### Hyperparameters
```python
n_estimators=100      # Number of boosting rounds
max_depth=6           # Maximum depth of each tree
learning_rate=0.1     # Step size for weight updates
objective='multi:softprob'  # Multi-class classification with probabilities
random_state=42       # For reproducibility
```

#### How It Works
- **Training**: Builds trees sequentially, each improving on the previous ensemble
- **Prediction**: Combines predictions from all trees with learned weights
- **Speed**: Optimized implementation makes it faster than standard gradient boosting

## Feature Engineering

### TF-IDF Vectorization
Text is converted to numerical features using TF-IDF (Term Frequency-Inverse Document Frequency):

```python
max_features=5000     # Top 5000 most important words
min_df=2              # Word must appear in at least 2 documents
max_df=0.95          # Ignore words appearing in >95% of documents
ngram_range=(1, 2)   # Use single words and word pairs
stop_words='english'  # Remove common words (the, is, at, etc.)
```

**Why TF-IDF?**
- **TF (Term Frequency)**: Measures how often a word appears in a document
- **IDF (Inverse Document Frequency)**: Reduces weight of common words
- **Result**: Important words (like "critical", "urgent") get higher weights

## Train-Test Split

```python
train_size = 80%  (38,710 records)
test_size = 20%   (9,678 records)
stratify = True   (maintains class distribution)
random_state = 42 (for reproducibility)
```

**Why Stratify?**
Ensures both training and test sets have the same proportion of Low, Medium, High, and Critical tickets.

## Accuracy Calculation

### Formula
```
Accuracy = (Number of Correct Predictions / Total Predictions) × 100
```

### Example
If the model predicts 9,200 out of 9,678 test tickets correctly:
```
Accuracy = (9,200 / 9,678) × 100 = 95.06%
```

### Step-by-Step Process
1. **Training Phase**: Model learns patterns from 38,710 training tickets
2. **Prediction Phase**: Model predicts priority for 9,678 unseen test tickets
3. **Comparison**: Compare predictions against actual priorities
4. **Count**: Count how many predictions match the actual values
5. **Calculate**: Divide correct predictions by total and multiply by 100

## Evaluation Metrics

### 1. Accuracy
- **Definition**: Overall percentage of correct predictions
- **Formula**: (TP + TN) / Total Predictions
- **Use**: Good for balanced datasets

### 2. F1 Score (Weighted)
- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use**: Better for imbalanced classes (weighted accounts for class distribution)

### 3. Precision (Weighted)
- **Definition**: Of all tickets predicted as "High", how many were actually "High"?
- **Formula**: True Positives / (True Positives + False Positives)
- **Use**: Measures how accurate positive predictions are

### 4. Recall (Weighted)
- **Definition**: Of all actual "High" tickets, how many did we correctly identify?
- **Formula**: True Positives / (True Positives + False Negatives)
- **Use**: Measures how well we find all instances of a class

### Why Weighted?
Since we have 4 classes (Low, Medium, High, Critical), we calculate metrics for each class and then average them weighted by the number of samples in each class.

## Model Comparison

Both models are trained on the same data and evaluated using the same metrics. This allows direct comparison:

```
Metric              Random Forest    XGBoost
-------------------------------------------------
Test Accuracy       XX.XX%          XX.XX%
Test F1 Score       XX.XX%          XX.XX%
Test Precision      XX.XX%          XX.XX%
Test Recall         XX.XX%          XX.XX%
```

## Azure ML Integration

### Metrics Logged to Azure ML Studio
Both models log the following to Azure ML:
- Training accuracy, F1, precision, recall
- Test accuracy, F1, precision, recall
- Confusion matrix (shows which classes are confused)
- Classification report (detailed per-class metrics)
- Feature importance (which words are most predictive)

### How to View Results
1. Go to Azure ML Studio
2. Navigate to Experiments → ticket-priority-classification
3. Click on the latest run
4. View metrics in the "Metrics" tab
5. Download artifacts (models, confusion matrices) from "Outputs + logs"

## Training Script

### Main Script: train_azure_ml.py

**Purpose**: Train both Random Forest and XGBoost models with comprehensive Azure ML logging

**Usage**:
```bash
python train_azure_ml.py
```

**Output**:
- Console logs showing training progress
- Azure ML metrics visible in Studio
- Saved model artifacts (.pkl files)
- Confusion matrices and classification reports

## Regression Testing (2% Threshold)

After training, the pipeline compares new model performance against baseline:

```python
improvement = current_accuracy - baseline_accuracy
improvement_percentage = improvement × 100

if improvement_percentage < 2%:
    # Pipeline fails - model not good enough
    exit(1)
else:
    # Pipeline continues - model approved
    continue()
```

**Why 2%?**
This ensures only models that show meaningful improvement are deployed. Small fluctuations (<2%) could be due to random variation rather than actual improvement.

## Summary

1. **Data Preparation**: 48,388 support tickets converted to TF-IDF features
2. **Model Training**: Both Random Forest and XGBoost trained with cross-validation
3. **Evaluation**: Comprehensive metrics calculated on unseen test data
4. **Comparison**: Models compared to identify best performer
5. **Azure ML Logging**: All metrics and artifacts logged for visualization
6. **Quality Gate**: 2% improvement threshold ensures only better models are deployed

## Next Steps

After training completes:
1. Review metrics in Azure ML Studio
2. Compare Random Forest vs XGBoost performance
3. Select best model for deployment
4. Submit to regression testing pipeline
5. If approved (>2% improvement), deploy to staging
6. Load test with Locust (100 concurrent users)
7. Deploy to production endpoint
