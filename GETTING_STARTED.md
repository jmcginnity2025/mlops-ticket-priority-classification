# Getting Started - MLOps Pipeline for CW2

## What We've Built

A complete MLOps CI/CD pipeline that:
1. **Preprocesses data** from your support tickets dataset
2. **Trains two model iterations** (baseline and improved)
3. **Compares performance** with regression testing
4. **Fails the pipeline** if new model is worse than previous
5. **Versions models** automatically on successful runs

## Project Structure

```
CW2_workflow/
├── preprocess.py                      # Data preprocessing
├── train.py                           # Model training (2 iterations)
├── evaluate.py                        # Evaluation & regression testing
├── .github/workflows/
│   └── ml-cicd-pipeline.yml          # CI/CD pipeline
├── processed_data/                    # Generated: preprocessed data
├── models/                            # Generated: trained models
│   ├── iteration_1/
│   │   ├── model.pkl
│   │   └── metrics.json
│   └── iteration_2/
│       ├── model.pkl
│       └── metrics.json
└── evaluation_results/                # Generated: evaluation reports
    └── evaluation_report.json
```

## How It Works

### Pipeline Flow
```
Git Commit to main
    ↓
[1] Preprocess Data
    ↓
[2] Train Iteration 1 (Random Forest - Baseline)
    ↓
[3] Train Iteration 2 (XGBoost - Improved)
    ↓
[4] Compare Models (Regression Test)
    ↓
    ├─→ New model BETTER or SIMILAR → PASS → Version & Tag
    └─→ New model WORSE → FAIL → Stop Pipeline
```

### Regression Testing Logic

**First Commit:**
- No baseline exists
- Checks absolute thresholds:
  - Accuracy ≥ 70%
  - F1 Score ≥ 65%

**Subsequent Commits:**
- Compares Iteration 2 vs Iteration 1
- Allows up to 2% performance drop
- Fails if performance drops more than 2%

## Testing Locally

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### Step 2: Run Preprocessing
```bash
python preprocess.py
```
This creates `processed_data/` folder with train/test splits.

### Step 3: Train Models
```bash
# Train both iterations
python train.py

# Or train specific iteration
python train.py --iteration 1
python train.py --iteration 2
```
This creates `models/` folder with trained models and metrics.

### Step 4: Evaluate Models
```bash
python evaluate.py
```
- Exit code 0 = PASSED (models meet requirements)
- Exit code 1 = FAILED (regression detected)

## Expected Output

### Preprocessing
```
======================================================================
PREPROCESSING PIPELINE - Support Ticket Priority Classification
======================================================================
Loading dataset...
Loaded 50000 rows, 70 columns

Selected 24 features:
  - Numeric: 12
  - Categorical: 9
  - Binary: 3

Cleaning data...
After cleaning: 49850 rows

Splitting data (80/20)...
Train set: 39880 samples
Test set: 9970 samples

All outputs saved to: processed_data/
======================================================================
```

### Training
```
======================================================================
TRAINING ITERATION 1: Baseline Random Forest
======================================================================
Training model...

Iteration 1 Results:
  Train Accuracy: 0.9856
  Test Accuracy:  0.8234
  Test F1 Score:  0.8187

Model saved to: models/iteration_1/

======================================================================
TRAINING ITERATION 2: Improved XGBoost
======================================================================
Training model...

Iteration 2 Results:
  Train Accuracy: 0.9912
  Test Accuracy:  0.8456
  Test F1 Score:  0.8423

Model saved to: models/iteration_2/
======================================================================
```

### Evaluation (Passing)
```
======================================================================
MODEL COMPARISON
======================================================================

Baseline Model (Iteration 1):
  Test Accuracy: 0.8234
  Test F1 Score: 0.8187

Current Model (Iteration 2):
  Test Accuracy: 0.8456
  Test F1 Score: 0.8423

Difference:
  Accuracy:  +0.0222 (+2.22%)
  F1 Score:  +0.0236 (+2.36%)

======================================================================
EVALUATION RESULT: PASSED
======================================================================

The model meets performance requirements.
```

### Evaluation (Failing - Regression Detected)
```
======================================================================
MODEL COMPARISON
======================================================================

Baseline Model (Iteration 1):
  Test Accuracy: 0.8456
  Test F1 Score: 0.8423

Current Model (Iteration 2):
  Test Accuracy: 0.7988
  Test F1 Score: 0.7945

Difference:
  Accuracy:  -0.0468 (-4.68%)
  F1 Score:  -0.0478 (-4.78%)

======================================================================
EVALUATION RESULT: FAILED
======================================================================

REGRESSION DETECTED!

Failure reasons:
  - Accuracy dropped by 4.68% (threshold: 2.00%)
  - F1 Score dropped by 4.78% (threshold: 2.00%)

The pipeline will be stopped.
```

## Committing to GitHub

Once you've tested locally and everything works:

```bash
# Add new files
git add preprocess.py train.py evaluate.py
git add .github/workflows/ml-cicd-pipeline.yml

# Commit (this will trigger the pipeline)
git commit -m "Add ML pipeline with regression testing"

# Push to main branch
git push origin main
```

The GitHub Actions pipeline will automatically:
1. Run preprocessing
2. Train both model iterations
3. Evaluate and compare
4. Version models if tests pass
5. Fail pipeline if regression detected

## Next Steps

After this basic pipeline works:
1. Integrate with Azure ML for remote training
2. Add MLflow for experiment tracking
3. Add data versioning
4. Add deployment to Azure endpoints
5. Add monitoring and drift detection

## Troubleshooting

**Problem:** `FileNotFoundError` for dataset
- **Fix:** Update `DATA_PATH` in `preprocess.py` to your dataset location

**Problem:** Import errors
- **Fix:** Run `pip install -r requirements.txt`

**Problem:** Pipeline fails on GitHub but works locally
- **Fix:** Check GitHub Actions logs, ensure data path is correct

## Questions?

This is a starting point! As you add more features, you can extend:
- The preprocessing (more feature engineering)
- The models (try different algorithms)
- The evaluation (add more metrics)
- The pipeline (add more stages)
