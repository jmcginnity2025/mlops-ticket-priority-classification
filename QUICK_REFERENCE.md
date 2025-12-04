# Quick Reference - MLOps Pipeline

## 🚀 Quick Start

### Local Testing (3 commands)
```bash
python preprocess.py    # Step 1: Prepare data
python train.py         # Step 2: Train both models
python evaluate.py      # Step 3: Compare & test (exit 0=pass, 1=fail)
```

### Commit to GitHub (Triggers Pipeline)
```bash
git add preprocess.py train.py evaluate.py .github/workflows/ml-cicd-pipeline.yml
git commit -m "Initial ML pipeline with regression testing"
git push origin main
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| [preprocess.py](preprocess.py) | Cleans data, creates train/test splits |
| [train.py](train.py) | Trains 2 models: RF (baseline), XGBoost (improved) |
| [evaluate.py](evaluate.py) | Compares models, FAILS if regression |
| [.github/workflows/ml-cicd-pipeline.yml](.github/workflows/ml-cicd-pipeline.yml) | CI/CD automation |

## 📊 Pipeline Flow

```
Commit to main
    ↓
Preprocess Data (processed_data/)
    ↓
Train Iteration 1: Random Forest (models/iteration_1/)
    ↓
Train Iteration 2: XGBoost (models/iteration_2/)
    ↓
Compare Models (evaluation_results/)
    ↓
    ├─→ Better/Similar (≤2% drop) → ✅ PASS → Version
    └─→ Worse (>2% drop) → ❌ FAIL → Stop
```

## 🎯 Current Results

```
Dataset: 48,837 support tickets (3 priority classes)
Features: 24 selected features

Model 1 (Random Forest):
  Accuracy: 86.84%
  F1 Score: 86.60%

Model 2 (XGBoost):
  Accuracy: 90.97% ⬆️ +4.13%
  F1 Score: 90.88% ⬆️ +4.28%

Result: ✅ PASSED (improvement detected)
```

## ⚙️ Configuration

### Regression Thresholds
- **Allowed drop**: 2% (in evaluate.py)
- **Min accuracy**: 70% (first commit only)
- **Min F1**: 65% (first commit only)

### Model Parameters

**Iteration 1 (Random Forest):**
```python
n_estimators=100
max_depth=10
```

**Iteration 2 (XGBoost):**
```python
n_estimators=200
max_depth=6
learning_rate=0.1
subsample=0.8
```

## 🔍 Check Results

```bash
# View evaluation report
cat evaluation_results/evaluation_report.json

# View model metrics
cat models/iteration_1/metrics.json
cat models/iteration_2/metrics.json
```

## ✅ What Works Now

- [x] Local preprocessing
- [x] Local training (2 iterations)
- [x] Local evaluation with regression testing
- [x] GitHub Actions workflow created
- [x] Pass/fail logic implemented

## 🚧 Next Steps

1. **Commit to GitHub** - Test the CI/CD pipeline
2. **Azure ML Setup** - Create workspace and compute
3. **Cloud Training** - Run training on Azure
4. **Deployment** - Deploy model as REST API
5. **Monitoring** - Add drift detection

## 📚 Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Detailed setup and examples
- [PROJECT_SUMMARY_CW2.md](PROJECT_SUMMARY_CW2.md) - Complete project overview
- This file - Quick reference

## 💡 Tips

1. **Test locally first** - Always run the 3 commands before committing
2. **Check exit codes** - `echo $?` after evaluate.py (0=pass, 1=fail)
3. **Watch GitHub Actions** - Go to Actions tab after pushing
4. **Iteration 2 is what counts** - Iter1 is just the baseline for comparison

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: dataset` | Update `DATA_PATH` in preprocess.py |
| `pip install errors` | Run `pip install -r requirements.txt` |
| `XGBoost class error` | Already fixed - labels remapped 1,2,3→0,1,2 |
| `evaluate.py fails` | Check models/ folder has both iterations |

## 📞 Need Help?

1. Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. Check logs: `python script.py 2>&1 | tee log.txt`
3. Check GitHub Actions logs (if pipeline fails)

---

**You're ready to commit and test the CI/CD pipeline!** 🎉
