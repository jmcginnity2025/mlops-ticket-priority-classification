# Push to GitHub - Quick Guide

## ✅ Your Fresh Repository is Ready!

**Location**: `C:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2\mlops-cw2-fresh`

**What's included**:
- ✅ All ML scripts (preprocess, train, evaluate)
- ✅ Azure ML scripts (train_azure, submit_training_job)
- ✅ Two GitHub workflows (local + Azure ML)
- ✅ Complete documentation
- ✅ Git initialized with initial commit

---

## 📝 Step-by-Step Instructions

### Step 1: Create GitHub Repository

1. Go to: https://github.com/new
2. Fill in:
   - **Repository name**: `mlops-cw2-fresh` (or your choice)
   - **Description**: MLOps pipeline for support ticket classification - CW2
   - **Visibility**: Public or Private (your choice)
   - ❌ **DON'T** check "Add README" (we already have one!)
   - ❌ **DON'T** add .gitignore (we have one!)
3. Click "Create repository"

### Step 2: Connect and Push

Open a terminal and run these commands:

```bash
# Navigate to the repo
cd "C:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2\mlops-cw2-fresh"

# Add your GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/mlops-cw2-fresh.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

**Replace `YOUR_USERNAME`** with your actual GitHub username!

---

## 🎯 What Happens Next

Once you push:

1. **Code uploads to GitHub** ✅
2. **GitHub Actions triggers automatically** 🚀
3. **Pipeline starts running** (ml-cicd-pipeline.yml)
4. **You can watch it in Actions tab** 👀

---

## 📊 Watch Your Pipeline

After pushing, go to:
```
https://github.com/YOUR_USERNAME/mlops-cw2-fresh/actions
```

You'll see:
- ✅ ML CI/CD Pipeline running
- ✅ Steps: Preprocess → Train → Evaluate
- ✅ Pass/Fail results

---

## 🔑 Optional: Setup Azure ML Pipeline

If you want the Azure ML pipeline to work (submits to Azure):

1. **Create Service Principal**:
   ```bash
   az ad sp create-for-rbac \
     --name "github-actions-mlops" \
     --role contributor \
     --scopes /subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg \
     --sdk-auth
   ```
   Copy the entire JSON output!

2. **Add GitHub Secrets**:
   - Go to: Repo → Settings → Secrets and variables → Actions
   - Add secret: `AZURE_CREDENTIALS` = paste JSON
   - Add secret: `AZURE_SUBSCRIPTION_ID` = `d5156f99-abd5-4af9-9e2d-a875ef22df46`

3. **Push again to trigger**:
   ```bash
   git commit --allow-empty -m "Trigger Azure ML pipeline"
   git push
   ```

---

## 🚨 Important Notes

### Dataset Path Issue

⚠️ The dataset path is hardcoded in `preprocess.py`:
```python
DATA_PATH = r"C:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2 New\cleaned_support_tickets - with context.csv"
```

**This won't work in GitHub Actions!**

### Two Options:

#### Option 1: Commit Dataset (Simple)
```bash
# Copy dataset to repo
cp "C:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2 New\cleaned_support_tickets - with context.csv" data/

# Update preprocess.py line 17 to:
DATA_PATH = "data/cleaned_support_tickets - with context.csv"

# Commit
git add .
git commit -m "Add dataset"
git push
```

#### Option 2: Use Azure ML Only (Recommended)
- Don't use local pipeline (it needs the dataset file)
- Only use Azure ML pipeline (dataset already in Azure!)
- See [GITHUB_SETUP.md](GITHUB_SETUP.md)

---

## ✅ Quick Test Commands

```bash
# Navigate to repo
cd "C:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2\mlops-cw2-fresh"

# Check git status
git status

# View commit
git log --oneline

# Add remote (replace YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/mlops-cw2-fresh.git

# Push
git push -u origin main
```

---

## 📚 What's in This Repo

```
mlops-cw2-fresh/
├── .github/workflows/
│   ├── ml-cicd-pipeline.yml      # Local training pipeline
│   └── azure-ml-pipeline.yml     # Azure ML pipeline
├── preprocess.py                 # Data preprocessing
├── train.py                      # Train 2 iterations
├── evaluate.py                   # Regression testing
├── train_azure.py                # Azure ML training
├── submit_training_job.py        # Submit to Azure
├── upload_dataset_azure.py       # Upload dataset
├── environment.yml               # Python environment
├── azure_config.json             # Azure configuration
├── requirements.txt              # Dependencies
├── README.md                     # Project overview
└── Documentation (7 files)       # Complete guides
```

---

## 🎓 For Your Coursework

**Take screenshots of**:
1. GitHub repo (showing code)
2. Actions tab (showing pipeline running)
3. Successful pipeline run
4. Azure ML Studio (showing training job)

**All requirements met**:
- ✅ CI/CD pipeline
- ✅ Automated testing
- ✅ Version control
- ✅ Cloud deployment ready
- ✅ MLflow tracking (Azure ML)

---

## 🆘 Troubleshooting

### Can't find dataset
→ See "Dataset Path Issue" above

### GitHub push fails
→ Make sure you replaced YOUR_USERNAME with your actual username

### Pipeline fails on GitHub
→ Check dataset is included OR use Azure ML pipeline instead

### Azure ML pipeline not triggering
→ Check GitHub secrets are set (see Optional section)

---

## 🚀 Ready to Push?

```bash
cd "C:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2\mlops-cw2-fresh"
git remote add origin https://github.com/YOUR_USERNAME/mlops-cw2-fresh.git
git push -u origin main
```

Then check: https://github.com/YOUR_USERNAME/mlops-cw2-fresh

**Good luck!** 🎉
