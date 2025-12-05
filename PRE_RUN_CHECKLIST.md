# Pre-Run Checklist ✅

Complete this checklist before running the MLOps pipeline.

## Environment Setup

- [ ] **Python 3.9+** installed
  ```bash
  python --version
  ```

- [ ] **Virtual environment** created and activated
  ```bash
  python -m venv venv
  # Windows: venv\Scripts\activate
  # Mac/Linux: source venv/bin/activate
  ```

- [ ] **Dependencies** installed
  ```bash
  pip install -r requirements.txt
  ```

## Azure Setup

- [ ] **Azure account** with active subscription

- [ ] **Azure CLI** installed and logged in
  ```bash
  az --version
  az login
  ```

- [ ] **Resource group** created (or will be created)
  ```bash
  az group create -n mlops-cw2-rg -l eastus
  ```

- [ ] **Azure ML workspace** created (or will be created)
  ```bash
  az ml workspace create -w ticket-priority-workspace -g mlops-cw2-rg
  ```

## Configuration

- [ ] **config/config.yaml** updated with:
  - [ ] Your Azure subscription ID
  - [ ] Correct resource group name
  - [ ] Correct workspace name
  - [ ] Correct data file path

  **Open config/config.yaml and replace:**
  ```yaml
  azure:
    subscription_id: "YOUR_SUBSCRIPTION_ID_HERE"  # ← CHANGE THIS!
  ```

## Data

- [ ] **Dataset file exists** at the path specified in config.yaml

  Default path: `data/cleaned_support_tickets - with context.csv`

  Check if file exists:
  ```bash
  # Windows
  dir "data\cleaned_support_tickets - with context.csv"

  # Mac/Linux
  ls -la "data/cleaned_support_tickets - with context.csv"
  ```

- [ ] **Data has required columns** (as per config.yaml):
  - Numeric features (org_users, past_30d_tickets, etc.)
  - Categorical features (day_of_week_num, company_size_cat, etc.)
  - Binary features (payment_impact_flag, data_loss_flag, has_runbook)
  - Target column (priority_cat or priority_score_internal)

## Directory Structure

- [ ] **All directories exist**:
  ```bash
  # These will be created automatically, but you can create them now:
  mkdir -p data/processed
  mkdir -p models
  mkdir -p evaluation_results
  mkdir -p monitoring
  mkdir -p mlruns
  ```

## Testing (Optional but Recommended)

- [ ] **Run unit tests** to verify setup
  ```bash
  pytest tests/test_preprocessing.py -v
  ```

- [ ] **Verify configuration** is valid
  ```bash
  python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
  ```

## Ready to Run!

Once all items are checked, you can run:

### Option 1: Full Pipeline (Recommended for First Run)
```bash
python run_pipeline.py --mode full --model_type multiclass --iterations 2
```

### Option 2: Step by Step

```bash
# Step 1: Preprocess
python run_pipeline.py --mode preprocess

# Step 2: Train
python run_pipeline.py --mode train --model_type multiclass --iterations 2

# Step 3: Evaluate
python run_pipeline.py --mode evaluate --model_type multiclass
```

## After Running

### Verify Outputs

- [ ] **Processed data** created in `data/processed/YYYYMMDD_HHMMSS/`
- [ ] **Models** saved in `models/multiclass_iteration_*/`
- [ ] **MLflow runs** visible:
  ```bash
  mlflow ui
  # Open http://localhost:5000
  ```
- [ ] **Evaluation results** in `evaluation_results/multiclass/`

### Next Steps

- [ ] Review MLflow experiments
- [ ] Check evaluation metrics
- [ ] Deploy to Azure (optional)
- [ ] Setup CI/CD pipeline
- [ ] Generate monitoring reports

## Troubleshooting

### If you see "Data file not found"
1. Check path in `config/config.yaml`
2. Verify file exists at that location
3. Use absolute path if needed

### If you see "Azure authentication failed"
```bash
az login
az account show
az account set --subscription YOUR_SUBSCRIPTION_ID
```

### If you see "Module not found"
```bash
pip install -r requirements.txt
```

### If tests fail
- Check Python version (3.9+)
- Ensure all dependencies installed
- Review error messages in detail

## Minimum Requirements to Run Locally (Without Azure)

If you just want to test locally without Azure deployment:

- [x] Python 3.9+
- [x] Dependencies installed
- [x] Data file exists
- [x] config.yaml path updated

You can run everything except deployment:
```bash
python run_pipeline.py --mode full --model_type multiclass --iterations 2
```

Skip deployment steps - the pipeline will work for:
- ✅ Data preprocessing
- ✅ Model training
- ✅ Evaluation
- ✅ MLflow tracking
- ⏭️ Deployment (requires Azure)
- ⏭️ Monitoring (works but deployment needed for full cycle)

## CI/CD Setup (Optional)

### For Azure DevOps
- [ ] Azure DevOps account
- [ ] Service connection to Azure
- [ ] Pipeline created from `azure-pipelines.yml`

### For GitHub Actions
- [ ] GitHub repository
- [ ] Azure credentials added to secrets
- [ ] Workflow enabled

## Final Check

Run this command to verify your setup:
```bash
python -c "
import yaml
import pandas as pd
from pathlib import Path

# Check config
config = yaml.safe_load(open('config/config.yaml'))
print('✓ Config loaded')

# Check data path
data_path = config['data']['raw_data_path']
if Path(data_path).exists():
    df = pd.read_csv(data_path)
    print(f'✓ Data loaded: {len(df)} rows, {len(df.columns)} columns')
else:
    print(f'✗ Data not found at: {data_path}')

print('✓ Setup verification complete!')
"
```

---

**Once all items are checked, you're ready to run the MLOps pipeline! 🚀**

See [QUICKSTART.md](QUICKSTART.md) for quick commands.
See [README.md](README.md) for detailed documentation.
