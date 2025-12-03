# Quick Start Guide

Get your MLOps pipeline running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] Azure account with active subscription
- [ ] Git installed
- [ ] Dataset file available

## Step 1: Setup (2 minutes)

```bash
# Clone and navigate to project
cd CW2_workflow

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure (1 minute)

Edit [config/config.yaml](config/config.yaml):

```yaml
azure:
  subscription_id: "YOUR_AZURE_SUBSCRIPTION_ID"  # Replace this!
  resource_group: "mlops-cw2-rg"
  workspace_name: "ticket-priority-workspace"

data:
  raw_data_path: "data/cleaned_support_tickets - with context.csv"  # Verify path!
```

## Step 3: Run Pipeline (2 minutes)

```bash
# Run complete pipeline (preprocessing + training + evaluation)
python run_pipeline.py --mode full --model_type multiclass --iterations 2
```

That's it! 🎉

## What Just Happened?

1. ✅ **Data Preprocessing**: Your data was cleaned, encoded, and split
2. ✅ **Model Training**: Trained 2 XGBoost models with different hyperparameters
3. ✅ **Experiment Tracking**: All metrics logged to MLflow
4. ✅ **Model Evaluation**: Performance compared against thresholds
5. ✅ **Regression Testing**: Checked for performance degradation

## View Results

### MLflow Experiments
```bash
mlflow ui
```
Open http://localhost:5000

### Evaluation Results
Check [evaluation_results/multiclass/](evaluation_results/multiclass/):
- `iteration_comparison.json` - Compare all iterations
- `evaluation_summary.json` - Best model summary
- `regression_test_report.json` - Performance checks

### Trained Models
Find models in [models/](models/):
- `multiclass_iteration_1/model.pkl`
- `multiclass_iteration_2/model.pkl`

## Next Steps

### Deploy to Azure (Optional)

First, create Azure ML workspace:
```bash
az login
az ml workspace create -w ticket-priority-workspace -g mlops-cw2-rg
```

Then deploy:
```bash
# Register model
python src/deploy_model.py --action register \
  --model_path models/multiclass_iteration_2/model.pkl \
  --model_name ticket-priority-classifier

# Deploy to Azure
python src/deploy_model.py --action deploy \
  --model_name ticket-priority-classifier
```

### Setup Monitoring

```bash
# Generate monitoring dashboard data
python src/monitor_model.py --action dashboard
```

### Setup CI/CD

#### Azure DevOps
1. Create new pipeline
2. Select `azure-pipelines.yml`
3. Configure Azure service connection
4. Run pipeline

#### GitHub Actions
1. Add Azure credentials to repository secrets
2. Push to main branch
3. Pipeline runs automatically

## Common Commands

```bash
# Preprocess data only
python run_pipeline.py --mode preprocess

# Train only
python run_pipeline.py --mode train --model_type multiclass --iterations 3

# Evaluate only
python run_pipeline.py --mode evaluate --model_type multiclass

# Run tests
pytest tests/

# Check data drift
python src/monitor_model.py --action drift --current_data data/new_data.csv
```

## Troubleshooting

### "Data file not found"
- Check path in `config/config.yaml`
- Ensure file exists at specified location

### "Azure authentication failed"
```bash
az login
az account set --subscription YOUR_SUBSCRIPTION_ID
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Need help?
- Check full [README.md](README.md)
- Review configuration in [config/config.yaml](config/config.yaml)
- Examine logs in terminal output

## Project Structure Quick Reference

```
CW2_workflow/
├── config/config.yaml          # ← Edit your settings here
├── run_pipeline.py             # ← Main entry point
├── src/                        # Core scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── deploy_model.py
│   └── monitor_model.py
├── models/                     # Trained models
├── evaluation_results/         # Evaluation reports
└── mlruns/                     # MLflow tracking
```

## Success Criteria for CW2

Your pipeline successfully demonstrates:

✅ **Model Development**
- Multiple training iterations
- Hyperparameter tuning
- Performance evaluation

✅ **CI/CD**
- Automated testing
- Build pipeline
- Multi-stage deployment

✅ **Deployment**
- Model registered in Azure ML
- REST API endpoint
- Deployment testing

✅ **Monitoring**
- Data drift detection
- Performance tracking
- Alert triggers

✅ **Retraining**
- Automated triggers
- Validation workflow
- Approval process

✅ **Governance**
- Model versioning
- Experiment tracking
- Audit trail

---

**Ready for more?** Check the full [README.md](README.md) for detailed documentation!
