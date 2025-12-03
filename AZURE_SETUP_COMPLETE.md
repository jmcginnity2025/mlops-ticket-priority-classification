# Azure ML Setup Complete ✅

## What's Already Done

✅ **Azure Subscription**: d5156f99-abd5-4af9-9e2d-a875ef22df46 (Azure for Students)
✅ **Resource Group**: mlops-cw2-rg
✅ **ML Workspace**: ticket-priority-workspace (France Central)
✅ **Compute Cluster**: CPU-Cluster (Standard_DS3_v2, 0-1 nodes)
✅ **Dataset Uploaded**: support_tickets (version 1)
✅ **Config Updated**: config.yaml has your subscription ID

## Next Steps - Run in Azure ML Studio

### Step 1: Open Azure ML Studio
1. Go to: https://ml.azure.com
2. Sign in with: McGinnity-J1@ulster.ac.uk
3. Select workspace: **ticket-priority-workspace**

### Step 2: Upload Notebook
1. Click **"Notebooks"** in left menu
2. Click the **upload icon** (↑) at the top
3. Upload this file: **`azure_ml_pipeline_notebook.ipynb`**
   - Location: `c:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2\CW2_workflow\azure_ml_pipeline_notebook.ipynb`

### Step 3: Create Compute Instance (if you don't have one)
1. Click **"Compute"** in left menu
2. Click **"Compute instances"** tab
3. If you don't see a running instance, click **"+ New"**:
   - Name: `mlops-compute-instance`
   - VM size: `Standard_DS3_v2` (or `Standard_DS2_v2` for lower cost)
   - Click **"Create"**
   - Wait 3-5 minutes for it to start

### Step 4: Run the Notebook
1. Click on **`azure_ml_pipeline_notebook.ipynb`** in the Notebooks section
2. At the top, select your compute instance from the dropdown
3. Click **"Run all"** or run cells one by one

## What the Notebook Will Do

The notebook will execute the complete MLOps pipeline:

1. ✅ **Load Dataset** - Access your uploaded `support_tickets` data
2. ✅ **Data Preprocessing** - Clean, encode, scale features
3. ✅ **Train Model - Iteration 1** - XGBoost with hyperparameter tuning
4. ✅ **Train Model - Iteration 2** - Different hyperparameters
5. ✅ **MLflow Tracking** - All metrics logged automatically
6. ✅ **Model Evaluation** - Compare both iterations
7. ✅ **Regression Testing** - Check performance thresholds
8. ✅ **Model Registration** - Register best model in Azure ML

**Estimated Runtime**: 15-30 minutes (depending on compute and data size)

## After Running the Notebook

### View Experiments
1. Click **"Experiments"** in left menu
2. Click **"ticket-priority-classification"**
3. You'll see 2 runs (one for each iteration)
4. Click on each run to view:
   - Parameters (hyperparameters used)
   - Metrics (accuracy, F1 score, etc.)
   - Outputs (models, plots)

### View Registered Model
1. Click **"Models"** in left menu
2. You'll see **"ticket-priority-classifier"**
3. Click on it to see version, tags, and metrics

### View Datasets
1. Click **"Data"** in left menu
2. You'll see **"support_tickets"**
3. Click to explore the data

## Optional: Deploy Model

After training, you can deploy the model:

1. Go to **Models** → **ticket-priority-classifier**
2. Click **"Deploy"** → **"Deploy to web service"**
3. Configure:
   - Name: `ticket-priority-endpoint`
   - Compute type: **Azure Container Instance (ACI)**
   - CPU: 1 core
   - Memory: 2 GB
4. Click **"Deploy"**
5. Wait 5-10 minutes

## Files in This Project

**Main Notebook** (Upload this to Azure ML Studio):
- `azure_ml_pipeline_notebook.ipynb` ← **Use this in Azure ML Studio**

**Configuration**:
- `config/config.yaml` - All settings (already configured)

**Alternative - Local Scripts** (if you want to run locally):
- `src/data_preprocessing.py`
- `src/train_model.py`
- `src/evaluate_model.py`
- `src/deploy_model.py`
- `src/monitor_model.py`
- `run_pipeline.py` - Main orchestrator

**Documentation**:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `AZURE_ML_STUDIO_GUIDE.md` - Azure ML Studio guide

## Troubleshooting

### Dataset not found
- Ensure it's named exactly: `support_tickets`
- Check in Azure ML Studio → Data

### Compute not starting
- Check quota in Azure Portal
- Try smaller VM size: `Standard_DS2_v2`

### Notebook kernel crashes
- Reduce data size for testing
- Restart kernel: Kernel → Restart

### Import errors in notebook
Add this cell at the top and run:
```python
!pip install xgboost scikit-learn mlflow pandas numpy scipy
```

## Cost Management

**Free Tier**:
- 4 hours/month free compute
- Stop compute instance when not in use

**Stop Compute**:
1. Go to Compute → Compute instances
2. Select your instance
3. Click **"Stop"**

Or in notebook:
```python
from azureml.core import Workspace
from azureml.core.compute import ComputeInstance

ws = Workspace.from_config()
compute = ComputeInstance(workspace=ws, name='your-compute-name')
compute.stop()
```

## Summary of Resources

| Resource | Name | Status |
|----------|------|--------|
| Subscription | Azure for Students | ✅ Active |
| Resource Group | mlops-cw2-rg | ✅ Created |
| ML Workspace | ticket-priority-workspace | ✅ Created |
| Location | France Central | ✅ |
| Compute Cluster | CPU-Cluster | ✅ Created |
| Dataset | support_tickets | ✅ Uploaded |
| Notebook | azure_ml_pipeline_notebook.ipynb | 📤 Ready to upload |

## Quick Links

- **Azure ML Studio**: https://ml.azure.com
- **Your Workspace**: https://ml.azure.com/?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/mlops-cw2-rg/providers/Microsoft.MachineLearningServices/workspaces/ticket-priority-workspace
- **Azure Portal**: https://portal.azure.com

## Need Help?

See detailed guides:
- `AZURE_ML_STUDIO_GUIDE.md` - Step-by-step Azure ML Studio guide
- `README.md` - Complete project documentation
- `QUICKSTART.md` - Quick start for local development

---

**You're all set! 🚀**

Next: Upload `azure_ml_pipeline_notebook.ipynb` to Azure ML Studio and run it!
