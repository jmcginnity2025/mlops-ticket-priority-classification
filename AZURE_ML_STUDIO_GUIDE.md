# Azure ML Studio Setup Guide

This guide shows you how to run the entire MLOps pipeline directly in Azure ML Studio.

## Prerequisites

- Azure account with active subscription
- Azure ML Workspace created
- Dataset file ready

## Step-by-Step Instructions

### Step 1: Create Azure ML Workspace

1. Go to [Azure Portal](https://portal.azure.com)
2. Click **"Create a resource"**
3. Search for **"Machine Learning"**
4. Click **"Create"**
5. Fill in details:
   - **Subscription**: Your subscription
   - **Resource Group**: Create new → `mlops-cw2-rg`
   - **Workspace Name**: `ticket-priority-workspace`
   - **Region**: Choose closest region (e.g., East US)
6. Click **"Review + Create"** → **"Create"**
7. Wait for deployment (2-3 minutes)

### Step 2: Launch Azure ML Studio

1. Go to your Azure ML Workspace in Azure Portal
2. Click **"Launch studio"**
3. You'll be redirected to [ml.azure.com](https://ml.azure.com)

### Step 3: Create Compute Instance

1. In Azure ML Studio, click **"Compute"** in left menu
2. Click **"Compute instances"** tab
3. Click **"+ New"**
4. Configure:
   - **Compute name**: `mlops-compute`
   - **Virtual machine size**: `Standard_DS3_v2` (or any suitable size)
   - **Enable SSH**: Optional
5. Click **"Create"**
6. Wait for compute to start (3-5 minutes)

### Step 4: Upload Dataset

#### Option A: Via UI
1. Click **"Data"** in left menu
2. Click **"+ Create"**
3. Choose **"From local files"**
4. Fill in:
   - **Name**: `support_tickets`
   - **Dataset type**: Tabular
   - **Datastore**: workspaceblobstore (default)
5. Click **"Next"**
6. **Upload files**: Click "Browse" and select your CSV file:
   `cleaned_support_tickets - with context.csv`
7. Click **"Next"** through remaining steps
8. Click **"Create"**

#### Option B: Via Notebook
```python
from azureml.core import Workspace, Dataset

ws = Workspace.from_config()
datastore = ws.get_default_datastore()

# Upload file
datastore.upload_files(
    files=['./data/cleaned_support_tickets - with context.csv'],
    target_path='data/',
    overwrite=True
)

# Create dataset
dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, 'data/cleaned_support_tickets - with context.csv')
)

# Register dataset
dataset.register(
    workspace=ws,
    name='support_tickets',
    description='Support tickets with priority'
)
```

### Step 5: Upload Notebook

1. Click **"Notebooks"** in left menu
2. Click **"Upload files"** button (upload icon)
3. Upload these files from your project:
   - `azure_ml_pipeline_notebook.ipynb`
   - `config/config.yaml` (optional)
4. Files will appear in "Users/your-username/" folder

### Step 6: Run the Pipeline

1. Click on `azure_ml_pipeline_notebook.ipynb` to open it
2. Select your compute instance at the top
3. Run cells one by one or click **"Run all"**

The notebook will:
- ✅ Connect to workspace
- ✅ Load and preprocess data
- ✅ Train 2 model iterations
- ✅ Track experiments with MLflow
- ✅ Evaluate and compare models
- ✅ Run performance regression tests
- ✅ Register best model

### Step 7: View Results

#### View Experiments
1. Click **"Experiments"** in left menu
2. Click on `ticket-priority-classification`
3. You'll see all runs (iterations)
4. Click on a run to see:
   - Parameters
   - Metrics
   - Artifacts
   - Charts

#### View Models
1. Click **"Models"** in left menu
2. You'll see `ticket-priority-classifier`
3. Click on it to see:
   - Model version
   - Tags (accuracy, F1 score, etc.)
   - Associated runs

### Step 8: Deploy Model (Optional)

Create a new notebook cell or new notebook:

```python
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment

# Get workspace and model
ws = Workspace.from_config()
model = Model(ws, name='ticket-priority-classifier')

# Create environment
env = Environment(name='deployment-env')
env.python.conda_dependencies.add_pip_package('scikit-learn')
env.python.conda_dependencies.add_pip_package('xgboost')
env.python.conda_dependencies.add_pip_package('azureml-defaults')

# Create scoring script
scoring_script = '''
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('ticket-priority-classifier')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_data = np.array(data['data'])
        predictions = model.predict(input_data)
        return json.dumps({'predictions': predictions.tolist()})
    except Exception as e:
        return json.dumps({'error': str(e)})
'''

with open('score.py', 'w') as f:
    f.write(scoring_script)

# Create inference config
inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)

# Create deployment config
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=2,
    auth_enabled=True,
    enable_app_insights=True
)

# Deploy
service = Model.deploy(
    workspace=ws,
    name='ticket-priority-endpoint',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)

print(f"Scoring URI: {service.scoring_uri}")
print(f"Swagger URI: {service.swagger_uri}")
```

### Step 9: Test Deployment

```python
import json

# Get service
service = Webservice(workspace=ws, name='ticket-priority-endpoint')

# Create test data (replace with actual feature values)
test_data = {
    'data': [
        [0.5, 0.3, 0.2, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.3] + [0]*12
    ]
}

# Make request
input_data = json.dumps(test_data)
response = service.run(input_data)

print(f"Prediction: {response}")
```

## Monitoring and Data Drift

### Setup Data Drift Monitor

```python
from azureml.datadrift import DataDriftDetector
from azureml.core import Dataset

# Get baseline and target datasets
baseline_dataset = Dataset.get_by_name(ws, 'support_tickets')

# Create drift detector
drift_detector = DataDriftDetector.create_from_datasets(
    workspace=ws,
    name='ticket-priority-drift-detector',
    baseline_data_set=baseline_dataset,
    target_data_set=baseline_dataset,  # Replace with new data
    compute_target='mlops-compute',
    frequency='Week',
    feature_list=None,  # Monitor all features
    drift_threshold=0.15
)

# Run drift detection
drift_run = drift_detector.run(
    target_date=datetime.now(),
    services=None
)

# View results
drift_metrics = drift_run.get_metrics()
print(drift_metrics)
```

## CI/CD Integration

### Setup Azure DevOps Pipeline

1. Create Azure DevOps account
2. Create new project
3. Go to **Pipelines** → **New Pipeline**
4. Connect to your Git repository
5. Use existing `azure-pipelines.yml`
6. Add service connection to Azure
7. Run pipeline

## File Structure in Azure ML Studio

After setup, your file structure in Azure ML Studio should look like:

```
Users/
└── your-username/
    ├── azure_ml_pipeline_notebook.ipynb
    ├── config/
    │   └── config.yaml
    └── (other uploaded files)

Data/
└── support_tickets (registered dataset)

Models/
└── ticket-priority-classifier

Experiments/
└── ticket-priority-classification/
    ├── multiclass_iteration_1
    └── multiclass_iteration_2

Endpoints/ (after deployment)
└── ticket-priority-endpoint
```

## Common Issues and Solutions

### Issue 1: Compute Instance Not Starting
**Solution**: Check quota in your subscription. Try smaller VM size.

### Issue 2: Dataset Upload Fails
**Solution**:
- Check file size (max 50 MB for UI upload)
- Use notebook upload method for larger files
- Ensure CSV is properly formatted

### Issue 3: Notebook Kernel Crashes
**Solution**:
- Reduce dataset size for testing
- Use smaller hyperparameter grid
- Increase compute instance size

### Issue 4: Import Errors
**Solution**:
```python
# Install required packages in notebook
!pip install xgboost scikit-learn mlflow pandas numpy scipy
```

### Issue 5: Authentication Errors
**Solution**:
- Ensure you're logged into correct Azure account
- Check workspace permissions
- Re-run workspace connection cell

## Cost Management

### Free Tier Options
- **Compute**: 4 hours/month free compute hours
- **Storage**: First 10 GB free
- **Experiments**: Unlimited free

### Cost Optimization Tips
1. Stop compute instance when not in use
2. Use lower-tier VM sizes for development
3. Delete old experiments and artifacts
4. Use burst compute for training
5. Set up auto-shutdown for compute

### Stop Compute Instance
```python
from azureml.core import Workspace
from azureml.core.compute import ComputeInstance

ws = Workspace.from_config()
compute = ComputeInstance(workspace=ws, name='mlops-compute')
compute.stop()
```

## Next Steps

1. ✅ Complete pipeline execution
2. ✅ Review all experiment runs
3. ✅ Deploy model to endpoint
4. ✅ Test deployed endpoint
5. ✅ Setup monitoring
6. ✅ Configure CI/CD
7. ✅ Document results for coursework

## Support Resources

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure ML SDK Reference](https://docs.microsoft.com/python/api/overview/azure/ml/)
- [Azure ML Studio](https://ml.azure.com)
- [Azure ML Pricing](https://azure.microsoft.com/pricing/details/machine-learning/)

## Checklist for Coursework Submission

- [ ] Workspace created and configured
- [ ] Dataset uploaded and registered
- [ ] Pipeline executed successfully (2+ iterations)
- [ ] MLflow experiments visible in Azure ML Studio
- [ ] Performance metrics meet thresholds
- [ ] Model registered in Model Registry
- [ ] Screenshots captured for report
- [ ] Deployment tested (optional but recommended)
- [ ] Monitoring configured
- [ ] Cost analysis reviewed

---

**Ready to Start?** Follow Step 1 above and work through each step sequentially!

For local development, see [QUICKSTART.md](QUICKSTART.md)
