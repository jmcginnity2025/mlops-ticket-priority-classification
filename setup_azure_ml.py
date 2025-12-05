"""
Complete Azure ML Setup Script
Uploads dataset, creates environment, and prepares workspace for training
"""
from azureml.core import Workspace, Dataset, Environment, Experiment
from azureml.core.conda_dependencies import CondaDependencies
import yaml

print("="*70)
print("AZURE ML SETUP")
print("="*70)

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to workspace
print("\n1. Connecting to workspace...")
ws = Workspace(
    subscription_id=config['azure']['subscription_id'],
    resource_group=config['azure']['resource_group'],
    workspace_name=config['azure']['workspace_name']
)

print(f"   ✓ Connected to: {ws.name}")
print(f"   ✓ Region: {ws.location}")

# Upload and register dataset
print("\n2. Uploading and registering dataset...")
datastore = ws.get_default_datastore()

# Upload file
print("   Uploading data file...")
datastore.upload_files(
    files=[config['data']['raw_data_path']],
    target_path='support_tickets_data/',
    overwrite=True,
    show_progress=True
)

# Create dataset
dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, 'support_tickets_data/cleaned_support_tickets - with context.csv')
)

# Register dataset
dataset_name = config['azure']['dataset_name']
registered_dataset = dataset.register(
    workspace=ws,
    name=dataset_name,
    description='Support tickets with priority for classification',
    create_new_version=True
)

print(f"   ✓ Dataset registered: {dataset_name}")
print(f"   ✓ Version: {registered_dataset.version}")

# Create environment
print("\n3. Creating ML environment...")
env_name = "mlops-training-env"
env = Environment(name=env_name)

# Add dependencies
conda_dep = CondaDependencies()
conda_dep.add_pip_package("azureml-defaults")
conda_dep.add_pip_package("scikit-learn>=1.0.0")
conda_dep.add_pip_package("xgboost>=1.5.0")
conda_dep.add_pip_package("pandas>=1.3.0")
conda_dep.add_pip_package("numpy>=1.21.0")
conda_dep.add_pip_package("mlflow>=2.0.0")
conda_dep.add_pip_package("azureml-mlflow")
conda_dep.add_pip_package("pyyaml>=6.0")
conda_dep.add_pip_package("joblib>=1.1.0")
conda_dep.add_pip_package("scipy>=1.7.0")

env.python.conda_dependencies = conda_dep

# Register environment
env.register(workspace=ws)
print(f"   ✓ Environment registered: {env_name}")

# Create experiment
print("\n4. Creating MLflow experiment...")
experiment_name = config['mlflow']['experiment_name']
experiment = Experiment(workspace=ws, name=experiment_name)
print(f"   ✓ Experiment created: {experiment_name}")

# Save workspace config
print("\n5. Saving workspace configuration...")
ws.write_config(path=".", file_name="config.json")
print(f"   ✓ Workspace config saved to config.json")

# Summary
print("\n" + "="*70)
print("SETUP COMPLETE!")
print("="*70)
print("\nWorkspace Details:")
print(f"  Name: {ws.name}")
print(f"  Subscription: {ws.subscription_id}")
print(f"  Resource Group: {ws.resource_group}")
print(f"  Location: {ws.location}")
print(f"  MLflow URI: {ws.get_mlflow_tracking_uri()}")

print("\nResources Created:")
print(f"  ✓ Dataset: {dataset_name} (v{registered_dataset.version})")
print(f"  ✓ Environment: {env_name}")
print(f"  ✓ Experiment: {experiment_name}")
print(f"  ✓ Compute: CPU-Cluster (existing)")

print("\nNext Steps:")
print("  1. Open Azure ML Studio: https://ml.azure.com")
print("  2. Navigate to your workspace: ticket-priority-workspace")
print("  3. Upload and run: azure_ml_pipeline_notebook.ipynb")
print("  4. Or run locally: python run_pipeline.py --mode full")

print("\n" + "="*70)
