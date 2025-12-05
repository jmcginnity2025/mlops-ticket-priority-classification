"""
Azure ML Studio Setup Script
Run this in Azure ML Studio to set up the complete MLOps pipeline
"""

from azureml.core import Workspace, Experiment, Dataset, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.data import OutputFileDatasetConfig
import os

# ============================================================================
# STEP 1: Connect to Workspace
# ============================================================================
print("Connecting to Azure ML Workspace...")

# This will automatically connect to the workspace in Azure ML Studio
ws = Workspace.from_config()

print(f"Workspace name: {ws.name}")
print(f"Resource group: {ws.resource_group}")
print(f"Location: {ws.location}")

# ============================================================================
# STEP 2: Create or Get Compute Target
# ============================================================================
print("\nSetting up compute target...")

compute_name = "cpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print(f"Found existing compute target: {compute_name}")
except ComputeTargetException:
    print(f"Creating new compute target: {compute_name}")

    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_DS3_V2',  # Use appropriate VM size
        min_nodes=0,
        max_nodes=4,
        idle_seconds_before_scaledown=300
    )

    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# ============================================================================
# STEP 3: Register Dataset
# ============================================================================
print("\nRegistering dataset...")

# Upload your data file to Azure ML Studio first
# Then register it as a dataset

datastore = ws.get_default_datastore()

# Register dataset (modify path as needed)
dataset_name = "support_tickets"

try:
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    print(f"Found existing dataset: {dataset_name}")
except:
    print(f"Dataset {dataset_name} not found. Please upload your data file to Azure ML Studio.")
    print("You can do this via the 'Datasets' tab in Azure ML Studio")

# ============================================================================
# STEP 4: Create Environment
# ============================================================================
print("\nCreating environment...")

env = Environment(name="mlops-env")

# Use conda dependencies
conda_dep = env.python.conda_dependencies
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

# Register environment
env.register(workspace=ws)
print(f"Environment registered: {env.name}")

# ============================================================================
# STEP 5: Create MLflow Experiment
# ============================================================================
print("\nCreating MLflow experiment...")

experiment_name = "ticket-priority-classification"
experiment = Experiment(workspace=ws, name=experiment_name)

print(f"Experiment created: {experiment_name}")

# ============================================================================
# STEP 6: Display Next Steps
# ============================================================================
print("\n" + "="*70)
print("AZURE ML STUDIO SETUP COMPLETE!")
print("="*70)

print("\nNext Steps:")
print("1. Upload your data file to Azure ML Studio:")
print("   - Go to 'Datasets' tab")
print("   - Click 'Create dataset' → 'From local files'")
print("   - Upload 'cleaned_support_tickets - with context.csv'")
print("   - Name it 'support_tickets'")

print("\n2. Upload all Python scripts from src/ folder:")
print("   - data_preprocessing.py")
print("   - train_model.py")
print("   - evaluate_model.py")
print("   - deploy_model.py")
print("   - monitor_model.py")

print("\n3. Upload config/config.yaml")

print("\n4. Create a Compute Instance or use existing one")

print("\n5. Run the pipeline using the notebook:")
print("   - Open 'azure_ml_pipeline_notebook.ipynb'")
print("   - Run all cells")

print("\n" + "="*70)
print("Workspace Information:")
print("="*70)
print(f"Workspace: {ws.name}")
print(f"Subscription: {ws.subscription_id}")
print(f"Resource Group: {ws.resource_group}")
print(f"Compute Target: {compute_name}")
print(f"Environment: {env.name}")
print(f"Experiment: {experiment_name}")
print("="*70)
