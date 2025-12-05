"""
Submit Training Job to Azure ML
"""
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import json

# Load config
with open("azure_config.json", 'r') as f:
    config = json.load(f)

print("="*70)
print("SUBMITTING TRAINING JOB TO AZURE ML")
print("="*70)

# Connect to Azure ML
print("\nConnecting to Azure ML...")
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=config['subscription_id'],
    resource_group_name=config['resource_group'],
    workspace_name=config['workspace_name']
)
print(f"Connected to: {config['workspace_name']}")

# Get dataset
print("\nGetting dataset...")
data_asset = ml_client.data.get(name="support-tickets-dataset", version="1")
print(f"Dataset: {data_asset.name} v{data_asset.version}")

# Create environment
print("\nCreating environment...")
env = Environment(
    name="mlops-training-env",
    description="Environment for ML training",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

# Create job
print("\nConfiguring training job...")
job = command(
    code="./",
    command="python train_azure.py --data_path ${{inputs.dataset}}",
    inputs={
        "dataset": Input(
            type="uri_file",
            path=data_asset.id
        )
    },
    environment=env,
    compute=config['compute_name'],
    experiment_name="cw2-ticket-priority-classification",
    display_name="train-both-iterations"
)

# Submit job
print("\nSubmitting job...")
print(f"Compute: {config['compute_name']}")
print(f"Experiment: cw2-ticket-priority-classification")

returned_job = ml_client.jobs.create_or_update(job)

print(f"\nJob submitted successfully!")
print(f"Job name: {returned_job.name}")
print(f"Job status: {returned_job.status}")
print(f"\nMonitor job in Azure ML Studio:")
print(f"https://ml.azure.com/runs/{returned_job.name}")
print(f"?wsid=/subscriptions/{config['subscription_id']}/resourceGroups/{config['resource_group']}/providers/Microsoft.MachineLearningServices/workspaces/{config['workspace_name']}")

print("\n" + "="*70)
print("To check job status, run:")
print(f"  az ml job show --name {returned_job.name} --resource-group {config['resource_group']} --workspace-name {config['workspace_name']}")
print("="*70)
