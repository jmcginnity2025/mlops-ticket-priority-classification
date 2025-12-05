"""
Simple script to upload dataset to Azure ML
"""
from azureml.core import Workspace, Dataset
import yaml

print("Uploading dataset to Azure ML...")

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to workspace
print("Connecting to workspace...")
ws = Workspace(
    subscription_id=config['azure']['subscription_id'],
    resource_group=config['azure']['resource_group'],
    workspace_name=config['azure']['workspace_name']
)
print(f"✓ Connected to: {ws.name}")

# Get default datastore
datastore = ws.get_default_datastore()

# Upload file
print("\nUploading file (this may take a few minutes)...")
file_path = config['data']['raw_data_path']

datastore.upload_files(
    files=[file_path],
    target_path='support_tickets_data/',
    overwrite=True,
    show_progress=True
)
print("✓ File uploaded")

# Create tabular dataset
print("\nCreating tabular dataset...")
dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, 'support_tickets_data/cleaned_support_tickets - with context.csv')
)

# Register dataset
dataset_name = 'support_tickets'
registered_dataset = dataset.register(
    workspace=ws,
    name=dataset_name,
    description='Support tickets with priority classification',
    create_new_version=True
)

print(f"✓ Dataset registered: {dataset_name}")
print(f"✓ Version: {registered_dataset.version}")
print(f"\n✅ Dataset upload complete!")
print(f"\nYou can now view it in Azure ML Studio:")
print(f"https://ml.azure.com → Data → {dataset_name}")
