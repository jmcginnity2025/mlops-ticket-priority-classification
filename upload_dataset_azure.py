"""
Upload Dataset to Azure ML
"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import json
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load config
with open("azure_config.json", 'r') as f:
    config = json.load(f)

# Dataset path
DATASET_PATH = r"C:\AI Masters\AI Masters\Infrastucture Module - Azure\CW2 New\cleaned_support_tickets - with context.csv"

print("="*70)
print("UPLOADING DATASET TO AZURE ML")
print("="*70)

# Connect to Azure ML
print("\nConnecting to Azure ML...")
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=config['subscription_id'],
    resource_group_name=config['resource_group'],
    workspace_name=config['workspace_name']
)
print(f"✅ Connected to workspace: {config['workspace_name']}")

# Create data asset
print(f"\nUploading dataset from: {DATASET_PATH}")
print("This may take a minute...")

data_asset = Data(
    name="support-tickets-dataset",
    version="1",
    description="Support ticket priority classification dataset",
    path=DATASET_PATH,
    type=AssetTypes.URI_FILE
)

# Upload
data_asset = ml_client.data.create_or_update(data_asset)

print(f"✅ Dataset uploaded successfully!")
print(f"   Name: {data_asset.name}")
print(f"   Version: {data_asset.version}")
print(f"   Path: {data_asset.path}")

print("\n" + "="*70)
print("DATASET UPLOAD COMPLETE")
print("="*70)
