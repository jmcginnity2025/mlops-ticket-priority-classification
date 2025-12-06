"""
Submit Azure ML Training Job for Random Forest and XGBoost Models

This script submits a training job to Azure ML that will:
1. Train both Random Forest and XGBoost classifiers
2. Use the actual dataset: cleaned_support_tickets - with context.csv (48,388 records)
3. Log comprehensive metrics to Azure ML Studio
4. Save trained models as artifacts

Usage:
    python submit_training_job.py
"""

from azure.ai.ml import MLClient
from azure.ai.ml import command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
import os

# Initialize Azure ML client
print("🔐 Authenticating with Azure...")
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP", "mlops-cw2-rg"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME", "ticket-priority-workspace")
)

print(f"✅ Connected to Azure ML workspace: {ml_client.workspace_name}")

# Submit job using job.yml configuration
print("\n📤 Submitting training job to Azure ML...")
print("   - Script: train_azure_ml.py")
print("   - Models: Random Forest + XGBoost")
print("   - Dataset: cleaned_support_tickets - with context.csv (48,388 records)")
print("   - Compute: MLOps-compute-instance")
print("   - Experiment: ticket-priority-classification-rf-xgboost")

# Load and submit the job
job = ml_client.jobs.create_or_update(
    ml_client.jobs.load("train_job.yml")
)

print(f"\n✅ Job submitted successfully!")
print(f"   - Job Name: {job.name}")
print(f"   - Status: {job.status}")
print(f"   - Studio URL: {job.studio_url}")

print("\n🔍 To view job progress:")
print(f"   1. Open Azure ML Studio: {job.studio_url}")
print(f"   2. Navigate to: Experiments → ticket-priority-classification-rf-xgboost")
print(f"   3. Click on run: {job.name}")
print(f"   4. View metrics in 'Metrics' tab")
print(f"   5. Download models from 'Outputs + logs' tab")

print("\n⏳ Streaming job logs (this may take 10-15 minutes)...")
print("=" * 80)

# Stream the job logs
ml_client.jobs.stream(job.name)

print("\n" + "=" * 80)
print("✅ TRAINING JOB COMPLETED")
print("=" * 80)

# Get final job status
final_job = ml_client.jobs.get(job.name)
print(f"\n📊 Final Status: {final_job.status}")

if final_job.status == "Completed":
    print("\n✅ SUCCESS! Both models trained successfully.")
    print("\n📈 Next steps:")
    print("   1. Review metrics in Azure ML Studio")
    print("   2. Compare Random Forest vs XGBoost performance")
    print("   3. Download trained models from artifacts")
    print("   4. Proceed with regression testing (2% threshold)")
    print("\n🔗 View results: " + final_job.studio_url)
else:
    print(f"\n❌ Job failed with status: {final_job.status}")
    print(f"   Check logs at: {final_job.studio_url}")
