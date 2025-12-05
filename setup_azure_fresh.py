"""
Fresh Azure ML Setup Script
Creates everything from scratch: Resource Group, Workspace, Compute
"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
import subprocess

# Configuration
CONFIG = {
    'subscription_id': 'd5156f99-abd5-4af9-9e2d-a875ef22df46',
    'resource_group': 'cw2-mlops-rg',
    'workspace_name': 'cw2-mlops-workspace',
    'location': 'francecentral',
    'compute_name': 'cpu-cluster',
}

def create_resource_group():
    """Create a new resource group"""
    print("\n" + "="*70)
    print("STEP 1: Creating Resource Group")
    print("="*70)

    cmd = [
        'az', 'group', 'create',
        '--name', CONFIG['resource_group'],
        '--location', CONFIG['location']
    ]

    print(f"\nCreating resource group: {CONFIG['resource_group']}")
    print(f"Location: {CONFIG['location']}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Resource group created successfully!")
        return True
    else:
        print(f"❌ Error: {result.stderr}")
        return False

def create_workspace():
    """Create Azure ML workspace using az ml"""
    print("\n" + "="*70)
    print("STEP 2: Creating Azure ML Workspace")
    print("="*70)

    cmd = [
        'az', 'ml', 'workspace', 'create',
        '--name', CONFIG['workspace_name'],
        '--resource-group', CONFIG['resource_group'],
        '--location', CONFIG['location']
    ]

    print(f"\nCreating workspace: {CONFIG['workspace_name']}")
    print("This may take 2-3 minutes...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Workspace created successfully!")
        return True
    else:
        print(f"❌ Error: {result.stderr}")
        return False

def create_compute():
    """Create compute cluster"""
    print("\n" + "="*70)
    print("STEP 3: Creating Compute Cluster")
    print("="*70)

    cmd = [
        'az', 'ml', 'compute', 'create',
        '--name', CONFIG['compute_name'],
        '--resource-group', CONFIG['resource_group'],
        '--workspace-name', CONFIG['workspace_name'],
        '--type', 'amlcompute',
        '--size', 'STANDARD_DS3_v2',
        '--min-instances', '0',
        '--max-instances', '2',
        '--idle-time-before-scale-down', '300'
    ]

    print(f"\nCreating compute: {CONFIG['compute_name']}")
    print(f"Size: STANDARD_DS3_v2 (4 cores, 14 GB RAM)")
    print(f"Scaling: 0-2 instances (scales to zero when idle)")
    print("This may take 3-5 minutes...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Compute cluster created successfully!")
        return True
    else:
        print(f"❌ Error: {result.stderr}")
        return False

def verify_setup():
    """Verify everything was created"""
    print("\n" + "="*70)
    print("STEP 4: Verifying Setup")
    print("="*70)

    # Check workspace
    result = subprocess.run([
        'az', 'ml', 'workspace', 'show',
        '--name', CONFIG['workspace_name'],
        '--resource-group', CONFIG['resource_group']
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Workspace verified: {CONFIG['workspace_name']}")
    else:
        print(f"❌ Workspace verification failed")
        return False

    # Check compute
    result = subprocess.run([
        'az', 'ml', 'compute', 'show',
        '--name', CONFIG['compute_name'],
        '--resource-group', CONFIG['resource_group'],
        '--workspace-name', CONFIG['workspace_name']
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Compute verified: {CONFIG['compute_name']}")
    else:
        print(f"❌ Compute verification failed")
        return False

    return True

def save_config():
    """Save configuration"""
    import json

    with open("azure_config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\n💾 Configuration saved to: azure_config.json")

def main():
    print("="*70)
    print("AZURE ML FRESH SETUP")
    print("="*70)
    print("\nThis will create:")
    print(f"  Resource Group: {CONFIG['resource_group']}")
    print(f"  Workspace: {CONFIG['workspace_name']}")
    print(f"  Compute: {CONFIG['compute_name']}")
    print(f"  Location: {CONFIG['location']}")
    print("\nEstimated time: 5-8 minutes")

    input("\nPress Enter to start or Ctrl+C to cancel...")

    # Create resource group
    if not create_resource_group():
        print("\n❌ Setup failed at Step 1")
        return

    # Create workspace
    if not create_workspace():
        print("\n❌ Setup failed at Step 2")
        return

    # Create compute
    if not create_compute():
        print("\n❌ Setup failed at Step 3")
        return

    # Verify
    if not verify_setup():
        print("\n❌ Verification failed")
        return

    # Save config
    save_config()

    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)
    print("\nYour Azure ML environment:")
    print(f"  Subscription: {CONFIG['subscription_id']}")
    print(f"  Resource Group: {CONFIG['resource_group']}")
    print(f"  Workspace: {CONFIG['workspace_name']}")
    print(f"  Compute: {CONFIG['compute_name']}")
    print(f"  Location: {CONFIG['location']}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Upload dataset to Azure ML")
    print("2. Create and submit training job")
    print("\nCommands:")
    print("  python upload_dataset_azure.py")
    print("  python submit_training_job.py")

if __name__ == "__main__":
    main()
