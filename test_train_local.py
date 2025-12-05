"""
Local test script to verify train_azure.py works before submitting to Azure ML
Tests the training script with the actual dataset
"""
import subprocess
import sys
import os

print("="*70)
print("LOCAL TEST: Training Script Validation")
print("="*70)

# Path to dataset
dataset_path = r"data\cleaned_support_tickets - with context.csv"

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"\nERROR: Dataset not found at: {dataset_path}")
    print("   Please ensure the dataset file exists in the data/ directory")
    sys.exit(1)

print(f"\nOK: Dataset found: {dataset_path}")
print("\n" + "="*70)
print("Running train_azure.py locally...")
print("="*70)

# Run the training script
try:
    result = subprocess.run(
        [sys.executable, "train_azure.py", "--data_path", dataset_path],
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )

    # Print output
    print("\n--- STDOUT ---")
    print(result.stdout)

    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

    # Check exit code
    print("\n" + "="*70)
    if result.returncode == 0:
        print("SUCCESS: Training script completed without errors!")
        print("="*70)
        print("\nSafe to commit and push to GitHub")
        sys.exit(0)
    else:
        print(f"FAILED: Training script exited with code {result.returncode}")
        print("="*70)
        print("\nDO NOT commit - fix errors first")
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("\nERROR: Training script timed out (>5 minutes)")
    print("   This might indicate an infinite loop or hang")
    sys.exit(1)

except Exception as e:
    print(f"\nERROR: {str(e)}")
    sys.exit(1)
