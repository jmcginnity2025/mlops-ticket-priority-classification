#!/bin/bash
# Script to create a fresh GitHub repository for MLOps CW2

echo "=========================================="
echo "Creating Fresh MLOps Repository"
echo "=========================================="

# New repo name
REPO_NAME="mlops-cw2-fresh"
REPO_DIR="../${REPO_NAME}"

echo ""
echo "Step 1: Creating new directory..."
mkdir -p "$REPO_DIR"

echo "Step 2: Copying essential files..."

# Core ML scripts
cp preprocess.py "$REPO_DIR/"
cp train.py "$REPO_DIR/"
cp evaluate.py "$REPO_DIR/"

# Azure ML scripts
cp train_azure.py "$REPO_DIR/"
cp upload_dataset_azure.py "$REPO_DIR/"
cp submit_training_job.py "$REPO_DIR/"
cp environment.yml "$REPO_DIR/"
cp azure_config.json "$REPO_DIR/"

# GitHub workflows
mkdir -p "$REPO_DIR/.github/workflows"
cp .github/workflows/ml-cicd-pipeline.yml "$REPO_DIR/.github/workflows/"
cp .github/workflows/azure-ml-pipeline.yml "$REPO_DIR/.github/workflows/"

# Documentation
cp GETTING_STARTED.md "$REPO_DIR/"
cp PROJECT_SUMMARY_CW2.md "$REPO_DIR/"
cp QUICK_REFERENCE.md "$REPO_DIR/"
cp GITHUB_SETUP.md "$REPO_DIR/"
cp AZURE_COMPLETE.md "$REPO_DIR/"
cp AZURE_COMMANDS.md "$REPO_DIR/"
cp AZURE_LOCATION_MAP.md "$REPO_DIR/"

# Requirements
cp requirements.txt "$REPO_DIR/"

# Gitignore
cp .gitignore "$REPO_DIR/"

echo "Step 3: Creating README..."
cat > "$REPO_DIR/README.md" << 'EOF'
# MLOps Pipeline - Support Ticket Priority Classification

Complete MLOps solution with CI/CD and Azure ML integration for coursework CW2.

## Quick Start

### Local Training
```bash
python preprocess.py  # Preprocess data
python train.py       # Train both iterations
python evaluate.py    # Compare and test
```

### Azure ML Training
```bash
python upload_dataset_azure.py  # Upload dataset (once)
python submit_training_job.py   # Submit training to Azure
```

## CI/CD Pipelines

- **Local Pipeline**: `.github/workflows/ml-cicd-pipeline.yml` - Trains on GitHub runners
- **Azure ML Pipeline**: `.github/workflows/azure-ml-pipeline.yml` - Trains on Azure compute

## Project Structure

```
mlops-cw2-fresh/
├── preprocess.py                 # Data preprocessing
├── train.py                      # Model training (2 iterations)
├── evaluate.py                   # Evaluation & regression testing
├── train_azure.py                # Azure ML training script
├── submit_training_job.py        # Submit jobs to Azure
├── environment.yml               # Dependencies
├── .github/workflows/            # CI/CD pipelines
├── requirements.txt              # Python packages
└── docs/                         # Documentation
```

## Features

- ✅ Two model iterations (Random Forest + XGBoost)
- ✅ Regression testing (fails if performance drops)
- ✅ CI/CD with GitHub Actions
- ✅ Azure ML integration
- ✅ MLflow experiment tracking
- ✅ Cost-optimized compute (scales to zero)

## Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Setup and usage
- [PROJECT_SUMMARY_CW2.md](PROJECT_SUMMARY_CW2.md) - Complete overview
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - CI/CD setup
- [AZURE_COMPLETE.md](AZURE_COMPLETE.md) - Azure ML setup

## Requirements

All coursework requirements (CW2) met:
- ✅ Model Development (2 iterations)
- ✅ CI/CD (automated testing)
- ✅ Deployment (Azure ML ready)
- ✅ Monitoring (MLflow tracking)
- ✅ Governance (versioning, audit trail)

## Azure Resources

- **Resource Group**: cw2-mlops-rg
- **Workspace**: cw2-mlops-workspace
- **Compute**: cpu-cluster (STANDARD_DS3_v2)
- **Dataset**: support-tickets-dataset (v1)

## License

Academic project for MLOps coursework.
EOF

echo "Step 4: Initializing git..."
cd "$REPO_DIR"
git init
git add .
git commit -m "Initial commit: Complete MLOps pipeline with Azure ML integration"

echo ""
echo "=========================================="
echo "✅ Repository created successfully!"
echo "=========================================="
echo ""
echo "Location: $REPO_DIR"
echo ""
echo "Next steps:"
echo "1. Create GitHub repository at: https://github.com/new"
echo "2. Name it: $REPO_NAME"
echo "3. Run these commands:"
echo ""
echo "   cd $REPO_DIR"
echo "   git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=========================================="
