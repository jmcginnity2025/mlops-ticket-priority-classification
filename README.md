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
