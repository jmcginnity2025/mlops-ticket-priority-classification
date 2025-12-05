# Azure ML Quick Commands

## Check Training Job Status

```bash
# Quick status check
az ml job show --name boring_rat_y4htjsxyd9 \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace \
  --query "status"

# Full details
az ml job show --name boring_rat_y4htjsxyd9 \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace

# Stream logs (watch live)
az ml job stream --name boring_rat_y4htjsxyd9 \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace
```

## List Jobs

```bash
# All jobs
az ml job list \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace \
  --output table

# Latest 5 jobs
az ml job list \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace \
  --max-results 5 \
  --output table
```

## Submit New Training Job

```bash
# From Python script
python submit_training_job.py

# Or directly
az ml job create --file job.yml \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace
```

## Check Compute

```bash
# Compute status
az ml compute show --name cpu-cluster \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace

# List all compute
az ml compute list \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace \
  --output table
```

## View Models

```bash
# List all models
az ml model list \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace \
  --output table

# Show specific model
az ml model show --name ticket-priority-classifier --version 1 \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace
```

## View Dataset

```bash
# List datasets
az ml data list \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace \
  --output table

# Show dataset details
az ml data show --name support-tickets-dataset --version 1 \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace
```

## Azure ML Studio Links

**Workspace:**
https://ml.azure.com/workspaces/cw2-mlops-workspace

**Current Job:**
https://ml.azure.com/runs/boring_rat_y4htjsxyd9

**Experiments:**
https://ml.azure.com/experiments

**Models:**
https://ml.azure.com/model/list

**Compute:**
https://ml.azure.com/compute/list

## Cleanup (When Done)

```bash
# Delete specific resources
az ml compute delete --name cpu-cluster \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace

# Delete entire resource group (CAUTION!)
az group delete --name cw2-mlops-rg
```

## Configuration

All config stored in: `azure_config.json`

```json
{
  "subscription_id": "d5156f99-abd5-4af9-9e2d-a875ef22df46",
  "resource_group": "cw2-mlops-rg",
  "workspace_name": "cw2-mlops-workspace",
  "compute_name": "cpu-cluster"
}
```
