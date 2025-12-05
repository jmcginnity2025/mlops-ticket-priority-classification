# Azure Resources Location Map 🗺️

## Overview

Everything is organized under one **Resource Group** for easy management and cleanup.

---

## 📦 Resource Group (Container for Everything)

**Name**: `cw2-mlops-rg`
**Location**: France Central

```
Azure Portal → Resource Groups → cw2-mlops-rg
```

**Direct Link**:
https://portal.azure.com/#@ulster.ac.uk/resource/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/overview

---

## 🏢 Azure ML Workspace (Main Hub)

**Name**: `cw2-mlops-workspace`
**Purpose**: Central hub for all ML operations

### What's Inside:
- Experiments & Runs
- Models
- Datasets
- Compute clusters
- Endpoints (for deployment)

**Azure Portal Location**:
```
Azure Portal → Machine Learning → cw2-mlops-workspace
```

**Azure ML Studio (Better Interface)**:
https://ml.azure.com

**Direct Workspace Link**:
https://ml.azure.com/home?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/cw2-mlops-workspace

---

## 🖥️ Compute Cluster

**Name**: `cpu-cluster`
**Type**: Azure ML Compute
**Size**: STANDARD_DS3_v2 (4 cores, 14GB RAM)
**Scaling**: 0-2 instances (currently scaled to 0 when idle)

**Location in Azure ML Studio**:
```
Azure ML Studio → Compute → Compute clusters → cpu-cluster
```

**Direct Link**:
https://ml.azure.com/compute/list?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/cw2-mlops-workspace

### What It Does:
- Runs your training jobs
- Automatically scales up when job starts
- Automatically scales down to 0 after 5 minutes idle (saves money!)

---

## 📊 Dataset

**Name**: `support-tickets-dataset`
**Version**: 1
**Size**: 20.2 MB (48,837 tickets)

**Location in Azure ML Studio**:
```
Azure ML Studio → Data → support-tickets-dataset
```

**Direct Link**:
https://ml.azure.com/data?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/cw2-mlops-workspace

### Storage Location:
The actual file is stored in:
- **Storage Account**: `cw2mlopsstorage40e16b0cc`
- **Container**: `azureml-blobstore-...`
- **Path**: `LocalUpload/...`

**To view storage directly**:
```
Azure Portal → Storage Accounts → cw2mlopsstorage40e16b0cc → Containers
```

---

## 🧪 Experiments & Training Jobs

**Experiment Name**: `cw2-ticket-priority-classification`
**Current Job**: `boring_rat_y4htjsxyd9`

**Location in Azure ML Studio**:
```
Azure ML Studio → Jobs → cw2-ticket-priority-classification
```

**Direct Links**:

**All Experiments**:
https://ml.azure.com/experiments?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/cw2-mlops-workspace

**Current Job**:
https://ml.azure.com/runs/boring_rat_y4htjsxyd9?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/cw2-mlops-workspace

### What's Logged (via MLflow):
- **Metrics**: accuracy, F1, precision, recall
- **Parameters**: model type, hyperparameters
- **Models**: Trained models (RF & XGBoost)
- **Logs**: Training output

---

## 📦 Models (After Training Completes)

**Location in Azure ML Studio**:
```
Azure ML Studio → Models
```

**Direct Link**:
https://ml.azure.com/model/list?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/cw2-mlops-workspace

### What You'll See:
Once training completes, you'll have:
- `iteration_1_random_forest` model
- `iteration_2_xgboost` model

Both logged automatically via MLflow!

---

## 🔐 Supporting Resources (Auto-Created)

These were created automatically with the workspace:

### 1. Storage Account
**Name**: `cw2mlopsstorage40e16b0cc`
**Purpose**: Stores datasets, models, logs

**Location**:
```
Azure Portal → Storage Accounts → cw2mlopsstorage40e16b0cc
```

### 2. Key Vault
**Name**: `cw2mlopskeyvaulta0ac8200`
**Purpose**: Stores secrets, credentials

**Location**:
```
Azure Portal → Key Vaults → cw2mlopskeyvaulta0ac8200
```

### 3. Application Insights
**Name**: `cw2mlopsinsights9c51bfd8`
**Purpose**: Monitoring, telemetry, logs

**Location**:
```
Azure Portal → Application Insights → cw2mlopsinsights9c51bfd8
```

### 4. Log Analytics Workspace
**Name**: `cw2mlopslogalytiffa7a7be`
**Purpose**: Centralized logging

**Location**:
```
Azure Portal → Log Analytics Workspaces → cw2mlopslogalytiffa7a7be
```

---

## 📍 Quick Navigation Map

```
Azure Portal (portal.azure.com)
│
├── Resource Groups
│   └── cw2-mlops-rg (Everything is here!)
│       ├── cw2-mlops-workspace (ML Workspace)
│       ├── cw2mlopsstorage40e16b0cc (Storage)
│       ├── cw2mlopskeyvaulta0ac8200 (Key Vault)
│       ├── cw2mlopsinsights9c51bfd8 (App Insights)
│       └── cw2mlopslogalytiffa7a7be (Log Analytics)
│
Azure ML Studio (ml.azure.com) - BETTER FOR ML WORK!
│
├── Home
├── Compute
│   └── cpu-cluster
├── Data
│   └── support-tickets-dataset (v1)
├── Jobs
│   └── cw2-ticket-priority-classification
│       └── boring_rat_y4htjsxyd9 (current run)
├── Models (after training)
│   ├── iteration_1_random_forest
│   └── iteration_2_xgboost
└── Endpoints (for deployment - future)
```

---

## 🎯 How to Find Your Training Job

### Option 1: Azure ML Studio (Recommended)

1. Go to: https://ml.azure.com
2. Select workspace: `cw2-mlops-workspace`
3. Click **Jobs** in left sidebar
4. Look for experiment: `cw2-ticket-priority-classification`
5. Click on job: `boring_rat_y4htjsxyd9`

### Option 2: Direct Link

Just click this:
https://ml.azure.com/runs/boring_rat_y4htjsxyd9?wsid=/subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/cw2-mlops-workspace

### Option 3: Azure Portal (Less useful for ML)

1. Go to: https://portal.azure.com
2. Search for `cw2-mlops-workspace`
3. Click on it
4. Click "Launch Studio" button

---

## 🔍 What to Look For in Azure ML Studio

### During Training:

**Jobs Page** → Your job → Check:
- ✅ Status (Preparing → Running → Completed)
- ✅ Logs (see what's happening)
- ✅ Metrics (graphs updating in real-time)
- ✅ Outputs (trained models)

### After Training:

**Experiments Page**:
- Compare both iterations
- View metrics side-by-side
- Download models

**Models Page**:
- Registered models
- Model versions
- Ready for deployment

---

## 💰 Cost Breakdown

All in Resource Group: `cw2-mlops-rg`

| Resource | Cost | Notes |
|----------|------|-------|
| Workspace | Free | No charge |
| Storage | ~$0.02/month | For 20MB dataset |
| Compute | ~$0.20/hour | Only when running! |
| Training Job | ~$0.05/run | ~15 min |
| Key Vault | Free tier | Minimal usage |
| App Insights | Free tier | Basic monitoring |

**Total**: Less than $1 from your Azure for Students credits!

---

## 🧹 Cleanup (When Done with Coursework)

**Delete Everything at Once**:
```bash
az group delete --name cw2-mlops-rg
```

This deletes:
- Workspace
- Compute
- Storage
- All models
- All experiments
- Everything else!

**Stops All Charges** ✅

---

## 📞 Need to Find Something?

### Azure Portal Search
1. Go to portal.azure.com
2. Use search bar at top
3. Search for: `cw2-mlops`
4. All your resources will show up!

### Azure ML Studio Search
1. Go to ml.azure.com
2. Select workspace
3. Use left sidebar to navigate
4. Everything organized by category

---

## 🎓 For Your Coursework Report

**Include these screenshots**:

1. **Resource Group** - Shows all components
2. **Training Job** - Shows it completed successfully
3. **Metrics Graph** - Comparison of both iterations
4. **Models** - Shows models are registered
5. **Compute** - Shows scalable infrastructure

**All found in**:
- Azure ML Studio: https://ml.azure.com
- Navigate using the map above!

---

**Quick Check**: Is your training job complete?
```bash
az ml job show --name boring_rat_y4htjsxyd9 \
  --resource-group cw2-mlops-rg \
  --workspace-name cw2-mlops-workspace \
  --query "status"
```

If it says **"Completed"**, go to Azure ML Studio and check out your results! 🎉
