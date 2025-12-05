# GitHub Actions Setup for Azure ML

## Current Pipelines

You now have **2 pipelines**:

### 1. Local Training Pipeline ✅
**File**: `.github/workflows/ml-cicd-pipeline.yml`
**What it does**:
- Trains models locally on GitHub runners
- Fast and free
- Good for testing

**Status**: Ready to use! No setup needed.

### 2. Azure ML Pipeline 🆕
**File**: `.github/workflows/azure-ml-pipeline.yml`
**What it does**:
- Submits training jobs to Azure ML
- Uses your Azure compute cluster
- Logs with MLflow
- Production-ready

**Status**: Needs GitHub Secrets setup (below)

---

## Setting Up Azure ML Pipeline

### Step 1: Create Azure Service Principal

Run this command to create credentials for GitHub to access Azure:

```bash
az ad sp create-for-rbac \
  --name "github-actions-mlops" \
  --role contributor \
  --scopes /subscriptions/d5156f99-abd5-4af9-9e2d-a875ef22df46/resourceGroups/cw2-mlops-rg \
  --sdk-auth
```

**Copy the entire JSON output!** It looks like this:

```json
{
  "clientId": "...",
  "clientSecret": "...",
  "subscriptionId": "d5156f99-abd5-4af9-9e2d-a875ef22df46",
  "tenantId": "...",
  "activeDirectoryEndpointUrl": "...",
  "resourceManagerEndpointUrl": "...",
  "activeDirectoryGraphResourceId": "...",
  "sqlManagementEndpointUrl": "...",
  "galleryEndpointUrl": "...",
  "managementEndpointUrl": "..."
}
```

### Step 2: Add GitHub Secrets

1. Go to your GitHub repo
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

Add these 2 secrets:

#### Secret 1: AZURE_CREDENTIALS
- **Name**: `AZURE_CREDENTIALS`
- **Value**: Paste the ENTIRE JSON from Step 1

#### Secret 2: AZURE_SUBSCRIPTION_ID
- **Name**: `AZURE_SUBSCRIPTION_ID`
- **Value**: `d5156f99-abd5-4af9-9e2d-a875ef22df46`

### Step 3: Test the Pipeline

```bash
# Make a small change and commit
git add .
git commit -m "Test Azure ML pipeline"
git push origin main
```

Then:
1. Go to your repo → **Actions** tab
2. You'll see the pipeline running
3. Click on it to watch progress

---

## How the Pipelines Work

### Local Pipeline Flow
```
Commit → GitHub → Preprocess → Train Locally → Evaluate → Pass/Fail
```

**Pros**:
- ✅ Fast (5-10 min)
- ✅ Free
- ✅ No setup needed

**Cons**:
- ❌ Limited compute
- ❌ No MLflow tracking
- ❌ Not production-ready

### Azure ML Pipeline Flow
```
Commit → GitHub → Submit to Azure ML → Train on Cluster → MLflow Logging → Pass/Fail
```

**Pros**:
- ✅ Powerful compute
- ✅ MLflow tracking
- ✅ Production-ready
- ✅ Scalable

**Cons**:
- ❌ Slower (10-15 min)
- ❌ Uses Azure credits
- ❌ Needs setup

---

## Choosing Which Pipeline to Use

### Use Local Pipeline (`ml-cicd-pipeline.yml`) when:
- Testing code changes quickly
- Developing locally
- Want fast feedback
- Don't need MLflow tracking

### Use Azure ML Pipeline (`azure-ml-pipeline.yml`) when:
- Production deployment
- Need experiment tracking
- Want to show coursework integration
- Need more compute power

---

## Quick Commands

### Check Pipeline Status
```bash
# View recent workflow runs
gh run list

# Watch current run
gh run watch
```

### Trigger Pipeline Manually
Go to: GitHub repo → Actions → Select workflow → Run workflow

Or via CLI:
```bash
gh workflow run "Azure ML Pipeline"
```

### View Logs
```bash
gh run view --log
```

---

## Current Status

### ✅ Ready to Use:
- Local training pipeline
- All scripts (preprocess, train, evaluate)
- Azure ML workspace
- Azure ML compute cluster
- Dataset uploaded

### ⏳ Needs Setup (Optional):
- GitHub Secrets for Azure ML pipeline
- Service Principal creation

---

## For Your Coursework

### Recommended Approach:

1. **Show Local Pipeline** (Easy)
   - Already working
   - Commit your code
   - Show GitHub Actions running
   - Screenshot the successful run

2. **Show Azure ML Integration** (Advanced)
   - Set up service principal (above)
   - Run Azure ML pipeline
   - Show MLflow tracking
   - Show models in Azure ML Studio

**Both approaches satisfy coursework requirements!**

---

## Troubleshooting

### Pipeline Fails with "Dataset not found"
- Make sure dataset is uploaded: `python upload_dataset_azure.py`
- Check dataset name: `support-tickets-dataset`

### Authentication Fails
- Verify GitHub secrets are set correctly
- Check service principal has contributor role
- Try: `az login` then re-create service principal

### Azure ML Job Fails
- Check logs in Azure ML Studio
- Verify compute cluster is running
- Check environment.yml dependencies

---

## Testing Your Setup

### Test Local Pipeline:
```bash
# 1. Make a change
echo "# Test" >> README.md

# 2. Commit
git add .
git commit -m "Test local pipeline"
git push

# 3. Watch on GitHub
# Go to: Actions tab → See workflow running
```

### Test Azure ML Pipeline:
```bash
# 1. Set up secrets (see Step 2 above)

# 2. Commit
git add .
git commit -m "Test Azure ML pipeline"
git push

# 3. Watch on GitHub AND Azure ML Studio
```

---

## What Gets Tracked

### Local Pipeline Tracks:
- Code changes
- Model metrics (in artifacts)
- Test results

### Azure ML Pipeline Tracks:
- Everything above PLUS:
- MLflow experiments
- Model lineage
- Compute usage
- Full logs in Azure ML

---

## Cost Considerations

### Local Pipeline:
- **Cost**: $0 (GitHub provides free runners)
- **Time**: 5-10 minutes

### Azure ML Pipeline:
- **Cost**: ~$0.05 per run (from student credits)
- **Time**: 10-15 minutes
- **Worth it**: For production & coursework demonstration!

---

## Quick Decision Guide

**Want to test code changes quickly?**
→ Use local pipeline (already works!)

**Want to demonstrate full MLOps for coursework?**
→ Set up Azure ML pipeline (15 min setup)

**Want both?**
→ Keep both! They don't conflict.

---

## Next Steps

1. ✅ Local pipeline is ready - just commit and push!
2. ⏳ Azure ML pipeline - follow Step 1-3 above
3. 📊 Check GitHub Actions tab after pushing
4. 🎓 Screenshot results for coursework

**Need help?** Check the troubleshooting section or let me know!
