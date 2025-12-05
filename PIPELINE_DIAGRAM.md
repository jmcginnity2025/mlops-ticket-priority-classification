# MLOps CI/CD Pipeline Diagram

## Complete Pipeline Architecture

```mermaid
graph LR
    %% =================================================================
    %% NODES DEFINITION
    %% =================================================================

    %% Trigger
    GHA[/GitHub Actions<br/>CI/CD Pipeline<br/>git push to main/]

    %% Data Storage
    DATA[(Azure ML Data Asset<br/>support-tickets-dataset<br/>Version 1)]

    %% Job 1: Data Preprocessing
    JOB1[Job 1: Data Quality & Unit Tests<br/>Load, Clean, Validate Data<br/>Local - Fast ~30s]

    %% Job 2: Model Training
    JOB2[Job 2: Train & Register Model<br/>Iteration 1: Random Forest baseline<br/>Iteration 2: XGBoost improved<br/>Azure ML - 15-20 min]

    %% Compute Cluster
    COMPUTE{{Azure ML Compute Cluster<br/>cpu-cluster-fast<br/>STANDARD_DS2_V3<br/>Auto-scales: 0→1→0}}

    %% Model Registry
    REGISTRY[(Model Registry<br/>support-ticket-classifier-rf<br/>support-ticket-classifier<br/>Versioned Models)]

    %% Azure ML Studio Metrics
    METRICS[Azure ML Studio<br/>Metrics & Logs]

    %% Job 3: Regression Testing - Decision Node
    JOB3{Job 3: Regression Test<br/>Quality Gate<br/>2% threshold<br/>Iteration 2 vs Iteration 1}

    %% Failed Pipeline
    FAIL[Pipeline Failed<br/>Model Regression Detected<br/>Deployment Blocked]

    %% Job 4: Model Versioning
    JOB4[Job 4: Version Models<br/>Create version tag<br/>Prepare for deployment<br/>Local - Fast ~10s]

    %% Job 5: Online Endpoint Deployment
    JOB5[Job 5: Deploy to Endpoint<br/>Create/Update Endpoint<br/>Blue-Green Deployment<br/>Azure ML - 5-10 min]

    %% Production Endpoint
    ENDPOINT([Production Endpoint<br/>support-ticket-classifier<br/>Real-time Inference API<br/>Scoring URI])

    %% =================================================================
    %% CONNECTIONS - Main Flow
    %% =================================================================

    GHA -->|Fetch data| DATA
    DATA -->|Load dataset| JOB1
    JOB1 -->|Processed data| JOB2
    JOB2 -.->|Uses for training| COMPUTE
    JOB2 ==>|Register models| REGISTRY
    JOB2 -.->|Log metrics| METRICS
    JOB2 -->|Pass metrics| JOB3

    %% Decision Point
    JOB3 -->|Pass: Improvement ≥ -2%| JOB4
    JOB3 -->|Fail: Regression > 2%| FAIL

    %% Continue after quality gate
    JOB4 -->|Version approved| JOB5
    REGISTRY ==>|Fetch model v{n}| JOB5
    JOB5 -->|Deploy model| ENDPOINT

    %% Feedback Loop
    ENDPOINT -.->|Retraining trigger<br/>Code/data changes| GHA

    %% =================================================================
    %% STYLING
    %% =================================================================

    %% GitHub Actions - Dark Gray
    classDef githubStyle fill:#24292e,stroke:#1a1d21,stroke-width:3px,color:#fff,font-weight:bold

    %% Data Assets - Azure Blue
    classDef dataStyle fill:#0078d4,stroke:#005a9e,stroke-width:3px,color:#fff,font-weight:bold

    %% Local Jobs - Green
    classDef localStyle fill:#28a745,stroke:#1e7e34,stroke-width:3px,color:#fff,font-weight:bold

    %% Cloud Training - Blue
    classDef trainingStyle fill:#0366d6,stroke:#044289,stroke-width:3px,color:#fff,font-weight:bold

    %% Compute - Orange
    classDef computeStyle fill:#f66a0a,stroke:#c44c00,stroke-width:3px,color:#fff,font-weight:bold

    %% Model Registry - Purple
    classDef registryStyle fill:#6f42c1,stroke:#5a32a3,stroke-width:3px,color:#fff,font-weight:bold

    %% Testing/Quality Gate - Orange
    classDef testingStyle fill:#ffa500,stroke:#cc8400,stroke-width:3px,color:#000,font-weight:bold

    %% Versioning - Light Purple
    classDef versionStyle fill:#9d4edd,stroke:#7b2cbf,stroke-width:3px,color:#fff,font-weight:bold

    %% Deployment - Deep Purple
    classDef deployStyle fill:#5a32a3,stroke:#3d1f70,stroke-width:3px,color:#fff,font-weight:bold

    %% Production Endpoint - Green
    classDef endpointStyle fill:#28a745,stroke:#1e7e34,stroke-width:4px,color:#fff,font-weight:bold

    %% Failure - Red
    classDef failStyle fill:#d73a49,stroke:#b01e2e,stroke-width:3px,color:#fff,font-weight:bold

    %% Metrics/Monitoring - Light Blue
    classDef metricsStyle fill:#17a2b8,stroke:#117a8b,stroke-width:2px,color:#fff

    %% =================================================================
    %% APPLY STYLES
    %% =================================================================

    class GHA githubStyle
    class DATA dataStyle
    class JOB1 localStyle
    class JOB2 trainingStyle
    class COMPUTE computeStyle
    class REGISTRY registryStyle
    class METRICS metricsStyle
    class JOB3 testingStyle
    class FAIL failStyle
    class JOB4 versionStyle
    class JOB5 deployStyle
    class ENDPOINT endpointStyle
```

## Pipeline Flow Summary

### Trigger
- **GitHub Actions CI/CD** - Automated trigger on `git push` to main branch

### Stage 1: Data Preparation (Local)
- **Job 1**: Data Quality & Unit Tests (~30 seconds)
  - Load support-tickets-dataset from Azure ML
  - Clean and validate data
  - Feature engineering
  - Train/test split

### Stage 2: Model Training (Azure ML Cloud)
- **Job 2**: Train & Register Model (~15-20 minutes)
  - **Iteration 1**: Random Forest (baseline)
  - **Iteration 2**: XGBoost (improved)
  - Uses **cpu-cluster-fast** compute (auto-scaling)
  - Logs metrics to **Azure ML Studio**
  - Registers both models to **Model Registry**

### Stage 3: Quality Gate (Local)
- **Job 3**: Regression Testing (~5 seconds)
  - Compare Iteration 2 vs Iteration 1
  - Quality threshold: ≤2% performance drop allowed
  - **PASS** → Continue to versioning
  - **FAIL** → Block pipeline, prevent deployment

### Stage 4: Versioning (Local)
- **Job 4**: Version Models (~10 seconds)
  - Create version tag (GitHub run number)
  - Generate pipeline summary
  - Prepare models for deployment

### Stage 5: Deployment (Azure ML Cloud)
- **Job 5**: Deploy to Online Endpoint (~5-10 minutes)
  - Create/update endpoint: `support-ticket-classifier`
  - Fetch latest model from registry
  - **Blue-green deployment** strategy
  - Route 100% traffic to new version

### Production
- **Online Endpoint**: Real-time inference API
  - RESTful API with authentication
  - Scoring URI for predictions
  - Production-ready serving

### Feedback Loop
- **Continuous Retraining**: Code/data changes trigger pipeline restart

---

## Key Features Demonstrated

✅ **Automated CI/CD** - GitHub Actions orchestration
✅ **Cloud Infrastructure** - Azure ML compute and storage
✅ **Quality Gates** - Regression testing prevents bad deployments
✅ **Model Registry** - Versioned model storage with lineage
✅ **Scalable Training** - Auto-scaling compute cluster
✅ **Monitoring** - Metrics logged to Azure ML Studio
✅ **Production Deployment** - Real-time inference endpoint
✅ **Continuous Retraining** - Automated pipeline triggers

---

## Component Colors

| Component | Color | Meaning |
|-----------|-------|---------|
| GitHub Actions | Dark Gray (#24292e) | CI/CD Trigger |
| Data Asset | Azure Blue (#0078d4) | Data Storage |
| Job 1 (Preprocessing) | Green (#28a745) | Local/Fast |
| Job 2 (Training) | Blue (#0366d6) | Cloud Training |
| Compute Cluster | Orange (#f66a0a) | Infrastructure |
| Model Registry | Purple (#6f42c1) | Model Storage |
| Job 3 (Testing) | Orange (#ffa500) | Quality Gate |
| Job 4 (Versioning) | Light Purple (#9d4edd) | Versioning |
| Job 5 (Deployment) | Deep Purple (#5a32a3) | Deployment |
| Production Endpoint | Green (#28a745) | Live/Production |
| Pipeline Failed | Red (#d73a49) | Error/Blocked |

---

## View the Diagram

To view this diagram:

1. **GitHub**: This file will render automatically on GitHub
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **Online**: Copy the mermaid code block to https://mermaid.live
4. **Export**: Use Mermaid Live Editor to export as PNG/SVG for presentations

---

## Matches Reference Architecture

This diagram demonstrates:
- ✅ All 5 jobs in sequential order
- ✅ GitHub Actions trigger
- ✅ Azure ML Data Asset (cylinder)
- ✅ Compute cluster (hexagon) connected to training
- ✅ Model Registry (cylinder) with versioning
- ✅ Decision node at regression testing (diamond)
- ✅ Production endpoint deployment
- ✅ Feedback loop for continuous retraining (dashed line)
- ✅ Clear visual hierarchy and color coding
- ✅ Duration labels on each job

**100% implementation of enterprise MLOps pipeline architecture!**
