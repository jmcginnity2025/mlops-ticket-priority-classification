# MLOps Pipeline for Support Ticket Priority Classification

This project implements a complete MLOps workflow for a machine learning model that classifies support ticket priorities. It demonstrates best practices in Model Development, CI/CD, Deployment, Monitoring, Retraining, and Governance.

## Project Overview

**Dataset**: Support Tickets with Priority Classification
**Task**: Multi-class classification (Low, Medium, High priority) and ranking
**Platform**: Azure Machine Learning
**Tools**: MLflow, XGBoost, scikit-learn, Azure DevOps/GitHub Actions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion & Versioning              │
│              (Azure ML Datasets, Data Validation)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Data Preprocessing Pipeline                   │
│        (Cleaning, Feature Engineering, Scaling)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Model Training & Experiment Tracking              │
│         (XGBoost, MLflow, Multiple Iterations)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Model Evaluation & Regression Testing              │
│       (Performance Metrics, Comparison, Validation)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              CI/CD Pipeline (Azure DevOps)                  │
│   (Automated Testing, Build, Staging, Production)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             Model Deployment (Azure ML)                     │
│         (ACI Endpoint, Authentication, Scoring)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│      Monitoring (Data Drift & Performance Tracking)         │
│              (PSI, KS-Test, Alert System)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Automated Retraining Triggers                     │
│       (Based on Drift & Performance Degradation)            │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
CW2_workflow/
│
├── config/
│   └── config.yaml                 # Configuration for all components
│
├── data/
│   ├── cleaned_support_tickets - with context.csv  # Raw data
│   └── processed/                  # Processed data with versioning
│       └── YYYYMMDD_HHMMSS/
│           ├── X_train.csv
│           ├── X_test.csv
│           ├── y_train.csv
│           ├── y_test.csv
│           └── metadata.json
│
├── src/
│   ├── data_preprocessing.py       # Data pipeline
│   ├── train_model.py              # Training with MLflow
│   ├── evaluate_model.py           # Evaluation & regression tests
│   ├── deploy_model.py             # Azure ML deployment
│   ├── monitor_model.py            # Drift & performance monitoring
│   └── score.py                    # Scoring script (auto-generated)
│
├── tests/
│   ├── test_data_validation.py    # Data quality tests
│   └── test_model_performance.py  # Model performance tests
│
├── models/                         # Trained models
│   ├── multiclass_iteration_1/
│   │   └── model.pkl
│   └── multiclass_iteration_2/
│       └── model.pkl
│
├── evaluation_results/             # Evaluation reports
│   └── multiclass/
│       ├── iteration_comparison.json
│       ├── model_comparison.json
│       ├── regression_test_report.json
│       └── evaluation_summary.json
│
├── monitoring/                     # Monitoring outputs
│   ├── drift_reports/
│   ├── performance_reports/
│   └── dashboard_data.json
│
├── mlruns/                         # MLflow tracking data
│
├── .github/workflows/
│   └── mlops-pipeline.yml          # GitHub Actions CI/CD
│
├── azure-pipelines.yml             # Azure DevOps CI/CD
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

### Prerequisites

1. **Azure Account**: Active Azure subscription
2. **Python**: Version 3.9 or higher
3. **Git**: For version control
4. **Azure CLI**: For Azure operations

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CW2_workflow
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Azure credentials**:
   Update `config/config.yaml` with your Azure subscription details:
   ```yaml
   azure:
     subscription_id: "YOUR_SUBSCRIPTION_ID"
     resource_group: "mlops-cw2-rg"
     workspace_name: "ticket-priority-workspace"
   ```

5. **Create Azure ML Workspace** (if not exists):
   ```bash
   az ml workspace create -w ticket-priority-workspace -g mlops-cw2-rg
   ```

## Usage

### 1. Data Preprocessing

Process and prepare data for training:

```bash
python src/data_preprocessing.py
```

This will:
- Load raw data from the configured path
- Clean and handle missing values
- Encode categorical features
- Scale numerical features
- Split into train/test sets
- Save versioned processed data

### 2. Model Training

Train models with MLflow experiment tracking:

```bash
# Train multiclass classification model (2 iterations)
python src/train_model.py --model_type multiclass --iterations 2

# Train ranking model
python src/train_model.py --model_type ranking --iterations 2
```

Features:
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- MLflow tracking (metrics, parameters, artifacts)
- Model versioning

View MLflow UI:
```bash
mlflow ui
```
Navigate to http://localhost:5000

### 3. Model Evaluation

Evaluate models and check for performance regression:

```bash
python src/evaluate_model.py --model_type multiclass
```

This will:
- Compare all model iterations
- Check performance against thresholds
- Generate evaluation reports
- Detect performance regression

### 4. Model Deployment

Deploy model to Azure ML:

```bash
# Register model
python src/deploy_model.py --action register \
  --model_path models/multiclass_iteration_2/model.pkl \
  --model_name ticket-priority-classifier

# Deploy to endpoint
python src/deploy_model.py --action deploy \
  --model_name ticket-priority-classifier \
  --endpoint_name ticket-priority-endpoint

# Test deployment
python src/deploy_model.py --action test \
  --endpoint_name ticket-priority-endpoint
```

### 5. Model Monitoring

Monitor for data drift and performance degradation:

```bash
# Detect data drift
python src/monitor_model.py --action drift \
  --current_data data/new_data.csv

# Generate monitoring dashboard
python src/monitor_model.py --action dashboard
```

Monitoring includes:
- Data drift detection (KS-test, Chi-squared, PSI)
- Performance tracking
- Automated retraining triggers

## CI/CD Pipeline

### Azure DevOps

1. Create a new pipeline in Azure DevOps
2. Connect to your repository
3. Use the existing `azure-pipelines.yml`
4. Configure service connection to Azure
5. Set up environments for staging and production

Pipeline stages:
1. **Build & Test**: Install dependencies, run unit tests
2. **Data Validation**: Validate data schema and quality
3. **Training**: Preprocess data, train models
4. **Model Validation**: Check performance thresholds
5. **Deploy to Staging**: Deploy to staging environment
6. **Deploy to Production**: Deploy to production (manual approval)
7. **Monitoring**: Setup monitoring and alerts

### GitHub Actions

1. Add repository secrets:
   - `AZURE_CREDENTIALS`: Azure service principal credentials
2. Push to main branch to trigger pipeline

## Key Features

### ✅ Model Development
- Automated data preprocessing pipeline
- Feature engineering and selection
- Multiple model iterations for comparison
- Hyperparameter tuning with GridSearchCV

### ✅ CI/CD
- Automated testing (unit, integration, data validation)
- Multi-stage deployment (staging → production)
- Blue-green deployment strategy
- Automated model registration

### ✅ Deployment
- Azure Container Instance deployment
- REST API endpoint with authentication
- Scoring script with error handling
- Deployment testing and validation

### ✅ Monitoring
- Data drift detection (PSI, KS-test, Chi-squared)
- Performance degradation alerts
- Prediction logging
- Dashboard data generation

### ✅ Retraining
- Automated trigger conditions:
  - Data drift threshold exceeded
  - Performance degradation detected
  - Scheduled retraining
- Validation before deployment
- Approval workflow

### ✅ Governance
- Model registry with versioning
- Experiment tracking with MLflow
- Audit trail for deployments
- Model explainability (SHAP)
- Compliance logging

## Performance Thresholds

Classification model requirements (from `config.yaml`):
- **Minimum Accuracy**: 75%
- **Minimum F1 Score**: 70%

Regression model requirements:
- **Maximum RMSE**: 30.0

## Monitoring Thresholds

- **Data Drift**: 15% of features drifted
- **Performance Degradation**: 5% drop in F1 score
- **PSI Interpretation**:
  - < 0.1: No significant change
  - 0.1 - 0.2: Moderate change
  - \> 0.2: Significant change (trigger alert)

## Troubleshooting

### Common Issues

1. **Azure authentication fails**:
   ```bash
   az login
   az account set --subscription <subscription-id>
   ```

2. **MLflow tracking URI not found**:
   Ensure `mlruns/` directory exists or set tracking URI:
   ```bash
   export MLFLOW_TRACKING_URI=file:./mlruns
   ```

3. **Data file not found**:
   Update path in `config/config.yaml` to point to your data file

4. **Model deployment fails**:
   Check Azure quota for compute resources

## Evaluation Criteria

This project addresses the following coursework requirements:

### Task 1: Solution Design ✅
- Uses pre-selected dataset from CW1
- Implements data versioning (Azure ML Datasets)
- Trains ML model with 2+ iterations
- Uses MLflow for experiment tracking
- Includes performance regression tests
- Considers scalability and deployment constraints

### Task 2: Implementation, Deployment, Testing ✅
- Fully implemented solution using Azure ML
- Deployed model as REST API endpoint
- Comprehensive testing suite
- Monitoring and alerting system
- Documentation and setup instructions

### Learning Outcomes ✅
1. **Technology Evaluation**: Demonstrates understanding of MLOps tools and their trade-offs
2. **ML Paradigms**: Implements classification and ranking models with proper evaluation
3. **Infrastructure**: Shows end-to-end MLOps pipeline with CI/CD, monitoring, and governance

## Future Enhancements

- [ ] Add A/B testing for model comparison
- [ ] Implement canary deployment strategy
- [ ] Add real-time prediction API with FastAPI
- [ ] Integrate with Application Insights for APM
- [ ] Add model interpretability dashboard
- [ ] Implement automated feature engineering
- [ ] Add data quality monitoring with Great Expectations

## License

This project is for academic purposes as part of the AI Masters Infrastructure Module Coursework.

## Contact

For questions or issues, please contact [your email/contact info].

---

**Academic Note**: This project demonstrates MLOps best practices for the Infrastructure Module Coursework CW2. All implementations follow industry standards and are designed for educational purposes.
# Testing CI/CD pipeline with new compute
Trigger pipeline
Trigger pipeline
Trigger workflow
