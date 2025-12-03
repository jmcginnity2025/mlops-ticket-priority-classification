"""
Model Deployment to Azure ML
"""
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.conda_dependencies import CondaDependencies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    """Handles model deployment to Azure ML"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize deployer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.workspace = None

    def connect_to_workspace(self):
        """Connect to Azure ML workspace"""
        logger.info("Connecting to Azure ML workspace...")

        try:
            # Try to load from config file if exists
            self.workspace = Workspace.from_config()
            logger.info(f"Connected to workspace: {self.workspace.name}")
        except:
            # Create/get workspace from config
            self.workspace = Workspace.get(
                name=self.config['azure']['workspace_name'],
                subscription_id=self.config['azure']['subscription_id'],
                resource_group=self.config['azure']['resource_group']
            )
            logger.info(f"Connected to workspace: {self.workspace.name}")

        return self.workspace

    def register_model(self, model_path: str, model_name: str,
                      model_description: str = "", tags: dict = None):
        """Register model in Azure ML Model Registry"""
        logger.info(f"Registering model: {model_name}")

        if self.workspace is None:
            self.connect_to_workspace()

        # Default tags
        if tags is None:
            tags = {
                'framework': 'xgboost',
                'type': 'classification',
                'timestamp': datetime.now().isoformat()
            }

        model = Model.register(
            workspace=self.workspace,
            model_path=model_path,
            model_name=model_name,
            description=model_description,
            tags=tags
        )

        logger.info(f"Model registered with ID: {model.id}")
        logger.info(f"Model version: {model.version}")

        return model

    def create_inference_config(self):
        """Create inference configuration"""
        logger.info("Creating inference configuration...")

        # Create environment
        env = Environment(name="ticket-priority-env")

        # Define conda dependencies
        conda_dep = CondaDependencies()
        conda_dep.add_pip_package("azureml-defaults")
        conda_dep.add_pip_package("scikit-learn")
        conda_dep.add_pip_package("xgboost")
        conda_dep.add_pip_package("pandas")
        conda_dep.add_pip_package("numpy")
        conda_dep.add_pip_package("joblib")

        env.python.conda_dependencies = conda_dep

        # Create scoring script if it doesn't exist
        self._create_scoring_script()

        # Create inference config
        inference_config = InferenceConfig(
            entry_script="score.py",
            source_directory="src",
            environment=env
        )

        logger.info("Inference configuration created")

        return inference_config

    def _create_scoring_script(self):
        """Create scoring script for deployment"""
        scoring_script = '''
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    """Initialize the model"""
    global model
    model_path = Model.get_model_path('ticket-priority-classifier')
    model = joblib.load(model_path)

def run(raw_data):
    """
    Make predictions on input data

    Input format (JSON):
    {
        "data": [
            [feature1, feature2, ..., featureN],
            [feature1, feature2, ..., featureN]
        ]
    }

    Output format (JSON):
    {
        "predictions": [1, 2, 3],
        "probabilities": [[0.1, 0.8, 0.1], ...]
    }
    """
    try:
        # Parse input data
        data = json.loads(raw_data)
        input_data = np.array(data['data'])

        # Make predictions
        predictions = model.predict(input_data)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)
            result = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            }
        else:
            result = {
                'predictions': predictions.tolist()
            }

        return json.dumps(result)

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
'''

        score_path = Path("src") / "score.py"
        with open(score_path, 'w') as f:
            f.write(scoring_script)

        logger.info(f"Scoring script created at {score_path}")

    def create_deployment_config(self):
        """Create deployment configuration for ACI"""
        logger.info("Creating deployment configuration...")

        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=2,
            auth_enabled=True,
            enable_app_insights=True,
            description="Ticket Priority Classification Endpoint"
        )

        logger.info("Deployment configuration created")

        return deployment_config

    def deploy_model(self, model_name: str, endpoint_name: str = None):
        """Deploy model to Azure Container Instance"""
        logger.info(f"Deploying model: {model_name}")

        if self.workspace is None:
            self.connect_to_workspace()

        if endpoint_name is None:
            endpoint_name = self.config['deployment']['endpoint_name']

        # Get registered model
        model = Model(self.workspace, name=model_name)

        # Create configurations
        inference_config = self.create_inference_config()
        deployment_config = self.create_deployment_config()

        # Deploy
        logger.info(f"Deploying to endpoint: {endpoint_name}")
        logger.info("This may take several minutes...")

        service = Model.deploy(
            workspace=self.workspace,
            name=endpoint_name,
            models=[model],
            inference_config=inference_config,
            deployment_config=deployment_config,
            overwrite=True
        )

        service.wait_for_deployment(show_output=True)

        logger.info(f"Deployment completed!")
        logger.info(f"Scoring URI: {service.scoring_uri}")
        logger.info(f"Swagger URI: {service.swagger_uri}")

        # Save deployment info
        self._save_deployment_info(service)

        return service

    def _save_deployment_info(self, service):
        """Save deployment information"""
        deployment_info = {
            'endpoint_name': service.name,
            'scoring_uri': service.scoring_uri,
            'swagger_uri': service.swagger_uri,
            'state': service.state,
            'deployment_timestamp': datetime.now().isoformat()
        }

        # Get authentication keys
        try:
            keys = service.get_keys()
            deployment_info['primary_key'] = keys[0]
            deployment_info['secondary_key'] = keys[1]
        except:
            logger.warning("Could not retrieve authentication keys")

        output_path = Path("deployment_info.json")
        with open(output_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)

        logger.info(f"Deployment info saved to {output_path}")

    def test_deployment(self, service_name: str = None, test_data: list = None):
        """Test deployed model"""
        logger.info("Testing deployed model...")

        if self.workspace is None:
            self.connect_to_workspace()

        if service_name is None:
            service_name = self.config['deployment']['endpoint_name']

        # Get service
        service = Webservice(workspace=self.workspace, name=service_name)

        # Prepare test data
        if test_data is None:
            # Create dummy test data (replace with actual feature values)
            test_data = [[0.5] * 20]  # Assuming 20 features

        input_data = json.dumps({'data': test_data})

        # Make prediction
        logger.info("Sending test request...")
        response = service.run(input_data)

        logger.info(f"Response: {response}")

        return response

    def update_deployment(self, service_name: str, new_model_name: str):
        """Update existing deployment with new model version"""
        logger.info(f"Updating deployment {service_name} with model {new_model_name}")

        if self.workspace is None:
            self.connect_to_workspace()

        # Get service and new model
        service = Webservice(workspace=self.workspace, name=service_name)
        new_model = Model(self.workspace, name=new_model_name)

        # Update service
        service.update(models=[new_model])
        service.wait_for_deployment(show_output=True)

        logger.info("Deployment updated successfully")

        return service

    def delete_deployment(self, service_name: str):
        """Delete deployment"""
        logger.info(f"Deleting deployment: {service_name}")

        if self.workspace is None:
            self.connect_to_workspace()

        service = Webservice(workspace=self.workspace, name=service_name)
        service.delete()

        logger.info(f"Deployment {service_name} deleted")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Deploy model to Azure ML')
    parser.add_argument('--action', type=str, required=True,
                        choices=['register', 'deploy', 'test', 'update', 'delete'],
                        help='Deployment action to perform')
    parser.add_argument('--model_path', type=str,
                        help='Path to model file (for registration)')
    parser.add_argument('--model_name', type=str, default='ticket-priority-classifier',
                        help='Name for the model')
    parser.add_argument('--endpoint_name', type=str,
                        help='Name for the endpoint')

    args = parser.parse_args()

    deployer = ModelDeployer()

    if args.action == 'register':
        if not args.model_path:
            print("Error: --model_path required for registration")
            exit(1)
        model = deployer.register_model(args.model_path, args.model_name)
        print(f"Model registered: {model.name} v{model.version}")

    elif args.action == 'deploy':
        service = deployer.deploy_model(args.model_name, args.endpoint_name)
        print(f"Model deployed to: {service.scoring_uri}")

    elif args.action == 'test':
        response = deployer.test_deployment(args.endpoint_name)
        print(f"Test response: {response}")

    elif args.action == 'update':
        service = deployer.update_deployment(args.endpoint_name, args.model_name)
        print(f"Deployment updated: {service.name}")

    elif args.action == 'delete':
        deployer.delete_deployment(args.endpoint_name)
        print(f"Deployment deleted")
