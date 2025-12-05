"""
Model Training Pipeline with MLflow Tracking
"""
import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import spearmanr
import logging
from pathlib import Path
import json
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training with MLflow experiment tracking"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def load_processed_data(self, data_dir: str = None):
        """Load processed training and test data"""
        if data_dir is None:
            # Find latest processed data
            processed_path = Path(self.config['data']['processed_data_path'])
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed data directory not found: {processed_path}")

            versions = sorted([d for d in processed_path.iterdir() if d.is_dir()])
            if not versions:
                raise FileNotFoundError("No processed data versions found")

            data_dir = versions[-1]
            logger.info(f"Loading latest processed data from: {data_dir}")
        else:
            data_dir = Path(data_dir)

        X_train = pd.read_csv(data_dir / "X_train.csv")
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

        logger.info(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_multiclass_model(self, X_train, y_train, X_test, y_test, iteration: int = 1):
        """Train multiclass classification model with MLflow tracking"""
        logger.info(f"Starting multiclass model training - Iteration {iteration}")

        with mlflow.start_run(run_name=f"multiclass_iteration_{iteration}"):
            # Log parameters
            mlflow.log_param("iteration", iteration)
            mlflow.log_param("model_type", "multiclass_classification")
            mlflow.log_param("algorithm", self.config['models']['multiclass']['algorithm'])
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("num_features", X_train.shape[1])

            # Prepare hyperparameters for grid search
            param_grid = self.config['models']['multiclass']['hyperparameters']
            cv_folds = self.config['training']['cross_validation_folds']

            # Initialize model
            base_model = XGBClassifier(
                random_state=self.config['data']['random_seed'],
                eval_metric='mlogloss',
                early_stopping_rounds=self.config['training']['early_stopping_rounds']
            )

            # Grid search
            logger.info("Performing grid search for hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=self.config['training']['metric_multiclass'],
                verbose=1,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")

            # Log best hyperparameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)

            mlflow.log_metric("cv_best_score", grid_search.best_score_)

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_classification_metrics(y_train, y_train_pred, y_test, y_test_pred)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

            # Log model
            if self.config['mlflow']['log_models']:
                mlflow.xgboost.log_model(best_model, "model")

            # Save model locally
            model_dir = Path("models") / f"multiclass_iteration_{iteration}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_dir / "model.pkl")

            logger.info(f"Model saved to {model_dir}")
            logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
            logger.info(f"Test F1 Score: {metrics['test_f1']:.4f}")

            return best_model, metrics

    def train_ranking_model(self, X_train, y_train, X_test, y_test, iteration: int = 1):
        """Train ranking/regression model with MLflow tracking"""
        logger.info(f"Starting ranking model training - Iteration {iteration}")

        with mlflow.start_run(run_name=f"ranking_iteration_{iteration}"):
            # Log parameters
            mlflow.log_param("iteration", iteration)
            mlflow.log_param("model_type", "ranking_regression")
            mlflow.log_param("algorithm", self.config['models']['ranking']['algorithm'])
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Prepare hyperparameters
            param_grid = self.config['models']['ranking']['hyperparameters']
            cv_folds = self.config['training']['cross_validation_folds']

            # Initialize model
            base_model = XGBRegressor(
                random_state=self.config['data']['random_seed'],
                early_stopping_rounds=self.config['training']['early_stopping_rounds']
            )

            # Grid search
            logger.info("Performing grid search for hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_

            # Log best hyperparameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            if self.config['mlflow']['log_models']:
                mlflow.xgboost.log_model(best_model, "model")

            # Save model locally
            model_dir = Path("models") / f"ranking_iteration_{iteration}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_dir / "model.pkl")

            logger.info(f"Model saved to {model_dir}")
            logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
            logger.info(f"Test R2 Score: {metrics['test_r2']:.4f}")

            return best_model, metrics

    def _calculate_classification_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate classification metrics"""
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
            'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
            'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
            'test_recall': recall_score(y_test, y_test_pred, average='weighted')
        }
        return metrics

    def _calculate_regression_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate regression metrics"""
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_spearman': spearmanr(y_train, y_train_pred)[0],
            'test_spearman': spearmanr(y_test, y_test_pred)[0]
        }
        return metrics

    def run_training_iterations(self, num_iterations: int = 2, model_type: str = 'multiclass'):
        """Run multiple training iterations for comparison"""
        logger.info(f"Running {num_iterations} training iterations for {model_type} model")

        X_train, X_test, y_train, y_test = self.load_processed_data()

        results = []

        for i in range(1, num_iterations + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"ITERATION {i}/{num_iterations}")
            logger.info(f"{'='*50}\n")

            if model_type == 'multiclass':
                model, metrics = self.train_multiclass_model(X_train, y_train, X_test, y_test, iteration=i)
            else:
                model, metrics = self.train_ranking_model(X_train, y_train, X_test, y_test, iteration=i)

            results.append({
                'iteration': i,
                'model': model,
                'metrics': metrics
            })

        # Save comparison results
        self._save_iteration_comparison(results, model_type)

        return results

    def _save_iteration_comparison(self, results, model_type):
        """Save comparison of multiple iterations"""
        comparison_dir = Path("evaluation_results") / model_type
        comparison_dir.mkdir(parents=True, exist_ok=True)

        comparison_data = {
            'model_type': model_type,
            'num_iterations': len(results),
            'timestamp': datetime.now().isoformat(),
            'iterations': []
        }

        for result in results:
            comparison_data['iterations'].append({
                'iteration': result['iteration'],
                'metrics': result['metrics']
            })

        with open(comparison_dir / "iteration_comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)

        logger.info(f"Iteration comparison saved to {comparison_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model with MLflow tracking')
    parser.add_argument('--model_type', type=str, default='multiclass',
                        choices=['multiclass', 'ranking'],
                        help='Type of model to train')
    parser.add_argument('--iterations', type=int, default=2,
                        help='Number of training iterations')

    args = parser.parse_args()

    trainer = ModelTrainer()
    results = trainer.run_training_iterations(
        num_iterations=args.iterations,
        model_type=args.model_type
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    for result in results:
        print(f"\nIteration {result['iteration']}:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
