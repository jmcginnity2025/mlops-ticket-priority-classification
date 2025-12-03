"""
Model Evaluation and Regression Testing
Detects performance degradation across model iterations
"""
import pandas as pd
import numpy as np
import yaml
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and performance regression detection"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_model(self, model_path: str):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model

    def load_test_data(self, data_dir: str = None):
        """Load test data"""
        if data_dir is None:
            processed_path = Path(self.config['data']['processed_data_path'])
            versions = sorted([d for d in processed_path.iterdir() if d.is_dir()])
            if not versions:
                raise FileNotFoundError("No processed data found")
            data_dir = versions[-1]

        data_dir = Path(data_dir)
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

        logger.info(f"Loaded test data: {X_test.shape}")
        return X_test, y_test

    def evaluate_classification_model(self, model, X_test, y_test, model_name: str = "model"):
        """Comprehensive evaluation for classification models"""
        logger.info(f"Evaluating classification model: {model_name}")

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro')
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report

        # Log results
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")

        return metrics

    def evaluate_regression_model(self, model, X_test, y_test, model_name: str = "model"):
        """Comprehensive evaluation for regression models"""
        logger.info(f"Evaluating regression model: {model_name}")

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'spearman_correlation': spearmanr(y_test, y_pred)[0]
        }

        # Log results
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"R2 Score: {metrics['r2_score']:.4f}")
        logger.info(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")

        return metrics

    def check_performance_regression(self, current_metrics: dict, baseline_metrics: dict,
                                    model_type: str = 'classification'):
        """
        Check if current model shows performance regression compared to baseline
        Returns: (passed, degradation_report)
        """
        logger.info("Checking for performance regression...")

        degradation_report = {
            'passed': True,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'checks': []
        }

        if model_type == 'classification':
            # Check accuracy threshold
            min_accuracy = self.config['evaluation']['min_accuracy']
            if current_metrics['accuracy'] < min_accuracy:
                degradation_report['passed'] = False
                degradation_report['checks'].append({
                    'metric': 'accuracy',
                    'threshold': min_accuracy,
                    'current_value': current_metrics['accuracy'],
                    'status': 'FAILED'
                })
            else:
                degradation_report['checks'].append({
                    'metric': 'accuracy',
                    'threshold': min_accuracy,
                    'current_value': current_metrics['accuracy'],
                    'status': 'PASSED'
                })

            # Check F1 score threshold
            min_f1 = self.config['evaluation']['min_f1_score']
            if current_metrics['f1_weighted'] < min_f1:
                degradation_report['passed'] = False
                degradation_report['checks'].append({
                    'metric': 'f1_weighted',
                    'threshold': min_f1,
                    'current_value': current_metrics['f1_weighted'],
                    'status': 'FAILED'
                })
            else:
                degradation_report['checks'].append({
                    'metric': 'f1_weighted',
                    'threshold': min_f1,
                    'current_value': current_metrics['f1_weighted'],
                    'status': 'PASSED'
                })

            # Compare with baseline if provided
            if baseline_metrics:
                accuracy_degradation = baseline_metrics['accuracy'] - current_metrics['accuracy']
                f1_degradation = baseline_metrics['f1_weighted'] - current_metrics['f1_weighted']

                degradation_report['checks'].append({
                    'metric': 'accuracy_vs_baseline',
                    'baseline_value': baseline_metrics['accuracy'],
                    'current_value': current_metrics['accuracy'],
                    'degradation': accuracy_degradation,
                    'status': 'DEGRADED' if accuracy_degradation > 0.05 else 'OK'
                })

                degradation_report['checks'].append({
                    'metric': 'f1_vs_baseline',
                    'baseline_value': baseline_metrics['f1_weighted'],
                    'current_value': current_metrics['f1_weighted'],
                    'degradation': f1_degradation,
                    'status': 'DEGRADED' if f1_degradation > 0.05 else 'OK'
                })

        else:  # regression
            # Check RMSE threshold
            max_rmse = self.config['evaluation']['max_rmse']
            if current_metrics['rmse'] > max_rmse:
                degradation_report['passed'] = False
                degradation_report['checks'].append({
                    'metric': 'rmse',
                    'threshold': max_rmse,
                    'current_value': current_metrics['rmse'],
                    'status': 'FAILED'
                })
            else:
                degradation_report['checks'].append({
                    'metric': 'rmse',
                    'threshold': max_rmse,
                    'current_value': current_metrics['rmse'],
                    'status': 'PASSED'
                })

            # Compare with baseline
            if baseline_metrics:
                rmse_degradation = current_metrics['rmse'] - baseline_metrics['rmse']
                degradation_report['checks'].append({
                    'metric': 'rmse_vs_baseline',
                    'baseline_value': baseline_metrics['rmse'],
                    'current_value': current_metrics['rmse'],
                    'degradation': rmse_degradation,
                    'status': 'DEGRADED' if rmse_degradation > 5.0 else 'OK'
                })

        # Log results
        if degradation_report['passed']:
            logger.info("✓ All performance checks PASSED")
        else:
            logger.warning("✗ Performance regression detected!")

        for check in degradation_report['checks']:
            status_symbol = "✓" if check['status'] in ['PASSED', 'OK'] else "✗"
            logger.info(f"{status_symbol} {check['metric']}: {check.get('current_value', 'N/A')}")

        return degradation_report['passed'], degradation_report

    def compare_model_iterations(self, model_type: str = 'multiclass'):
        """Compare performance across multiple model iterations"""
        logger.info(f"Comparing {model_type} model iterations...")

        models_dir = Path("models")
        if model_type == 'multiclass':
            pattern = "multiclass_iteration_*"
        else:
            pattern = "ranking_iteration_*"

        model_dirs = sorted(models_dir.glob(pattern))

        if len(model_dirs) < 2:
            logger.warning("Need at least 2 model iterations to compare")
            return None

        X_test, y_test = self.load_test_data()

        results = []
        for model_dir in model_dirs:
            model_path = model_dir / "model.pkl"
            if model_path.exists():
                model = self.load_model(model_path)

                if model_type == 'multiclass':
                    metrics = self.evaluate_classification_model(
                        model, X_test, y_test, model_name=model_dir.name
                    )
                else:
                    metrics = self.evaluate_regression_model(
                        model, X_test, y_test, model_name=model_dir.name
                    )

                results.append(metrics)

        # Save comparison
        output_dir = Path("evaluation_results") / model_type
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "model_comparison.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Model comparison saved to {output_dir / 'model_comparison.json'}")

        # Check for regression
        if len(results) >= 2:
            baseline = results[0]
            current = results[-1]

            passed, report = self.check_performance_regression(
                current, baseline,
                model_type='classification' if model_type == 'multiclass' else 'regression'
            )

            with open(output_dir / "regression_test_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)

        return results

    def generate_evaluation_report(self, model_type: str = 'multiclass'):
        """Generate comprehensive evaluation report"""
        logger.info(f"Generating evaluation report for {model_type} model...")

        results = self.compare_model_iterations(model_type)

        if not results:
            logger.error("No results to report")
            return

        output_dir = Path("evaluation_results") / model_type

        # Create summary report
        report = {
            'model_type': model_type,
            'num_iterations': len(results),
            'timestamp': datetime.now().isoformat(),
            'summary': {}
        }

        if model_type == 'multiclass':
            report['summary'] = {
                'best_iteration': max(results, key=lambda x: x['f1_weighted'])['model_name'],
                'best_accuracy': max(r['accuracy'] for r in results),
                'best_f1': max(r['f1_weighted'] for r in results),
                'avg_accuracy': np.mean([r['accuracy'] for r in results]),
                'avg_f1': np.mean([r['f1_weighted'] for r in results])
            }
        else:
            report['summary'] = {
                'best_iteration': min(results, key=lambda x: x['rmse'])['model_name'],
                'best_rmse': min(r['rmse'] for r in results),
                'best_r2': max(r['r2_score'] for r in results),
                'avg_rmse': np.mean([r['rmse'] for r in results]),
                'avg_r2': np.mean([r['r2_score'] for r in results])
            }

        with open(output_dir / "evaluation_summary.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {output_dir}")
        logger.info(f"Best iteration: {report['summary']['best_iteration']}")

        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model and check for regressions')
    parser.add_argument('--model_type', type=str, default='multiclass',
                        choices=['multiclass', 'ranking'],
                        help='Type of model to evaluate')

    args = parser.parse_args()

    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(model_type=args.model_type)

    if report:
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        for key, value in report['summary'].items():
            print(f"{key}: {value}")
