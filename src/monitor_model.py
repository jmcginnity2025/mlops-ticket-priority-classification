"""
Model Monitoring for Data Drift and Performance Degradation
"""
import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Handles model monitoring for data drift and performance issues"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize monitor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_baseline_data(self):
        """Load baseline training data for drift detection"""
        logger.info("Loading baseline data...")

        processed_path = Path(self.config['data']['processed_data_path'])
        versions = sorted([d for d in processed_path.iterdir() if d.is_dir()])

        if not versions:
            raise FileNotFoundError("No processed data found")

        baseline_dir = versions[0]  # Use first version as baseline

        X_train = pd.read_csv(baseline_dir / "X_train.csv")
        logger.info(f"Loaded baseline data: {X_train.shape}")

        return X_train

    def detect_numerical_drift(self, baseline_data: pd.DataFrame,
                               current_data: pd.DataFrame,
                               feature: str,
                               threshold: float = 0.05):
        """
        Detect drift in numerical features using Kolmogorov-Smirnov test

        Returns: (is_drifted, p_value, drift_score)
        """
        if feature not in baseline_data.columns or feature not in current_data.columns:
            logger.warning(f"Feature {feature} not found in data")
            return False, 1.0, 0.0

        baseline_values = baseline_data[feature].dropna()
        current_values = current_data[feature].dropna()

        # Perform KS test
        statistic, p_value = ks_2samp(baseline_values, current_values)

        is_drifted = p_value < threshold
        drift_score = statistic

        return is_drifted, p_value, drift_score

    def detect_categorical_drift(self, baseline_data: pd.DataFrame,
                                 current_data: pd.DataFrame,
                                 feature: str,
                                 threshold: float = 0.05):
        """
        Detect drift in categorical features using Chi-squared test

        Returns: (is_drifted, p_value, drift_score)
        """
        if feature not in baseline_data.columns or feature not in current_data.columns:
            logger.warning(f"Feature {feature} not found in data")
            return False, 1.0, 0.0

        # Get value counts
        baseline_counts = baseline_data[feature].value_counts()
        current_counts = current_data[feature].value_counts()

        # Align categories
        all_categories = set(baseline_counts.index) | set(current_counts.index)

        baseline_aligned = [baseline_counts.get(cat, 0) for cat in all_categories]
        current_aligned = [current_counts.get(cat, 0) for cat in all_categories]

        # Create contingency table
        contingency_table = np.array([baseline_aligned, current_aligned])

        # Perform chi-squared test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            is_drifted = p_value < threshold
            drift_score = chi2
        except:
            return False, 1.0, 0.0

        return is_drifted, p_value, drift_score

    def calculate_psi(self, baseline_data: pd.Series, current_data: pd.Series, bins: int = 10):
        """
        Calculate Population Stability Index (PSI)

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        # Create bins
        min_val = min(baseline_data.min(), current_data.min())
        max_val = max(baseline_data.max(), current_data.max())

        breakpoints = np.linspace(min_val, max_val, bins + 1)

        # Calculate distributions
        baseline_dist = np.histogram(baseline_data, bins=breakpoints)[0] / len(baseline_data)
        current_dist = np.histogram(current_data, bins=breakpoints)[0] / len(current_data)

        # Avoid division by zero
        baseline_dist = np.where(baseline_dist == 0, 0.0001, baseline_dist)
        current_dist = np.where(current_dist == 0, 0.0001, current_dist)

        # Calculate PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))

        return psi

    def run_data_drift_detection(self, current_data: pd.DataFrame):
        """
        Run comprehensive data drift detection

        Returns: drift_report
        """
        logger.info("Running data drift detection...")

        baseline_data = self.load_baseline_data()

        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'drift_threshold': self.config['monitoring']['data_drift']['drift_threshold'],
            'features_analyzed': 0,
            'features_drifted': 0,
            'overall_drift_detected': False,
            'feature_details': []
        }

        # Analyze numerical features
        numeric_features = self.config['data']['numeric_features']
        for feature in numeric_features:
            if feature in baseline_data.columns and feature in current_data.columns:
                is_drifted, p_value, drift_score = self.detect_numerical_drift(
                    baseline_data, current_data, feature
                )

                # Calculate PSI
                psi = self.calculate_psi(
                    baseline_data[feature].dropna(),
                    current_data[feature].dropna()
                )

                feature_info = {
                    'feature': feature,
                    'type': 'numerical',
                    'is_drifted': is_drifted,
                    'p_value': p_value,
                    'ks_statistic': drift_score,
                    'psi': psi,
                    'psi_interpretation': self._interpret_psi(psi)
                }

                drift_report['feature_details'].append(feature_info)
                drift_report['features_analyzed'] += 1

                if is_drifted or psi >= 0.2:
                    drift_report['features_drifted'] += 1
                    logger.warning(f"Drift detected in {feature}: PSI={psi:.4f}, p-value={p_value:.4f}")

        # Analyze categorical features
        categorical_features = self.config['data']['categorical_features']
        for feature in categorical_features:
            if feature in baseline_data.columns and feature in current_data.columns:
                is_drifted, p_value, drift_score = self.detect_categorical_drift(
                    baseline_data, current_data, feature
                )

                feature_info = {
                    'feature': feature,
                    'type': 'categorical',
                    'is_drifted': is_drifted,
                    'p_value': p_value,
                    'chi2_statistic': drift_score
                }

                drift_report['feature_details'].append(feature_info)
                drift_report['features_analyzed'] += 1

                if is_drifted:
                    drift_report['features_drifted'] += 1
                    logger.warning(f"Drift detected in {feature}: p-value={p_value:.4f}")

        # Overall drift assessment
        drift_percentage = drift_report['features_drifted'] / drift_report['features_analyzed']
        drift_report['drift_percentage'] = drift_percentage

        if drift_percentage > self.config['monitoring']['data_drift']['drift_threshold']:
            drift_report['overall_drift_detected'] = True
            logger.warning(f"OVERALL DATA DRIFT DETECTED: {drift_percentage:.2%} features drifted")
        else:
            logger.info(f"No significant overall drift: {drift_percentage:.2%} features drifted")

        # Save report
        self._save_drift_report(drift_report)

        return drift_report

    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI value"""
        if psi < 0.1:
            return "No significant change"
        elif psi < 0.2:
            return "Moderate change"
        else:
            return "Significant change"

    def _save_drift_report(self, report: dict):
        """Save drift detection report"""
        output_dir = Path("monitoring") / "drift_reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"drift_report_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Drift report saved to {report_path}")

    def monitor_model_performance(self, predictions: np.ndarray, actual: np.ndarray,
                                  model_type: str = 'classification'):
        """
        Monitor model performance and detect degradation

        Returns: performance_report
        """
        logger.info("Monitoring model performance...")

        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'num_predictions': len(predictions),
            'metrics': {},
            'degradation_detected': False,
            'alerts': []
        }

        if model_type == 'classification':
            # Calculate metrics
            accuracy = accuracy_score(actual, predictions)
            f1 = f1_score(actual, predictions, average='weighted')

            performance_report['metrics'] = {
                'accuracy': accuracy,
                'f1_weighted': f1
            }

            # Check thresholds
            min_accuracy = self.config['monitoring']['model_performance']['accuracy_threshold']
            min_f1 = self.config['monitoring']['model_performance']['f1_threshold']

            if accuracy < min_accuracy:
                performance_report['degradation_detected'] = True
                performance_report['alerts'].append({
                    'metric': 'accuracy',
                    'current_value': accuracy,
                    'threshold': min_accuracy,
                    'message': f"Accuracy {accuracy:.4f} below threshold {min_accuracy}"
                })
                logger.warning(f"Performance degradation: Accuracy {accuracy:.4f} < {min_accuracy}")

            if f1 < min_f1:
                performance_report['degradation_detected'] = True
                performance_report['alerts'].append({
                    'metric': 'f1_score',
                    'current_value': f1,
                    'threshold': min_f1,
                    'message': f"F1 Score {f1:.4f} below threshold {min_f1}"
                })
                logger.warning(f"Performance degradation: F1 {f1:.4f} < {min_f1}")

        else:  # regression
            from sklearn.metrics import mean_squared_error, r2_score

            rmse = np.sqrt(mean_squared_error(actual, predictions))
            r2 = r2_score(actual, predictions)

            performance_report['metrics'] = {
                'rmse': rmse,
                'r2_score': r2
            }

            max_rmse = self.config['evaluation']['max_rmse']

            if rmse > max_rmse:
                performance_report['degradation_detected'] = True
                performance_report['alerts'].append({
                    'metric': 'rmse',
                    'current_value': rmse,
                    'threshold': max_rmse,
                    'message': f"RMSE {rmse:.4f} exceeds threshold {max_rmse}"
                })

        # Save performance report
        self._save_performance_report(performance_report)

        return performance_report

    def _save_performance_report(self, report: dict):
        """Save performance monitoring report"""
        output_dir = Path("monitoring") / "performance_reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"performance_report_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report saved to {report_path}")

    def should_trigger_retraining(self, drift_report: dict = None,
                                  performance_report: dict = None) -> tuple:
        """
        Determine if model retraining should be triggered

        Returns: (should_retrain, reasons)
        """
        logger.info("Evaluating retraining triggers...")

        should_retrain = False
        reasons = []

        retraining_config = self.config['retraining']['trigger_conditions']

        # Check data drift
        if drift_report and drift_report.get('overall_drift_detected'):
            drift_pct = drift_report.get('drift_percentage', 0)
            threshold = retraining_config['data_drift_threshold']

            if drift_pct > threshold:
                should_retrain = True
                reasons.append(f"Data drift detected: {drift_pct:.2%} > {threshold:.2%}")

        # Check performance degradation
        if performance_report and performance_report.get('degradation_detected'):
            should_retrain = True
            reasons.append("Performance degradation detected")

            for alert in performance_report.get('alerts', []):
                reasons.append(f"  - {alert['message']}")

        # Log decision
        if should_retrain:
            logger.warning("RETRAINING TRIGGERED")
            for reason in reasons:
                logger.warning(f"  {reason}")
        else:
            logger.info("No retraining needed")

        return should_retrain, reasons

    def generate_monitoring_dashboard_data(self):
        """Generate data for monitoring dashboard"""
        logger.info("Generating monitoring dashboard data...")

        # Collect latest reports
        drift_reports_dir = Path("monitoring") / "drift_reports"
        performance_reports_dir = Path("monitoring") / "performance_reports"

        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'drift_summary': None,
            'performance_summary': None,
            'retraining_recommendation': None
        }

        # Latest drift report
        if drift_reports_dir.exists():
            drift_reports = sorted(drift_reports_dir.glob("*.json"))
            if drift_reports:
                with open(drift_reports[-1], 'r') as f:
                    dashboard_data['drift_summary'] = json.load(f)

        # Latest performance report
        if performance_reports_dir.exists():
            perf_reports = sorted(performance_reports_dir.glob("*.json"))
            if perf_reports:
                with open(perf_reports[-1], 'r') as f:
                    dashboard_data['performance_summary'] = json.load(f)

        # Retraining recommendation
        should_retrain, reasons = self.should_trigger_retraining(
            dashboard_data['drift_summary'],
            dashboard_data['performance_summary']
        )

        dashboard_data['retraining_recommendation'] = {
            'should_retrain': should_retrain,
            'reasons': reasons
        }

        # Save dashboard data
        output_path = Path("monitoring") / "dashboard_data.json"
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        logger.info(f"Dashboard data saved to {output_path}")

        return dashboard_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor model for drift and performance')
    parser.add_argument('--action', type=str, required=True,
                        choices=['drift', 'performance', 'dashboard'],
                        help='Monitoring action to perform')
    parser.add_argument('--current_data', type=str,
                        help='Path to current data CSV (for drift detection)')

    args = parser.parse_args()

    monitor = ModelMonitor()

    if args.action == 'drift':
        if not args.current_data:
            print("Error: --current_data required for drift detection")
            exit(1)

        current_data = pd.read_csv(args.current_data)
        report = monitor.run_data_drift_detection(current_data)

        print("\n" + "="*50)
        print("DATA DRIFT DETECTION RESULTS")
        print("="*50)
        print(f"Features analyzed: {report['features_analyzed']}")
        print(f"Features drifted: {report['features_drifted']}")
        print(f"Drift percentage: {report['drift_percentage']:.2%}")
        print(f"Overall drift: {'YES' if report['overall_drift_detected'] else 'NO'}")

    elif args.action == 'dashboard':
        dashboard = monitor.generate_monitoring_dashboard_data()
        print("\n" + "="*50)
        print("MONITORING DASHBOARD")
        print("="*50)
        if dashboard['retraining_recommendation']['should_retrain']:
            print("⚠ RETRAINING RECOMMENDED")
            for reason in dashboard['retraining_recommendation']['reasons']:
                print(f"  {reason}")
        else:
            print("✓ Model performing well, no action needed")
