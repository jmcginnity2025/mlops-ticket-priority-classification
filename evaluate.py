"""
Model Evaluation Script with Regression Testing
Compares new model against previous baseline
FAILS pipeline if new model performs worse
"""
import json
import os
import sys
from datetime import datetime

MODELS_DIR = "models"
EVAL_DIR = "evaluation_results"

# Minimum thresholds for acceptable model performance
MIN_ACCURACY = 0.70  # 70%
MIN_F1_SCORE = 0.65  # 65%

def load_metrics(iteration):
    """Load metrics for a specific iteration"""
    metrics_path = f"{MODELS_DIR}/iteration_{iteration}/metrics.json"

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, 'r') as f:
        return json.load(f)

def compare_models(current_metrics, baseline_metrics):
    """
    Compare current model with baseline
    Returns: (passed, comparison_report)
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    # Extract key metrics
    current_acc = current_metrics['test_accuracy']
    current_f1 = current_metrics['test_f1']
    baseline_acc = baseline_metrics['test_accuracy']
    baseline_f1 = baseline_metrics['test_f1']

    # Calculate differences
    acc_diff = current_acc - baseline_acc
    f1_diff = current_f1 - baseline_f1

    print(f"\nBaseline Model (Iteration {baseline_metrics['iteration']}):")
    print(f"  Test Accuracy: {baseline_acc:.4f}")
    print(f"  Test F1 Score: {baseline_f1:.4f}")

    print(f"\nCurrent Model (Iteration {current_metrics['iteration']}):")
    print(f"  Test Accuracy: {current_acc:.4f}")
    print(f"  Test F1 Score: {current_f1:.4f}")

    print(f"\nDifference:")
    print(f"  Accuracy:  {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
    print(f"  F1 Score:  {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")

    # Determine if regression occurred
    regression_threshold = -0.02  # Allow up to 2% drop
    passed = True
    reasons = []

    if acc_diff < regression_threshold:
        passed = False
        reasons.append(f"Accuracy dropped by {abs(acc_diff*100):.2f}% (threshold: {abs(regression_threshold*100):.2f}%)")

    if f1_diff < regression_threshold:
        passed = False
        reasons.append(f"F1 Score dropped by {abs(f1_diff*100):.2f}% (threshold: {abs(regression_threshold*100):.2f}%)")

    # Create comparison report
    comparison_report = {
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'iteration': baseline_metrics['iteration'],
            'model_type': baseline_metrics['model_type'],
            'test_accuracy': baseline_acc,
            'test_f1': baseline_f1
        },
        'current': {
            'iteration': current_metrics['iteration'],
            'model_type': current_metrics['model_type'],
            'test_accuracy': current_acc,
            'test_f1': current_f1
        },
        'differences': {
            'accuracy': float(acc_diff),
            'f1_score': float(f1_diff)
        },
        'regression_threshold': regression_threshold,
        'passed': passed,
        'failure_reasons': reasons if not passed else []
    }

    return passed, comparison_report

def evaluate_single_model(metrics):
    """
    Evaluate a single model against absolute thresholds
    Used for first commit when there's no baseline
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION (First Commit - No Baseline)")
    print("="*70)

    test_acc = metrics['test_accuracy']
    test_f1 = metrics['test_f1']

    print(f"\nModel Performance:")
    print(f"  Test Accuracy: {test_acc:.4f} (min: {MIN_ACCURACY:.2f})")
    print(f"  Test F1 Score: {test_f1:.4f} (min: {MIN_F1_SCORE:.2f})")

    passed = True
    reasons = []

    if test_acc < MIN_ACCURACY:
        passed = False
        reasons.append(f"Accuracy {test_acc:.4f} below minimum {MIN_ACCURACY:.2f}")

    if test_f1 < MIN_F1_SCORE:
        passed = False
        reasons.append(f"F1 Score {test_f1:.4f} below minimum {MIN_F1_SCORE:.2f}")

    evaluation_report = {
        'timestamp': datetime.now().isoformat(),
        'model': {
            'iteration': metrics['iteration'],
            'model_type': metrics['model_type'],
            'test_accuracy': test_acc,
            'test_f1': test_f1
        },
        'thresholds': {
            'min_accuracy': MIN_ACCURACY,
            'min_f1_score': MIN_F1_SCORE
        },
        'passed': passed,
        'failure_reasons': reasons if not passed else []
    }

    return passed, evaluation_report

def main():
    """Main evaluation function"""
    print("="*70)
    print("MODEL EVALUATION & REGRESSION TESTING")
    print("="*70)

    # Create evaluation directory
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Load metrics for both iterations
    iter1_metrics = load_metrics(1)
    iter2_metrics = load_metrics(2)

    # Determine evaluation strategy
    if iter2_metrics is None:
        print("\nERROR: No metrics found for iteration 2")
        print("Make sure training has been completed.")
        sys.exit(1)

    # Compare iteration 2 against iteration 1 (baseline)
    if iter1_metrics is not None:
        # We have a baseline - compare models
        passed, report = compare_models(iter2_metrics, iter1_metrics)
    else:
        # No baseline - evaluate standalone
        passed, report = evaluate_single_model(iter2_metrics)

    # Save evaluation report
    report_path = f"{EVAL_DIR}/evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation report saved to: {report_path}")

    # Print final result
    print("\n" + "="*70)
    if passed:
        print("EVALUATION RESULT: PASSED")
        print("="*70)
        print("\nThe model meets performance requirements.")
        sys.exit(0)
    else:
        print("EVALUATION RESULT: FAILED")
        print("="*70)
        print("\nREGRESSION DETECTED!")
        print("\nFailure reasons:")
        for reason in report['failure_reasons']:
            print(f"  - {reason}")
        print("\nThe pipeline will be stopped.")
        sys.exit(1)

if __name__ == "__main__":
    main()
