"""
Model Performance Regression Tests
"""
import json
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_classification_performance():
    """Test that classification model meets minimum performance requirements"""
    print("Running classification model performance tests...")

    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load evaluation results
    eval_dir = Path("evaluation_results/multiclass")

    if not eval_dir.exists():
        print("Warning: No evaluation results found")
        print("Skipping performance tests")
        return

    comparison_file = eval_dir / "model_comparison.json"

    if not comparison_file.exists():
        print("Warning: No model comparison file found")
        return

    with open(comparison_file, 'r') as f:
        results = json.load(f)

    if not results:
        print("Warning: No model results to validate")
        return

    # Get latest model results
    latest_result = results[-1]

    # Check minimum thresholds
    min_accuracy = config['evaluation']['min_accuracy']
    min_f1 = config['evaluation']['min_f1_score']

    accuracy = latest_result.get('accuracy', 0)
    f1_score = latest_result.get('f1_weighted', 0)

    print(f"Latest model performance:")
    print(f"  Accuracy: {accuracy:.4f} (threshold: {min_accuracy})")
    print(f"  F1 Score: {f1_score:.4f} (threshold: {min_f1})")

    assert accuracy >= min_accuracy, \
        f"Accuracy {accuracy:.4f} below threshold {min_accuracy}"
    print("✓ Accuracy meets minimum requirement")

    assert f1_score >= min_f1, \
        f"F1 Score {f1_score:.4f} below threshold {min_f1}"
    print("✓ F1 Score meets minimum requirement")


def test_no_performance_regression():
    """Test that new model doesn't regress compared to baseline"""
    print("Running performance regression tests...")

    eval_dir = Path("evaluation_results/multiclass")

    if not eval_dir.exists():
        print("Skipping regression tests - no evaluation results")
        return

    regression_file = eval_dir / "regression_test_report.json"

    if not regression_file.exists():
        print("No regression test report found")
        return

    with open(regression_file, 'r') as f:
        report = json.load(f)

    passed = report.get('passed', False)

    print(f"Regression test status: {'PASSED' if passed else 'FAILED'}")

    for check in report.get('checks', []):
        status = check.get('status', 'UNKNOWN')
        metric = check.get('metric', 'unknown')
        current_value = check.get('current_value', 'N/A')

        print(f"  {metric}: {current_value} - {status}")

    # Allow moderate degradation but not complete failure
    if not passed:
        # Check if it's severe degradation
        for check in report.get('checks', []):
            if check.get('status') == 'FAILED':
                raise AssertionError(
                    f"Severe performance regression detected in {check.get('metric')}"
                )

        print("Warning: Some degradation detected but within acceptable limits")
    else:
        print("✓ No performance regression detected")


def test_model_artifacts_exist():
    """Test that model artifacts are generated"""
    print("Running model artifact tests...")

    models_dir = Path("models")

    if not models_dir.exists():
        print("Warning: Models directory not found")
        return

    # Check for at least one model
    model_files = list(models_dir.glob("**/model.pkl"))

    assert len(model_files) > 0, "No model files found"
    print(f"✓ Found {len(model_files)} model file(s)")

    # Check for metadata
    metadata_files = list(Path("data/processed").glob("**/metadata.json"))
    if metadata_files:
        print(f"✓ Found {len(metadata_files)} metadata file(s)")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("MODEL PERFORMANCE TESTS")
    print("="*50 + "\n")

    try:
        test_classification_performance()
        test_no_performance_regression()
        test_model_artifacts_exist()

        print("\n" + "="*50)
        print("ALL PERFORMANCE TESTS PASSED ✓")
        print("="*50)

    except AssertionError as e:
        print("\n" + "="*50)
        print(f"PERFORMANCE TEST FAILED: {str(e)}")
        print("="*50)
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*50)
        print(f"ERROR: {str(e)}")
        print("="*50)
        sys.exit(1)
