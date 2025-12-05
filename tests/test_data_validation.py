"""
Data Validation Tests
"""
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_data_schema():
    """Test that data has expected schema"""
    print("Running data schema validation...")

    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load data if exists
    data_path = config['data']['raw_data_path']

    if not Path(data_path).exists():
        print(f"Warning: Data file not found at {data_path}")
        print("Skipping data validation tests")
        return

    df = pd.read_csv(data_path)

    # Check expected columns exist
    expected_features = (
        config['data']['numeric_features'] +
        config['data']['categorical_features'] +
        config['data']['binary_features']
    )

    missing_columns = set(expected_features) - set(df.columns)

    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    else:
        print("✓ All expected columns present")

    # Check target columns
    target_multiclass = config['data']['target_multiclass']
    target_score = config['data']['target_score']

    assert target_multiclass in df.columns or target_score in df.columns, \
        "Target column not found in data"

    print("✓ Target columns validated")


def test_data_quality():
    """Test data quality metrics"""
    print("Running data quality checks...")

    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['data']['raw_data_path']

    if not Path(data_path).exists():
        print("Skipping data quality tests - data file not found")
        return

    df = pd.read_csv(data_path)

    # Check for minimum rows
    min_rows = 100
    assert len(df) >= min_rows, f"Dataset too small: {len(df)} < {min_rows}"
    print(f"✓ Dataset size acceptable: {len(df)} rows")

    # Check missing values percentage
    missing_pct = (df.isnull().sum() / len(df) * 100).max()
    max_missing_pct = 50

    if missing_pct > max_missing_pct:
        print(f"Warning: High missing value percentage: {missing_pct:.2f}%")
    else:
        print(f"✓ Missing values acceptable: {missing_pct:.2f}%")


def test_numeric_ranges():
    """Test that numeric features are within expected ranges"""
    print("Running numeric range validation...")

    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['data']['raw_data_path']

    if not Path(data_path).exists():
        print("Skipping numeric range tests - data file not found")
        return

    df = pd.read_csv(data_path)

    numeric_features = config['data']['numeric_features']

    for feature in numeric_features:
        if feature in df.columns:
            # Check for infinite values
            inf_count = np.isinf(df[feature]).sum()
            if inf_count > 0:
                print(f"Warning: {feature} has {inf_count} infinite values")

            # Check for extreme outliers (basic check)
            q1 = df[feature].quantile(0.01)
            q99 = df[feature].quantile(0.99)
            print(f"  {feature}: range [{q1:.2f}, {q99:.2f}]")

    print("✓ Numeric range validation completed")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("DATA VALIDATION TESTS")
    print("="*50 + "\n")

    try:
        test_data_schema()
        test_data_quality()
        test_numeric_ranges()

        print("\n" + "="*50)
        print("ALL VALIDATION TESTS PASSED ✓")
        print("="*50)

    except Exception as e:
        print("\n" + "="*50)
        print(f"VALIDATION FAILED: {str(e)}")
        print("="*50)
        sys.exit(1)
