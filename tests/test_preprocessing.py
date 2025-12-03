"""
Unit tests for data preprocessing pipeline
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return DataPreprocessor()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)

        data = {
            # Numeric features
            'org_users': np.random.randint(10, 1000, 100),
            'past_30d_tickets': np.random.randint(0, 50, 100),
            'customers_affected': np.random.randint(1, 100, 100),
            'error_rate_pct': np.random.uniform(0, 100, 100),
            'downtime_min': np.random.randint(0, 500, 100),
            'description_length': np.random.randint(50, 500, 100),
            'resolution_time_hours': np.random.uniform(0.5, 48, 100),
            'customer_satisfaction_score': np.random.uniform(1, 5, 100),
            'revenue_dollars': np.random.uniform(1000, 100000, 100),
            'api_calls_per_day': np.random.randint(100, 10000, 100),
            'team_size': np.random.randint(1, 20, 100),
            'satisfaction_score_0_10': np.random.randint(0, 11, 100),

            # Categorical features
            'day_of_week_num': np.random.randint(0, 7, 100),
            'company_size_cat': np.random.choice([1, 2, 3], 100),
            'industry_cat': np.random.choice([1, 2, 3, 4], 100),
            'customer_tier_cat': np.random.choice([1, 2, 3], 100),
            'region_cat': np.random.choice([1, 2, 3, 4], 100),
            'product_area_cat': np.random.choice([1, 2, 3], 100),
            'booking_channel_cat': np.random.choice([1, 2], 100),
            'reported_by_role_cat': np.random.choice([1, 2, 3], 100),
            'customer_sentiment_cat': np.random.choice([1, 2, 3], 100),

            # Binary features
            'payment_impact_flag': np.random.choice([0, 1], 100),
            'data_loss_flag': np.random.choice([0, 1], 100),
            'has_runbook': np.random.choice([0, 1], 100),

            # Target
            'priority_cat': np.random.choice([1, 2, 3], 100),
            'priority_score_internal': np.random.uniform(0, 100, 100)
        }

        return pd.DataFrame(data)

    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initializes correctly"""
        assert preprocessor is not None
        assert preprocessor.config is not None
        assert preprocessor.scaler is not None
        assert isinstance(preprocessor.label_encoders, dict)

    def test_clean_data_removes_duplicates(self, preprocessor, sample_data):
        """Test that clean_data removes duplicate rows"""
        # Add duplicate rows
        data_with_dupes = pd.concat([sample_data, sample_data.iloc[:5]], ignore_index=True)

        cleaned = preprocessor.clean_data(data_with_dupes)

        assert len(cleaned) < len(data_with_dupes)
        assert cleaned.duplicated().sum() == 0

    def test_clean_data_handles_missing_values(self, preprocessor, sample_data):
        """Test that clean_data handles missing values"""
        # Introduce missing values
        sample_data.loc[0:5, 'org_users'] = np.nan
        sample_data.loc[10:15, 'company_size_cat'] = np.nan

        cleaned = preprocessor.clean_data(sample_data)

        # Check no missing values remain
        assert cleaned['org_users'].isnull().sum() == 0
        assert cleaned['company_size_cat'].isnull().sum() == 0

    def test_encode_features(self, preprocessor, sample_data):
        """Test feature encoding"""
        encoded = preprocessor.encode_features(sample_data.copy(), fit=True)

        # Check that encoders were created
        categorical_features = preprocessor.config['data']['categorical_features']
        for feature in categorical_features:
            if feature in sample_data.columns:
                assert feature in preprocessor.label_encoders

        # Check that values are numeric
        for feature in categorical_features:
            if feature in encoded.columns:
                assert pd.api.types.is_numeric_dtype(encoded[feature])

    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling"""
        scaled = preprocessor.scale_features(sample_data.copy(), fit=True)

        numeric_features = preprocessor.config['data']['numeric_features']
        available_features = [f for f in numeric_features if f in sample_data.columns]

        # Check that scaled values have mean close to 0 and std close to 1
        for feature in available_features:
            mean = scaled[feature].mean()
            std = scaled[feature].std()

            assert abs(mean) < 0.1, f"{feature} mean {mean} not close to 0"
            assert abs(std - 1.0) < 0.1, f"{feature} std {std} not close to 1"

    def test_prepare_features_target(self, preprocessor, sample_data):
        """Test feature and target preparation"""
        X, y = preprocessor.prepare_features_target(sample_data, task_type='multiclass')

        # Check shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)

        # Check target is correct
        assert y.name == preprocessor.config['data']['target_multiclass']

    def test_split_data(self, preprocessor, sample_data):
        """Test train/test split"""
        X, y = preprocessor.prepare_features_target(sample_data, task_type='multiclass')

        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        # Check split ratio
        test_size = preprocessor.config['data']['train_test_split']
        expected_test_size = int(len(X) * test_size)

        assert len(X_test) == expected_test_size
        assert len(y_test) == expected_test_size

        # Check no overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert len(train_indices.intersection(test_indices)) == 0

    def test_data_types_after_preprocessing(self, preprocessor, sample_data):
        """Test that data types are correct after preprocessing"""
        # Encode and scale
        encoded = preprocessor.encode_features(sample_data.copy(), fit=True)
        scaled = preprocessor.scale_features(encoded, fit=True)

        # All values should be numeric
        for column in scaled.columns:
            assert pd.api.types.is_numeric_dtype(scaled[column]), \
                f"Column {column} is not numeric after preprocessing"

    def test_no_data_leakage_in_scaling(self, preprocessor, sample_data):
        """Test that scaling doesn't cause data leakage"""
        X, y = preprocessor.prepare_features_target(sample_data, task_type='multiclass')
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        # Fit on train
        X_train_scaled = pd.DataFrame(
            preprocessor.scaler.fit_transform(X_train),
            columns=X_train.columns
        )

        # Transform test with same scaler
        X_test_scaled = pd.DataFrame(
            preprocessor.scaler.transform(X_test),
            columns=X_test.columns
        )

        # Test mean and std might not be exactly 0 and 1
        # (since fitted on train only)
        test_means = X_test_scaled.mean()

        # The test set should have different statistics than training
        # (not exactly 0 mean, 1 std if distributions differ)
        assert not all(abs(test_means) < 0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
