"""
Data Preprocessing Pipeline for Support Ticket Priority Classification
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import os
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        logger.info(f"Loading data from {self.config['data']['raw_data_path']}")

        try:
            df = pd.read_csv(self.config['data']['raw_data_path'])
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.config['data']['raw_data_path']}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("Cleaning data...")

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {original_len - len(df)} duplicate records")

        # Handle missing values
        numeric_features = self.config['data']['numeric_features']
        categorical_features = self.config['data']['categorical_features']
        binary_features = self.config['data']['binary_features']

        # Fill numeric missing values with median
        for col in numeric_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_val}")

        # Fill categorical missing values with mode
        for col in categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_val}")

        # Fill binary features with 0
        for col in binary_features:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(0, inplace=True)

        return df

    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")

        categorical_features = self.config['data']['categorical_features']

        for col in categorical_features:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    else:
                        logger.warning(f"No encoder found for {col}, skipping...")

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("Scaling numeric features...")

        numeric_features = self.config['data']['numeric_features']
        available_features = [f for f in numeric_features if f in df.columns]

        if fit:
            df[available_features] = self.scaler.fit_transform(df[available_features])
        else:
            df[available_features] = self.scaler.transform(df[available_features])

        return df

    def prepare_features_target(self, df: pd.DataFrame, task_type: str = 'multiclass'):
        """Prepare features and target variables"""
        logger.info(f"Preparing features and target for {task_type} task...")

        # Get all feature columns
        feature_cols = (
            self.config['data']['numeric_features'] +
            self.config['data']['categorical_features'] +
            self.config['data']['binary_features']
        )

        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Get target
        if task_type == 'multiclass':
            target_col = self.config['data']['target_multiclass']
        else:
            target_col = self.config['data']['target_score']

        X = df[feature_cols]
        y = df[target_col]

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    def split_data(self, X, y):
        """Split data into train and test sets"""
        test_size = self.config['data']['train_test_split']
        random_seed = self.config['data']['random_seed']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test, version: str = None):
        """Save processed data with versioning"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = Path(self.config['data']['processed_data_path']) / version
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        X_train.to_csv(output_dir / "X_train.csv", index=False)
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
        y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)

        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
            'class_distribution_train': y_train.value_counts().to_dict(),
            'class_distribution_test': y_test.value_counts().to_dict()
        }

        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved processed data to {output_dir}")

        return output_dir

    def run_pipeline(self, task_type: str = 'multiclass'):
        """Run complete preprocessing pipeline"""
        logger.info("Starting data preprocessing pipeline...")

        # Load data
        df = self.load_data()

        # Clean data
        df = self.clean_data(df)

        # Encode categorical features
        df = self.encode_features(df, fit=True)

        # Prepare features and target
        X, y = self.prepare_features_target(df, task_type=task_type)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Scale features
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # Save processed data
        output_dir = self.save_processed_data(X_train, X_test, y_train, y_test)

        logger.info("Data preprocessing pipeline completed successfully!")

        return X_train, X_test, y_train, y_test, output_dir


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, output_dir = preprocessor.run_pipeline(task_type='multiclass')
    print(f"\nProcessed data saved to: {output_dir}")
