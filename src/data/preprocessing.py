"""Data preprocessing module for loan default prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


class DataPreprocessor:
    """Data preprocessing pipeline for loan default data."""
    
    def __init__(self):
        self.settings = settings
        self.params = settings.get_data_params()
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load raw data from CSV files."""
        logger.info("Loading raw data...")
        
        # Load features and target
        X_path = self.settings.data_paths["raw"] / "X.csv"
        y_path = self.settings.data_paths["raw"] / "y.csv"
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Raw data files not found at {X_path} or {y_path}")
        
        X = pd.read_csv(X_path)
        y_df = pd.read_csv(y_path)
        
        # Extract target column (assuming first column is target)
        target_col = y_df.columns[0]
        y = y_df[target_col]
        
        logger.info(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
        logger.info(f"Target column: {target_col}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline based on feature types."""
        logger.info("Creating preprocessing pipeline...")
        
        # Get feature lists from parameters
        categorical_features = self.params.get("categorical_features", [])
        numerical_features = self.params.get("numerical_features", [])
        
        # Verify features exist in data
        available_features = X.columns.tolist()
        categorical_features = [f for f in categorical_features if f in available_features]
        numerical_features = [f for f in numerical_features if f in available_features]
        
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")
        
        # Define transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any features not specified
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def prepare_target(self, y: pd.Series) -> np.ndarray:
        """Prepare target variable for training."""
        logger.info("Preparing target variable...")
        
        # Create label encoder if not exists
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
        
        logger.info(f"Target encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return y_encoded
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data...")
        
        test_size = self.params.get("test_size", 0.2)
        validation_size = self.params.get("validation_size", 0.2)
        random_state = self.params.get("random_state", 42)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data split sizes:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples") 
        logger.info(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def generate_data_quality_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Generate data quality metrics."""
        logger.info("Generating data quality metrics...")
        
        metrics = {
            "dataset_info": {
                "n_samples": int(len(X)),
                "n_features": int(X.shape[1]),
                "target_distribution": y.value_counts().to_dict(),
                "missing_values_per_feature": X.isnull().sum().to_dict(),
                "duplicate_rows": int(X.duplicated().sum()),
                "data_types": X.dtypes.astype(str).to_dict()
            },
            "feature_statistics": {
                "numerical_features_stats": X.select_dtypes(include=[np.number]).describe().to_dict(),
                "categorical_features_counts": {
                    col: X[col].value_counts().head(10).to_dict() 
                    for col in X.select_dtypes(include=['object']).columns
                }
            },
            "quality_checks": {
                "missing_value_percentage": float((X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100),
                "duplicate_percentage": float((X.duplicated().sum() / len(X)) * 100),
                "target_class_balance": {
                    "majority_class_ratio": float(y.value_counts().max() / len(y)),
                    "minority_class_ratio": float(y.value_counts().min() / len(y))
                }
            }
        }
        
        return metrics
    
    def save_processed_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> None:
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        processed_path = self.settings.data_paths["processed"]
        
        # Combine features and target for saving
        train_df = X_train.copy()
        train_df[self.params.get("target_column", "class")] = y_train
        
        val_df = X_val.copy()
        val_df[self.params.get("target_column", "class")] = y_val
        
        test_df = X_test.copy()
        test_df[self.params.get("target_column", "class")] = y_test
        
        # Save datasets
        train_df.to_csv(processed_path / "train.csv", index=False)
        val_df.to_csv(processed_path / "validation.csv", index=False)
        test_df.to_csv(processed_path / "test.csv", index=False)
        
        logger.info(f"Saved processed data to {processed_path}")
    
    def save_preprocessor(self) -> None:
        """Save the preprocessing pipeline."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call create_preprocessing_pipeline first.")
        
        models_path = self.settings.model_paths["models"]
        
        # Save preprocessor and label encoder
        joblib.dump(self.preprocessor, models_path / "preprocessor.joblib")
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, models_path / "label_encoder.joblib")
        
        logger.info(f"Saved preprocessor to {models_path}")
    
    def process_data(self) -> None:
        """Main data processing pipeline."""
        logger.info("Starting data preprocessing pipeline...")
        
        try:
            # Load raw data
            X, y = self.load_raw_data()
            
            # Generate data quality metrics
            metrics = self.generate_data_quality_metrics(X, y)
            
            # Save metrics with numpy type conversion
            metrics_path = self.settings.model_paths["metrics"]
            with open(metrics_path / "data_quality.json", "w") as f:
                json.dump(convert_numpy_types(metrics), f, indent=2)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Create and fit preprocessing pipeline
            preprocessor = self.create_preprocessing_pipeline(X_train)
            
            # Prepare target variable
            y_train_encoded = self.prepare_target(y_train)
            y_val_encoded = self.prepare_target(y_val)
            y_test_encoded = self.prepare_target(y_test)
            
            # Save processed data
            self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Save preprocessor
            self.save_preprocessor()
            
            logger.info("Data preprocessing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise


def main():
    """Main function for running data preprocessing."""
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.process_data()


if __name__ == "__main__":
    main()