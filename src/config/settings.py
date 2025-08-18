"""Configuration settings for the loan default prediction system."""

import os
from pathlib import Path
from typing import Dict, Any
import yaml

class Settings:
    """Main application settings."""
    
    def __init__(self):
        # Project paths
        self.project_root = Path(__file__).parent.parent.parent
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        
        # Database
        self.database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/loan_default_monitoring")
        
        # MLflow
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.mlflow_experiment_name = "loan_default_prediction"
        self.mlflow_model_name = "loan_default_classifier"
        
        # API
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
    
    @property
    def data_paths(self) -> Dict[str, Path]:
        """Get data paths."""
        return {
            "raw": self.project_root / "data" / "raw",
            "processed": self.project_root / "data" / "processed",
            "features": self.project_root / "data" / "features",
        }
    
    @property
    def model_paths(self) -> Dict[str, Path]:
        """Get model paths."""
        return {
            "models": self.project_root / "models",
            "metrics": self.project_root / "metrics",
            "plots": self.project_root / "plots",
        }
    def get_data_params(self):
        """Get data processing parameters."""
        return {
            "test_size": 0.2,
            "validation_size": 0.2,
            "random_state": 42,
            "target_column": "class",
            "categorical_features": [
                "Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute7",
                "Attribute9", "Attribute10", "Attribute12", "Attribute14", "Attribute15",
                "Attribute17", "Attribute19", "Attribute20"
            ],
            "numerical_features": [
                "Attribute2", "Attribute5", "Attribute8", "Attribute11", 
                "Attribute13", "Attribute16", "Attribute18"
            ]
        }
    def get_training_params(self):
        """Get model training parameters."""
        return {
            "algorithm": "RandomForestClassifier",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "n_jobs": -1
            },
            "cv_folds": 5,
            "scoring_metric": "roc_auc"
        }
    
    def get_evaluation_params(self):
        """Get model evaluation parameters."""
        return {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            "threshold": 0.5,
            "retraining_thresholds": {
                "min_auc": 0.75,
                "min_accuracy": 0.70
            }
        }
    
    def ensure_directories(self):
        """Ensure all necessary directories exist."""
        # Create data directories
        for path in self.data_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Create model directories  
        for path in self.model_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directories created successfully")

# Global settings instance
settings = Settings()

def get_settings():
    """Get the global settings instance."""
    return settings