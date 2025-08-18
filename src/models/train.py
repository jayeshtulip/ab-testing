"""Model training module with MLflow integration."""
import joblib
import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ModelTrainer:
    """Model training class with MLflow tracking."""
    
    def __init__(self):
        self.settings = settings
        
        # Initialize MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        
    def load_processed_data(self):
        """Load processed training data."""
        logger.info("Loading processed data...")
        
        processed_path = settings.data_paths["processed"]
        
        # Load datasets
        train_df = pd.read_csv(processed_path / "train.csv")
        val_df = pd.read_csv(processed_path / "validation.csv")
        test_df = pd.read_csv(processed_path / "test.csv")
        
        # Separate features and target
        target_col = "class"
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        logger.info(f"Loaded data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_preprocessor(self):
        """Load the fitted preprocessor."""
        logger.info("Loading preprocessor...")
        
        models_path = settings.model_paths["models"]
        
        # Load preprocessor and label encoder
        self.preprocessor = joblib.load(models_path / "preprocessor.joblib")
        
        if (models_path / "label_encoder.joblib").exists():
            self.label_encoder = joblib.load(models_path / "label_encoder.joblib")
        
        logger.info("Preprocessor loaded successfully")
    
    def create_model(self):
        """Create the machine learning model."""
        logger.info("Creating model...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Created RandomForestClassifier")
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model and return validation metrics."""
        logger.info("Training model...")
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_val_pred)),
            "precision": float(precision_score(y_val, y_val_pred, average='weighted')),
            "recall": float(recall_score(y_val, y_val_pred, average='weighted')),
            "f1_score": float(f1_score(y_val, y_val_pred, average='weighted')),
            "roc_auc": float(roc_auc_score(y_val, y_val_proba))
        }
        
        logger.info("Training completed. Validation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model_artifacts(self):
        """Save model and related artifacts."""
        logger.info("Saving model artifacts...")
        
        models_path = settings.model_paths["models"]
        
        # Save the trained model
        model_file = models_path / "model.joblib"
        joblib.dump(self.model, model_file)
        
        # Create model metadata
        metadata = {
            "model_type": self.model.__class__.__name__,
            "model_parameters": self.model.get_params(),
            "training_timestamp": datetime.now().isoformat(),
            "feature_count": self.model.n_features_in_,
            "classes": self.model.classes_.tolist() if hasattr(self.model, 'classes_') else None
        }
        
        metadata_file = models_path / "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model artifacts saved to {models_path}")
        
        return {
            "model": str(model_file),
            "metadata": str(metadata_file)
        }
    
    def log_to_mlflow(self, metrics, artifacts):
        """Log training results to MLflow."""
        logger.info("Logging to MLflow...")
        
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(self.model.get_params())
            mlflow.log_param("algorithm", "RandomForestClassifier")
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model to MLflow registry
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=settings.mlflow_model_name,
                input_example=np.random.random((1, self.model.n_features_in_))
            )
            
            # Add tags
            mlflow.set_tag("model_type", "RandomForestClassifier")
            mlflow.set_tag("training_type", "initial")
            
            run_id = run.info.run_id
            
        logger.info(f"MLflow run completed: {run_id}")
        logger.info(f"MLflow UI: {settings.mlflow_tracking_uri}/#/experiments")
        
        return run_id
    
    def train_full_pipeline(self):
        """Execute the complete training pipeline."""
        logger.info("Starting full training pipeline...")
        try:
            # Load data and preprocessor
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_processed_data()
            self.load_preprocessor()
        
            # Check if preprocessor is fitted, if not fit it on training data
            from sklearn.utils.validation import check_is_fitted
            try:
                check_is_fitted(self.preprocessor)
                logger.info("Preprocessor is already fitted")
                # Transform data using fitted preprocessor
                X_train_processed = self.preprocessor.transform(X_train)
            except:
                logger.info("Fitting preprocessor on training data...")
                # Fit and transform training data
                X_train_processed = self.preprocessor.fit_transform(X_train)
            
                # Save the fitted preprocessor after fitting (using the same path as loading)
                models_path = settings.model_paths["models"]
                preprocessor_path = models_path / "preprocessor.joblib"
                joblib.dump(self.preprocessor, preprocessor_path)
                logger.info(f"Fitted preprocessor saved to {preprocessor_path}")
        
            # Transform validation data (preprocessor should be fitted by now)
            X_val_processed = self.preprocessor.transform(X_val)
        
            # Encode target if needed
            if self.label_encoder is not None:
                y_train_encoded = self.label_encoder.transform(y_train)
                y_val_encoded = self.label_encoder.transform(y_val)
            else:
                y_train_encoded = y_train
                y_val_encoded = y_val
        
            # Train model
            validation_metrics = self.train_model(X_train_processed, y_train_encoded, 
                                            X_val_processed, y_val_encoded)
        
            # Save model artifacts
            artifacts = self.save_model_artifacts()
        
            # Save training metrics
            metrics_path = settings.model_paths["metrics"]
            with open(metrics_path / "train_metrics.json", "w") as f:
                json.dump(validation_metrics, f, indent=2)
        
            # Log to MLflow
            run_id = self.log_to_mlflow(validation_metrics, artifacts)
        
            result = {
                "status": "success",
                "metrics": validation_metrics,
                "run_id": run_id,
                "artifacts": artifacts
            }
        
            logger.info("Training pipeline completed successfully!")
            return result
        
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main function for training."""
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run training pipeline
    result = trainer.train_full_pipeline()
    
    print("\n" + "="*50)
    print("🎉 TRAINING RESULTS")
    print("="*50)
    for metric, value in result["metrics"].items():
        print(f"📊 {metric.upper()}: {value:.4f}")
    print(f"🔗 MLflow Run ID: {result['run_id']}")
    print(f"🌐 MLflow UI: {settings.mlflow_tracking_uri}")
    print("="*50)


if __name__ == "__main__":
    main()