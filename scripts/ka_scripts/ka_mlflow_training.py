# Ka-MLOps MLflow Integration
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ka_modules.ka_model_training import KaLoanDefaultModel
from ka_modules.ka_data_preprocessing import KaLendingClubPreprocessor

class KaMLflowTrainer:
    def __init__(self, mlflow_tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("ka_loan_default_prediction")
        self.client = MlflowClient()
        
    def train_and_register_ka_model(self):
        '''Complete Ka training pipeline with MLflow tracking'''
        
        with mlflow.start_run(run_name=f"ka_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            print(f" Starting Ka MLflow run: {run.info.run_id}")
            
            # Log run info
            mlflow.log_param("ka_model_type", "RandomForest")
            mlflow.log_param("ka_data_source", "synthetic_lending_club")
            mlflow.log_param("ka_training_date", datetime.now().isoformat())
            
            # Load processed data
            print(" Loading Ka processed data...")
            X_train = pd.read_csv('data/processed/ka_files/ka_X_train.csv')
            X_test = pd.read_csv('data/processed/ka_files/ka_X_test.csv')
            y_train = pd.read_csv('data/processed/ka_files/ka_y_train.csv').values.ravel()
            y_test = pd.read_csv('data/processed/ka_files/ka_y_test.csv').values.ravel()
            
            # Log data statistics
            mlflow.log_metric("ka_total_samples", len(X_train) + len(X_test))
            mlflow.log_metric("ka_train_samples", len(X_train))
            mlflow.log_metric("ka_test_samples", len(X_test))
            mlflow.log_metric("ka_feature_count", X_train.shape[1])
            mlflow.log_metric("ka_default_rate", y_train.mean())
            
            # Train Ka model
            print(" Training Ka model...")
            ka_model = KaLoanDefaultModel()
            
            # Log model parameters
            mlflow.log_param("ka_n_estimators", ka_model.model.n_estimators)
            mlflow.log_param("ka_max_depth", ka_model.model.max_depth)
            mlflow.log_param("ka_class_weight", str(ka_model.model.class_weight))
            
            # Train and get metrics
            metrics = ka_model.train(X_train, y_train, X_test, y_test)
            
            # Log Ka metrics to MLflow
            for metric_name, metric_value in metrics.items():
                if metric_name != 'training_date':
                    mlflow.log_metric(f"ka_{metric_name}", metric_value)
            
            # Log Ka feature importance
            feature_importance = ka_model.get_feature_importance(X_train.columns)
            
            # Save feature importance as artifact
            importance_file = f"ka_feature_importance_{run.info.run_id}.csv"
            feature_importance.to_csv(importance_file, index=False)
            mlflow.log_artifact(importance_file, "ka_feature_analysis")
            os.remove(importance_file)  # Clean up temp file
            
            # Save Ka model
            ka_model.save_model(f"ka_model_{run.info.run_id}.pkl")
            mlflow.log_artifact(f"ka_model_{run.info.run_id}.pkl", "ka_model")
            os.remove(f"ka_model_{run.info.run_id}.pkl")  # Clean up temp file
            
            # Log Ka model with MLflow
            mlflow.sklearn.log_model(
                ka_model.model, 
                "ka_random_forest_model",
                registered_model_name="ka_loan_default_production"
            )
            
            # Register model if performance is good
            if metrics['f1_score'] >= 0.65:  # Ka performance threshold
                print(f" Ka model meets performance threshold! F1: {metrics['f1_score']:.3f}")
                
                # Get the model version
                model_name = "ka_loan_default_production"
                latest_version = self.client.get_latest_versions(model_name, stages=["None"])[0]
                
                # Transition to staging
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Staging"
                )
                
                print(f" Ka model registered as version {latest_version.version} in Staging")
                return latest_version
            else:
                print(f" Ka model performance below threshold. F1: {metrics['f1_score']:.3f}")
                return None

    def promote_ka_to_production(self, model_name, version):
        '''Promote Ka model from staging to production'''
        
        # Archive current production model
        try:
            current_prod = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )[0]
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod.version,
                stage="Archived"
            )
            print(f" Archived previous Ka production model v{current_prod.version}")
        except IndexError:
            print(" No previous Ka production model to archive")
        
        # Promote to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f" Ka model version {version} promoted to Production!")

def main():
    '''Main Ka MLflow training pipeline'''
    print(' Ka-MLOps MLflow Training Pipeline')
    print('=' * 50)
    
    try:
        # Initialize Ka MLflow trainer
        ka_trainer = KaMLflowTrainer()
        
        # Train and register Ka model
        model_version = ka_trainer.train_and_register_ka_model()
        
        if model_version:
            print(f"\n Ka MLflow training completed successfully!")
            print(f" Model version: {model_version.version}")
            print(f"  Model stage: {model_version.current_stage}")
            
            # Optionally promote to production (manual step)
            user_input = input("\n Promote Ka model to Production? (y/N): ").lower().strip()
            if user_input == 'y':
                ka_trainer.promote_ka_to_production("ka_loan_default_production", model_version.version)
                print(" Ka model is now in Production!")
            else:
                print(" Ka model remains in Staging for further validation")
        else:
            print(" Ka model training failed or performance insufficient")
    
    except Exception as e:
        print(f" Ka MLflow training error: {e}")
        return None
    
    return model_version

if __name__ == "__main__":
    main()
