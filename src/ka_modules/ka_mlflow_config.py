# MLflow Configuration for Ka Training
import mlflow
import mlflow.sklearn
import os
from datetime import datetime

class KaMLflowConfig:
    def __init__(self):
        # Your MLflow server URL
        self.MLFLOW_TRACKING_URI = "http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com"
        self.EXPERIMENT_NAME = "ka-loan-default-retraining"
        self.MODEL_NAME = "KaLoanDefaultModel"
        
    def setup_mlflow(self):
        '''Setup MLflow connection'''
        try:
            mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.EXPERIMENT_NAME)
            
            # Test connection
            mlflow.search_experiments()
            print(f" MLflow connected: {self.MLFLOW_TRACKING_URI}")
            return True
        except Exception as e:
            print(f" MLflow connection failed: {e}")
            return False
    
    def start_run(self, run_name=None):
        '''Start MLflow run'''
        if not run_name:
            run_name = f"ka-retraining-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_metrics(self, metrics_dict):
        '''Log metrics to MLflow'''
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
    
    def log_params(self, params_dict):
        '''Log parameters to MLflow'''
        for key, value in params_dict.items():
            mlflow.log_param(key, value)
    
    def log_model(self, model, model_path="ka_model"):
        '''Log model to MLflow'''
        mlflow.sklearn.log_model(
            model, 
            model_path,
            registered_model_name=self.MODEL_NAME
        )
        print(f" Model logged to MLflow: {model_path}")
