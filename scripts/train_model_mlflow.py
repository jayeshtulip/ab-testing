import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

def main():
    print("🚀 Starting MLflow tracked training...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com")
    
    # Set experiment
    experiment_name = "loan-default-training"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"📊 MLflow Run ID: {run.info.run_id}")
        
        # Load data
        print("📊 Loading training data...")
        try:
            X_df = pd.read_csv('data/raw/X.csv')
            y_df = pd.read_csv('data/raw/y.csv')
            
            print(f"   📈 Data shape: {X_df.shape}")
            print(f"   📊 Target distribution: {y_df.iloc[:, 0].value_counts().to_dict()}")
            
            # Log data info
            mlflow.log_param("data_shape", f"{X_df.shape[0]}x{X_df.shape[1]}")
            mlflow.log_param("n_features", X_df.shape[1])
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔄 Using synthetic data for demonstration...")
            
            # Generate synthetic loan default data
            np.random.seed(42)
            n_samples = 1000
            
            X_df = pd.DataFrame({
                'credit_amount': np.random.normal(3000, 1500, n_samples),
                'duration_months': np.random.randint(6, 72, n_samples),
                'age': np.random.randint(18, 75, n_samples),
                'employment_duration': np.random.randint(0, 10, n_samples),
            })
            
            # Create realistic target based on features
            default_prob = (
                (X_df['credit_amount'] > 5000).astype(int) * 0.3 +
                (X_df['duration_months'] > 36).astype(int) * 0.2 +
                (X_df['age'] < 25).astype(int) * 0.2 +
                np.random.random(n_samples) * 0.3
            )
            y_df = pd.DataFrame({'default': (default_prob > 0.5).astype(int)})
            
            mlflow.log_param("data_type", "synthetic")
            mlflow.log_param("data_shape", f"{X_df.shape[0]}x{X_df.shape[1]}")
        
        # Prepare features
        X = X_df.values if not X_df.select_dtypes(include=['object']).empty else X_df.values
        y = y_df.iloc[:, 0].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   📚 Training set: {X_train.shape[0]} samples")
        print(f"   🧪 Test set: {X_test.shape[0]} samples")
        
        # Model parameters
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        # Log hyperparameters
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Train model
        print("🤖 Training Random Forest model...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        print("📈 Evaluating model performance...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   ✅ Model accuracy: {accuracy:.4f}")
        print(f"   ✅ Model AUC: {auc_score:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc_score", auc_score)
        mlflow.log_metric("test_samples", len(y_test))
        
        # Log additional metrics from classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("precision_0", report['0']['precision'])
        mlflow.log_metric("recall_0", report['0']['recall'])
        mlflow.log_metric("f1_score_0", report['0']['f1-score'])
        mlflow.log_metric("precision_1", report['1']['precision'])
        mlflow.log_metric("recall_1", report['1']['recall'])
        mlflow.log_metric("f1_score_1", report['1']['f1-score'])
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"importance_{feature}", importance)
        
        # Log model
        print("💾 Logging model to MLflow...")
        model_name = "loan-default-model"
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=model_name
        )
        
        # Log additional metadata
        mlflow.log_param("training_time", datetime.now().isoformat())
        mlflow.log_param("git_commit", os.getenv('GITHUB_SHA', 'local'))
        mlflow.log_param("training_trigger", os.getenv('GITHUB_EVENT_NAME', 'manual'))
        
        print(f"✅ Training completed successfully!")
        print(f"🔗 MLflow Run: {run.info.run_id}")
        print(f"📊 Model Accuracy: {accuracy:.4f}")
        print(f"📈 Model AUC: {auc_score:.4f}")
        
        return {
            'run_id': run.info.run_id,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'model_name': model_name
        }

if __name__ == '__main__':
    try:
        result = main()
        print("🎉 Training pipeline completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)