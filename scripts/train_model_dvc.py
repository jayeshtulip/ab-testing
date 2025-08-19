import pandas as pd
import numpy as np
import json
import yaml
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_params():
    """Load parameters from params.yaml"""
    if os.path.exists('params.yaml'):
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params
    else:
        # Default parameters
        return {
            'model_training': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        }

def load_processed_data():
    """Load processed data from DVC pipeline"""
    print("📊 Loading processed data...")
    
    try:
        # Load processed features and target
        X_processed = pd.read_csv('data/processed/X_processed.csv')
        y_processed = pd.read_csv('data/processed/y_processed.csv')
        
        # Load preprocessing metadata
        with open('data/processed/preprocessing_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"   ✅ Loaded processed data: {X_processed.shape}")
        print(f"   ✅ Loaded target data: {y_processed.shape}")
        
        # Split back into train/test based on split column
        train_mask = X_processed['split'] == 'train'
        test_mask = X_processed['split'] == 'test'
        
        # Remove split column from features
        X_train = X_processed[train_mask].drop('split', axis=1)
        X_test = X_processed[test_mask].drop('split', axis=1)
        y_train = y_processed[y_processed['split'] == 'train']['target']
        y_test = y_processed[y_processed['split'] == 'test']['target']
        
        print(f"   📚 Training set: {X_train.shape}")
        print(f"   🧪 Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, metadata
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return None, None, None, None, None

def train_model(X_train, y_train, params):
    """Train the model with specified parameters"""
    print("🤖 Training Random Forest model...")
    
    model_params = params['model_training']
    
    model = RandomForestClassifier(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        random_state=model_params['random_state']
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print(f"   ✅ Model trained with {model_params}")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model and generate metrics"""
    print("📊 Evaluating model...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Calculate AUC
    try:
        if len(np.unique(y_test)) == 2:
            auc_score = roc_auc_score(y_test, y_test_prob[:, 1])
        else:
            auc_score = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
    except:
        auc_score = 0.0
    
    # Classification report
    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'auc_score': float(auc_score),
        'precision': float(class_report['weighted avg']['precision']),
        'recall': float(class_report['weighted avg']['recall']),
        'f1_score': float(class_report['weighted avg']['f1-score'])
    }
    
    print(f"   📈 Training Accuracy: {train_accuracy:.4f}")
    print(f"   📈 Test Accuracy: {test_accuracy:.4f}")
    print(f"   📈 AUC Score: {auc_score:.4f}")
    print(f"   📈 F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics, conf_matrix, y_test_prob

def generate_plots(model, X_train, X_test, y_test, y_test_prob, conf_matrix):
    """Generate plots for DVC"""
    print("📊 Generating plots...")
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # 1. Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save as JSON for DVC plots
    feature_importance_data = {
        'data': [
            {'feature': row['feature'], 'importance': row['importance']}
            for _, row in feature_importance.head(10).iterrows()
        ]
    }
    
    with open('plots/feature_importance.json', 'w') as f:
        json.dump(feature_importance_data, f, indent=2)
    
    # 2. Confusion matrix
    conf_matrix_data = {
        'data': [
            {'actual': int(i), 'predicted': int(j), 'count': int(conf_matrix[i][j])}
            for i in range(conf_matrix.shape[0])
            for j in range(conf_matrix.shape[1])
        ]
    }
    
    with open('plots/confusion_matrix.json', 'w') as f:
        json.dump(conf_matrix_data, f, indent=2)
    
    # 3. ROC curve (for binary classification)
    if len(np.unique(y_test)) == 2 and y_test_prob.shape[1] == 2:
        # Convert target to 0/1 if it's 1/2
        y_test_binary = y_test.copy()
        unique_vals = sorted(np.unique(y_test))
        if unique_vals == [1, 2]:
            y_test_binary = y_test - 1  # Convert 1,2 to 0,1
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_test_prob[:, 1])
        
        roc_data = {
            'data': [
                {'fpr': float(fpr[i]), 'tpr': float(tpr[i])}
                for i in range(len(fpr))
            ]
        }
        
        with open('plots/roc_curve.json', 'w') as f:
            json.dump(roc_data, f, indent=2)
        
        # 4. Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test_binary, y_test_prob[:, 1])
        
        pr_data = {
            'data': [
                {'precision': float(precision[i]), 'recall': float(recall[i])}
                for i in range(len(precision))
            ]
        }
        
        with open('plots/precision_recall.json', 'w') as f:
            json.dump(pr_data, f, indent=2)
    
    print("   ✅ Generated plots for DVC visualization")

def save_model_and_artifacts(model, metrics, metadata):
    """Save model and artifacts"""
    print("💾 Saving model and artifacts...")
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/model.pkl')
    print("   ✅ Saved model.pkl")
    
    # Save preprocessing pipeline (copy from processed data)
    if os.path.exists('data/processed/preprocessing_pipeline.joblib'):
        import shutil
        shutil.copy('data/processed/preprocessing_pipeline.joblib', 'models/preprocessing_pipeline.joblib')
        print("   ✅ Copied preprocessing_pipeline.joblib")
    
    # Save training metrics
    train_metrics = {
        **metrics,
        'training_timestamp': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'data_samples': metadata.get('train_size', 0) + metadata.get('test_size', 0)
    }
    
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print("   ✅ Saved train_metrics.json")

def log_to_mlflow(model, metrics, params):
    """Log experiment to MLflow"""
    print("📝 Logging to MLflow...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com")
    
    # Set experiment
    experiment_name = "loan-default-dvc-training"
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass  # Experiment already exists
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"dvc_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(params['model_training'])
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="loan-default-dvc-model"
        )
        
        # Log artifacts
        if os.path.exists('models/preprocessing_pipeline.joblib'):
            mlflow.log_artifact('models/preprocessing_pipeline.joblib')
        
        if os.path.exists('metrics/train_metrics.json'):
            mlflow.log_artifact('metrics/train_metrics.json')
        
        print("   ✅ Logged to MLflow successfully")

def main():
    """Main training function"""
    print("🚀 Starting DVC Model Training Pipeline")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    print(f"📋 Model parameters: {params['model_training']}")
    
    # Load processed data
    X_train, X_test, y_train, y_test, metadata = load_processed_data()
    if X_train is None:
        print("❌ Failed to load processed data")
        return False
    
    # Train model
    model = train_model(X_train, y_train, params)
    
    # Evaluate model
    metrics, conf_matrix, y_test_prob = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Generate plots
    generate_plots(model, X_train, X_test, y_test, y_test_prob, conf_matrix)
    
    # Save model and artifacts
    save_model_and_artifacts(model, metrics, metadata)
    
    # Log to MLflow
    log_to_mlflow(model, metrics, params)
    
    print("=" * 60)
    print("✅ DVC Model training completed successfully!")
    print(f"📊 Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"📊 AUC Score: {metrics['auc_score']:.4f}")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)