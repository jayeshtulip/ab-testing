import pandas as pd
import numpy as np
import json
import yaml
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
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
        return {
            'model_evaluation': {
                'classification_threshold': 0.5,
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc']
            }
        }

def load_model_and_data():
    """Load trained model and processed data"""
    print("📊 Loading model and processed data...")
    
    try:
        # Load model
        model = joblib.load('models/model.pkl')
        print("   ✅ Model loaded successfully")
        
        # Load processed data
        X_processed = pd.read_csv('data/processed/X_processed.csv')
        y_processed = pd.read_csv('data/processed/y_processed.csv')
        
        # Split back into train/test
        train_mask = X_processed['split'] == 'train'
        test_mask = X_processed['split'] == 'test'
        
        X_train = X_processed[train_mask].drop('split', axis=1)
        X_test = X_processed[test_mask].drop('split', axis=1)
        y_train = y_processed[y_processed['split'] == 'train']['target']
        y_test = y_processed[y_processed['split'] == 'test']['target']
        
        print(f"   ✅ Training data: {X_train.shape}")
        print(f"   ✅ Test data: {X_test.shape}")
        
        return model, X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Error loading model and data: {e}")
        return None, None, None, None, None

def evaluate_model_performance(model, X_train, X_test, y_train, y_test, params):
    """Comprehensive model evaluation"""
    print("📊 Performing comprehensive model evaluation...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    metrics = {}
    
    # Basic metrics
    metrics['train_accuracy'] = float(accuracy_score(y_train, y_train_pred))
    metrics['test_accuracy'] = float(accuracy_score(y_test, y_test_pred))
    metrics['precision'] = float(precision_score(y_test, y_test_pred, average='weighted'))
    metrics['recall'] = float(recall_score(y_test, y_test_pred, average='weighted'))
    metrics['f1_score'] = float(f1_score(y_test, y_test_pred, average='weighted'))
    
    # AUC Score
    try:
        if len(np.unique(y_test)) == 2:
            # For binary classification, convert to 0/1 if needed
            y_test_binary = y_test.copy()
            unique_vals = sorted(np.unique(y_test))
            if unique_vals == [1, 2]:
                y_test_binary = y_test - 1
            metrics['auc_score'] = float(roc_auc_score(y_test_binary, y_test_prob[:, 1]))
        else:
            metrics['auc_score'] = float(roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
    except Exception as e:
        print(f"   ⚠️ Could not calculate AUC: {e}")
        metrics['auc_score'] = 0.0
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Classification Report
    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Model complexity metrics
    metrics['n_estimators'] = model.n_estimators if hasattr(model, 'n_estimators') else 0
    metrics['max_depth'] = model.max_depth if hasattr(model, 'max_depth') else 0
    metrics['n_features'] = X_test.shape[1]
    metrics['n_samples_train'] = len(X_train)
    metrics['n_samples_test'] = len(X_test)
    
    # Overfitting metrics
    metrics['overfitting_ratio'] = metrics['train_accuracy'] - metrics['test_accuracy']
    
    print(f"   📈 Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   📈 Precision: {metrics['precision']:.4f}")
    print(f"   📈 Recall: {metrics['recall']:.4f}")
    print(f"   📈 F1 Score: {metrics['f1_score']:.4f}")
    print(f"   📈 AUC Score: {metrics['auc_score']:.4f}")
    print(f"   📈 Overfitting Ratio: {metrics['overfitting_ratio']:.4f}")
    
    return metrics, conf_matrix, y_test_prob, class_report

def generate_evaluation_plots(model, X_test, y_test, y_test_prob, conf_matrix):
    """Generate evaluation plots for DVC"""
    print("📊 Generating evaluation plots...")
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # 1. ROC Curve (for binary classification)
    if len(np.unique(y_test)) == 2 and y_test_prob.shape[1] == 2:
        # Convert target to 0/1 if needed
        y_test_binary = y_test.copy()
        unique_vals = sorted(np.unique(y_test))
        if unique_vals == [1, 2]:
            y_test_binary = y_test - 1
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_test_prob[:, 1])
        
        roc_data = {
            'data': [
                {'fpr': float(fpr[i]), 'tpr': float(tpr[i])}
                for i in range(len(fpr))
            ],
            'auc': float(roc_auc_score(y_test_binary, y_test_prob[:, 1]))
        }
        
        with open('plots/roc_curve.json', 'w') as f:
            json.dump(roc_data, f, indent=2)
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test_binary, y_test_prob[:, 1])
        
        pr_data = {
            'data': [
                {'precision': float(precision[i]), 'recall': float(recall[i])}
                for i in range(len(precision))
            ]
        }
        
        with open('plots/precision_recall.json', 'w') as f:
            json.dump(pr_data, f, indent=2)
    
    # 3. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
        if X_test.columns is not None:
            feature_names = list(X_test.columns)
        
        importance_data = {
            'data': [
                {'feature': feature_names[i], 'importance': float(model.feature_importances_[i])}
                for i in range(len(model.feature_importances_))
            ]
        }
        
        # Sort by importance
        importance_data['data'] = sorted(importance_data['data'], 
                                       key=lambda x: x['importance'], 
                                       reverse=True)
        
        # Save top 15 features
        with open('plots/feature_importance_eval.json', 'w') as f:
            json.dump({'data': importance_data['data'][:15]}, f, indent=2)
    
    print("   ✅ Generated evaluation plots")

def save_evaluation_metrics(metrics, class_report):
    """Save evaluation metrics"""
    print("💾 Saving evaluation metrics...")
    
    # Ensure metrics directory exists
    os.makedirs('metrics', exist_ok=True)
    
    # Add metadata
    eval_metrics = {
        **metrics,
        'evaluation_timestamp': datetime.now().isoformat(),
        'evaluation_type': 'comprehensive_evaluation',
        'class_report': class_report
    }
    
    # Save evaluation metrics
    with open('metrics/eval_metrics.json', 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    print("   ✅ Saved eval_metrics.json")

def generate_evaluation_summary(metrics):
    """Generate evaluation summary"""
    print("📋 Evaluation Summary:")
    print("=" * 50)
    print(f"🎯 Model Performance:")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   AUC Score: {metrics['auc_score']:.4f}")
    print()
    print(f"📊 Model Complexity:")
    print(f"   Features: {metrics['n_features']}")
    print(f"   Training Samples: {metrics['n_samples_train']}")
    print(f"   Test Samples: {metrics['n_samples_test']}")
    print()
    print(f"🔍 Overfitting Analysis:")
    print(f"   Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   Overfitting Ratio: {metrics['overfitting_ratio']:.4f}")
    
    if metrics['overfitting_ratio'] > 0.1:
        print("   ⚠️ Potential overfitting detected!")
    else:
        print("   ✅ Good generalization")
    
    print("=" * 50)

def main():
    """Main evaluation function"""
    print("🚀 Starting DVC Model Evaluation Pipeline")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    print(f"📋 Evaluation parameters: {params.get('model_evaluation', {})}")
    
    # Load model and data
    model, X_train, X_test, y_train, y_test = load_model_and_data()
    if model is None:
        print("❌ Failed to load model and data")
        return False
    
    # Evaluate model
    metrics, conf_matrix, y_test_prob, class_report = evaluate_model_performance(
        model, X_train, X_test, y_train, y_test, params
    )
    
    # Generate plots
    generate_evaluation_plots(model, X_test, y_test, y_test_prob, conf_matrix)
    
    # Save metrics
    save_evaluation_metrics(metrics, class_report)
    
    # Generate summary
    generate_evaluation_summary(metrics)
    
    print("=" * 60)
    print("✅ Model evaluation completed successfully!")
    print(f"📊 Final Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"📊 Final AUC Score: {metrics['auc_score']:.4f}")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)