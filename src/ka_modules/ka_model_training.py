# Ka-MLOps Model Training Pipeline
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import json
from pathlib import Path
from datetime import datetime

class KaLoanDefaultModel:
    def __init__(self):
        # Random Forest with class balancing for better performance
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
    def train(self, X_train, y_train, X_test, y_test):
        '''Train the Ka Random Forest model'''
        print(' Training Ka Random Forest model...')
        print('=' * 50)
        
        # Train model
        print(' Fitting Random Forest...')
        self.model.fit(X_train, y_train)
        
        # Make predictions
        print(' Making predictions...')
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print results
        self._print_results(metrics, y_test, y_pred)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        '''Calculate comprehensive metrics for Ka model'''
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'training_date': datetime.now().isoformat()
        }
    
    def _print_results(self, metrics, y_test, y_pred):
        '''Print Ka model training results'''
        print(' Ka Model Performance:')
        print('-' * 30)
        print(f' Accuracy:  {metrics["accuracy"]:.4f}')
        print(f' Precision: {metrics["precision"]:.4f}')
        print(f' Recall:    {metrics["recall"]:.4f}')
        print(f' F1 Score:  {metrics["f1_score"]:.4f}')
        print(f' AUC-ROC:   {metrics["auc_roc"]:.4f}')
        
        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f'\n Confusion Matrix:')
        print(f'               Predicted')
        print(f'Actual    No Default  Default')
        print(f'No Default     {cm[0,0]:5d}     {cm[0,1]:5d}')
        print(f'Default        {cm[1,0]:5d}     {cm[1,1]:5d}')
    
    def get_feature_importance(self, feature_names):
        '''Get Ka feature importance for interpretability'''
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X):
        '''Make Ka predictions'''
        return self.model.predict(X)
    
    def predict_proba(self, X):
        '''Get Ka prediction probabilities'''
        return self.model.predict_proba(X)
    
    def save_model(self, path):
        '''Save Ka model'''
        joblib.dump(self.model, path)
        print(f' Saved Ka model to: {path}')

def main():
    '''Main Ka training pipeline'''
    print(' Ka-MLOps Model Training Pipeline')
    print('=' * 50)
    
    # Load processed data
    print(' Loading Ka processed data...')
    X_train = pd.read_csv('data/processed/ka_files/ka_X_train.csv')
    X_test = pd.read_csv('data/processed/ka_files/ka_X_test.csv')
    y_train = pd.read_csv('data/processed/ka_files/ka_y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/ka_files/ka_y_test.csv').values.ravel()
    
    print(f' Training data: {X_train.shape}')
    print(f' Test data: {X_test.shape}')
    
    # Train Ka model
    ka_model = KaLoanDefaultModel()
    metrics = ka_model.train(X_train, y_train, X_test, y_test)
    
    # Feature importance analysis
    print('\n Ka Feature Importance Analysis:')
    print('-' * 40)
    feature_importance = ka_model.get_feature_importance(X_train.columns)
    
    # Show top 10 features
    print(' Top 10 Most Important Features:')
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f'  {i:2d}. {row["feature"]:<25} {row["importance"]:.4f}')
    
    # Save model and metrics
    print('\n Saving Ka model and results...')
    
    # Create directories
    Path('models/ka_models').mkdir(parents=True, exist_ok=True)
    Path('metrics/ka_metrics').mkdir(parents=True, exist_ok=True)
    Path('reports/ka_reports').mkdir(parents=True, exist_ok=True)
    
    # Save model
    ka_model.save_model('models/ka_models/ka_loan_default_model.pkl')
    
    # Save metrics
    with open('metrics/ka_metrics/ka_train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(' Saved metrics to: metrics/ka_metrics/ka_train_metrics.json')
    
    # Save feature importance
    feature_importance.to_csv('reports/ka_reports/ka_feature_importance.csv', index=False)
    print(' Saved feature importance to: reports/ka_reports/ka_feature_importance.csv')
    
    # Final summary
    print('\n Ka Model Training Summary:')
    print('=' * 40)
    f1_score_val = metrics['f1_score']
    if f1_score_val >= 0.75:
        status = ' Excellent'
    elif f1_score_val >= 0.65:
        status = ' Good'
    else:
        status = ' Needs Improvement'
    
    print(f' F1 Score: {f1_score_val:.3f} ({status})')
    print(f' Model Status: Ready for deployment!')
    print(f' Model saved to: models/ka_models/')
    
    return ka_model, metrics

if __name__ == '__main__':
    model, metrics = main()
