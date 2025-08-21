# Ka-MLOps Model Evaluation Script
import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate_ka_model():
    '''Evaluate the trained Ka model and generate reports'''
    print(' Ka Model Evaluation Pipeline')
    print('=' * 40)
    
    # Load test data
    print(' Loading Ka test data...')
    X_test = pd.read_csv('data/processed/ka_files/ka_X_test.csv')
    y_test = pd.read_csv('data/processed/ka_files/ka_y_test.csv').values.ravel()
    
    # Load Ka model
    print(' Loading Ka model...')
    ka_model = joblib.load('models/ka_models/ka_loan_default_model.pkl')
    
    # Make predictions
    print(' Making Ka predictions...')
    y_pred = ka_model.predict(X_test)
    y_pred_proba = ka_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'ka_accuracy': float((y_pred == y_test).mean()),
        'ka_precision': float(precision_score(y_test, y_pred)),
        'ka_recall': float(recall_score(y_test, y_pred)),
        'ka_f1_score': float(f1_score(y_test, y_pred)),
        'ka_auc_roc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    # Print results
    print(' Ka Model Performance:')
    print('-' * 30)
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # Save metrics
    Path('metrics/ka_metrics').mkdir(parents=True, exist_ok=True)
    with open('metrics/ka_metrics/ka_test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate performance report
    generate_ka_performance_report(y_test, y_pred, y_pred_proba, metrics)
    
    print(' Ka model evaluation completed!')
    return metrics

def generate_ka_performance_report(y_test, y_pred, y_pred_proba, metrics):
    '''Generate Ka HTML performance report'''
    print(' Generating Ka performance report...')
    
    # Create plots
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ka Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('Ka Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0,1].plot(fpr, tpr, label=f'Ka ROC (AUC = {metrics["ka_auc_roc"]:.3f})', linewidth=2)
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('Ka ROC Curve')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Prediction Distribution
    axes[1,0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Non-Default', color='green')
    axes[1,0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Default', color='red')
    axes[1,0].set_xlabel('Ka Default Probability')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Ka Prediction Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metric_values = [metrics['ka_accuracy'], metrics['ka_precision'], 
                    metrics['ka_recall'], metrics['ka_f1_score'], metrics['ka_auc_roc']]
    
    bars = axes[1,1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
    axes[1,1].set_title('Ka Model Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    Path('reports/ka_reports').mkdir(parents=True, exist_ok=True)
    plt.savefig('reports/ka_reports/ka_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate HTML report
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ka Model Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            .metric {{ background: #ecf0f1; padding: 15px; margin: 10px; border-radius: 8px; display: inline-block; min-width: 150px; text-align: center; }}
            .good {{ background: #d5f4e6; border-left: 5px solid #27ae60; }}
            .warning {{ background: #fef9e7; border-left: 5px solid #f39c12; }}
            .bad {{ background: #fadbd8; border-left: 5px solid #e74c3c; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
            .chart {{ text-align: center; margin: 30px 0; }}
            .status {{ padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> Ka Model Performance Report</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric {'good' if metrics['ka_f1_score'] > 0.7 else 'warning' if metrics['ka_f1_score'] > 0.5 else 'bad'}">
                    <div class="metric-value">{metrics['ka_f1_score']:.3f}</div>
                    <div class="metric-label">Ka F1 Score</div>
                </div>
                <div class="metric {'good' if metrics['ka_precision'] > 0.7 else 'warning' if metrics['ka_precision'] > 0.5 else 'bad'}">
                    <div class="metric-value">{metrics['ka_precision']:.3f}</div>
                    <div class="metric-label">Ka Precision</div>
                </div>
                <div class="metric {'good' if metrics['ka_recall'] > 0.7 else 'warning' if metrics['ka_recall'] > 0.5 else 'bad'}">
                    <div class="metric-value">{metrics['ka_recall']:.3f}</div>
                    <div class="metric-label">Ka Recall</div>
                </div>
                <div class="metric {'good' if metrics['ka_auc_roc'] > 0.8 else 'warning' if metrics['ka_auc_roc'] > 0.7 else 'bad'}">
                    <div class="metric-value">{metrics['ka_auc_roc']:.3f}</div>
                    <div class="metric-label">Ka AUC-ROC</div>
                </div>
            </div>
            
            <div class="chart">
                <h2> Ka Performance Visualization</h2>
                <img src="ka_model_performance.png" alt="Ka Model Performance Charts" style="max-width: 100%; border-radius: 8px;">
            </div>
            
            <div class="status {'good' if metrics['ka_f1_score'] > 0.7 else 'warning' if metrics['ka_f1_score'] > 0.5 else 'bad'}">
                <h2> Ka Model Status</h2>
                <p>{' Ka model meets production criteria (F1 > 0.7)' if metrics['ka_f1_score'] > 0.7 else ' Ka model performance needs monitoring' if metrics['ka_f1_score'] > 0.5 else ' Ka model needs improvement'}</p>
            </div>
            
            <div class="summary">
                <h2> Ka Summary</h2>
                <p><strong>Total Test Samples:</strong> {len(y_test):,}</p>
                <p><strong>Default Rate:</strong> {y_test.mean():.1%}</p>
                <p><strong>Ka Model Type:</strong> Random Forest Classifier</p>
                <p><strong>Ka System Status:</strong> {' Production Ready' if metrics['ka_f1_score'] > 0.7 else ' Monitoring Required' if metrics['ka_f1_score'] > 0.5 else ' Needs Improvement'}</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    with open('reports/ka_reports/ka_model_performance.html', 'w') as f:
        f.write(html_content)
    
    print(' Ka performance report saved to: reports/ka_reports/ka_model_performance.html')

if __name__ == "__main__":
    evaluate_ka_model()
