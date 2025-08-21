# Ka-MLOps System Status Checker
import os
import pandas as pd
import requests
from pathlib import Path
import json

def check_ka_system_status():
    '''Check the complete Ka-MLOps system status'''
    print('🔍 Ka-MLOps System Status Check')
    print('=' * 50)
    
    status = {
        'data': False,
        'preprocessing': False,
        'model': False,
        'api': False,
        'reports': False
    }
    
    # Check Ka data
    print('1.  Checking Ka data files...')
    data_files = [
        'data/raw/ka_lending_club_dataset.csv',
        'data/processed/ka_files/ka_X_train.csv',
        'data/processed/ka_files/ka_X_test.csv',
        'data/processed/ka_files/ka_y_train.csv',
        'data/processed/ka_files/ka_y_test.csv'
    ]
    
    data_ok = all(Path(f).exists() for f in data_files)
    if data_ok:
        df = pd.read_csv('data/raw/ka_lending_club_dataset.csv')
        print(f'    Ka dataset: {len(df):,} samples, {len(df.columns)} features')
        status['data'] = True
    else:
        print('    Ka data files missing')
    
    # Check Ka preprocessing
    print('2.  Checking Ka preprocessing...')
    preprocessor_file = 'models/ka_models/ka_preprocessor.pkl'
    if Path(preprocessor_file).exists():
        print('    Ka preprocessor available')
        status['preprocessing'] = True
    else:
        print('    Ka preprocessor missing')
    
    # Check Ka model
    print('3.  Checking Ka model...')
    model_file = 'models/ka_models/ka_loan_default_model.pkl'
    if Path(model_file).exists():
        print('    Ka model available')
        
        # Check model metrics
        metrics_file = 'metrics/ka_metrics/ka_train_metrics.json'
        if Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            f1_score = metrics.get('f1_score', 0)
            print(f'    Ka F1 Score: {f1_score:.3f}')
            status['model'] = True
        else:
            print('    Ka model metrics missing')
    else:
        print('    Ka model missing')
    
    # Check Ka API
    print('4.  Checking Ka API...')
    try:
        response = requests.get('http://localhost:8000/ka-health', timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f'    Ka API: {health_data["status"]}')
            print(f'    Model loaded: {health_data["model_loaded"]}')
            status['api'] = True
        else:
            print(f'    Ka API error: {response.status_code}')
    except Exception as e:
        print('    Ka API not accessible (server may be down)')
    
    # Check Ka reports
    print('5.  Checking Ka reports...')
    report_file = 'reports/ka_reports/ka_model_performance.html'
    if Path(report_file).exists():
        print('    Ka performance report available')
        status['reports'] = True
    else:
        print('    Ka performance report missing')
    
    # Overall status
    print('\n Ka-MLOps System Overview:')
    print('=' * 30)
    
    total_components = len(status)
    working_components = sum(status.values())
    system_health = working_components / total_components
    
    for component, working in status.items():
        icon = '' if working else ''
        print(f'   {icon} Ka {component.capitalize()}')
    
    print(f'\n Ka System Health: {system_health:.1%} ({working_components}/{total_components} components)')
    
    if system_health >= 0.8:
        print(' Ka-MLOps system is healthy!')
    elif system_health >= 0.6:
        print(' Ka-MLOps system has some issues')
    else:
        print(' Ka-MLOps system needs attention')
    
    # Next steps
    print('\n Next Steps:')
    if not status['data']:
        print('    Run: python src/ka_modules/ka_data_preprocessing.py')
    if not status['model']:
        print('    Run: python src/ka_modules/ka_model_training.py')
    if not status['api']:
        print('    Start: python scripts/ka_scripts/ka_start_server.py')
    if not status['reports']:
        print('    Run: python scripts/ka_scripts/ka_evaluate_model.py')
    
    if all(status.values()):
        print('    Ka-MLOps system is fully operational!')
    
    return status

if __name__ == "__main__":
    check_ka_system_status()
