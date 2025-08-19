import os
import yaml
import json
import subprocess
from pathlib import Path

def check_dvc_status():
    """Check current DVC status and configuration"""
    print("🔍 Checking DVC status...")
    
    # Check if DVC is initialized
    if os.path.exists('.dvc'):
        print("✅ DVC is initialized")
    else:
        print("❌ DVC not initialized")
        return False
    
    # Check DVC config
    dvc_config_path = '.dvc/config'
    if os.path.exists(dvc_config_path):
        print("✅ DVC config exists")
        with open(dvc_config_path, 'r') as f:
            config_content = f.read()
            print("📋 Current DVC config:")
            print(config_content)
    else:
        print("⚠️ No DVC config found")
    
    # Check remote storage
    try:
        result = subprocess.run(['dvc', 'remote', 'list'], 
                              capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print("✅ DVC remotes configured:")
            print(result.stdout)
        else:
            print("⚠️ No DVC remotes configured")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error checking DVC remotes: {e}")
    
    # Check tracked files
    dvc_files = list(Path('.').rglob('*.dvc'))
    print(f"📁 Found {len(dvc_files)} DVC tracked files:")
    for dvc_file in dvc_files:
        print(f"   - {dvc_file}")
    
    return True

def configure_s3_remote():
    """Configure S3 remote for DVC using your existing MLflow S3 bucket"""
    print("\n🔧 Configuring S3 remote for DVC...")
    
    # Your existing S3 bucket from MLflow setup
    s3_bucket = "mlflow-artifacts-365021531163-ap-south-1"
    s3_region = "ap-south-1"
    
    # Configure DVC remote
    remote_name = "s3-storage"
    s3_path = f"s3://{s3_bucket}/dvc-data"
    
    commands = [
        ['dvc', 'remote', 'add', '-d', remote_name, s3_path],
        ['dvc', 'remote', 'modify', remote_name, 'region', s3_region],
        ['dvc', 'config', 'core.analytics', 'false']  # Disable analytics
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {' '.join(cmd)} - {e}")
            print(f"   Output: {e.stdout}")
            print(f"   Error: {e.stderr}")
    
    print(f"✅ S3 remote configured: {s3_path}")

def create_dvcyaml_pipeline():
    """Create DVC pipeline configuration"""
    print("\n📝 Creating DVC pipeline configuration...")
    
    # Create dvc.yaml file for pipeline definition
    pipeline_config = {
        'stages': {
            'data_preprocessing': {
                'cmd': 'python scripts/preprocess_data.py',
                'deps': [
                    'data/raw/X.csv',
                    'data/raw/y.csv',
                    'scripts/preprocess_data.py'
                ],
                'outs': [
                    'data/processed/X_processed.csv',
                    'data/processed/y_processed.csv',
                    'data/processed/preprocessing_metadata.json'
                ]
            },
            'train_model': {
                'cmd': 'python scripts/train_model_dvc.py',
                'deps': [
                    'data/processed/X_processed.csv',
                    'data/processed/y_processed.csv',
                    'data/processed/preprocessing_metadata.json',
                    'scripts/train_model_dvc.py'
                ],
                'outs': [
                    'models/model.pkl',
                    'models/preprocessing_pipeline.joblib'
                ],
                'metrics': [
                    'metrics/train_metrics.json'
                ],
                'plots': [
                    'plots/confusion_matrix.json',
                    'plots/feature_importance.json'
                ]
            },
            'evaluate_model': {
                'cmd': 'python scripts/evaluate_model.py',
                'deps': [
                    'models/model.pkl',
                    'models/preprocessing_pipeline.joblib',
                    'data/processed/X_processed.csv',
                    'data/processed/y_processed.csv'
                ],
                'metrics': [
                    'metrics/eval_metrics.json'
                ],
                'plots': [
                    'plots/roc_curve.json',
                    'plots/precision_recall.json'
                ]
            }
        }
    }
    
    # Write dvc.yaml
    with open('dvc.yaml', 'w') as f:
        yaml.dump(pipeline_config, f, default_flow_style=False, indent=2)
    
    print("✅ Created dvc.yaml pipeline configuration")
    
    # Create .dvcignore file
    dvcignore_content = """
# DVC ignore file
*.pyc
__pycache__/
.pytest_cache/
.coverage
.env
*.log
.DS_Store
.vscode/
.idea/
*.tmp
"""
    
    with open('.dvcignore', 'w') as f:
        f.write(dvcignore_content.strip())
    
    print("✅ Created .dvcignore file")

def create_directory_structure():
    """Create necessary directory structure for DVC pipeline"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        'data/processed',
        'models',
        'metrics',
        'plots',
        'scripts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_params_yaml():
    """Create params.yaml for experiment parameters"""
    print("\n⚙️ Creating params.yaml...")
    
    params = {
        'data_preprocessing': {
            'test_size': 0.2,
            'random_state': 42,
            'scale_features': True
        },
        'model_training': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        },
        'model_evaluation': {
            'classification_threshold': 0.5,
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }
    
    with open('params.yaml', 'w') as f:
        yaml.dump(params, f, default_flow_style=False, indent=2)
    
    print("✅ Created params.yaml")

def main():
    """Main function to set up enhanced DVC configuration"""
    print("🚀 Setting up Enhanced DVC Configuration")
    print("=" * 60)
    
    # Check current status
    if not check_dvc_status():
        print("❌ Please initialize DVC first with: dvc init")
        return False
    
    # Configure S3 remote
    configure_s3_remote()
    
    # Create directory structure
    create_directory_structure()
    
    # Create DVC pipeline
    create_dvcyaml_pipeline()
    
    # Create parameters file
    create_params_yaml()
    
    print("\n" + "=" * 60)
    print("✅ Enhanced DVC configuration completed!")
    print("\n📋 Next steps:")
    print("1. Create data preprocessing script")
    print("2. Create DVC-compatible training script")
    print("3. Set up AWS credentials for S3 access")
    print("4. Run DVC pipeline")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)