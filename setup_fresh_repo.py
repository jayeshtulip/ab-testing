import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def clean_old_files():
    """Clean up old files and prepare for fresh repository"""
    print("🧹 Cleaning up old files...")
    
    # Files and directories to clean up
    cleanup_items = [
        '.git',  # Remove old git history
        '*.log',
        '__pycache__',
        '*.pyc',
        '.pytest_cache',
        'mlruns',
        'mlartifacts',
        '*.tmp'
    ]
    
    for item in cleanup_items:
        if '*' in item:
            # Handle wildcard patterns
            import glob
            for file_path in glob.glob(item, recursive=True):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"   🗑️ Removed file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"   🗑️ Removed directory: {file_path}")
                except Exception as e:
                    print(f"   ⚠️ Could not remove {file_path}: {e}")
        else:
            try:
                if os.path.exists(item):
                    if os.path.isfile(item):
                        os.remove(item)
                        print(f"   🗑️ Removed file: {item}")
                    elif os.path.isdir(item):
                        shutil.rmtree(item)
                        print(f"   🗑️ Removed directory: {item}")
            except Exception as e:
                print(f"   ⚠️ Could not remove {item}: {e}")

def create_fresh_gitignore():
    """Create a comprehensive .gitignore for ML projects"""
    print("📝 Creating fresh .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/
.env

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# MLflow
mlruns/
mlartifacts/

# Large model files (use DVC instead)
*.pkl
*.joblib
*.h5
*.pb
models/*.pkl
models/*.joblib

# Data files (tracked by DVC)
data/raw/*.csv
data/processed/*.csv
!data/raw/*.dvc
!data/processed/*.dvc

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp

# AWS credentials (NEVER commit these!)
.aws/
aws_credentials.txt
credentials.json

# Secrets
secrets.yaml
.secrets

# Test artifacts
.pytest_cache/
.coverage
htmlcov/
.tox/

# DVC
/models
/data/raw
/data/processed
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ Created comprehensive .gitignore")

def create_directory_structure():
    """Create proper directory structure for ML project"""
    print("📁 Creating directory structure...")
    
    directories = [
        '.github/workflows',
        '.github/ISSUE_TEMPLATE',
        'scripts',
        'tests',
        'docs',
        'config',
        'data/raw',
        'data/processed',
        'models',
        'metrics',
        'plots',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    # Create __init__.py files for Python packages
    init_dirs = ['scripts', 'tests']
    for directory in init_dirs:
        init_file = Path(directory) / '__init__.py'
        init_file.touch()

def create_essential_files():
    """Create essential project files"""
    print("📄 Creating essential project files...")
    
    # README.md
    readme_content = f"""# Loan Default Prediction MLOps Pipeline

A complete MLOps pipeline for loan default prediction featuring:

- **MLflow** for experiment tracking and model registry
- **DVC** for data versioning and pipeline automation
- **GitHub Actions** for CI/CD
- **AWS** infrastructure (RDS, S3, EKS)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/loan-default-mlops.git
cd loan-default-mlops

# Install dependencies
pip install -r requirements.txt

# Initialize DVC and pull data
dvc pull

# Run the pipeline
dvc repro
```

## Model Performance

- **Test Accuracy**: 78%
- **AUC Score**: 79.27%
- **Model Type**: Random Forest

## Infrastructure

- **MLflow**: http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com
- **S3 Bucket**: mlflow-artifacts-365021531163-ap-south-1
- **Region**: ap-south-1

## CI/CD Pipeline

- Automated training on code changes
- Quality gates with testing
- Multi-environment deployment
- Continuous monitoring

## Documentation

- [Setup Guide](docs/setup.md)
- [Pipeline Documentation](docs/pipeline.md)
- [Troubleshooting](docs/troubleshooting.md)

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # requirements.txt
    requirements_content = """# Core ML libraries
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# MLflow for experiment tracking
mlflow>=2.8.0
boto3>=1.28.0

# DVC for data versioning
dvc>=3.30.0
dvc[s3]>=3.30.0

# Data processing
PyYAML>=6.0
joblib>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Code quality
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0

# Utilities
python-dotenv>=1.0.0
requests>=2.31.0
psutil>=5.9.0

# AWS CLI
awscli>=1.29.0
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    # params.yaml
    params_content = """data_preprocessing:
  test_size: 0.2
  random_state: 42
  scale_features: true

model_training:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
  min_samples_leaf: 2
  random_state: 42

model_evaluation:
  classification_threshold: 0.5
  metrics: ['accuracy', 'precision', 'recall', 'f1', 'auc']
"""
    
    with open('params.yaml', 'w', encoding='utf-8') as f:
        f.write(params_content)
    
    print("✅ Created README.md")
    print("✅ Created requirements.txt")
    print("✅ Created params.yaml")

def initialize_git():
    """Initialize Git repository"""
    print("🔧 Initializing Git repository...")
    
    try:
        # Initialize git
        subprocess.run(['git', 'init'], check=True)
        subprocess.run(['git', 'branch', '-M', 'main'], check=True)
        
        # Set basic config (user should customize)
        subprocess.run(['git', 'config', 'user.name', 'MLOps Pipeline'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'mlops@example.com'], check=True)
        
        print("✅ Git repository initialized")
        print("💡 Remember to set your own git config:")
        print("   git config user.name 'Your Name'")
        print("   git config user.email 'your.email@example.com'")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git initialization failed: {e}")

def create_github_actions_structure():
    """Create GitHub Actions workflow structure"""
    print("🔧 Creating GitHub Actions structure...")
    
    # Create workflow directory
    workflows_dir = Path('.github/workflows')
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files
    workflow_files = [
        'training-pipeline.yml',
        'deployment.yml',
        'testing-quality.yml', 
        'monitoring.yml'
    ]
    
    for workflow_file in workflow_files:
        placeholder_path = workflows_dir / workflow_file
        with open(placeholder_path, 'w', encoding='utf-8') as f:
            f.write(f"""# {workflow_file.replace('-', ' ').replace('.yml', '').title()} Workflow
# This file will be replaced with the actual workflow content
# Please save the corresponding GitHub Actions workflow artifact here

name: {workflow_file.replace('-', ' ').replace('.yml', '').title()}

on:
  workflow_dispatch:

jobs:
  placeholder:
    runs-on: ubuntu-latest
    steps:
    - name: Placeholder
      run: echo "Replace this with actual workflow content"
""")
        print(f"   📄 Created placeholder: {workflow_file}")
    
    # Create issue template
    issue_template = """---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior.

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Windows, macOS, Linux]
- Python version: [e.g. 3.10]
- MLflow version: [e.g. 2.8.0]

**Additional context**
Any other context about the problem.
"""
    
    with open('.github/ISSUE_TEMPLATE/bug_report.md', 'w', encoding='utf-8') as f:
        f.write(issue_template)
    
    print("✅ Created GitHub Actions structure")

def create_setup_instructions():
    """Create setup instructions"""
    print("📋 Creating setup instructions...")
    
    instructions = f"""# Setup Instructions for CI/CD Pipeline

## Current Status ✅
- Repository structure created
- Git initialized
- Basic files created
- GitHub Actions structure ready

## Next Steps:

### 1. Replace Workflow Files
Copy the following artifacts to `.github/workflows/`:
- GitHub Actions - Training Pipeline Workflow → training-pipeline.yml
- GitHub Actions - Deployment Workflow → deployment.yml  
- GitHub Actions - Testing and Quality Workflow → testing-quality.yml
- GitHub Actions - Monitoring Workflow → monitoring.yml

### 2. Copy Your Existing Scripts
Copy your working scripts to the `scripts/` directory:
- preprocess_data.py
- train_model_dvc.py
- evaluate_model.py
- deploy_model_working.py
- (and any other scripts you have)

### 3. Copy DVC Configuration
Copy your existing DVC files:
- dvc.yaml
- dvc.lock (if exists)
- .dvc/config

### 4. Set Up Data
Copy your data DVC files:
- data/raw/X.csv.dvc
- data/raw/y.csv.dvc

### 5. Create GitHub Repository
```bash
# Create repository on GitHub.com
# Then connect local repo:
git remote add origin https://github.com/YOUR_USERNAME/loan-default-mlops.git
```

### 6. Set GitHub Secrets
In GitHub repository settings, add:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- (Optional) SLACK_WEBHOOK_URL

### 7. First Commit and Push
```bash
git add .
git commit -m "Initial MLOps pipeline setup"
git push -u origin main
```

### 8. Test CI/CD Pipeline
Make a small change and push to trigger the pipeline:
```bash
echo "# CI/CD Test" >> README.md
git add README.md
git commit -m "Test CI/CD pipeline"
git push
```

## Files Created:
- ✅ Repository structure
- ✅ .gitignore
- ✅ README.md
- ✅ requirements.txt
- ✅ params.yaml
- ✅ GitHub Actions structure
- ✅ Issue templates

## Next Session:
When ready, copy your existing working files and push to GitHub to activate the complete CI/CD pipeline!

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('SETUP_INSTRUCTIONS.md', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Created SETUP_INSTRUCTIONS.md")

def main():
    """Main setup function"""
    print("🚀 Setting up fresh MLOps repository with CI/CD")
    print("=" * 60)
    
    # Clean up old files
    clean_old_files()
    
    # Create fresh structure
    create_fresh_gitignore()
    create_directory_structure()
    create_essential_files()
    create_github_actions_structure()
    
    # Initialize Git
    initialize_git()
    
    # Create instructions
    create_setup_instructions()
    
    print("\n" + "=" * 60)
    print("✅ Fresh repository setup completed!")
    print("\n📋 Next steps:")
    print("1. Copy your existing working scripts to scripts/ directory")
    print("2. Replace placeholder workflow files with actual ones")
    print("3. Copy DVC configuration files")
    print("4. Create GitHub repository and push")
    print("5. Set up GitHub secrets")
    print("6. Test the CI/CD pipeline!")
    print("\n📄 See SETUP_INSTRUCTIONS.md for detailed steps")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)