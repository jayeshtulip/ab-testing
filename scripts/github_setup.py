import os
import json
import subprocess
from pathlib import Path

def check_git_status():
    """Check current Git status"""
    print("🔍 Checking Git repository status...")
    
    try:
        # Check if Git is initialized
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository is initialized")
            print("📋 Current status:")
            print(result.stdout)
            return True
        else:
            print("❌ Git repository not initialized")
            return False
    except FileNotFoundError:
        print("❌ Git not installed")
        return False

def initialize_git_repo():
    """Initialize Git repository if not exists"""
    print("🔧 Initializing Git repository...")
    
    commands = [
        ['git', 'init'],
        ['git', 'branch', '-M', 'main'],
        ['git', 'config', 'user.name', 'MLOps Pipeline'],
        ['git', 'config', 'user.email', 'mlops@example.com']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ {' '.join(cmd)} - {e}")

def create_gitignore():
    """Create comprehensive .gitignore file"""
    print("📝 Creating .gitignore file...")
    
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

# Model artifacts (large files - use DVC instead)
*.pkl
*.joblib
*.h5
*.pb

# Data files (use DVC instead)
*.csv
*.parquet
*.json
!params.yaml
!dvc.yaml
!.dvc/

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp

# AWS credentials (never commit these!)
.aws/
aws_credentials.txt

# Secrets
secrets.yaml
.secrets
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def create_github_actions_structure():
    """Create GitHub Actions directory structure"""
    print("📁 Creating GitHub Actions directory structure...")
    
    # Create .github/workflows directory
    workflows_dir = Path('.github/workflows')
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Create other necessary directories
    directories = [
        '.github/ISSUE_TEMPLATE',
        '.github/PULL_REQUEST_TEMPLATE',
        'tests',
        'docs',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")

def create_readme():
    """Create comprehensive README.md"""
    print("📝 Creating README.md...")
    
    readme_content = """# Loan Default Prediction MLOps Pipeline

## Overview

A complete MLOps pipeline for loan default prediction using:
- **MLflow** for experiment tracking and model registry
- **DVC** for data versioning and pipeline automation  
- **GitHub Actions** for CI/CD
- **AWS** for cloud infrastructure (RDS, S3, EKS)

## Architecture

```
Data → DVC Pipeline → MLflow → Model Registry → Deployment
  ↓         ↓           ↓           ↓            ↓
GitHub → CI/CD → Testing → Validation → Production
```

## Model Performance

- **Test Accuracy**: 78%
- **AUC Score**: 79.27%
- **Model Type**: Random Forest
- **Features**: 20 (13 categorical, 7 numerical)

## Pipeline Stages

### 1. Data Preprocessing
- Categorical encoding with LabelEncoder
- Feature scaling with StandardScaler
- Train/test split with stratification

### 2. Model Training
- Random Forest classifier
- Hyperparameter optimization
- MLflow experiment tracking

### 3. Model Evaluation
- Comprehensive metrics calculation
- ROC curve and precision-recall analysis
- Feature importance analysis

## Setup & Usage

### Prerequisites
- Python 3.8+
- AWS account with configured credentials
- Git and DVC installed

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd loan-default-mlops

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc pull

# Run pipeline
dvc repro

# View experiments
mlflow ui
```

### Running Individual Stages
```bash
# Data preprocessing only
python scripts/preprocess_data.py

# Model training only  
python scripts/train_model_dvc.py

# Model evaluation only
python scripts/evaluate_model.py
```

## Configuration

### Parameters (params.yaml)
```yaml
data_preprocessing:
  test_size: 0.2
  random_state: 42
  scale_features: true

model_training:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
  min_samples_leaf: 2
  random_state: 42
```

### DVC Pipeline (dvc.yaml)
- **data_preprocessing**: Raw data → Processed data
- **train_model**: Processed data → Trained model
- **evaluate_model**: Model → Metrics & plots

## Infrastructure

### MLflow Server
- **URL**: http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com
- **Backend**: PostgreSQL on AWS RDS
- **Artifacts**: S3 bucket

### DVC Remote Storage
- **S3 Bucket**: mlflow-artifacts-365021531163-ap-south-1
- **Region**: ap-south-1

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_preprocessing.py
pytest tests/test_model.py
pytest tests/test_pipeline.py
```

## Monitoring

- Model performance metrics tracked in MLflow
- Data drift detection with DVC
- Pipeline execution monitoring via GitHub Actions

## Deployment

### Manual Deployment
```bash
python scripts/deploy_model_working.py
```

### Automated Deployment (CI/CD)
- Triggered on push to `main` branch
- Automatic model validation
- Staging environment deployment
- Production deployment with approval

## Documentation

- [Setup Guide](docs/setup.md)
- [Pipeline Documentation](docs/pipeline.md)
- [API Documentation](docs/api.md)
- [Troubleshooting](docs/troubleshooting.md)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with MLflow, DVC, and GitHub Actions
- Deployed on AWS infrastructure
- German Credit Dataset for loan default prediction
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created README.md")

def create_requirements_txt():
    """Create requirements.txt with all dependencies"""
    print("📝 Creating requirements.txt...")
    
    requirements = """# Core ML libraries
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0

# MLflow
mlflow>=2.0.0
boto3>=1.24.0

# DVC
dvc>=2.45.0
dvc[s3]>=2.45.0

# Data processing
PyYAML>=6.0
joblib>=1.1.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Code quality
black>=22.0.0
flake8>=5.0.0
isort>=5.10.0

# Utilities
python-dotenv>=0.19.0
requests>=2.28.0

# AWS
awscli>=1.25.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def create_github_templates():
    """Create GitHub issue and PR templates"""
    print("📝 Creating GitHub templates...")
    
    # Issue template
    issue_template = """---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
- OS: [e.g. Windows, macOS, Linux]
- Python version: [e.g. 3.8.10]
- MLflow version: [e.g. 2.0.1]
- DVC version: [e.g. 2.45.0]

**Additional context**
Add any other context about the problem here.
"""
    
    with open('.github/ISSUE_TEMPLATE/bug_report.md', 'w') as f:
        f.write(issue_template)
    
    # PR template
    pr_template = """## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Model Performance
If this change affects model performance:
- [ ] Model accuracy maintained or improved
- [ ] Training time acceptable
- [ ] Pipeline execution successful

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive information committed
"""
    
    with open('.github/PULL_REQUEST_TEMPLATE/pull_request_template.md', 'w') as f:
        f.write(pr_template)
    
    print("✅ Created GitHub templates")

def main():
    """Main setup function"""
    print("🚀 Setting up GitHub Repository for CI/CD")
    print("=" * 60)
    
    # Check Git status
    git_exists = check_git_status()
    
    if not git_exists:
        initialize_git_repo()
    
    # Create necessary files and directories
    create_gitignore()
    create_github_actions_structure()
    create_readme()
    create_requirements_txt()
    create_github_templates()
    
    print("\n" + "=" * 60)
    print("✅ GitHub repository setup completed!")
    print("\n📋 Next steps:")
    print("1. Create GitHub repository online")
    print("2. Add AWS credentials as GitHub secrets")
    print("3. Create GitHub Actions workflows")
    print("4. Push code to GitHub")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)