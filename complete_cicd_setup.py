import os
import json
from pathlib import Path

def check_current_status():
    """Check the current status of the repository"""
    print("🔍 Checking current repository status...")
    
    # Check directory structure
    expected_dirs = ['.github/workflows', 'data/raw', 'models', 'metrics', 'scripts']
    
    print("\n📁 Directory structure:")
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - Missing")
            # Create missing directories
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ Created {dir_path}")
    
    # Check for key files
    key_files = ['dvc.yaml', 'params.yaml', 'requirements.txt', 'README.md']
    
    print("\n📄 Key files:")
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
    
    # Check workflow files
    workflow_dir = Path('.github/workflows')
    if workflow_dir.exists():
        workflows = list(workflow_dir.glob('*.yml')) + list(workflow_dir.glob('*.yaml'))
        print(f"\n⚙️ Current workflows ({len(workflows)}):")
        for workflow in workflows:
            print(f"   📄 {workflow.name}")
    else:
        print("\n❌ No .github/workflows directory found")
    
    return True

def create_workflow_files():
    """Create all required workflow files"""
    print("\n🔧 Creating GitHub Actions workflow files...")
    
    workflows_dir = Path('.github/workflows')
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the 4 main workflows with placeholder content
    workflows = {
        'training-pipeline.yml': '''name: Training Pipeline

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'scripts/**'
      - 'params.yaml'
      - 'dvc.yaml'
      - 'data/raw/**'
  pull_request:
    branches: [ main ]
    paths:
      - 'scripts/**'
      - 'params.yaml'
      - 'dvc.yaml'
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'Experiment name for this training run'
        required: true
        default: 'manual-training'

env:
  PYTHON_VERSION: '3.10'
  MLFLOW_TRACKING_URI: http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com

jobs:
  training:
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: 📦 Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Placeholder Training Step
      run: |
        echo "🚧 This is a placeholder training workflow"
        echo "📋 Replace this file with the complete Training Pipeline Workflow content"
        echo "💡 Experiment: ${{ github.event.inputs.experiment_name || 'automated' }}"
''',
        
        'deployment.yml': '''name: Model Deployment

on:
  workflow_run:
    workflows: ["Training Pipeline"]
    types: [completed]
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

env:
  PYTHON_VERSION: '3.10'
  MLFLOW_TRACKING_URI: http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: github.event.workflow_run.conclusion == 'success' || github.event.inputs.environment == 'staging'
    environment: staging
    
    steps:
    - name: 📥 Checkout code
      uses: actions/checkout@v4
    
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Placeholder Deployment Step
      run: |
        echo "🚧 This is a placeholder deployment workflow"
        echo "📋 Replace this file with the complete Deployment Workflow content"
        echo "💡 Environment: ${{ github.event.inputs.environment || 'staging' }}"
''',
        
        'testing-quality.yml': '''name: Testing & Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:

env:
  PYTHON_VERSION: '3.10'

jobs:
  code-quality:
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout code
      uses: actions/checkout@v4
    
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: 📦 Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install black flake8 isort pytest-cov
    
    - name: Placeholder Testing Step
      run: |
        echo "🚧 This is a placeholder testing workflow"
        echo "📋 Replace this file with the complete Testing & Quality Workflow content"
        echo "💡 Running basic quality checks..."
''',
        
        'monitoring.yml': '''name: Model Monitoring

on:
  schedule:
    - cron: '0 */6 * * *'
  workflow_dispatch:
    inputs:
      check_type:
        description: 'Type of monitoring check'
        required: true
        default: 'all'
        type: choice
        options:
        - all
        - performance
        - drift
        - infrastructure

env:
  PYTHON_VERSION: '3.10'
  MLFLOW_TRACKING_URI: http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com

jobs:
  infrastructure-health:
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout code
      uses: actions/checkout@v4
    
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Placeholder Monitoring Step
      run: |
        echo "🚧 This is a placeholder monitoring workflow"
        echo "📋 Replace this file with the complete Monitoring Workflow content"
        echo "💡 Check type: ${{ github.event.inputs.check_type || 'all' }}"
'''
    }
    
    # Create each workflow file
    created_count = 0
    for filename, content in workflows.items():
        workflow_path = workflows_dir / filename
        
        try:
            with open(workflow_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"   ✅ Created: {filename}")
            created_count += 1
        except Exception as e:
            print(f"   ❌ Failed to create {filename}: {e}")
    
    return created_count == len(workflows)

def create_github_secrets_guide():
    """Create a comprehensive guide for setting up GitHub secrets"""
    print("\n🔐 Creating GitHub secrets setup guide...")
    
    secrets_guide = {
        "repository_setup": {
            "repository_url": "https://github.com/YOUR_USERNAME/mlops_production",
            "secrets_path": "Settings → Secrets and variables → Actions → New repository secret"
        },
        "required_secrets": {
            "AWS_ACCESS_KEY_ID": {
                "description": "Your AWS Access Key ID for S3 and MLflow access",
                "priority": "CRITICAL",
                "used_by": ["DVC data pulling/pushing", "MLflow artifact storage", "S3 bucket access"]
            },
            "AWS_SECRET_ACCESS_KEY": {
                "description": "Your AWS Secret Access Key",
                "priority": "CRITICAL", 
                "used_by": ["DVC data pulling/pushing", "MLflow artifact storage", "S3 bucket access"]
            }
        },
        "optional_secrets": {
            "SLACK_WEBHOOK_URL": {
                "description": "Slack webhook URL for pipeline notifications",
                "priority": "LOW",
                "used_by": ["Notification workflows", "Alert systems"]
            },
            "NOTIFICATION_EMAIL": {
                "description": "Email for critical alerts",
                "priority": "LOW",
                "used_by": ["Email notifications", "Critical alerts"]
            }
        },
        "setup_instructions": [
            "1. Go to https://github.com/YOUR_USERNAME/mlops_production",
            "2. Click the 'Settings' tab at the top of the repository page",
            "3. In the left sidebar, click 'Secrets and variables' → 'Actions'",
            "4. Click the 'New repository secret' button",
            "5. Enter the secret name (exactly as shown above)",
            "6. Enter the secret value",
            "7. Click 'Add secret'",
            "8. Repeat for each required secret"
        ],
        "security_best_practices": [
            "🔒 Never commit secrets to your repository",
            "🔒 Use the principle of least privilege for AWS IAM",
            "🔒 Rotate secrets regularly (at least quarterly)",
            "🔒 Monitor secret usage in GitHub Actions logs",
            "🔒 Use environment-specific secrets for staging/production"
        ],
        "testing_secrets": [
            "After adding secrets, test by running:",
            "1. Go to Actions tab in your GitHub repository",
            "2. Select any workflow and click 'Run workflow'", 
            "3. Check the workflow logs for secret access",
            "4. Ensure no secret values are printed in logs"
        ]
    }
    
    try:
        with open('GITHUB_SECRETS_SETUP.json', 'w', encoding='utf-8') as f:
            json.dump(secrets_guide, f, indent=2)
        
        print("   ✅ Created GITHUB_SECRETS_SETUP.json")
        
        # Also create a markdown version for easier reading
        markdown_guide = """# GitHub Secrets Setup Guide

## Required Secrets

Go to: https://github.com/YOUR_USERNAME/mlops_production/settings/secrets/actions

### Critical Secrets (Required)

1. **AWS_ACCESS_KEY_ID**
   - Your AWS Access Key ID
   - Used for: DVC data access, MLflow artifacts, S3 operations

2. **AWS_SECRET_ACCESS_KEY**
   - Your AWS Secret Access Key
   - Used for: DVC data access, MLflow artifacts, S3 operations

### Optional Secrets

3. **SLACK_WEBHOOK_URL** (Optional)
   - Slack webhook for notifications
   - Used for: Pipeline notifications, alerts

## Setup Steps

1. Go to your GitHub repository
2. Click **Settings** tab
3. Click **Secrets and variables** → **Actions** in left sidebar
4. Click **New repository secret**
5. Add each secret name and value
6. Click **Add secret**

## Test Your Setup

After adding secrets, test the pipeline:

```bash
# Make a small change
echo "# Test CI/CD" >> README.md
git add README.md
git commit -m "Test CI/CD pipeline"
git push origin main
```

Then check the **Actions** tab to see workflows running.

## Security Notes

- 🔒 Never commit secrets to your repository
- 🔒 Rotate secrets regularly
- 🔒 Monitor secret usage in workflow logs
- 🔒 Use minimum required AWS permissions
"""
        
        with open('SECRETS_SETUP.md', 'w', encoding='utf-8') as f:
            f.write(markdown_guide)
        
        print("   ✅ Created SECRETS_SETUP.md")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create secrets guide: {e}")
        return False

def create_next_steps_guide():
    """Create a comprehensive next steps guide"""
    print("\n📋 Creating next steps guide...")
    
    next_steps_content = """# 🚀 Complete CI/CD Pipeline Setup - Next Steps

## Current Status ✅

Your repository structure is ready for CI/CD! Here's what was created:

- ✅ GitHub Actions workflow files (placeholders)
- ✅ Directory structure for MLOps pipeline
- ✅ GitHub secrets setup guide
- ✅ Documentation and instructions

## Immediate Next Steps

### 1. 🔄 Replace Workflow Placeholders

The workflow files created are placeholders. Replace them with the actual workflow content:

**Replace these files in `.github/workflows/`:**

1. **training-pipeline.yml** → Copy "GitHub Actions - Training Pipeline Workflow" artifact
2. **deployment.yml** → Copy "GitHub Actions - Deployment Workflow" artifact  
3. **testing-quality.yml** → Copy "GitHub Actions - Testing and Quality Workflow" artifact
4. **monitoring.yml** → Copy "GitHub Actions - Monitoring Workflow" artifact

### 2. 🔐 Set Up GitHub Secrets

**CRITICAL**: Add these secrets to your GitHub repository:

1. Go to: https://github.com/YOUR_USERNAME/mlops_production/settings/secrets/actions
2. Add `AWS_ACCESS_KEY_ID` (your AWS access key)
3. Add `AWS_SECRET_ACCESS_KEY` (your AWS secret key)

See `SECRETS_SETUP.md` for detailed instructions.

### 3. 📄 Copy Your Existing Scripts

Copy your working scripts to the `scripts/` directory:

```bash
# Copy your existing scripts
cp /path/to/your/preprocess_data.py scripts/
cp /path/to/your/train_model_dvc.py scripts/
cp /path/to/your/evaluate_model.py scripts/
cp /path/to/your/deploy_model_working.py scripts/
```

### 4. 📊 Copy DVC Configuration

Copy your existing DVC files:

```bash
# Copy DVC pipeline configuration
cp /path/to/your/dvc.yaml .
cp /path/to/your/params.yaml .

# Copy DVC directory (if different location)
cp -r /path/to/your/.dvc .
```

### 5. 🧪 Test the Pipeline

After completing steps 1-4:

```bash
# Stage all changes
git add .

# Commit the complete setup
git commit -m "Complete CI/CD pipeline setup with workflows"

# Push to trigger the pipeline
git push origin main
```

### 6. 📊 Monitor Results

1. Go to your GitHub repository
2. Click the **Actions** tab
3. Watch the workflows execute
4. Check for any errors in the logs
5. Verify successful completion

## Expected Workflow Behavior

### 🤖 Training Pipeline
- **Triggers**: Push to main, PR creation, manual dispatch
- **Actions**: 
  - Pulls data from DVC
  - Runs preprocessing → training → evaluation
  - Validates model performance
  - Registers model in MLflow
  - Pushes artifacts to S3

### 🚀 Deployment Pipeline
- **Triggers**: After successful training, manual dispatch
- **Actions**:
  - Validates model meets criteria
  - Deploys to staging automatically
  - Requires manual approval for production
  - Runs health checks and validation

### 🧪 Testing Pipeline
- **Triggers**: Every push/PR, scheduled daily
- **Actions**:
  - Code quality checks (Black, flake8, isort)
  - Unit tests and integration tests
  - Security scanning
  - Performance tests

### 📊 Monitoring Pipeline
- **Triggers**: Every 6 hours, manual dispatch
- **Actions**:
  - Checks MLflow server health
  - Monitors model performance
  - Detects data drift
  - Sends alerts for issues

## Troubleshooting

### Common Issues:

1. **Secrets not working**
   - Verify secret names match exactly
   - Check AWS credentials are valid
   - Ensure proper AWS permissions

2. **Workflows not triggering**
   - Check trigger conditions in workflow files
   - Verify branch names (main vs master)
   - Check file paths in trigger conditions

3. **DVC errors**
   - Verify DVC remote is configured
   - Check S3 bucket permissions
   - Ensure data files exist

4. **MLflow connection issues**
   - Verify MLflow server URL is accessible
   - Check network connectivity
   - Validate tracking URI format

## Success Criteria

When everything is working, you'll have:

- 🔄 Automatic training triggered by code changes
- 🚀 Automated deployment with proper gates
- 🧪 Continuous testing and quality assurance
- 📊 24/7 monitoring and alerting
- 📈 Complete MLOps workflow automation

## Next Phase: Production Optimization

After CI/CD is working:

1. **Performance Optimization**
   - Optimize model training time
   - Implement caching strategies
   - Parallel processing for large datasets

2. **Advanced Monitoring**
   - Custom metrics and dashboards
   - Real-time alerting systems
   - Data drift detection refinement

3. **Security Hardening**
   - Implement secret rotation
   - Add compliance checks
   - Security scanning automation

4. **Documentation**
   - API documentation generation
   - Runbook creation
   - Knowledge base development

---

🎉 **Congratulations on building a production-ready MLOps pipeline!**

For questions or issues, check the logs in GitHub Actions or review the troubleshooting section above.
"""
    
    try:
        with open('NEXT_STEPS.md', 'w', encoding='utf-8') as f:
            f.write(next_steps_content)
        
        print("   ✅ Created NEXT_STEPS.md")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create next steps guide: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Complete CI/CD Pipeline Setup")
    print("=" * 60)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Check current status
    if check_current_status():
        success_count += 1
    
    # Step 2: Create workflow files
    if create_workflow_files():
        success_count += 1
    
    # Step 3: Create secrets guide
    if create_github_secrets_guide():
        success_count += 1
    
    # Step 4: Create next steps guide
    if create_next_steps_guide():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Setup completed: {success_count}/{total_steps} steps successful")
    
    if success_count == total_steps:
        print("\n🎉 COMPLETE CI/CD SETUP SUCCESSFUL!")
        print("\n📋 IMMEDIATE NEXT STEPS:")
        print("1. 🔄 Replace workflow placeholder files with actual content")
        print("2. 🔐 Set up GitHub secrets (see SECRETS_SETUP.md)")
        print("3. 📄 Copy your existing scripts to scripts/ directory")
        print("4. 🧪 Push changes and test the pipeline")
        print("\n📖 See NEXT_STEPS.md for detailed instructions")
    else:
        print("\n⚠️ Some steps failed. Check the errors above and retry.")
    
    print("=" * 60)
    
    return success_count == total_steps

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)