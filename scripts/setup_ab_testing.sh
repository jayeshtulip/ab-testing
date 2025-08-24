# =============================================================================
# Manual Windows Setup for A/B Testing MLOps Pipeline
# Run these commands one by one in PowerShell
# =============================================================================

# First, let's create the basic directory structure
Write-Host "🏗️ Creating project structure..." -ForegroundColor Green

# Create directories
$directories = @(
    "src\ab_testing",
    "tests", 
    "data",
    "models\control",
    "models\treatment",
    "results",
    "metrics", 
    "plots",
    "logs",
    "monitoring\grafana\dashboards",
    "monitoring\grafana\provisioning",
    "deployments\kubernetes",
    "experiments\configs",
    "notebooks",
    "scripts"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "✅ Created: $dir" -ForegroundColor Green
}

# Create __init__.py files
New-Item -ItemType File -Path "src\__init__.py" -Force | Out-Null
New-Item -ItemType File -Path "src\ab_testing\__init__.py" -Force | Out-Null
New-Item -ItemType File -Path "tests\__init__.py" -Force | Out-Null

Write-Host "✅ Created Python package structure" -ForegroundColor Green

# =============================================================================
# Create requirements.txt
# =============================================================================

Write-Host "📦 Creating requirements.txt..." -ForegroundColor Cyan

$requirements = @"
# Core ML and Data Science
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
xgboost==2.0.1
joblib==1.3.2

# API and Web Framework  
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# Database and Caching
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
redis==5.0.1
aioredis==2.0.1

# MLOps and Experiment Tracking
mlflow==2.8.1
dvc[s3]==3.30.0

# AWS Integration
boto3==1.34.0
botocore==1.34.0

# Monitoring and Metrics
prometheus-client==0.19.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Utilities
python-dotenv==1.0.0
click==8.1.7
pyyaml==6.0.1
requests==2.31.0
matplotlib==3.7.2
seaborn==0.13.0

# Windows specific
psutil==5.9.0
"@

$requirements | Out-File -FilePath "requirements.txt" -Encoding UTF8
Write-Host "✅ Created requirements.txt" -ForegroundColor Green

# =============================================================================
# Create .env template
# =============================================================================

Write-Host "⚙️ Creating .env template..." -ForegroundColor Cyan

$envTemplate = @"
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1

# Database Configuration
DATABASE_URL=sqlite:///data/abtest.db
REDIS_URL=redis://localhost:6379

# MLflow Configuration  
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=ab_testing_experiments

# S3 Configuration for DVC
DVC_REMOTE_URL=s3://your-dvc-bucket/ab-testing/

# API Configuration
HOST=0.0.0.0
PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# Windows specific paths
PYTHONPATH=./src
TEMP_DIR=./temp
"@

$envTemplate | Out-File -FilePath ".env.template" -Encoding UTF8
Write-Host "✅ Created .env.template" -ForegroundColor Green

# Copy to .env if it doesn't exist
if (-not (Test-Path ".env")) {
    Copy-Item ".env.template" ".env"
    Write-Host "✅ Created .env file from template" -ForegroundColor Green
}

# =============================================================================
# Create .gitignore  
# =============================================================================

Write-Host "📋 Creating .gitignore..." -ForegroundColor Cyan

$gitignore = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
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
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Windows specific
*.bat
*.cmd
desktop.ini
Thumbs.db
*.lnk

# IDEs
.vscode/settings.json
.vscode/launch.json
.idea/
*.swp
*.swo

# Environment variables
.env
.env.local

# Data and Models
/data/*.csv
/data/*.db
/models/*.pkl
/models/*.joblib

# Logs
logs/
*.log

# Jupyter Notebooks
.ipynb_checkpoints/

# MLflow
mlruns/

# DVC
.dvc/cache/

# Temporary files
tmp/
temp/
*.tmp

# Test cache
.pytest_cache/
"@

$gitignore | Out-File -FilePath ".gitignore" -Encoding UTF8
Write-Host "✅ Created .gitignore" -ForegroundColor Green

# =============================================================================
# Install Python dependencies
# =============================================================================

Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

try {
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Some dependencies may have failed. This is normal." -ForegroundColor Yellow
    Write-Host "You can install them individually later if needed." -ForegroundColor Yellow
}

# =============================================================================
# Create VS Code configuration
# =============================================================================

Write-Host "⚙️ Creating VS Code configuration..." -ForegroundColor Cyan

# Create .vscode directory
New-Item -ItemType Directory -Path ".vscode" -Force | Out-Null

# VS Code settings.json
$vscodeSettings = @"
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        "mlruns": true,
        ".dvc/cache": true
    },
    "python.envFile": "`${workspaceFolder}/.env"
}
"@

$vscodeSettings | Out-File -FilePath ".vscode\settings.json" -Encoding UTF8

# VS Code tasks.json
$vscodeTasks = @"
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Install Dependencies",
            "type": "shell",
            "command": "python",
            "args": ["-m", "pip", "install", "-r", "requirements.txt"],
            "group": "build"
        },
        {
            "label": "Run Tests",
            "type": "shell", 
            "command": "python",
            "args": ["-m", "pytest", "tests/", "-v"],
            "group": "test",
            "options": {
                "env": {
                    "PYTHONPATH": "./src"
                }
            }
        },
        {
            "label": "Start API",
            "type": "shell",
            "command": "python", 
            "args": ["-m", "uvicorn", "src.ab_testing.ab_testing_api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            "group": "build",
            "options": {
                "env": {
                    "PYTHONPATH": "./src"
                }
            }
        }
    ]
}
"@

$vscodeTasks | Out-File -FilePath ".vscode\tasks.json" -Encoding UTF8

Write-Host "✅ Created VS Code configuration" -ForegroundColor Green

# =============================================================================
# Create Docker Compose for local development
# =============================================================================

Write-Host "🐳 Creating Docker Compose configuration..." -ForegroundColor Cyan

$dockerCompose = @"
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: abtest_db
      POSTGRES_USER: abtest_user  
      POSTGRES_PASSWORD: password123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U abtest_user -d abtest_db"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  postgres_data:
"@

$dockerCompose | Out-File -FilePath "docker-compose.yml" -Encoding UTF8
Write-Host "✅ Created docker-compose.yml" -ForegroundColor Green

# =============================================================================
# Create basic Python modules
# =============================================================================

Write-Host "🐍 Creating basic Python modules..." -ForegroundColor Cyan

# Create a simple data generator
$dataGenerator = @"
"""
Simple data generator for A/B testing
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
from datetime import datetime
import os

class SyntheticDataGenerator:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_baseline_data(self, n_samples: int = 1000, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate baseline synthetic data"""
        
        # Create realistic loan default features
        loan_amount = np.random.lognormal(10, 1, n_samples)
        income = np.random.lognormal(11, 0.8, n_samples)
        credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
        debt_to_income = np.random.beta(2, 5, n_samples).clip(0, 1)
        employment_years = np.random.exponential(5, n_samples).clip(0, 40)
        
        X = np.column_stack([loan_amount, income, credit_score, debt_to_income, employment_years])
        
        # Generate target based on features (more realistic relationship)
        risk_score = (
            (X[:, 0] / X[:, 1]) * 0.3 +  # debt-to-income ratio influence
            ((850 - X[:, 2]) / 100) * 0.4 +  # credit score influence (inverted)
            (X[:, 3]) * 0.2 +  # debt-to-income direct influence
            np.random.random(n_samples) * 0.1  # random noise
        )
        
        # Convert to binary classification (15% default rate)
        threshold = np.percentile(risk_score, 85)
        y = (risk_score > threshold).astype(int)
        
        feature_names = ['loan_amount', 'income', 'credit_score', 'debt_to_income', 'employment_years']
        
        return X, y, feature_names
    
    def save_data(self, X: np.ndarray, y: np.ndarray, filename: str = "data/baseline_data.csv"):
        """Save data to CSV file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df = pd.DataFrame(X, columns=['loan_amount', 'income', 'credit_score', 'debt_to_income', 'employment_years'])
        df['target'] = y
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    X, y, features = generator.generate_baseline_data(1000, 5)
    generator.save_data(X, y)
    print(f"Generated {len(X)} samples with {len(features)} features")
    print(f"Target distribution: {np.bincount(y)}")
"@

$dataGenerator | Out-File -FilePath "src\ab_testing\data_generator.py" -Encoding UTF8

# Create a simple API
$simpleAPI = @"
"""
Simple A/B Testing API for Windows
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import from our package
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

app = FastAPI(
    title="A/B Testing API - Windows Edition",
    description="Simple A/B Testing API for loan default prediction",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    user_id: str
    features: List[float]
    experiment_id: Optional[int] = None

class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float] = None
    model_version: str = "dummy"
    experiment_group: str = "control"

# Simple in-memory storage
experiments = {}
user_assignments = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the API"""
    # Create dummy model for testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Create and train a dummy model
    X_dummy = np.random.random((100, 5))
    y_dummy = np.random.choice([0, 1], 100)
    
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    X_scaled = scaler.fit_transform(X_dummy)
    model.fit(X_scaled, y_dummy)
    
    # Store globally
    app.state.model = model
    app.state.scaler = scaler
    
    logger.info("API initialized with dummy model")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction"""
    try:
        # Simple A/B assignment based on user_id hash
        user_hash = hash(request.user_id) % 100
        group = "treatment" if user_hash < 50 else "control"
        
        # Use the dummy model
        features = np.array([request.features])
        features_scaled = app.state.scaler.transform(features)
        
        prediction = app.state.model.predict(features_scaled)[0]
        probability = app.state.model.predict_proba(features_scaled)[0][1]
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version="dummy_v1.0",
            experiment_group=group
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "platform": "Windows",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "A/B Testing API is running on Windows!",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
"@

$simpleAPI | Out-File -FilePath "src\ab_testing\ab_testing_api.py" -Encoding UTF8

Write-Host "✅ Created Python modules" -ForegroundColor Green

# =============================================================================
# Create test script
# =============================================================================

Write-Host "🧪 Creating test script..." -ForegroundColor Cyan

$testScript = @"
#!/usr/bin/env python3
"""
Quick test script for Windows
"""
import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
        
        import pandas as pd
        print("✅ pandas imported successfully")
        
        from sklearn.ensemble import RandomForestClassifier
        print("✅ scikit-learn imported successfully")
        
        import fastapi
        print("✅ FastAPI imported successfully")
        
        from ab_testing.data_generator import SyntheticDataGenerator
        print("✅ Custom data generator imported successfully")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation"""
    print("\n🧪 Testing data generation...")
    
    try:
        from ab_testing.data_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        X, y, features = generator.generate_baseline_data(100, 5)
        
        assert X.shape == (100, 5), f"Expected (100, 5), got {X.shape}"
        assert y.shape == (100,), f"Expected (100,), got {y.shape}"
        assert len(features) == 5, f"Expected 5 features, got {len(features)}"
        
        print("✅ Data generation test passed")
        print(f"   Generated {len(X)} samples with {len(features)} features")
        print(f"   Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_model_training():
    """Test simple model training"""
    print("\n🧪 Testing model training...")
    
    try:
        from ab_testing.data_generator import SyntheticDataGenerator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Generate data
        generator = SyntheticDataGenerator()
        X, y, _ = generator.generate_baseline_data(200, 5)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("✅ Model training test passed")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Running A/B Testing Pipeline Tests on Windows")
    print("=" * 55)
    
    tests = [
        test_imports,
        test_data_generation,
        test_model_training
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 55)
    print(f"🎯 Test Results")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your setup is working correctly.")
        print("Next steps:")
        print("1. Start the API: python src/ab_testing/ab_testing_api.py")
        print("2. Open VS Code: code .")
        print("3. Check the API docs: http://localhost:8000/docs")
        return 0
    else:
        print(f"\n⚠️ {failed} tests failed. Please check your setup.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
"@

$testScript | Out-File -FilePath "scripts\test_ab_pipeline.py" -Encoding UTF8

# Create batch file for easy execution  
$batchFile = @"
@echo off
echo 🧪 Running A/B Testing Tests
echo ============================

set PYTHONPATH=%~dp0\..\src

python "%~dp0\test_ab_pipeline.py"

pause
"@

$batchFile | Out-File -FilePath "scripts\test_ab_pipeline.bat" -Encoding ASCII

Write-Host "✅ Created test scripts" -ForegroundColor Green

# =============================================================================
# Create PowerShell setup script for next time
# =============================================================================

Write-Host "📜 Creating setup script for future use..." -ForegroundColor Cyan

$setupScript = @'
# A/B Testing Setup Script for Windows
param(
    [switch]$SkipDocker,
    [switch]$SkipTests
)

Write-Host "🎯 A/B Testing MLOps Setup for Windows" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Install dependencies if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
}

# Start Docker services if not skipped
if (-not $SkipDocker) {
    Write-Host "🐳 Starting Docker services..." -ForegroundColor Cyan
    try {
        docker-compose up -d postgres redis
        Write-Host "✅ Docker services started" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Docker services failed to start" -ForegroundColor Yellow
    }
}

# Run tests if not skipped
if (-not $SkipTests) {
    Write-Host "🧪 Running tests..." -ForegroundColor Cyan
    python scripts\test_ab_pipeline.py
}

Write-Host "🎉 Setup completed!" -ForegroundColor Green
Write-Host "Next: code . (to open in VS Code)" -ForegroundColor Blue
'@

$setupScript | Out-File -FilePath "scripts\setup_ab_testing.ps1" -Encoding UTF8

Write-Host "✅ Created setup script" -ForegroundColor Green

# =============================================================================
# Final status and next steps
# =============================================================================

Write-Host "`n🎉 A/B Testing MLOps Project Created Successfully!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

Write-Host "`n📋 What was created:" -ForegroundColor Blue
Write-Host "✅ Project structure with all directories" -ForegroundColor White  
Write-Host "✅ requirements.txt with all dependencies" -ForegroundColor White
Write-Host "✅ VS Code configuration (.vscode folder)" -ForegroundColor White
Write-Host "✅ Docker Compose configuration" -ForegroundColor White
Write-Host "✅ Basic Python modules (data generator, API)" -ForegroundColor White
Write-Host "✅ Test scripts" -ForegroundColor White
Write-Host "✅ Environment configuration (.env)" -ForegroundColor White

Write-Host "`n🚀 Next Steps:" -ForegroundColor Blue
Write-Host "1. Test the setup:" -ForegroundColor Yellow
Write-Host "   .\scripts\test_ab_pipeline.bat" -ForegroundColor White

Write-Host "2. Open in VS Code:" -ForegroundColor Yellow  
Write-Host "   code ." -ForegroundColor White

Write-Host "3. Start the API:" -ForegroundColor Yellow
Write-Host "   python src\ab_testing\ab_testing_api.py" -ForegroundColor White

Write-Host "4. Check API docs:" -ForegroundColor Yellow
Write-Host "   http://localhost:8000/docs" -ForegroundColor White

Write-Host "`n🎯 Your A/B Testing pipeline is ready!" -ForegroundColor Green\