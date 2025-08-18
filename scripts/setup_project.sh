#!/bin/bash

# Loan Default MLOps Project Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Loan Default MLOps Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Windows (WSL or Git Bash)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || -n "$WSL_DISTRO_NAME" ]]; then
    WINDOWS=true
    print_status "Detected Windows environment"
else
    WINDOWS=false
    print_status "Detected Unix-like environment"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check Prerequisites
print_step "Checking prerequisites..."

# Check Python
if command_exists python; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    print_status "Python version: $PYTHON_VERSION"
else
    print_error "Python is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check pip
if ! command_exists pip; then
    print_error "pip is not installed. Please install pip first."
    exit 1
fi

# Check git
if ! command_exists git; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check Docker (optional but recommended)
if command_exists docker; then
    print_status "Docker is available"
    DOCKER_AVAILABLE=true
else
    print_warning "Docker is not available. Some features will be limited."
    DOCKER_AVAILABLE=false
fi

# Check kubectl (optional)
if command_exists kubectl; then
    print_status "kubectl is available"
    KUBECTL_AVAILABLE=true
else
    print_warning "kubectl is not available. Kubernetes deployment will be limited."
    KUBECTL_AVAILABLE=false
fi

# Step 2: Create Project Structure
print_step "Creating project directory structure..."

# Create main directories
mkdir -p data/{raw,processed,features,external}
mkdir -p src/{config,data,models,api/{routes,middleware,schemas},monitoring,utils}
mkdir -p frontend/{pages,components,static/{css,images}}
mkdir -p tests/{unit,integration,e2e}
mkdir -p infrastructure/{terraform,kubernetes,docker}
mkdir -p mlflow/experiments
mkdir -p monitoring/{prometheus,grafana/{dashboards,provisioning},alerts}
mkdir -p cicd/{.github/workflows,jenkins,scripts}
mkdir -p scripts
mkdir -p docs/images
mkdir -p notebooks
mkdir -p models
mkdir -p artifacts
mkdir -p metrics
mkdir -p plots
mkdir -p reports
mkdir -p logs

print_status "Directory structure created successfully"

# Step 3: Create Virtual Environment
print_step "Setting up Python virtual environment..."

if [[ "$WINDOWS" == true ]]; then
    python -m venv venv
    source venv/Scripts/activate
else
    python3 -m venv venv
    source venv/bin/activate
fi

print_status "Virtual environment created and activated"

# Step 4: Upgrade pip and install requirements
print_step "Installing Python dependencies..."

pip install --upgrade pip
pip install wheel setuptools

# Install requirements if file exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Requirements installed from requirements.txt"
else
    print_warning "requirements.txt not found. Installing basic dependencies..."
    pip install pandas numpy scikit-learn mlflow fastapi uvicorn streamlit pytest
fi

# Step 5: Create __init__.py files
print_step "Creating Python package structure..."

find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;
find frontend -type d -exec touch {}/__init__.py \;

print_status "Python package structure created"

# Step 6: Create environment file
print_step "Creating environment configuration..."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Created .env file from template"
        print_warning "Please edit .env file with your configuration"
    else
        cat > .env << EOF
# Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=loan_default_monitoring
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=change-this-secret-key

# AWS (fill in your values)
AWS_REGION=us-east-1
S3_DATA_BUCKET=loan-default-data
S3_MODEL_BUCKET=loan-default-models

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
METRICS_PORT=8001
EOF
        print_status "Created basic .env file"
        print_warning "Please configure .env file with your AWS and database settings"
    fi
fi

# Step 7: Initialize Git repository
print_step "Initializing Git repository..."

if [ ! -d ".git" ]; then
    git init
    
    # Create .gitignore
    cat > .gitignore << EOF
# Python
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
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.production

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Data files (use DVC instead)
data/raw/*.csv
data/processed/*.csv
data/features/*.csv

# Model files (use MLflow instead)
models/*.joblib
models/*.pkl

# Temporary files
tmp/
temp/

# Jupyter Notebooks
.ipynb_checkpoints/

# MLflow
mlruns/

# DVC
.dvc/tmp/

# Terraform
*.tfstate
*.tfstate.*
.terraform/

# Kubernetes secrets
*secret*.yaml

# Docker
.docker/

# Coverage reports
htmlcov/
.coverage
.pytest_cache/
EOF
    
    git add .gitignore
    git commit -m "Initial commit: project structure"
    print_status "Git repository initialized"
else
    print_status "Git repository already exists"
fi

# Step 8: Initialize DVC
print_step "Initializing DVC for data versioning..."

if command_exists dvc; then
    if [ ! -d ".dvc" ]; then
        dvc init --no-scm
        print_status "DVC initialized"
    else
        print_status "DVC already initialized"
    fi
else
    print_warning "DVC not installed. Install with: pip install dvc[s3]"
fi

# Step 9: Set up MLflow
print_step "Setting up MLflow..."

# Create MLflow configuration
cat > mlflow_config.py << EOF
import mlflow
import os

def setup_mlflow():
    """Setup MLflow tracking"""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    experiment_name = "loan_default_prediction"
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass  # Experiment already exists
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow configured with URI: {tracking_uri}")

if __name__ == "__main__":
    setup_mlflow()
EOF

print_status "MLflow configuration created"

# Step 10: Create Docker files
print_step "Creating Docker configuration..."

# API Dockerfile
cat > infrastructure/docker/Dockerfile.api << EOF
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY .env .env

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Frontend Dockerfile
cat > infrastructure/docker/Dockerfile.frontend << EOF
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY frontend/ ./frontend/
COPY src/ ./src/
COPY .env .env

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
EOF

# Docker Compose
cat > infrastructure/docker/docker-compose.yml << EOF
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: loan_default_monitoring
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mlflow:
    image: python:3.10-slim
    working_dir: /app
    command: >
      bash -c "pip install mlflow psycopg2-binary boto3 &&
               mlflow server --host 0.0.0.0 --port 5000 
               --backend-store-uri postgresql://postgres:password@postgres:5432/loan_default_monitoring
               --default-artifact-root ./mlruns"
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    volumes:
      - mlflow_data:/app/mlruns

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning

volumes:
  postgres_data:
  mlflow_data:
  prometheus_data:
  grafana_data:
EOF

print_status "Docker configuration created"

# Step 11: Create basic Kubernetes manifests
print_step "Creating Kubernetes manifests..."

# Namespace
cat > infrastructure/kubernetes/namespace.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: loan-default
  labels:
    name: loan-default
EOF

# API Deployment
cat > infrastructure/kubernetes/deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loan-default-api
  namespace: loan-default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: loan-default-api
  template:
    metadata:
      labels:
        app: loan-default-api
    spec:
      containers:
      - name: api
        image: loan-default-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
EOF

# Service
cat > infrastructure/kubernetes/service.yaml << EOF
apiVersion: v1
kind: Service
metadata:
  name: loan-default-api-service
  namespace: loan-default
spec:
  selector:
    app: loan-default-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
EOF

print_status "Kubernetes manifests created"

# Step 12: Create GitHub Actions workflow
print_step "Creating CI/CD pipeline..."

mkdir -p .github/workflows

cat > .github/workflows/ci.yml << EOF
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: password
          POSTGRES_DB: loan_default_monitoring
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -f infrastructure/docker/Dockerfile.api -t loan-default-api .
        docker build -f infrastructure/docker/Dockerfile.frontend -t loan-default-frontend .
EOF

print_status "CI/CD pipeline created"

# Step 13: Create setup completion summary
print_step "Setup completion summary..."

cat > SETUP_COMPLETE.md << EOF
# 🎉 Loan Default MLOps Project Setup Complete!

## ✅ What was set up:

### 1. Project Structure
- Complete directory structure for MLOps project
- Python package structure with __init__.py files
- Data, models, and artifacts directories

### 2. Development Environment
- Python virtual environment created
- Dependencies installed
- Environment configuration (.env file)

### 3. Version Control
- Git repository initialized
- .gitignore configured for ML projects
- DVC initialized for data versioning

### 4. MLflow Configuration
- MLflow tracking setup
- Experiment configuration

### 5. Containerization
- Docker configurations for API and frontend
- Docker Compose for local development
- Multi-service setup with PostgreSQL, MLflow, Prometheus, Grafana

### 6. Kubernetes
- Basic deployment manifests
- Service and ingress configurations
- Namespace setup

### 7. CI/CD Pipeline
- GitHub Actions workflow
- Automated testing and building

## 🚀 Next Steps:

### Immediate Actions:
1. **Configure .env file** with your AWS credentials and database settings
2. **Add your data** to data/raw/ directory (X.csv and y.csv)
3. **Start local services**: \`docker-compose -f infrastructure/docker/docker-compose.yml up -d\`

### Development Workflow:
1. **Activate virtual environment**: \`source venv/bin/activate\` (Linux/Mac) or \`venv\\Scripts\\activate\` (Windows)
2. **Run data preprocessing**: \`python src/data/preprocessing.py\`
3. **Train model**: \`python src/models/train.py\`
4. **Start API**: \`python src/api/main.py\`
5. **Start frontend**: \`streamlit run frontend/app.py\`

### Testing:
- **Run unit tests**: \`pytest tests/unit/\`
- **Run integration tests**: \`pytest tests/integration/\`
- **Run end-to-end tests**: \`pytest tests/e2e/\`

### Monitoring:
- **MLflow UI**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Frontend**: http://localhost:8501

### Production Deployment:
1. Configure AWS credentials and S3 buckets
2. Set up Kubernetes cluster
3. Configure monitoring and alerting
4. Deploy using CI/CD pipeline

## 📚 Documentation:
- Check docs/ directory for detailed guides
- API documentation at /docs endpoint
- Architecture diagrams in docs/images/

## 🆘 Need Help?
- Check logs in logs/ directory
- Review configuration in .env file
- Ensure all services are running with docker-compose

Happy coding! 🚀
EOF

print_status "Setup summary created in SETUP_COMPLETE.md"

# Step 14: Final checks and recommendations
print_step "Final setup verification..."

echo ""
echo "==================== SETUP COMPLETE ===================="
echo ""
print_status "✅ Project structure created"
print_status "✅ Virtual environment set up"
print_status "✅ Git repository initialized"
print_status "✅ Docker configuration ready"
print_status "✅ Kubernetes manifests created"
print_status "✅ CI/CD pipeline configured"
echo ""

print_warning "🔧 REQUIRED ACTIONS:"
echo "   1. Edit .env file with your configuration"
echo "   2. Add your data files to data/raw/"
echo "   3. Configure AWS credentials"
echo "   4. Start local services: docker-compose up -d"
echo ""

print_status "🎯 NEXT COMMANDS TO RUN:"
if [[ "$WINDOWS" == true ]]; then
    echo "   venv\\Scripts\\activate"
else
    echo "   source venv/bin/activate"
fi
echo "   python src/data/preprocessing.py"
echo "   python src/models/train.py"
echo "   python src/api/main.py"
echo ""

print_status "📖 Read SETUP_COMPLETE.md for detailed next steps"
echo ""
echo "🚀 Your MLOps project is ready for development!"
echo "=========================================================="