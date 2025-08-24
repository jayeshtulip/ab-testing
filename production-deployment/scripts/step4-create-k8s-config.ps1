# Step 4: Create Kubernetes Secrets and Config

Write-Host " Step 4: Creating Kubernetes Configuration" -ForegroundColor Yellow

$dbUser = "mlops_user"
$dbPassword = "mlops_password"
$dbName = "ab_experiments"

# Step 4.1: Create Kubernetes Secret
Write-Host "`n=== Creating Kubernetes Secret ===" -ForegroundColor Cyan
kubectl create secret generic ab-testing-secrets -n loan-default `
  --from-literal=DB_PASSWORD=$dbPassword `
  --from-literal=DB_USER=$dbUser `
  --dry-run=client -o yaml | kubectl apply -f -

Write-Host " Created ab-testing-secrets" -ForegroundColor Green

# Step 4.2: Create ConfigMap YAML
Write-Host "`n=== Creating ConfigMap ===" -ForegroundColor Cyan

# Create kubernetes folder if it doesn't exist
if (!(Test-Path "production-deployment/kubernetes")) {
    New-Item -ItemType Directory -Path "production-deployment/kubernetes" -Force | Out-Null
}

$configMapYaml = @"
apiVersion: v1
kind: ConfigMap
metadata:
  name: ab-testing-config
  namespace: loan-default
data:
  # Database Configuration
  DB_HOST: "postgres.loan-default.svc.cluster.local"
  DB_PORT: "5432"
  DB_NAME: "ab_experiments"
  DB_SCHEMA: "ab_testing"
  
  # Service URLs
  MLOPS_API_URL: "http://ka-mlops-api-service.loan-default.svc.cluster.local"
  LOAN_API_URL: "http://loan-api-service.loan-default.svc.cluster.local:8002"
  
  # A/B Testing Settings
  DEFAULT_TRAFFIC_SPLIT: "0.5"
  EXPERIMENT_NAME: "loan_default_rf_vs_gb"
  CONTROL_MODEL: "random_forest_v1.0"
  TREATMENT_MODEL: "gradient_boost_v1.0"
  
  # Drift Detection
  DRIFT_THRESHOLD_PSI: "0.1"
  DRIFT_THRESHOLD_KS: "0.05"
  DRIFT_CHECK_INTERVAL: "300"
  
  # Application Settings
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  API_PORT: "8003"
  
  # MLflow Integration
  MLFLOW_TRACKING_URI: "http://ka-mlops-api-service.loan-default.svc.cluster.local"
"@

# Save ConfigMap YAML
$configMapYaml | Out-File -FilePath "production-deployment/kubernetes/ab-testing-config.yaml" -Encoding utf8

# Apply ConfigMap
kubectl apply -f "production-deployment/kubernetes/ab-testing-config.yaml"
Write-Host " Created and applied ab-testing-config ConfigMap" -ForegroundColor Green

# Step 4.3: Verify Kubernetes resources
Write-Host "`n=== Verifying Kubernetes Resources ===" -ForegroundColor Cyan

# Check secret
$secretCheck = kubectl get secret ab-testing-secrets -n loan-default -o name 2>&1
if ($secretCheck -match "secret/ab-testing-secrets") {
    Write-Host " Secret ab-testing-secrets exists" -ForegroundColor Green
} else {
    Write-Host " Secret not found: $secretCheck" -ForegroundColor Red
}

# Check configmap
$configCheck = kubectl get configmap ab-testing-config -n loan-default -o name 2>&1
if ($configCheck -match "configmap/ab-testing-config") {
    Write-Host " ConfigMap ab-testing-config exists" -ForegroundColor Green
} else {
    Write-Host " ConfigMap not found: $configCheck" -ForegroundColor Red
}

# Show config details
Write-Host "`n=== Configuration Details ===" -ForegroundColor Cyan
kubectl get configmap ab-testing-config -n loan-default -o yaml | Select-String -Pattern "data:" -A 20

Write-Host "`n Step 4 Complete!" -ForegroundColor Green
Write-Host " Kubernetes secrets created" -ForegroundColor Green
Write-Host " ConfigMap created and applied" -ForegroundColor Green
Write-Host " All configuration ready for A/B testing service" -ForegroundColor Green
