# Windows PowerShell Cleanup Script (Fixed)
# clean_project_fixed.ps1

Write-Host "Starting production-safe cleanup..." -ForegroundColor Green
Write-Host "Only removing files NOT used in successful pipeline runs" -ForegroundColor Yellow

# Function to safely remove file if it exists
function Safe-Remove-File {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        Write-Host "Removing: $FilePath" -ForegroundColor Red
        Remove-Item $FilePath -Force
    } else {
        Write-Host "Not found: $FilePath" -ForegroundColor Gray
    }
}

# Function to safely remove directory if it exists and is empty
function Safe-Remove-Directory {
    param([string]$DirPath)
    
    if (Test-Path $DirPath) {
        $items = Get-ChildItem $DirPath -Force
        if ($items.Count -eq 0) {
            Write-Host "Removing empty directory: $DirPath" -ForegroundColor Red
            Remove-Item $DirPath -Force
        } else {
            Write-Host "Directory not empty, keeping: $DirPath" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Directory not found: $DirPath" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "STEP 1: Remove duplicate workflow files..." -ForegroundColor Cyan
Safe-Remove-File ".github\workflows\ka_mlops_cicd_fixed.yml"
Safe-Remove-File ".github\workflows\ka_mlops_cicd_fixed.yml.backup"
Safe-Remove-File ".github\workflows\ka_mlops_cicd_fixed.yml.broken"

Write-Host ""
Write-Host "STEP 2: Remove duplicate Dockerfile variants..." -ForegroundColor Cyan
Safe-Remove-File "Dockerfile.webhook"
Safe-Remove-File "auto_retrain_dockerfile"
Safe-Remove-File "complete_dockerfile"
Safe-Remove-File "complex_auto_retrain_dockerfile"
Safe-Remove-File "corrected_dockerfile"
Safe-Remove-File "fixed_auto_retrain_dockerfile"
Safe-Remove-File "full_auto_retrain_dockerfile"
Safe-Remove-File "minimal_dockerfile"
Safe-Remove-File "numpy_dockerfile"
Safe-Remove-File "prometheus_dockerfile"
Safe-Remove-File "webhook_dockerfile"

Write-Host ""
Write-Host "STEP 3: Remove duplicate API files..." -ForegroundColor Cyan
Safe-Remove-File "src\api_enhanced_fixed.py"
Safe-Remove-File "src\api_prometheus.py"
Safe-Remove-File "src\ka_api\auto_retrain_main.py"
Safe-Remove-File "src\ka_api\complete_auto_retrain.py"
Safe-Remove-File "src\ka_api\corrected_auto_retrain.py"
Safe-Remove-File "src\ka_api\fixed_auto_retrain.py"
Safe-Remove-File "src\ka_api\full_auto_retrain_api.py"
Safe-Remove-File "src\ka_api\minimal_api.py"
Safe-Remove-File "src\ka_api\numpy_api.py"
Safe-Remove-File "src\ka_api\prometheus_api.py"
Safe-Remove-File "src\ka_api\simple_auto_retrain.py"

Write-Host ""
Write-Host "STEP 4: Remove duplicate requirements files..." -ForegroundColor Cyan
Safe-Remove-File "requirements_enhanced.txt"
Safe-Remove-File "src\requirements_enhanced_fixed.txt"

Write-Host ""
Write-Host "STEP 5: Remove unused webhook files..." -ForegroundColor Cyan
Safe-Remove-File "webhook.py"
Safe-Remove-File "webhook_handler.py"

Write-Host ""
Write-Host "STEP 6: Remove duplicate Kubernetes configs..." -ForegroundColor Cyan
Safe-Remove-File "webhook-deploy.yaml"
Safe-Remove-File "webhook-k8s-fixed.yaml"
Safe-Remove-File "webhook-k8s.yaml"
Safe-Remove-File "metrics-api-fixed.yaml"
Safe-Remove-File "retraining-webhook.yaml"

Write-Host ""
Write-Host "STEP 7: Remove monitoring configs..." -ForegroundColor Cyan
Write-Host "These files are for monitoring setup:" -ForegroundColor Yellow
Write-Host "   - alertmanager.yaml"
Write-Host "   - grafana.yaml"
Write-Host "   - ka_alert_rules.yml"
Write-Host "   - prometheus-config.yaml"
Write-Host "   - prometheus-stack.yaml"
Write-Host "   - postgres-fixed.yaml"

$deleteMonitoring = Read-Host "Delete monitoring config files? (y/n)"
if ($deleteMonitoring -eq "y") {
    Safe-Remove-File "alertmanager.yaml"
    Safe-Remove-File "grafana.yaml"
    Safe-Remove-File "ka_alert_rules.yml"
    Safe-Remove-File "prometheus-config.yaml"
    Safe-Remove-File "prometheus-stack.yaml"
    Safe-Remove-File "postgres-fixed.yaml"
    Write-Host "Monitoring configs removed" -ForegroundColor Green
} else {
    Write-Host "Keeping monitoring configs" -ForegroundColor Green
}

Write-Host ""
Write-Host "STEP 8: Remove utility files..." -ForegroundColor Cyan
Safe-Remove-File "simple_fix.ps1"

Write-Host ""
Write-Host "STEP 9: Remove old monitoring code..." -ForegroundColor Cyan
Safe-Remove-File "src\monitoring\__init__.py"
Safe-Remove-File "src\monitoring\drift_monitor.py"
Safe-Remove-File "src\monitoring\metrics_collector.py"
Safe-Remove-File "src\monitoring\model_monitor.py"
Safe-Remove-Directory "src\monitoring"

Write-Host ""
Write-Host "CLEANUP COMPLETED!" -ForegroundColor Green
Write-Host ""
Write-Host "REMAINING ESSENTIAL FILES:" -ForegroundColor Cyan
Write-Host "- .github\workflows\mlops-auto-retrain.yml  (Active pipeline)"
Write-Host "- src\ka_api\database_auto_retrain.py       (Deployed API)"
Write-Host "- src\ka_api\loan-api-deployment.yaml       (K8s config)"
Write-Host "- src\trigger_github_workflow.py            (Trigger script)"
Write-Host "- models\                                   (DVC tracked)"
Write-Host "- data\                                     (DVC tracked)"
Write-Host "- .dvc\                                     (DVC config)"
Write-Host "- dvc.yaml                                  (DVC pipeline)"
Write-Host "- docs\                                     (Documentation)"
Write-Host "- scripts\                                  (Test scripts)"
Write-Host "- README.md                                 (Documentation)"
Write-Host ""
Write-Host "Your project is now clean and contains only actively used files!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review the remaining files"
Write-Host "2. git add ."
Write-Host "3. git commit -m 'Clean up unused duplicate files'"
Write-Host "4. git push origin main"

# Show what files remain
Write-Host ""
Write-Host "Remaining files in project:" -ForegroundColor Magenta
Get-ChildItem -Recurse -File | Select-Object FullName | ForEach-Object { 
    $_.FullName.Replace((Get-Location).Path + "\", "") 
} | Sort-Object