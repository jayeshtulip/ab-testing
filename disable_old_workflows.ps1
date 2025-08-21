# disable_old_workflows.ps1
# Script to disable old workflow files and keep only ka_mlops_cicd_fixed.yml active

Write-Host "Disabling Old Workflows Script" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

# Define workflow files to disable
$workflowsToDisable = @(
    "training-pipeline.yml",
    "deployment.yml", 
    "monitoring.yml",
    "model-training.yml",
    "testing-quality.yml"
)

# Define the disable header to add
$disableHeader = @"
# ===========================================
# DISABLED - REPLACED BY ka_mlops_cicd_fixed.yml
# ===========================================
# This workflow has been replaced by the Ka-MLOps pipeline
# Only keeping for reference - will not run automatically
# Last disabled: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

on:
  workflow_dispatch:
    inputs:
      reason:
        description: 'Manual run reason (legacy testing only)'
        required: false
        default: 'Legacy testing - use ka_mlops_cicd_fixed.yml instead'

# ==================================================
# ORIGINAL TRIGGERS DISABLED - COMMENTED OUT BELOW:
# ==================================================
"@

$workflowDir = ".github\workflows"

# Check if workflows directory exists
if (-not (Test-Path $workflowDir)) {
    Write-Host "ERROR: Workflows directory not found: $workflowDir" -ForegroundColor Red
    exit 1
}

Write-Host "Found workflows directory: $workflowDir" -ForegroundColor Green

# Process each workflow file
foreach ($workflow in $workflowsToDisable) {
    $filePath = Join-Path $workflowDir $workflow
    
    if (Test-Path $filePath) {
        Write-Host "Processing: $workflow" -ForegroundColor Yellow
        
        # Read the original content
        $originalContent = Get-Content $filePath -Raw
        
        # Check if already disabled
        if ($originalContent -match "DISABLED - REPLACED BY ka_mlops_cicd_fixed.yml") {
            Write-Host "  Already disabled: $workflow" -ForegroundColor Green
            continue
        }
        
        # Find the original 'on:' section and comment it out
        $modifiedContent = $originalContent -replace "^on:", "# DISABLED - ORIGINAL TRIGGER:`n# on:"
        
        # Add disable header at the top
        $newContent = $disableHeader + "`n`n" + $modifiedContent
        
        # Create backup
        $backupPath = $filePath + ".backup"
        Copy-Item $filePath $backupPath
        Write-Host "  Backup created: $workflow.backup" -ForegroundColor Cyan
        
        # Write the modified content
        Set-Content $filePath $newContent -Encoding UTF8
        Write-Host "  SUCCESS: Disabled $workflow" -ForegroundColor Green
        
    } else {
        Write-Host "  Not found: $workflow" -ForegroundColor Yellow
    }
}

# Update the test script reference
$testScriptPath = "scripts\test_phase2c_pipeline.py"
if (Test-Path $testScriptPath) {
    Write-Host "Updating test script reference..." -ForegroundColor Yellow
    
    $testContent = Get-Content $testScriptPath -Raw
    $updatedTestContent = $testContent -replace "'training-pipeline.yml',", "# 'training-pipeline.yml',  # DISABLED"
    
    if ($testContent -ne $updatedTestContent) {
        # Create backup
        Copy-Item $testScriptPath ($testScriptPath + ".backup")
        Set-Content $testScriptPath $updatedTestContent -Encoding UTF8
        Write-Host "  Updated test script reference" -ForegroundColor Green
    } else {
        Write-Host "  No changes needed in test script" -ForegroundColor Blue
    }
}

# Show summary
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "========" -ForegroundColor Cyan

$allWorkflows = Get-ChildItem $workflowDir -Filter "*.yml" | Where-Object { $_.Name -ne "ka_mlops_cicd_fixed.yml" }
$disabledCount = 0
$activeCount = 0

foreach ($workflow in $allWorkflows) {
    $content = Get-Content $workflow.FullName -Raw
    if ($content -match "DISABLED - REPLACED BY ka_mlops_cicd_fixed.yml") {
        Write-Host "  DISABLED: $($workflow.Name)" -ForegroundColor Red
        $disabledCount++
    } else {
        Write-Host "  ACTIVE: $($workflow.Name)" -ForegroundColor Yellow
        $activeCount++
    }
}

Write-Host "  PRIMARY: ka_mlops_cicd_fixed.yml" -ForegroundColor Green

Write-Host "`nResults:" -ForegroundColor Cyan
Write-Host "  • Disabled workflows: $disabledCount" -ForegroundColor Red
Write-Host "  • Other active workflows: $activeCount" -ForegroundColor Yellow  
Write-Host "  • Primary Ka pipeline: 1" -ForegroundColor Green

if ($activeCount -eq 0) {
    Write-Host "`nSUCCESS: Only Ka pipeline will run automatically!" -ForegroundColor Green
} else {
    Write-Host "`nWARNING: $activeCount other workflows still active" -ForegroundColor Yellow
}

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "1. Review the disabled workflows in VS Code" -ForegroundColor White
Write-Host "2. Commit the changes:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Gray
Write-Host "   git commit -m `"Disable legacy workflows - Ka pipeline only`"" -ForegroundColor Gray
Write-Host "   git push origin main" -ForegroundColor Gray
Write-Host "3. Test your next commit - should trigger only 1 workflow!" -ForegroundColor White

Write-Host "`nScript completed successfully!" -ForegroundColor Green