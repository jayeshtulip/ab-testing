# Step 2: Test PostgreSQL Connection

Write-Host " Step 2: Testing PostgreSQL Connection" -ForegroundColor Yellow

$dbUser = "mlops_user"
$dbPassword = "mlops_password"
$existingDb = "mlops"

Write-Host "`nTesting PostgreSQL connection..." -ForegroundColor Cyan
$connectionResult = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $existingDb -c "SELECT current_database(), current_user;" 2>&1

if ($connectionResult -match "mlops") {
    Write-Host " PostgreSQL connection successful!" -ForegroundColor Green
    Write-Host $connectionResult -ForegroundColor White
} else {
    Write-Host "❌ Connection failed: $connectionResult" -ForegroundColor Red
}

Write-Host "`n✅ Step 2 Complete - Ready for Step 3!" -ForegroundColor Green
