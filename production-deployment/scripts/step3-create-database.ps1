# Step 3: Create A/B Testing Database

Write-Host " Step 3: Creating A/B Testing Database" -ForegroundColor Yellow

$dbUser = "mlops_user"
$dbPassword = "mlops_password"
$abTestingDb = "ab_experiments"

Write-Host "`nCreating database: $abTestingDb" -ForegroundColor Cyan

# Create the database
$createDbResult = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword createdb -U $dbUser $abTestingDb 2>&1

if ($createDbResult -match "already exists") {
    Write-Host " Database already exists - continuing..." -ForegroundColor Yellow
} elseif ($createDbResult -eq "" -or $createDbResult -eq $null) {
    Write-Host " Database created successfully!" -ForegroundColor Green
} else {
    Write-Host "Result: $createDbResult" -ForegroundColor Cyan
}

# Verify database connection
Write-Host "`nVerifying database connection..." -ForegroundColor Cyan
$verifyDb = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -c "SELECT current_database();" 2>&1

if ($verifyDb -match $abTestingDb) {
    Write-Host " Successfully connected to ab_experiments!" -ForegroundColor Green
} else {
    Write-Host " Connection failed: $verifyDb" -ForegroundColor Red
    exit 1
}

# Create schema SQL
$schemaSQL = @"
-- Create A/B Testing Schema
CREATE SCHEMA IF NOT EXISTS ab_testing;

-- Experiments table
CREATE TABLE IF NOT EXISTS ab_testing.experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Experiment groups
CREATE TABLE IF NOT EXISTS ab_testing.experiment_groups (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES ab_testing.experiments(id),
    name VARCHAR(100) NOT NULL,
    model_version VARCHAR(100),
    traffic_allocation DECIMAL(3,2) DEFAULT 0.5,
    UNIQUE(experiment_id, name)
);

-- Metrics table
CREATE TABLE IF NOT EXISTS ab_testing.experiment_metrics (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES ab_testing.experiments(id),
    group_id INTEGER REFERENCES ab_testing.experiment_groups(id),
    metric_name VARCHAR(100),
    metric_value DECIMAL(10,4),
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- User assignments
CREATE TABLE IF NOT EXISTS ab_testing.user_assignments (
    user_id VARCHAR(100),
    experiment_id INTEGER REFERENCES ab_testing.experiments(id),
    group_id INTEGER REFERENCES ab_testing.experiment_groups(id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, experiment_id)
);

-- Drift measurements
CREATE TABLE IF NOT EXISTS ab_testing.drift_measurements (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    drift_score DECIMAL(6,4),
    drift_type VARCHAR(50),
    measured_at TIMESTAMP DEFAULT NOW()
);

-- Insert test experiment
INSERT INTO ab_testing.experiments (name, status, config) 
VALUES ('loan_default_rf_vs_gb', 'active', '{"models": {"control": "random_forest_v1.0", "treatment": "gradient_boost_v1.0"}}')
ON CONFLICT (name) DO UPDATE SET status = 'active';

-- Insert groups
WITH exp AS (SELECT id FROM ab_testing.experiments WHERE name = 'loan_default_rf_vs_gb')
INSERT INTO ab_testing.experiment_groups (experiment_id, name, model_version, traffic_allocation)
SELECT exp.id, 'control', 'random_forest_v1.0', 0.5 FROM exp
UNION ALL
SELECT exp.id, 'treatment', 'gradient_boost_v1.0', 0.5 FROM exp
ON CONFLICT (experiment_id, name) DO UPDATE SET model_version = EXCLUDED.model_version;

SELECT 'Schema setup complete!' as status;
"@

# Save and execute SQL
$schemaSQL | Out-File -FilePath "production-deployment/sql/ab_schema.sql" -Encoding utf8
Write-Host "`nCreating database schema..." -ForegroundColor Cyan

# Copy and execute SQL
kubectl cp "production-deployment/sql/ab_schema.sql" "loan-default/postgres-f989955f-h4rjr:/tmp/ab_schema.sql"
$schemaResult = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -f /tmp/ab_schema.sql 2>&1

Write-Host "Schema creation result:" -ForegroundColor White
Write-Host $schemaResult -ForegroundColor Gray

# Verify tables
Write-Host "`nVerifying tables created..." -ForegroundColor Cyan
$tablesCheck = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -c "\dt ab_testing.*" 2>&1
Write-Host $tablesCheck -ForegroundColor Gray

Write-Host "`n Step 3 Complete!" -ForegroundColor Green
Write-Host " Database ab_experiments created" -ForegroundColor Green
Write-Host " Schema ab_testing with tables ready" -ForegroundColor Green
Write-Host " Test experiment configured" -ForegroundColor Green
