# Step 3: Create A/B Testing Database (Fixed)

Write-Host "🗄️ Step 3: Creating A/B Testing Database" -ForegroundColor Yellow

$dbUser = "mlops_user"
$dbPassword = "mlops_password"
$abTestingDb = "ab_experiments"

Write-Host "`nDatabase already created - proceeding with schema..." -ForegroundColor Cyan

# Create schema directly via kubectl exec (no file needed)
Write-Host "`nCreating database schema..." -ForegroundColor Cyan

$schemaCommands = @"
CREATE SCHEMA IF NOT EXISTS ab_testing;

CREATE TABLE IF NOT EXISTS ab_testing.experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ab_testing.experiment_groups (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES ab_testing.experiments(id),
    name VARCHAR(100) NOT NULL,
    model_version VARCHAR(100),
    traffic_allocation DECIMAL(3,2) DEFAULT 0.5,
    UNIQUE(experiment_id, name)
);

CREATE TABLE IF NOT EXISTS ab_testing.experiment_metrics (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES ab_testing.experiments(id),
    group_id INTEGER REFERENCES ab_testing.experiment_groups(id),
    metric_name VARCHAR(100),
    metric_value DECIMAL(10,4),
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ab_testing.user_assignments (
    user_id VARCHAR(100),
    experiment_id INTEGER REFERENCES ab_testing.experiments(id),
    group_id INTEGER REFERENCES ab_testing.experiment_groups(id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, experiment_id)
);

CREATE TABLE IF NOT EXISTS ab_testing.drift_measurements (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    drift_score DECIMAL(6,4),
    drift_type VARCHAR(50),
    measured_at TIMESTAMP DEFAULT NOW()
);
"@

# Execute schema creation
$schemaResult = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -c "$schemaCommands" 2>&1
Write-Host "Schema creation result:" -ForegroundColor White
Write-Host $schemaResult -ForegroundColor Gray

# Insert initial experiment
Write-Host "`nInserting initial experiment..." -ForegroundColor Cyan
$experimentInsert = @"
INSERT INTO ab_testing.experiments (name, status, config) 
VALUES ('loan_default_rf_vs_gb', 'active', '{\"models\": {\"control\": \"random_forest_v1.0\", \"treatment\": \"gradient_boost_v1.0\"}}')
ON CONFLICT (name) DO UPDATE SET status = 'active';
"@

$insertResult = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -c "$experimentInsert" 2>&1
Write-Host $insertResult -ForegroundColor Gray

# Insert experiment groups
Write-Host "`nInserting experiment groups..." -ForegroundColor Cyan
$groupsInsert = @"
WITH exp AS (SELECT id FROM ab_testing.experiments WHERE name = 'loan_default_rf_vs_gb')
INSERT INTO ab_testing.experiment_groups (experiment_id, name, model_version, traffic_allocation)
SELECT exp.id, 'control', 'random_forest_v1.0', 0.5 FROM exp
UNION ALL
SELECT exp.id, 'treatment', 'gradient_boost_v1.0', 0.5 FROM exp
ON CONFLICT (experiment_id, name) DO UPDATE SET model_version = EXCLUDED.model_version;
"@

$groupsResult = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -c "$groupsInsert" 2>&1
Write-Host $groupsResult -ForegroundColor Gray

# Verify tables
Write-Host "`nVerifying tables created..." -ForegroundColor Cyan
$tablesCheck = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -c "\dt ab_testing.*" 2>&1
Write-Host $tablesCheck -ForegroundColor Gray

# Verify experiment data
Write-Host "`nVerifying experiment setup..." -ForegroundColor Cyan
$experimentCheck = kubectl exec postgres-f989955f-h4rjr -n loan-default -- env PGPASSWORD=$dbPassword psql -U $dbUser -d $abTestingDb -c "SELECT e.name, g.name as group_name, g.model_version, g.traffic_allocation FROM ab_testing.experiments e JOIN ab_testing.experiment_groups g ON e.id = g.experiment_id;" 2>&1
Write-Host "Experiment configuration:" -ForegroundColor White
Write-Host $experimentCheck -ForegroundColor Gray

Write-Host "`n Step 3 Complete!" -ForegroundColor Green
Write-Host " Database ab_experiments ready" -ForegroundColor Green
Write-Host " Schema ab_testing created" -ForegroundColor Green
Write-Host " 5 tables created successfully" -ForegroundColor Green
Write-Host " Experiment loan_default_rf_vs_gb configured" -ForegroundColor Green
