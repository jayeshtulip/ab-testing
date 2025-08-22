#!/usr/bin/env python3
"""
GitHub Workflow Trigger Script
Triggers MLOps auto-retraining workflow via GitHub API
"""

import os
import requests
import json
import sys
from datetime import datetime

# Get GitHub token from environment variable (SECURE METHOD)
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    print("❌ Error: GITHUB_TOKEN environment variable not set")
    print("Set it with: export GITHUB_TOKEN=your_token_here")
    sys.exit(1)

# Repository configuration
REPO_OWNER = "jayeshtulip"
REPO_NAME = "mlops_production"
WORKFLOW_FILE = "mlops-auto-retrain.yml"

def trigger_workflow(reason="manual_trigger", performance_data=None, alert_name="ModelPerformanceDegraded"):
    """
    Trigger the MLOps auto-retraining workflow
    
    Args:
        reason (str): Reason for retraining (prometheus_alert, grafana_alert, manual_trigger, etc.)
        performance_data (dict): Performance metrics data
        alert_name (str): Name of the alert that triggered retraining
    """
    
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    # Prepare inputs for workflow
    inputs = {
        "reason": reason,
        "performance_data": json.dumps(performance_data or {}),
        "alert_name": alert_name
    }
    
    payload = {
        "ref": "main",
        "inputs": inputs
    }
    
    print(f"🚀 Triggering MLOps workflow...")
    print(f"📊 Reason: {reason}")
    print(f"🚨 Alert: {alert_name}")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 204:
            print("✅ Workflow triggered successfully!")
            print(f"🔗 Check status at: https://github.com/{REPO_OWNER}/{REPO_NAME}/actions")
            return True
        else:
            print(f"❌ Failed to trigger workflow")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error triggering workflow: {e}")
        return False

def trigger_prometheus_alert(f1_score=0.65, accuracy=0.70, alert_name="ModelPerformanceDegraded"):
    """Simulate Prometheus alert trigger"""
    performance_data = {
        "f1_score": f1_score,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "source": "prometheus"
    }
    
    return trigger_workflow(
        reason="prometheus_alert",
        performance_data=performance_data,
        alert_name=alert_name
    )

def trigger_grafana_alert(metrics_data=None):
    """Simulate Grafana alert trigger"""
    default_metrics = {
        "model_drift": 0.15,
        "prediction_latency": 250,
        "error_rate": 0.05,
        "timestamp": datetime.now().isoformat(),
        "source": "grafana"
    }
    
    return trigger_workflow(
        reason="grafana_alert",
        performance_data=metrics_data or default_metrics,
        alert_name="ModelDriftDetected"
    )

def trigger_data_drift():
    """Trigger retraining due to data drift"""
    drift_data = {
        "drift_score": 0.23,
        "affected_features": ["credit_score", "income", "debt_to_income"],
        "timestamp": datetime.now().isoformat(),
        "source": "drift_monitor"
    }
    
    return trigger_workflow(
        reason="data_drift",
        performance_data=drift_data,
        alert_name="DataDriftDetected"
    )

if __name__ == "__main__":
    print("🔧 GitHub Workflow Trigger Tool")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python trigger_github_workflow.py <trigger_type>")
        print("")
        print("Available trigger types:")
        print("  prometheus    - Simulate Prometheus performance alert")
        print("  grafana       - Simulate Grafana monitoring alert")
        print("  drift         - Simulate data drift detection")
        print("  manual        - Manual trigger")
        print("")
        print("Examples:")
        print("  python trigger_github_workflow.py prometheus")
        print("  python trigger_github_workflow.py grafana")
        print("  python trigger_github_workflow.py drift")
        print("  python trigger_github_workflow.py manual")
        sys.exit(1)
    
    trigger_type = sys.argv[1].lower()
    
    if trigger_type == "prometheus":
        success = trigger_prometheus_alert()
    elif trigger_type == "grafana":
        success = trigger_grafana_alert()
    elif trigger_type == "drift":
        success = trigger_data_drift()
    elif trigger_type == "manual":
        success = trigger_workflow(reason="manual_trigger")
    else:
        print(f"❌ Unknown trigger type: {trigger_type}")
        sys.exit(1)
    
    if success:
        print("🎉 Workflow trigger completed successfully!")
    else:
        print("💥 Workflow trigger failed!")
        sys.exit(1)