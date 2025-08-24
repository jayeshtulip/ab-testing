import requests
import time
import json
from datetime import datetime

def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=3)
        return response.status_code == 200, response.json()
    except:
        return False, None

def run_experiment_batch(batch_size=20):
    """Run a batch of experiment requests"""
    
    # Diverse user profiles
    profiles = [
        [30000, 60000, 720, 0.2, 5],   # Conservative loan
        [80000, 90000, 680, 0.3, 7],   # Moderate loan  
        [150000, 100000, 600, 0.6, 2], # Risky loan
        [200000, 80000, 550, 0.8, 1],  # Very risky loan
        [40000, 120000, 780, 0.15, 12] # Safe loan
    ]
    
    results = {"control": [], "treatment": []}
    
    for i in range(batch_size):
        user_id = f"user_{i:03d}"
        features = profiles[i % len(profiles)]
        
        try:
            response = requests.post(
                "http://127.0.0.1:8001/predict",
                json={"user_id": user_id, "features": features},
                timeout=2
            )
            
            if response.status_code == 200:
                result = response.json()
                group = result['experiment_group']
                probability = result['probability']
                results[group].append(probability)
                
        except:
            continue
    
    return results

def print_dashboard():
    """Print a simple dashboard"""
    print("\\n" + "="*60)
    print(" A/B TESTING DASHBOARD")
    print("="*60)
    print(f" Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check health
    healthy, health_data = check_api_health()
    if healthy:
        print(" API Status: HEALTHY")
        print(f" Control Model: {health_data.get('control_model', 'Unknown')}")
        print(f" Treatment Model: {health_data.get('treatment_model', 'Unknown')}")
    else:
        print(" API Status: UNHEALTHY")
        return
    
    # Run experiments
    print("\\n Running experiment batch...")
    results = run_experiment_batch(20)
    
    if results['control'] or results['treatment']:
        print("\\n RESULTS:")
        
        if results['control']:
            avg_control = sum(results['control']) / len(results['control'])
            print(f" Control Group: {len(results['control'])} requests, Avg Risk: {avg_control:.1%}")
        
        if results['treatment']:
            avg_treatment = sum(results['treatment']) / len(results['treatment'])
            print(f" Treatment Group: {len(results['treatment'])} requests, Avg Risk: {avg_treatment:.1%}")
        
        if results['control'] and results['treatment']:
            diff = avg_treatment - avg_control
            print(f" Difference: {diff:+.1%}")
    
    print("="*60)

if __name__ == "__main__":
    print(" A/B Testing Monitor - Press Ctrl+C to stop")
    
    try:
        while True:
            print_dashboard()
            print("  Updating in 30 seconds...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\\n Dashboard stopped")
