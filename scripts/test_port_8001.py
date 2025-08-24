import requests
import json

# Test scenarios
scenarios = [
    {"name": "Low Risk", "user_id": "alice", "features": [50000, 150000, 800, 0.1, 10]},
    {"name": "High Risk", "user_id": "charlie", "features": [100000, 40000, 500, 0.7, 0.5]}
]

print("🧪 Testing Enhanced A/B API on Port 8001")
print("=" * 50)

for scenario in scenarios:
    try:
        response = requests.post("http://127.0.0.1:8001/predict", json=scenario, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"\n{scenario['name']} - User: {scenario['user_id']}")
            print(f"  Group: {result['experiment_group']}")
            print(f"  Model: {result['model_version']}")
            print(f"  Default Probability: {result['probability']:.1%}")
            print(f"  Risk Score: {result['risk_score']:.1f}/100")
            if 'feature_interpretation' in result:
                interp = result['feature_interpretation']
                print(f"  Loan: {interp['loan_amount']}, Income: {interp['annual_income']}")
                print(f"  Credit: {interp['credit_score']}, Debt Ratio: {interp['debt_to_income_ratio']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the enhanced server is running on port 8001")
        break
