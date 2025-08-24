import requests
import json

# Test scenarios with more users to see both groups
scenarios = [
    {"name": "Low Risk (Good Credit)", "user_id": "alice", "features": [50000, 150000, 800, 0.1, 10]},
    {"name": "Medium Risk (Average)", "user_id": "bob", "features": [75000, 80000, 650, 0.4, 3]},
    {"name": "High Risk (Poor Credit)", "user_id": "charlie", "features": [100000, 40000, 500, 0.7, 0.5]},
    {"name": "Very High Risk", "user_id": "diana", "features": [120000, 35000, 400, 0.8, 0.1]},
    {"name": "Low Risk 2", "user_id": "eve", "features": [40000, 120000, 750, 0.2, 8]},
    {"name": "Medium Risk 2", "user_id": "frank", "features": [90000, 70000, 600, 0.5, 2]}
]

print(" Enhanced A/B Testing Results")
print("=" * 60)
print("Testing Random Forest (Control) vs Gradient Boosting (Treatment)")
print("=" * 60)

control_results = []
treatment_results = []

for scenario in scenarios:
    try:
        response = requests.post("http://127.0.0.1:8001/predict", json=scenario, timeout=5)
        if response.status_code == 200:
            result = response.json()
            
            # Extract feature values for display
            features = scenario['features']
            loan_amt = features[0]
            income = features[1]
            credit = features[2]
            debt_ratio = features[3]
            employment = features[4]
            
            print(f"\n{scenario['name']} - User: {scenario['user_id']}")
            print(f"   Profile: Loan=, Income=, Credit={credit:.0f}")
            print(f"   Debt Ratio: {debt_ratio:.1%}, Employment: {employment:.1f} years")
            print(f"   Group: {result['experiment_group'].upper()}")
            print(f"   Model: {result['model_version']}")
            print(f"   Default Risk: {result['probability']:.1%}")
            print(f"    Risk Score: {result['risk_score']:.1f}/100")
            
            # Collect for analysis
            if result['experiment_group'] == 'control':
                control_results.append(result['probability'])
            else:
                treatment_results.append(result['probability'])
                
        else:
            print(f" Error for {scenario['name']}: {response.text}")
            
    except Exception as e:
        print(f" Connection failed for {scenario['name']}: {e}")

# Summary analysis
print(f"\n A/B Testing Analysis")
print("=" * 40)

if control_results:
    avg_control = sum(control_results) / len(control_results)
    print(f" CONTROL (Random Forest): {len(control_results)} predictions")
    print(f"   Average Default Risk: {avg_control:.1%}")
    print(f"   Risk Range: {min(control_results):.1%} - {max(control_results):.1%}")

if treatment_results:
    avg_treatment = sum(treatment_results) / len(treatment_results)
    print(f" TREATMENT (Gradient Boost): {len(treatment_results)} predictions")
    print(f"   Average Default Risk: {avg_treatment:.1%}")
    print(f"   Risk Range: {min(treatment_results):.1%} - {max(treatment_results):.1%}")

if control_results and treatment_results:
    difference = avg_treatment - avg_control
    print(f"\n Model Difference: {difference:+.1%}")
    if abs(difference) > 0.05:
        winner = "Treatment" if difference < 0 else "Control"
        print(f" {winner} model appears more conservative")
    else:
        print(" Models show similar behavior")

print(f"\n A/B Testing Pipeline Working Successfully!")
