#!/usr/bin/env python3
"""
Test different loan scenarios
"""
import requests
import json

# Test scenarios with different risk profiles
scenarios = [
    {
        "name": "Low Risk - High Income, Good Credit",
        "user_id": "low_risk_alice",
        "features": [50000, 150000, 800, 0.1, 10]  # 50k loan, 150k income, 800 credit, 10% debt ratio, 10 years employment
    },
    {
        "name": "Medium Risk - Average Profile", 
        "user_id": "medium_risk_bob",
        "features": [75000, 80000, 650, 0.4, 3]   # 75k loan, 80k income, 650 credit, 40% debt ratio, 3 years employment
    },
    {
        "name": "High Risk - Low Income, Poor Credit",
        "user_id": "high_risk_charlie", 
        "features": [100000, 40000, 500, 0.7, 0.5]  # 100k loan, 40k income, 500 credit, 70% debt ratio, 6 months employment
    },
    {
        "name": "Very High Risk - Extreme Case",
        "user_id": "very_high_risk_diana",
        "features": [120000, 35000, 400, 0.8, 0]   # 120k loan, 35k income, 400 credit, 80% debt ratio, unemployed
    }
]

def test_scenario(scenario):
    """Test a specific scenario"""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=scenario,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n{scenario['name']}:")
            print(f"  User: {scenario['user_id']}")
            print(f"  Group: {result['experiment_group']}")
            print(f"  Model: {result['model_version']}")
            print(f"  Prediction: {'Default' if result['prediction'] == 1 else 'No Default'}")
            print(f"  Default Probability: {result['probability']:.1%}")
            print(f"  Risk Score: {result['risk_score']:.1f}/100")
            return result
        else:
            print(f"Error for {scenario['name']}: {response.text}")
            return None
            
    except Exception as e:
        print(f"Failed to test {scenario['name']}: {e}")
        return None

if __name__ == "__main__":
    print(" Testing Enhanced A/B API with Different Risk Profiles")
    print("=" * 60)
    
    results = []
    for scenario in scenarios:
        result = test_scenario(scenario)
        if result:
            results.append(result)
    
    # Compare groups
    if results:
        print(f"\n A/B Testing Results Summary:")
        print("=" * 40)
        
        control_results = [r for r in results if r['experiment_group'] == 'control']
        treatment_results = [r for r in results if r['experiment_group'] == 'treatment']
        
        if control_results:
            avg_control_prob = sum(r['probability'] for r in control_results) / len(control_results)
            print(f"Control Group (Random Forest): {len(control_results)} predictions, Avg Default Prob: {avg_control_prob:.1%}")
        
        if treatment_results:
            avg_treatment_prob = sum(r['probability'] for r in treatment_results) / len(treatment_results)
            print(f"Treatment Group (Gradient Boost): {len(treatment_results)} predictions, Avg Default Prob: {avg_treatment_prob:.1%}")
