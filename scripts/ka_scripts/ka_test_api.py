# Ka-MLOps API Test Script
import requests
import json

def test_ka_api():
    '''Test the Ka API endpoints'''
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Ka-MLOps API")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing Ka health endpoint...")
    try:
        response = requests.get(f"{base_url}/ka-health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health: {health_data['status']}")
            print(f"   📊 Model loaded: {health_data['model_loaded']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"    Health check error: {e}")
    
    # Test prediction endpoint
    print("\n2. Testing Ka prediction endpoint...")
    sample_request = {
        "loan_amnt": 15000.0,
        "int_rate": 12.5,
        "annual_inc": 65000.0,
        "dti": 18.5,
        "fico_range_low": 720,
        "fico_range_high": 724,
        "installment": 450.0,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 8,
        "pub_rec": 0,
        "revol_bal": 5000.0,
        "revol_util": 25.0,
        "total_acc": 15,
        "mort_acc": 1,
        "pub_rec_bankruptcies": 0,
        "term": " 36 months",
        "grade": "B",
        "emp_length": "5 years",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA"
    }
    
    try:
        response = requests.post(f"{base_url}/ka-predict", json=sample_request)
        if response.status_code == 200:
            pred_data = response.json()
            print(f"    Prediction: {pred_data['prediction']}")
            print(f"    Default probability: {pred_data['default_probability']}")
            print(f"    Confidence: {pred_data['confidence']}")
            print(f"     Risk factors: {pred_data['risk_factors']}")
        else:
            print(f"    Prediction failed: {response.status_code}")
            print(f"    Response: {response.text}")
    except Exception as e:
        print(f"    Prediction error: {e}")
    
    # Test model info endpoint
    print("\n3. Testing Ka model info endpoint...")
    try:
        response = requests.get(f"{base_url}/ka-model-info")
        if response.status_code == 200:
            model_data = response.json()
            print(f"    Model type: {model_data['model_type']}")
            print(f"    Features: {model_data['features_count']}")
            print(f"    System: {model_data['ka_system']}")
        else:
            print(f"    Model info failed: {response.status_code}")
    except Exception as e:
        print(f"    Model info error: {e}")
    
    print("\n Ka API testing completed!")

if __name__ == "__main__":
    test_ka_api()
