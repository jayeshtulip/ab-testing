"""
Quick test script for Enhanced API
Run this to verify your enhanced API is working
"""

import requests
import json
import time

def test_enhanced_api():
    """Test the enhanced API endpoints"""
    base_url = "http://localhost:8001"
    
    print(" Testing Enhanced MLOps API")
    print("=" * 50)
    
    # Test 1: Root endpoint
    try:
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f" Root endpoint working")
            print(f"   API Version: {data.get('version')}")
            print(f"   Features: {len(data.get('features', []))} enhanced features")
        else:
            print(f" Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f" Root endpoint error: {e}")
    
    # Test 2: Health check
    try:
        print("\n2. Testing health check...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f" Health check working")
            print(f"   Status: {data.get('status')}")
            print(f"   Pipeline Type: {data.get('pipeline_type')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
        else:
            print(f" Health check failed: {response.status_code}")
    except Exception as e:
        print(f" Health check error: {e}")
    
    # Test 3: Enhanced prediction
    try:
        print("\n3. Testing enhanced prediction...")
        test_data = {
            "loan_amount": 50000,
            "income": 75000,
            "credit_score": 720,
            "debt_to_income": 0.3,
            "employment_years": 5,
            "age": 30,
            "education_level": 3,
            "loan_purpose": 1
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", json=test_data)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f" Enhanced prediction working")
            print(f"   Prediction: {data.get('prediction')}")
            print(f"   Probability: {data.get('probability'):.3f}")
            print(f"   Risk Level: {data.get('risk_level')}")
            print(f"   Confidence: {data.get('confidence_score', 0):.3f}")
            print(f"   Features Used: {data.get('feature_count')}")
            print(f"   Response Time: {(end_time - start_time)*1000:.1f}ms")
        else:
            print(f" Enhanced prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f" Enhanced prediction error: {e}")
    
    # Test 4: Model info
    try:
        print("\n4. Testing model info...")
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f" Model info working")
            print(f"   Model Version: {data.get('model_version')}")
            print(f"   Model Type: {data.get('model_type')}")
            print(f"   Pipeline Type: {data.get('pipeline_type')}")
            print(f"   Hyperparameter Optimized: {data.get('hyperparameter_optimization')}")
        else:
            print(f" Model info failed: {response.status_code}")
    except Exception as e:
        print(f" Model info error: {e}")
    
    # Test 5: Metrics endpoint
    try:
        print("\n5. Testing metrics endpoint...")
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            print(f" Metrics endpoint working")
            print(f"   Metrics available for Prometheus")
        else:
            print(f" Metrics endpoint failed: {response.status_code}")
    except Exception as e:
        print(f" Metrics endpoint error: {e}")
    
    print("\n" + "=" * 50)
    print(" Enhanced API testing completed!")
    print("\nNext steps:")
    print("1. Compare with your basic API")
    print("2. Test batch predictions")
    print("3. Monitor metrics in production")

if __name__ == "__main__":
    test_enhanced_api()
