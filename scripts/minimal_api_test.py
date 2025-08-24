"""
Minimal API test
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

print("Testing imports...")
try:
    from ab_testing.ab_testing_api import app
    print("✅ API app imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("Testing with TestClient...")
try:
    from fastapi.testclient import TestClient
    
    print("Creating test client...")
    client = TestClient(app)
    print("✅ Test client created")
    
    print("Testing health endpoint...")
    response = client.get("/health")
    print(f"Health status: {response.status_code}")
    if response.status_code == 200:
        print(f"Health response: {response.json()}")
        print(" Health endpoint working!")
    else:
        print(f"Health error: {response.text}")
    
    print("Testing prediction endpoint...")
    pred_data = {
        "user_id": "test_user", 
        "features": [50000, 75000, 720, 0.3, 5]
    }
    response = client.post("/predict", json=pred_data)
    print(f"Prediction status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f" Prediction successful!")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Group: {result['experiment_group']}")
    else:
        print(f"Prediction error: {response.text}")
        
except Exception as e:
    print(f" TestClient failed: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")
