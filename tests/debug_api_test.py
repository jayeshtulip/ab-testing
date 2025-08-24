"""
Debug version of API test
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ab_testing.ab_testing_api import app

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

def test_predict_endpoint_debug(client):
    """Debug version of prediction test"""
    # First check health to see if model is loaded
    health_response = client.get("/health")
    print(f"Health response: {health_response.json()}")
    
    prediction_data = {
        "user_id": "test_user_123",
        "features": [50000, 75000, 720, 0.3, 5]
    }
    
    response = client.post("/predict", json=prediction_data)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code != 200:
        # Let's see what the error is
        print(f"Error response: {response.json()}")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
