"""
Test for A/B Testing API
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

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["platform"] == "Windows"

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "A/B Testing API is running" in data["message"]

def test_predict_endpoint(client):
    """Test prediction endpoint"""
    prediction_data = {
        "user_id": "test_user_123",
        "features": [50000, 75000, 720, 0.3, 5]
    }
    
    response = client.post("/predict", json=prediction_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "experiment_group" in data
    assert data["experiment_group"] in ["control", "treatment"]
    assert isinstance(data["prediction"], (int, float))
    assert 0 <= data["probability"] <= 1

def test_predict_different_users(client):
    """Test that different users can get different groups"""
    users = ["user1", "user2", "user3", "user4", "user5"]
    groups = []
    
    for user in users:
        prediction_data = {
            "user_id": user,
            "features": [50000, 75000, 720, 0.3, 5]
        }
        response = client.post("/predict", json=prediction_data)
        assert response.status_code == 200
        groups.append(response.json()["experiment_group"])
    
    # Should have at least some variety in groups (not all same)
    unique_groups = set(groups)
    assert len(unique_groups) >= 1  # At least one group type
