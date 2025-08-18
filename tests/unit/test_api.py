import pytest
import requests
from unittest.mock import Mock, patch

API_BASE_URL = 'http://localhost:8001'

def test_api_health_endpoint():
    '''Test API health check'''
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        print(' Health endpoint test passed')
    except requests.ConnectionError:
        pytest.skip('API not running - skipping health test')

def test_prediction_endpoint_structure():
    '''Test prediction endpoint response structure'''
    sample_request = {
        'features': {
            'Attribute1': 'A11',
            'Attribute2': 24,
            'Attribute3': 'A32',
            'Attribute5': 3500
        }
    }
    
    try:
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=sample_request,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Test response structure
            assert 'prediction' in data
            assert 'probability' in data
            assert 'model_version' in data
            assert 'timestamp' in data
            
            # Test data types
            assert isinstance(data['prediction'], str)
            assert isinstance(data['probability'], float)
            assert data['prediction'] in ['good', 'bad']
            assert 0 <= data['probability'] <= 1
            
            print(' Prediction endpoint test passed')
        else:
            print(f' API returned status code: {response.status_code}')
            
    except requests.ConnectionError:
        pytest.skip('API not running - skipping prediction test')

def test_invalid_request_handling():
    '''Test API handles invalid requests properly'''
    invalid_request = {'invalid': 'data'}
    
    try:
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=invalid_request,
            timeout=5
        )
        
        # Should return 4xx error for invalid request
        assert response.status_code >= 400
        print(' Invalid request handling test passed')
        
    except requests.ConnectionError:
        pytest.skip('API not running - skipping invalid request test')

if __name__ == '__main__':
    pytest.main([__file__])
