import pytest
import requests
import time
import json

API_URL = 'http://localhost:8001'
STREAMLIT_URL = 'http://localhost:8501'

class TestAPIIntegration:
    '''Integration tests for the API'''
    
    def test_full_prediction_pipeline(self):
        '''Test complete prediction pipeline'''
        # Test data
        test_cases = [
            {
                'features': {
                    'Attribute1': 'A11', 'Attribute2': 12, 'Attribute3': 'A32', 'Attribute5': 1500
                },
                'expected_prediction': ['good', 'bad']  # Either is valid
            },
            {
                'features': {
                    'Attribute1': 'A14', 'Attribute2': 48, 'Attribute3': 'A34', 'Attribute5': 8000
                },
                'expected_prediction': ['good', 'bad']
            }
        ]
        
        try:
            for i, test_case in enumerate(test_cases):
                print(f'Testing case {i+1}...')
                
                response = requests.post(
                    f'{API_URL}/predict',
                    json=test_case,
                    timeout=10
                )
                
                assert response.status_code == 200, f'Test case {i+1} failed with status {response.status_code}'
                
                data = response.json()
                assert data['prediction'] in test_case['expected_prediction']
                assert 0 <= data['probability'] <= 1
                
                print(f' Test case {i+1} passed: {data["prediction"]} ({data["probability"]:.2f})')
                
        except requests.ConnectionError:
            pytest.skip('API not running - skipping integration test')
    
    def test_api_performance(self):
        '''Test API response time'''
        sample_request = {
            'features': {
                'Attribute1': 'A11',
                'Attribute2': 24,
                'Attribute3': 'A32',
                'Attribute5': 3500
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(f'{API_URL}/predict', json=sample_request, timeout=5)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < 2.0  # Should respond within 2 seconds
            
            print(f' Performance test passed: {response_time:.2f}s response time')
            
        except requests.ConnectionError:
            pytest.skip('API not running - skipping performance test')

if __name__ == '__main__':
    pytest.main([__file__])
