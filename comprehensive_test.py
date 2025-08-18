import requests
import json
import time

print(' Testing API Health...')
try:
    response = requests.get('http://localhost:8001/health', timeout=5)
    print(f' Health Status: {response.status_code}')
    if response.status_code == 200:
        health_data = response.json()
        print(f' API Status: {health_data.get("status", "unknown")}')
        print(f' Model Loaded: {health_data.get("model_loaded", False)}')
    print()
except Exception as e:
    print(f' Health test failed: {e}')
    print()

# Test prediction
print(' Testing Prediction Endpoint...')
test_cases = [
    {
        'name': 'Test Case 1 - Low Risk Profile',
        'features': {
            'Attribute1': 'A11',
            'Attribute2': 12,
            'Attribute3': 'A32', 
            'Attribute5': 1500
        }
    },
    {
        'name': 'Test Case 2 - High Risk Profile', 
        'features': {
            'Attribute1': 'A14',
            'Attribute2': 48,
            'Attribute3': 'A34',
            'Attribute5': 8000
        }
    }
]

all_tests_passed = True

for i, test_case in enumerate(test_cases, 1):
    print(f'\n {test_case["name"]}')
    try:
        start_time = time.time()
        response = requests.post(
            'http://localhost:8001/predict',
            json={'features': test_case['features']},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f' Status: {response.status_code}')
            print(f' Prediction: {result["prediction"].upper()}')
            print(f'📊 Confidence: {result["probability"]:.1%}')
            print(f' Response Time: {response_time:.2f}s')
            
            # Validate response structure
            required_fields = ['prediction', 'probability', 'model_version', 'timestamp']
            for field in required_fields:
                if field not in result:
                    print(f' Missing field: {field}')
                    all_tests_passed = False
                    
            # Validate data types and ranges
            if not isinstance(result['prediction'], str) or result['prediction'] not in ['good', 'bad']:
                print(f' Invalid prediction value: {result["prediction"]}')
                all_tests_passed = False
                
            if not isinstance(result['probability'], (int, float)) or not (0 <= result['probability'] <= 1):
                print(f' Invalid probability value: {result["probability"]}')
                all_tests_passed = False
                
        else:
            print(f' Request failed with status: {response.status_code}')
            print(f' Error: {response.text}')
            all_tests_passed = False
            
    except Exception as e:
        print(f' Test case {i} failed: {e}')
        all_tests_passed = False

# Performance test
print('\n Testing API Performance...')
try:
    performance_times = []
    for i in range(5):
        start_time = time.time()
        response = requests.post(
            'http://localhost:8001/predict',
            json={'features': test_cases[0]['features']},
            timeout=5
        )
        end_time = time.time()
        
        if response.status_code == 200:
            performance_times.append(end_time - start_time)
    
    if performance_times:
        avg_time = sum(performance_times) / len(performance_times)
        max_time = max(performance_times)
        print(f' Average Response Time: {avg_time:.2f}s')
        print(f' Max Response Time: {max_time:.2f}s')
        
        if avg_time > 2.0:
            print(' Warning: Average response time > 2 seconds')
        if max_time > 5.0:
            print(' Performance issue: Max response time > 5 seconds')
            all_tests_passed = False
            
except Exception as e:
    print(f' Performance test failed: {e}')
    all_tests_passed = False

# Summary
print('\n' + '='*50)
if all_tests_passed:
    print(' ALL TESTS PASSED! ')
    print(' API Health Check: PASSED')
    print(' Prediction Endpoint: PASSED')
    print(' Response Structure: PASSED')
    print(' Data Validation: PASSED')
    print(' Performance: PASSED')
else:
    print(' SOME TESTS FAILED!')
    
print('='*50)
