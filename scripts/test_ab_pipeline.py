#!/usr/bin/env python3
"""
Quick test script for Windows
"""
import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print(" numpy imported successfully")
        
        import pandas as pd
        print(" pandas imported successfully")
        
        from sklearn.ensemble import RandomForestClassifier
        print(" scikit-learn imported successfully")
        
        import fastapi
        print(" FastAPI imported successfully")
        
        from ab_testing.data_generator import SyntheticDataGenerator
        print(" Custom data generator imported successfully")
        
    except ImportError as e:
        print(f" Import error: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation"""
    print("\n Testing data generation...")
    
    try:
        from ab_testing.data_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        X, y, features = generator.generate_baseline_data(100, 5)
        
        assert X.shape == (100, 5), f"Expected (100, 5), got {X.shape}"
        assert y.shape == (100,), f"Expected (100,), got {y.shape}"
        assert len(features) == 5, f"Expected 5 features, got {len(features)}"
        
        print(" Data generation test passed")
        print(f"   Generated {len(X)} samples with {len(features)} features")
        print(f"   Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")
        
        return True
        
    except Exception as e:
        print(f" Data generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print(" Running A/B Testing Pipeline Tests on Windows")
    print("=" * 55)
    
    tests = [test_imports, test_data_generation]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f" Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 55)
    print(f" Test Results")
    print(f" Passed: {passed}")
    print(f" Failed: {failed}")
    
    if failed == 0:
        print("\n All tests passed! Your setup is working correctly.")
        print("Next steps:")
        print("1. Start the API: python src/ab_testing/ab_testing_api.py")
        print("2. Open VS Code: code .")
        print("3. Check the API docs: http://localhost:8000/docs")
        return 0
    else:
        print(f"\n {failed} tests failed. Please check your setup.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
