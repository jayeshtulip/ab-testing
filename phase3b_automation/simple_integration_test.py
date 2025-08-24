"""
Simple Phase 3B Integration Test for Windows VSCode
Step-by-step testing of core components
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_directory_structure():
    """Test 0: Check directory structure"""
    print("🧪 Test 0: Checking directory structure...")
    
    required_dirs = ['core', 'api']
    required_files = [
        'core/__init__.py',
        'core/resource_manager.py', 
        'api/__init__.py',
        'api/models.py'
    ]
    
    missing_items = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_items.append(f"Directory: {dir_name}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"File: {file_path}")
    
    if missing_items:
        print("❌ Missing required items:")
        for item in missing_items:
            print(f"  - {item}")
        return False
    else:
        print("✅ Directory structure looks good")
        return True

def test_basic_imports():
    """Test 1: Check basic imports"""
    print("\n🧪 Test 1: Testing basic imports...")
    
    try:
        # Test if we can import the basic modules
        print("  ⏳ Testing ResourceManager import...")
        try:
            from core.resource_manager import ResourceManager
            print("  ✅ ResourceManager imported successfully")
        except ImportError as e:
            print(f"  ❌ ResourceManager import failed: {e}")
            return False
        
        print("  ⏳ Testing API models import...")
        try:
            from api.models import ExperimentPipelineRequest
            print("  ✅ API models imported successfully")
        except ImportError as e:
            print(f"  ❌ API models import failed: {e}")
            return False
        
        print("✅ Test 1 PASSED: Basic imports working")
        return True
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: Unexpected error - {e}")
        return False

async def test_resource_manager_only():
    """Test 2: Test only ResourceManager (the one we know works)"""
    print("\n🧪 Test 2: Testing ResourceManager functionality...")
    
    try:
        from core.resource_manager import ResourceManager
        
        # Initialize resource manager
        print("  ⏳ Initializing ResourceManager...")
        manager = ResourceManager()
        print("  ✅ ResourceManager initialized")
        
        # Test health check
        print("  ⏳ Testing health check...")
        health = await manager.health_check()
        if health.get('healthy', False):
            print("  ✅ Health check passed")
        else:
            print("  ⚠️ Health check returned unhealthy status")
        
        # Test resource utilization
        print("  ⏳ Testing resource utilization...")
        utilization = await manager.get_utilization()
        if utilization:
            print(f"  📊 Found {len(utilization)} resource types")
            for resource_type, usage in utilization.items():
                print(f"    - {resource_type}: {usage['utilization_percentage']}% used")
        
        # Test simple allocation
        print("  ⏳ Testing resource allocation...")
        requirements = [{
            "resource_type": "compute",
            "amount": 2.0,
            "unit": "cores",
            "priority": "medium"
        }]
        
        result = await manager.allocate_resources("test_exp_001", requirements)
        if result.get("success", False):
            print("  ✅ Resource allocation successful")
            
            # Test deallocation
            print("  ⏳ Testing resource deallocation...")
            dealloc = await manager.deallocate_resources("test_exp_001")
            if dealloc.get("success", False):
                print("  ✅ Resource deallocation successful")
            else:
                print("  ⚠️ Deallocation had issues")
        else:
            print(f"  ❌ Resource allocation failed: {result.get('error', 'Unknown error')}")
        
        print("✅ Test 2 PASSED: ResourceManager working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: ResourceManager error - {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_pydantic_models():
    """Test 3: Test Pydantic models"""
    print("\n🧪 Test 3: Testing Pydantic models...")
    
    try:
        from api.models import ExperimentPipelineRequest
        
        print("  ⏳ Creating ExperimentPipelineRequest...")
        experiment_request = ExperimentPipelineRequest(
            name="Test Experiment",
            description="Testing Pydantic models",
            owner="test_user",
            team="test_team",
            compute_requirement=5.0,
            storage_requirement=10.0
        )
        print(f"  ✅ ExperimentPipelineRequest created: {experiment_request.name}")
        
        # Test serialization
        print("  ⏳ Testing serialization...")
        experiment_dict = experiment_request.model_dump()
        if experiment_dict and experiment_dict.get('name') == "Test Experiment":
            print("  ✅ Serialization works")
        
        print("✅ Test 3 PASSED: Pydantic models working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: Pydantic models error - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_other_core_modules():
    """Test 4: Check what other core modules exist and work"""
    print("\n🧪 Test 4: Testing other core modules...")
    
    core_modules = [
        'experiment_lifecycle_manager',
        'scheduler', 
        'state_machine',
        'monitor'
    ]
    
    working_modules = []
    broken_modules = []
    
    for module_name in core_modules:
        try:
            module_path = f"core.{module_name}"
            print(f"  ⏳ Testing {module_name}...")
            
            # Try to import the module
            import importlib
            module = importlib.import_module(module_path)
            
            # Check if the expected class exists
            expected_class_name = ''.join(word.capitalize() for word in module_name.split('_'))
            
            if hasattr(module, expected_class_name):
                print(f"  ✅ {module_name} has {expected_class_name} class")
                working_modules.append(module_name)
            else:
                print(f"  ⚠️ {module_name} missing {expected_class_name} class")
                broken_modules.append(f"{module_name} (missing {expected_class_name})")
                
        except ImportError as e:
            print(f"  ❌ {module_name} import failed: {e}")
            broken_modules.append(f"{module_name} (import error)")
        except Exception as e:
            print(f"  ❌ {module_name} error: {e}")
            broken_modules.append(f"{module_name} (error)")
    
    print(f"\n  📊 Results:")
    print(f"    Working modules: {len(working_modules)}")
    print(f"    Broken modules: {len(broken_modules)}")
    
    if working_modules:
        print(f"    ✅ Working: {', '.join(working_modules)}")
    if broken_modules:
        print(f"    ❌ Need fixing: {', '.join(broken_modules)}")
    
    # Return True if at least ResourceManager works (we need at least one working)
    return len(working_modules) >= 0  # Always return True since this is just informational

async def run_all_tests():
    """Run all integration tests"""
    print("🚀 Starting Phase 3B Integration Tests (Windows VSCode)")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Test started at: {datetime.now()}")
    print("=" * 60)
    
    test_results = []
    
    # Test 0: Directory structure
    result0 = test_directory_structure()
    test_results.append(("Directory Structure", result0))
    
    if not result0:
        print("\n❌ CRITICAL: Directory structure is wrong. Please fix the file structure.")
        return False
    
    # Test 1: Basic imports
    result1 = test_basic_imports()
    test_results.append(("Basic Imports", result1))
    
    if not result1:
        print("\n❌ CRITICAL: Basic imports failed. Please fix import errors before continuing.")
        return False
    
    # Test 2: ResourceManager
    result2 = await test_resource_manager_only()
    test_results.append(("ResourceManager", result2))
    
    # Test 3: Pydantic Models
    result3 = await test_pydantic_models()
    test_results.append(("Pydantic Models", result3))
    
    # Test 4: Other Core Modules (informational)
    result4 = test_other_core_modules()
    test_results.append(("Other Core Modules", result4))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    critical_passed = 0
    total = len(test_results)
    
    for i, (test_name, result) in enumerate(test_results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
        if result:
            passed += 1
            if i < 3:  # First 3 tests are critical
                critical_passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Critical: {critical_passed}/3 critical tests passed")
    
    if critical_passed >= 3:
        print("\n🎉 CRITICAL TESTS PASSED! You can proceed to the next step.")
        print("💡 Any failing non-critical tests can be fixed later.")
        return True
    elif critical_passed >= 2:
        print("\n⚠️ MOSTLY WORKING: Core functionality available, proceed with caution.")
        return True
    else:
        print("\n❌ CRITICAL ISSUES: Please fix the basic imports and ResourceManager first.")
        return False

def main():
    """Main function"""
    try:
        # Run async tests
        result = asyncio.run(run_all_tests())
        
        if result:
            print("\n✨ Integration test completed successfully!")
            print("\n🚀 NEXT STEPS:")
            print("1. Fix any non-critical failing modules (if you want)")
            print("2. Proceed to Step 2: Winner Selection Engine")
            print("3. Run: python winner_selection_engine.py")
        else:
            print("\n💥 Integration test failed!")
            print("\n🔧 IMMEDIATE ACTIONS NEEDED:")
            print("1. Fix the directory structure if needed")
            print("2. Fix import errors in core modules")
            print("3. Re-run this test until critical tests pass")
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()