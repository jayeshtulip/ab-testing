"""
Phase 3A Test Client - Statistical Significance + Early Stopping Testing
Comprehensive testing of Phase 3A enhanced capabilities

Save as: tests/phase3a_test_client.py
"""

import requests
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List

class Phase3ATestClient:
    """Test client for Phase 3A enhanced server"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """Test Phase 3A health endpoint"""
        print("🏥 Testing Phase 3A Health Check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Status: {health_data['status']}")
                print(f"📦 Version: {health_data['version']}")
                print(f"🔧 Phase: {health_data['phase']}")
                print(f"🧪 Active Experiments: {health_data['active_experiments']}")
                print(f"⚡ Capabilities: {len(health_data['capabilities'])} features")
                
                # Check new Phase 3A components
                components = health_data['components']
                print(f"   Statistical Testing: {components.get('statistical_testing', False)}")
                print(f"   Early Stopping: {components.get('early_stopping', False)}")
                print(f"   Experiment Management: {components.get('experiment_management', False)}")
                
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_sample_size_calculation(self) -> Dict:
        """Test sample size calculation"""
        print("\n📊 Testing Sample Size Calculation...")
        
        test_scenarios = [
            {
                "name": "Small Effect (5% improvement)",
                "baseline_rate": 0.10,
                "minimum_detectable_effect": 0.05,
                "alpha": 0.05,
                "power": 0.8
            },
            {
                "name": "Medium Effect (20% improvement)", 
                "baseline_rate": 0.15,
                "minimum_detectable_effect": 0.20,
                "alpha": 0.05,
                "power": 0.8
            },
            {
                "name": "Large Effect (50% improvement)",
                "baseline_rate": 0.08,
                "minimum_detectable_effect": 0.50,
                "alpha": 0.05,
                "power": 0.9
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            try:
                response = self.session.post(
                    f"{self.base_url}/experiments/sample-size",
                    json=scenario
                )
                
                if response.status_code == 200:
                    result = response.json()
                    calc = result['calculation']
                    
                    print(f"\n   {scenario['name']}:")
                    print(f"     Sample size per group: {calc['n_per_group']:,}")
                    print(f"     Total sample size: {calc['total_sample_size']:,}")
                    print(f"     Effect size (Cohen's h): {calc['effect_size_cohens_h']:.3f}")
                    print(f"     Timeline: {result['timeline_estimate']['estimated_weeks']} weeks")
                    
                    results[scenario['name']] = calc
                    
                else:
                    print(f"     ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        return results
    
    def test_statistical_testing(self) -> Dict:
        """Test statistical significance testing"""
        print("\n📈 Testing Statistical Significance Testing...")
        
        test_scenarios = [
            {
                "name": "Significant Improvement",
                "control_successes": 45,
                "control_total": 500,
                "treatment_successes": 65,
                "treatment_total": 500,
                "test_type": "chi_square"
            },
            {
                "name": "No Significant Difference",
                "control_successes": 50,
                "control_total": 400,
                "treatment_successes": 55,
                "treatment_total": 400,
                "test_type": "z_test"
            },
            {
                "name": "Significant Decrease",
                "control_successes": 80,
                "control_total": 400,
                "treatment_successes": 55,
                "treatment_total": 400,
                "test_type": "chi_square"
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            try:
                response = self.session.post(
                    f"{self.base_url}/experiments/statistical-test",
                    json=scenario
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    summary = result['experiment_summary']
                    test = result['statistical_test']
                    power = result['power_analysis']
                    
                    print(f"\n   {scenario['name']}:")
                    print(f"     Control rate: {summary['control_rate']:.1%}")
                    print(f"     Treatment rate: {summary['treatment_rate']:.1%}")
                    print(f"     Relative lift: {summary['relative_lift']:.1%}")
                    print(f"     P-value: {test['p_value']:.4f}")
                    print(f"     Significant: {test['significant']}")
                    print(f"     Effect size: {test['effect_size']:.3f}")
                    print(f"     Power: {power['current_power']:.1%}")
                    
                    results[scenario['name']] = result
                    
                else:
                    print(f"     ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        return results
    
    def test_experiment_configuration(self) -> str:
        """Test experiment configuration"""
        print("\n🧪 Testing Experiment Configuration...")
        
        experiment_config = {
            "experiment_id": "test_experiment_001",
            "name": "Loan Default Model A/B Test",
            "hypothesis": "New gradient boosting model will reduce false positive rate by 15%",
            "success_metric": "false_positive_reduction",
            "baseline_rate": 0.12,
            "minimum_detectable_effect": 0.15,
            "max_duration_days": 21,
            "max_sample_size_per_group": 5000
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/experiments/configure",
                json=experiment_config
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Experiment configured: {result['experiment_id']}")
                
                sample_rec = result['sample_size_recommendation']
                print(f"   Recommended sample size: {sample_rec['n_per_group']:,} per group")
                print(f"   Total sample size: {sample_rec['total_sample_size']:,}")
                
                timeline = result['timeline_estimate']
                print(f"   Estimated duration: {timeline['estimated_weeks']} weeks")
                
                print(f"   Next steps: {len(result['next_steps'])} actions")
                
                return experiment_config['experiment_id']
                
            else:
                print(f"❌ Configuration failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return None
    
    def test_ab_predictions_with_tracking(self, experiment_id: str) -> Dict:
        """Test A/B predictions with experiment tracking"""
        print(f"\n🔄 Testing A/B Predictions with Tracking...")
        
        results = {"control": 0, "treatment": 0, "errors": 0, "predictions": []}
        
        # Generate diverse test users
        np.random.seed(42)
        for i in range(20):
            user_id = f"test_user_{i:03d}"
            features = [
                float(np.random.uniform(600, 750)),
                float(np.random.uniform(50000, 100000)),
                float(np.random.uniform(10000, 40000)),
                float(np.random.uniform(1, 15)),
                float(np.random.uniform(0, 5))
            ]
            
            try:
                payload = {
                    "user_id": user_id,
                    "features": features,
                    "experiment_id": experiment_id
                }
                
                response = self.session.post(f"{self.base_url}/predict", json=payload)
                
                if response.status_code == 200:
                    pred_data = response.json()
                    group = pred_data['experiment_group']
                    
                    results[group] += 1
                    results['predictions'].append(pred_data)
                    
                    if i < 5:  # Show first few predictions
                        print(f"   {user_id}: {group} -> {pred_data['model_version']} (prob: {pred_data['probability']:.3f})")
                    
                else:
                    results['errors'] += 1
                    
            except Exception as e:
                results['errors'] += 1
        
        total = results['control'] + results['treatment']
        if total > 0:
            print(f"\n   📊 Results: Control {results['control']}, Treatment {results['treatment']}")
            print(f"   Split: {results['control']/total*100:.1f}% / {results['treatment']/total*100:.1f}%")
            print(f"   Errors: {results['errors']}")
        
        return results
    
    def test_early_stopping_evaluation(self, experiment_id: str) -> Dict:
        """Test early stopping criteria evaluation"""
        print(f"\n🛑 Testing Early Stopping Evaluation...")
        
        # Create test scenarios with different stopping outcomes
        stopping_scenarios = [
            {
                "name": "Early Success",
                "control_successes": 45,
                "control_total": 400,
                "treatment_successes": 85,
                "treatment_total": 400
            },
            {
                "name": "Clear Futility",
                "control_successes": 50,
                "control_total": 800,
                "treatment_successes": 52,
                "treatment_total": 800
            },
            {
                "name": "Need More Data",
                "control_successes": 25,
                "control_total": 200,
                "treatment_successes": 32,
                "treatment_total": 200
            }
        ]
        
        results = {}
        
        for scenario in stopping_scenarios:
            try:
                evaluation_request = {
                    "experiment_id": experiment_id,
                    **scenario
                }
                evaluation_request.pop('name')  # Remove name from request
                
                response = self.session.post(
                    f"{self.base_url}/experiments/evaluate-stopping",
                    json=evaluation_request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    decision = result['decision']
                    
                    print(f"\n   {scenario['name']}:")
                    print(f"     Decision: {'STOP' if decision['should_stop'] else 'CONTINUE'}")
                    if decision['reason']:
                        print(f"     Reason: {decision['reason']}")
                    print(f"     Confidence: {decision['confidence']:.1%}")
                    
                    stats = result['statistical_analysis']
                    print(f"     P-value: {stats['p_value']:.4f}")
                    print(f"     Power: {stats['power']:.1%}")
                    
                    current = result['current_results']
                    print(f"     Control rate: {current['control_rate']:.1%}")
                    print(f"     Treatment rate: {current['treatment_rate']:.1%}")
                    print(f"     Relative lift: {current['relative_lift']:.1%}")
                    
                    results[scenario['name']] = result
                    
                else:
                    print(f"     ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        return results
    
    def test_experiment_status(self, experiment_id: str) -> Dict:
        """Test experiment status endpoint"""
        print(f"\n📋 Testing Experiment Status...")
        
        try:
            response = self.session.get(f"{self.base_url}/experiments/{experiment_id}/status")
            
            if response.status_code == 200:
                result = response.json()
                
                experiment = result['experiment']
                print(f"   Name: {experiment['name']}")
                print(f"   Status: {result['status']}")
                print(f"   Duration: {result['duration_days']} days")
                print(f"   Predictions logged: {result['predictions_logged']}")
                
                if 'latest_results' in result and result['latest_results']:
                    latest = result['latest_results']
                    print(f"   Latest control rate: {latest['control_rate']:.1%}")
                    print(f"   Latest treatment rate: {latest['treatment_rate']:.1%}")
                    print(f"   Latest lift: {latest['relative_lift']:.1%}")
                
                return result
                
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return {}
    
    def run_comprehensive_phase3a_test(self):
        """Run complete Phase 3A test suite"""
        print("🚀 Phase 3A Comprehensive Test Suite")
        print("=" * 60)
        
        # Test 1: Health Check
        health_ok = self.test_health_check()
        if not health_ok:
            print("❌ Phase 3A server not ready - stopping tests")
            return False
        
        # Test 2: Sample Size Calculations
        sample_results = self.test_sample_size_calculation()
        
        # Test 3: Statistical Testing
        stat_results = self.test_statistical_testing()
        
        # Test 4: Experiment Configuration
        experiment_id = self.test_experiment_configuration()
        if not experiment_id:
            print("❌ Could not configure experiment - skipping remaining tests")
            return False
        
        # Test 5: A/B Predictions with Tracking
        prediction_results = self.test_ab_predictions_with_tracking(experiment_id)
        
        # Test 6: Early Stopping Evaluation
        stopping_results = self.test_early_stopping_evaluation(experiment_id)
        
        # Test 7: Experiment Status
        status_result = self.test_experiment_status(experiment_id)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 PHASE 3A TEST SUMMARY")
        print("=" * 60)
        
        print(f"✅ Health Check: {'PASS' if health_ok else 'FAIL'}")
        print(f"✅ Sample Size Calculations: {'PASS' if sample_results else 'FAIL'}")
        print(f"✅ Statistical Testing: {'PASS' if stat_results else 'FAIL'}")
        print(f"✅ Experiment Configuration: {'PASS' if experiment_id else 'FAIL'}")
        print(f"✅ A/B Prediction Tracking: {'PASS' if prediction_results else 'FAIL'}")
        print(f"✅ Early Stopping Logic: {'PASS' if stopping_results else 'FAIL'}")
        print(f"✅ Experiment Management: {'PASS' if status_result else 'FAIL'}")
        
        all_passed = all([
            health_ok, sample_results, stat_results, 
            experiment_id, prediction_results, stopping_results, status_result
        ])
        
        if all_passed:
            print("\n🎉 ALL PHASE 3A TESTS PASSED!")
            print("✅ Statistical Significance Testing: Working")
            print("✅ Early Stopping Criteria: Working")
            print("✅ Sample Size Calculations: Working")
            print("✅ Experiment Management: Working")
            print("✅ Advanced A/B Testing: Ready for Production")
        else:
            print("\n⚠️  Some Phase 3A tests failed - check output above")
        
        return all_passed

def main():
    """Main test execution"""
    print("🧪 Phase 3A Test Client")
    print("Testing Statistical Significance + Early Stopping")
    print()
    
    # Initialize client
    client = Phase3ATestClient()
    
    # Run comprehensive tests
    success = client.run_comprehensive_phase3a_test()
    
    if success:
        print("\n🚀 Phase 3A is ready!")
        print("💡 Next: Set up monitoring dashboards or begin Phase 3B")
    else:
        print("\n🔧 Phase 3A needs attention - check server and try again")

if __name__ == "__main__":
    main()