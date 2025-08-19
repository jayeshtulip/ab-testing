#!/usr/bin/env python3
"""
Model Validation Script for Phase 2C
Validates model performance against quality gates before deployment
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import mlflow
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Phase 2C Model Validator with production-ready quality gates"""
    
    def __init__(self, mlflow_uri: Optional[str] = None):
        self.mlflow_uri = mlflow_uri or os.getenv(
            'MLFLOW_TRACKING_URI', 
            'http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com'
        )
        
        # Phase 2C Quality Gates - Production Standards
        self.quality_gates = {
            'accuracy': {
                'minimum': 0.75,      # Must exceed 75%
                'target': 0.80,       # Target 80%
                'excellent': 0.85     # Excellent 85%+
            },
            'f1_score': {
                'minimum': 0.70,      # Must exceed 70%
                'target': 0.75,       # Target 75%
                'excellent': 0.80     # Excellent 80%+
            },
            'precision': {
                'minimum': 0.65,      # Must exceed 65%
                'target': 0.70,       # Target 70%
                'excellent': 0.75     # Excellent 75%+
            },
            'recall': {
                'minimum': 0.65,      # Must exceed 65%
                'target': 0.70,       # Target 70%
                'excellent': 0.75     # Excellent 75%+
            }
        }
        
        # Business rules
        self.business_rules = {
            'max_performance_degradation': 0.05,  # Max 5% performance drop vs production
            'min_training_samples': 100,          # Minimum training samples required
            'max_overfitting_ratio': 0.20         # Max 20% gap between train/test
        }
        
    def test_mlflow_connection(self) -> bool:
        """Test connection to MLflow server"""
        try:
            logger.info(f"🔗 Testing MLflow connection: {self.mlflow_uri}")
            mlflow.set_tracking_uri(self.mlflow_uri)
            
            client = mlflow.MlflowClient()
            try:
                experiments = client.search_experiments()  # Fixed: newer MLflow API
            except AttributeError:
                experiments = client.list_experiments()    # Fallback for older versions
            
            logger.info(f"✅ MLflow connection successful! Found {len(experiments)} experiments")
            return True
            
        except Exception as e:
            logger.error(f"❌ MLflow connection failed: {e}")
            return False
    
    def get_latest_model(self, model_name: str = "loan-default-model") -> Tuple[str, str]:
        """Get latest model version and run ID"""
        try:
            client = mlflow.MlflowClient()
            
            # Try different stages in order of preference
            for stage in ["Production", "Staging", "None"]:
                try:
                    models = client.get_latest_versions(model_name, stages=[stage])
                    if models:
                        model = models[0]
                        logger.info(f"📦 Found {stage} model: {model_name} v{model.version}")
                        return model.version, model.run_id
                except Exception as e:
                    logger.debug(f"No {stage} model found: {e}")
                    continue
            
            raise Exception(f"No model versions found for {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error getting model: {e}")
            raise
    
    def get_model_metrics(self, run_id: str) -> Dict:
        """Get model metrics from MLflow run"""
        try:
            client = mlflow.MlflowClient()
            run = client.get_run(run_id)
            return dict(run.data.metrics)
            
        except Exception as e:
            logger.error(f"❌ Error getting model metrics: {e}")
            raise
    
    def validate_metrics(self, metrics: Dict, model_version: str) -> Dict:
        """Validate model metrics against quality gates"""
        logger.info(f"🔍 Validating model v{model_version} against quality gates")
        
        validation_results = {}
        overall_passed = True
        
        # Validate each metric
        for metric_name, thresholds in self.quality_gates.items():
            value = metrics.get(metric_name, 0)
            
            # Determine status
            if value >= thresholds['excellent']:
                status = 'EXCELLENT'
                passed = True
            elif value >= thresholds['target']:
                status = 'GOOD'
                passed = True
            elif value >= thresholds['minimum']:
                status = 'ACCEPTABLE'
                passed = True
            else:
                status = 'FAILED'
                passed = False
                overall_passed = False
            
            validation_results[metric_name] = {
                'value': value,
                'status': status,
                'passed': passed,
                'thresholds': thresholds
            }
            
            # Log result with emoji
            status_emoji = "✅" if passed else "❌"
            logger.info(f"   {metric_name}: {value:.4f} (min: {thresholds['minimum']:.2f}) {status_emoji} {status}")
        
        # Overall assessment
        if overall_passed:
            # Count excellent and good metrics
            excellent_count = sum(1 for r in validation_results.values() if r['status'] == 'EXCELLENT')
            good_count = sum(1 for r in validation_results.values() if r['status'] in ['EXCELLENT', 'GOOD'])
            
            if excellent_count >= len(validation_results) // 2:
                overall_status = 'EXCELLENT'
                recommendation = 'DEPLOY_TO_PRODUCTION'
            elif good_count >= len(validation_results) * 0.75:
                overall_status = 'GOOD'
                recommendation = 'DEPLOY_TO_STAGING'
            else:
                overall_status = 'ACCEPTABLE'
                recommendation = 'DEPLOY_WITH_MONITORING'
        else:
            overall_status = 'FAILED'
            recommendation = 'REJECT_DEPLOYMENT'
        
        return {
            'individual_results': validation_results,
            'overall_status': overall_status,
            'overall_passed': overall_passed,
            'recommendation': recommendation
        }
    
    def compare_with_production(self, current_metrics: Dict, model_name: str = "loan-default-model") -> Optional[Dict]:
        """Compare current model with production model"""
        try:
            client = mlflow.MlflowClient()
            
            # Get production model
            production_models = client.get_latest_versions(model_name, stages=["Production"])
            
            if not production_models:
                logger.info("ℹ️  No production model found for comparison")
                return None
            
            prod_model = production_models[0]
            prod_run = client.get_run(prod_model.run_id)
            prod_metrics = dict(prod_run.data.metrics)
            
            logger.info(f"📊 Comparing with Production model v{prod_model.version}")
            
            comparisons = {}
            significant_degradation = False
            
            for metric in ['accuracy', 'f1_score']:
                if metric in current_metrics and metric in prod_metrics:
                    current_val = current_metrics[metric]
                    prod_val = prod_metrics[metric]
                    diff = current_val - prod_val
                    pct_change = (diff / prod_val) * 100 if prod_val > 0 else 0
                    
                    # Check for significant degradation
                    if pct_change < -self.business_rules['max_performance_degradation'] * 100:
                        significant_degradation = True
                    
                    comparisons[metric] = {
                        'current': current_val,
                        'production': prod_val,
                        'difference': diff,
                        'percent_change': pct_change,
                        'improved': diff > 0
                    }
                    
                    trend = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                    logger.info(f"   {metric}: {current_val:.4f} vs {prod_val:.4f} ({diff:+.4f}, {pct_change:+.1f}%) {trend}")
            
            if significant_degradation:
                logger.warning("⚠️  WARNING: Significant performance degradation detected!")
            
            return {
                'production_version': prod_model.version,
                'comparisons': comparisons,
                'significant_degradation': significant_degradation
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Could not compare with production model: {e}")
            return None
    
    def validate_model(self, model_name: str = "loan-default-model") -> Dict:
        """Main validation function"""
        try:
            logger.info("🚀 Starting Phase 2C Model Validation")
            
            # Check for force deploy
            force_deploy = os.getenv('FORCE_DEPLOY', 'false').lower() == 'true'
            if force_deploy:
                logger.warning("🚨 Force deployment enabled - skipping validation gates")
                return {
                    'passed': True,
                    'status': 'FORCE_DEPLOYED',
                    'recommendation': 'FORCE_DEPLOYMENT',
                    'message': 'Validation bypassed due to force deployment flag'
                }
            
            # Test MLflow connection
            if not self.test_mlflow_connection():
                raise Exception("Cannot connect to MLflow server")
            
            # Get latest model
            model_version, run_id = self.get_latest_model(model_name)
            
            # Get model metrics
            metrics = self.get_model_metrics(run_id)
            
            # Validate metrics
            validation_result = self.validate_metrics(metrics, model_version)
            
            # Compare with production
            comparison_result = self.compare_with_production(metrics, model_name)
            
            # Create comprehensive result
            result = {
                'passed': validation_result['overall_passed'],
                'model_version': model_version,
                'run_id': run_id,
                'metrics': metrics,
                'validation': validation_result,
                'comparison': comparison_result,
                'timestamp': datetime.now().isoformat(),
                'mlflow_uri': self.mlflow_uri
            }
            
            # Log final result
            if validation_result['overall_passed']:
                logger.info(f"✅ Model validation PASSED - {validation_result['overall_status']}")
                logger.info(f"🚀 Recommendation: {validation_result['recommendation']}")
            else:
                logger.error(f"❌ Model validation FAILED - {validation_result['overall_status']}")
                logger.error(f"🚫 Recommendation: {validation_result['recommendation']}")
            
            return result
            
        except Exception as e:
            logger.error(f"💥 Model validation error: {e}")
            return {
                'passed': False,
                'status': 'ERROR',
                'recommendation': 'VALIDATION_ERROR',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_validation_report(self, result: Dict, filepath: str = 'model_validation_report.json'):
        """Save validation results to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"📄 Validation report saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Could not save validation report: {e}")


def main():
    """Main function for standalone execution"""
    try:
        # Initialize validator
        validator = ModelValidator()
        
        # Run validation
        result = validator.validate_model()
        
        # Save report
        validator.save_validation_report(result)
        
        # Print summary for GitHub Actions
        print(f"\n{'='*50}")
        print("📊 VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Status: {'PASSED' if result['passed'] else 'FAILED'}")
        print(f"Model Version: {result.get('model_version', 'N/A')}")
        print(f"Recommendation: {result.get('recommendation', 'N/A')}")
        
        if 'metrics' in result:
            print(f"\n📈 Model Metrics:")
            for metric, value in result['metrics'].items():
                print(f"   {metric}: {value:.4f}")
        
        print(f"{'='*50}")
        
        # Exit with appropriate code
        sys.exit(0 if result['passed'] else 1)
        
    except Exception as e:
        logger.error(f"💥 Validation script failed: {e}")
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()