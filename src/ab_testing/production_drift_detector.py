"""
Production-ready Drift Detection Engine
Implements KS test, PSI, and Chi-square as per the plan
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy import Column, Float, DateTime, Text, Boolean, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import json
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DriftMeasurement(Base):
    __tablename__ = 'drift_measurements'
    
    id = Column(Integer, primary_key=True)
    feature_name = Column(String(100), nullable=False)
    drift_score = Column(Float, nullable=False)
    drift_type = Column(String(50), nullable=False)  # 'feature', 'prediction', 'concept'
    test_method = Column(String(50), nullable=False)
    p_value = Column(Float)
    threshold = Column(Float, default=0.05)
    is_drift_detected = Column(Boolean, default=False)
    baseline_stats = Column(JSON)
    current_stats = Column(JSON)
    measured_at = Column(DateTime, default=datetime.utcnow)
    additional_info = Column(JSON)

class ProductionDriftDetector:
    """Production-grade drift detection as specified in the implementation plan"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(" Drift Detection Engine initialized")
        
    def calculate_psi(self, baseline: np.array, current: np.array, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI) - Key metric from plan"""
        
        # Create bins based on baseline distribution
        try:
            bin_edges = np.histogram_bin_edges(baseline, bins=bins)
            
            # Calculate proportions for each bin
            baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
            current_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            baseline_prop = (baseline_counts + epsilon) / (len(baseline) + bins * epsilon)
            current_prop = (current_counts + epsilon) / (len(current) + bins * epsilon)
            
            # Calculate PSI
            psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
            
            return psi
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def detect_feature_drift(self, 
                           feature_name: str,
                           baseline_data: np.array, 
                           current_data: np.array,
                           threshold: float = 0.05) -> Dict:
        """Feature drift detection using KS test and PSI (as per plan)"""
        
        results = {
            'feature_name': feature_name,
            'drift_detected': False,
            'drift_score': 0.0,
            'p_value': 1.0,
            'test_method': 'unknown',
            'baseline_stats': {},
            'current_stats': {},
            'psi_score': 0.0
        }
        
        try:
            # Basic statistics
            results['baseline_stats'] = {
                'mean': float(np.mean(baseline_data)),
                'std': float(np.std(baseline_data)),
                'min': float(np.min(baseline_data)),
                'max': float(np.max(baseline_data)),
                'count': len(baseline_data)
            }
            
            results['current_stats'] = {
                'mean': float(np.mean(current_data)),
                'std': float(np.std(current_data)),
                'min': float(np.min(current_data)),
                'max': float(np.max(current_data)),
                'count': len(current_data)
            }
            
            # Determine if data is continuous or categorical
            baseline_unique = len(np.unique(baseline_data))
            current_unique = len(np.unique(current_data))
            
            if baseline_unique > 20 and current_unique > 20:
                # Continuous data - use KS test and PSI (as specified in plan)
                ks_statistic, p_value = ks_2samp(baseline_data, current_data)
                psi_score = self.calculate_psi(baseline_data, current_data)
                
                results.update({
                    'test_method': 'ks_test_and_psi',
                    'drift_score': float(ks_statistic),
                    'p_value': float(p_value),
                    'psi_score': float(psi_score),
                    'drift_detected': p_value < threshold or psi_score > 0.1  # PSI threshold per plan
                })
                
                logger.info(f"Feature {feature_name}: KS={ks_statistic:.4f}, PSI={psi_score:.4f}, Drift={results['drift_detected']}")
                
            else:
                # Categorical data - use Chi-square test (as specified in plan)
                try:
                    baseline_counts = pd.Series(baseline_data).value_counts()
                    current_counts = pd.Series(current_data).value_counts()
                    
                    # Align categories
                    all_categories = set(baseline_counts.index) | set(current_counts.index)
                    baseline_aligned = [baseline_counts.get(cat, 0) for cat in all_categories]
                    current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                    
                    contingency_table = np.array([baseline_aligned, current_aligned])
                    
                    if contingency_table.sum() > 0:
                        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                        
                        results.update({
                            'test_method': 'chi_square',
                            'drift_score': float(chi2_stat),
                            'p_value': float(p_value),
                            'drift_detected': p_value < threshold
                        })
                        
                        logger.info(f"Feature {feature_name}: Chi2={chi2_stat:.4f}, Drift={results['drift_detected']}")
                        
                except Exception as e:
                    logger.warning(f"Chi-square test failed for {feature_name}: {e}")
            
        except Exception as e:
            logger.error(f"Feature drift detection failed for {feature_name}: {e}")
        
        return results
    
    def detect_prediction_drift(self, 
                              baseline_predictions: np.array,
                              current_predictions: np.array,
                              threshold: float = 0.05) -> Dict:
        """Prediction drift analysis (as per plan)"""
        
        try:
            # For binary classification
            if set(np.unique(baseline_predictions)) == {0, 1}:
                baseline_rate = np.mean(baseline_predictions)
                current_rate = np.mean(current_predictions)
                
                # Use proportion test
                from scipy.stats import proportions_ztest
                
                baseline_successes = int(np.sum(baseline_predictions))
                current_successes = int(np.sum(current_predictions))
                
                counts = np.array([baseline_successes, current_successes])
                nobs = np.array([len(baseline_predictions), len(current_predictions)])
                
                z_stat, p_value = proportions_ztest(counts, nobs)
                
                drift_detected = p_value < threshold
                
                logger.info(f"Prediction drift: {baseline_rate:.3f} -> {current_rate:.3f}, Drift={drift_detected}")
                
                return {
                    'drift_type': 'prediction_drift',
                    'test_method': 'proportion_test',
                    'baseline_rate': float(baseline_rate),
                    'current_rate': float(current_rate),
                    'z_statistic': float(z_stat),
                    'p_value': float(p_value),
                    'drift_detected': drift_detected,
                    'rate_change': float(current_rate - baseline_rate)
                }
            
            else:
                # For regression or continuous predictions
                ks_statistic, p_value = ks_2samp(baseline_predictions, current_predictions)
                psi_score = self.calculate_psi(baseline_predictions, current_predictions)
                
                drift_detected = p_value < threshold or psi_score > 0.1
                
                return {
                    'drift_type': 'prediction_drift',
                    'test_method': 'ks_test_and_psi',
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value),
                    'psi_score': float(psi_score),
                    'drift_detected': drift_detected,
                    'baseline_mean': float(np.mean(baseline_predictions)),
                    'current_mean': float(np.mean(current_predictions))
                }
        
        except Exception as e:
            logger.error(f"Prediction drift detection failed: {e}")
            return {'drift_type': 'prediction_drift', 'drift_detected': False, 'error': str(e)}
    
    def detect_concept_drift(self,
                           baseline_features: np.array,
                           baseline_labels: np.array,
                           current_features: np.array,
                           current_labels: np.array,
                           threshold: float = 0.05) -> Dict:
        """Concept drift detection - Performance degradation (as per plan)"""
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            # Train model on baseline data
            baseline_model = RandomForestClassifier(n_estimators=50, random_state=42)
            baseline_model.fit(baseline_features, baseline_labels)
            
            # Test on current data
            current_predictions = baseline_model.predict(current_features)
            current_accuracy = accuracy_score(current_labels, current_predictions)
            
            # Compare with baseline performance
            baseline_predictions = baseline_model.predict(baseline_features)
            baseline_accuracy = accuracy_score(baseline_labels, baseline_predictions)
            
            accuracy_drop = baseline_accuracy - current_accuracy
            drift_detected = accuracy_drop > threshold
            
            logger.info(f"Concept drift: {baseline_accuracy:.3f} -> {current_accuracy:.3f}, Drop={accuracy_drop:.3f}")
            
            return {
                'drift_type': 'concept_drift',
                'test_method': 'model_performance_comparison',
                'baseline_accuracy': float(baseline_accuracy),
                'current_accuracy': float(current_accuracy),
                'accuracy_drop': float(accuracy_drop),
                'drift_detected': drift_detected,
                'relative_performance_drop': float(accuracy_drop / baseline_accuracy) if baseline_accuracy > 0 else 0
            }
        
        except Exception as e:
            logger.error(f"Concept drift detection failed: {e}")
            return {'drift_type': 'concept_drift', 'drift_detected': False, 'error': str(e)}
    
    def comprehensive_drift_analysis(self, 
                                   feature_data: Dict[str, Tuple[np.array, np.array]],
                                   prediction_data: Tuple[np.array, np.array] = None,
                                   concept_data: Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]] = None,
                                   save_to_db: bool = True) -> Dict:
        """Comprehensive drift monitoring as specified in the plan"""
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'feature_drift': {},
            'prediction_drift': None,
            'concept_drift': None,
            'summary': {
                'total_features': len(feature_data),
                'features_with_drift': 0,
                'any_drift_detected': False,
                'drift_alert_level': 'GREEN'
            }
        }
        
        session = self.Session()
        
        try:
            # 1. Feature Drift Detection
            for feature_name, (baseline, current) in feature_data.items():
                drift_result = self.detect_feature_drift(feature_name, baseline, current)
                results['feature_drift'][feature_name] = drift_result
                
                if drift_result['drift_detected']:
                    results['summary']['features_with_drift'] += 1
                    results['summary']['any_drift_detected'] = True
                
                if save_to_db:
                    drift_measurement = DriftMeasurement(
                        feature_name=feature_name,
                        drift_score=drift_result['drift_score'],
                        drift_type='feature',
                        test_method=drift_result['test_method'],
                        p_value=drift_result.get('p_value'),
                        is_drift_detected=drift_result['drift_detected'],
                        baseline_stats=drift_result['baseline_stats'],
                        current_stats=drift_result['current_stats'],
                        additional_info=drift_result
                    )
                    session.add(drift_measurement)
            
            # 2. Prediction Drift Detection
            if prediction_data:
                baseline_pred, current_pred = prediction_data
                pred_drift_result = self.detect_prediction_drift(baseline_pred, current_pred)
                results['prediction_drift'] = pred_drift_result
                
                if pred_drift_result['drift_detected']:
                    results['summary']['any_drift_detected'] = True
                
                if save_to_db:
                    drift_measurement = DriftMeasurement(
                        feature_name='predictions',
                        drift_score=pred_drift_result.get('ks_statistic', pred_drift_result.get('z_statistic', 0)),
                        drift_type='prediction',
                        test_method=pred_drift_result['test_method'],
                        p_value=pred_drift_result.get('p_value'),
                        is_drift_detected=pred_drift_result['drift_detected'],
                        additional_info=pred_drift_result
                    )
                    session.add(drift_measurement)
            
            # 3. Concept Drift Detection
            if concept_data:
                (baseline_X, baseline_y), (current_X, current_y) = concept_data
                concept_drift_result = self.detect_concept_drift(baseline_X, baseline_y, current_X, current_y)
                results['concept_drift'] = concept_drift_result
                
                if concept_drift_result['drift_detected']:
                    results['summary']['any_drift_detected'] = True
                
                if save_to_db:
                    drift_measurement = DriftMeasurement(
                        feature_name='concept',
                        drift_score=concept_drift_result.get('accuracy_drop', 0),
                        drift_type='concept',
                        test_method=concept_drift_result['test_method'],
                        is_drift_detected=concept_drift_result['drift_detected'],
                        additional_info=concept_drift_result
                    )
                    session.add(drift_measurement)
            
            # Set alert level based on drift severity
            if results['summary']['features_with_drift'] > 2:
                results['summary']['drift_alert_level'] = 'RED'
            elif results['summary']['any_drift_detected']:
                results['summary']['drift_alert_level'] = 'YELLOW'
            
            session.commit()
            
            logger.info(f" Drift analysis complete: {results['summary']['features_with_drift']}/{results['summary']['total_features']} features with drift")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Drift analysis failed: {e}")
            raise
        finally:
            session.close()
        
        return results

if __name__ == "__main__":
    # Test the drift detection engine
    detector = ProductionDriftDetector("sqlite:///drift_monitoring.db")
    
    # Generate test data
    np.random.seed(42)
    baseline_data = np.random.normal(0, 1, 1000)
    drifted_data = np.random.normal(0.5, 1.2, 1000)  # Shifted mean and variance
    
    # Test feature drift
    result = detector.detect_feature_drift('test_feature', baseline_data, drifted_data)
    print(f"Feature drift detected: {result['drift_detected']}")
    print(f"Drift score: {result['drift_score']:.4f}")
    print(f"PSI score: {result.get('psi_score', 0):.4f}")
