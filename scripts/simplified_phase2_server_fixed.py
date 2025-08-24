"""
Simplified Phase 2 Server - A/B Testing + Drift Detection
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict
import numpy as np
from datetime import datetime
import logging
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    user_id: str
    features: List[float]
    experiment_id: Optional[int] = None

class DriftCheckRequest(BaseModel):
    feature_data: Dict[str, Dict[str, List[float]]]
    threshold: float = 0.05

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: float
    probability: Optional[float] = None
    model_version: str = "dummy"
    experiment_group: str = "control"
    risk_score: Optional[float] = None
    drift_status: str = "monitoring"

app = FastAPI(
    title="Phase 2: A/B Testing + Drift Detection",
    description="Simplified implementation of Phase 2 components",
    version="2.1.0"
)

# Global variables
CONTROL_MODEL = None
TREATMENT_MODEL = None
SCALER = None
MODEL_LOADED = False

def initialize_models():
    global CONTROL_MODEL, TREATMENT_MODEL, SCALER, MODEL_LOADED
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        logger.info(" Initializing Phase 2 System...")
        
        # Generate training data
        np.random.seed(42)
        n_samples = 1500
        
        # Loan features
        X = np.random.random((n_samples, 5)) * 100
        risk_scores = (X[:, 0] + X[:, 3] - X[:, 2] + np.random.normal(0, 10, n_samples))
        y = (risk_scores > np.percentile(risk_scores, 75)).astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train two different models
        control_model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
        treatment_model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
        
        control_model.fit(X_scaled, y)
        treatment_model.fit(X_scaled, y)
        
        CONTROL_MODEL = control_model
        TREATMENT_MODEL = treatment_model
        SCALER = scaler
        MODEL_LOADED = True
        
        logger.info(" Phase 2 models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return False

# Initialize on startup
initialize_models()

def simple_drift_detection(baseline_data: List[float], current_data: List[float]) -> Dict:
    """Simplified drift detection using KS test"""
    try:
        baseline = np.array(baseline_data)
        current = np.array(current_data)
        
        # KS test
        ks_statistic, p_value = ks_2samp(baseline, current)
        
        # PSI calculation (simplified)
        baseline_mean = np.mean(baseline)
        current_mean = np.mean(current)
        baseline_std = np.std(baseline)
        current_std = np.std(current)
        
        # Simple PSI approximation
        psi_score = abs(current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        drift_detected = bool(p_value < 0.05 or psi_score > 0.5)
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': float(ks_statistic),
            'p_value': float(p_value),
            'psi_score': float(psi_score),
            'test_method': 'ks_test_and_psi',
            'baseline_stats': {
                'mean': float(baseline_mean),
                'std': float(baseline_std),
                'count': len(baseline)
            },
            'current_stats': {
                'mean': float(current_mean),
                'std': float(current_std),
                'count': len(current)
            }
        }
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return {'drift_detected': False, 'error': str(e)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_with_monitoring(request: PredictionRequest):
    """Phase 2: A/B predictions with drift monitoring"""
    try:
        if not MODEL_LOADED:
            raise HTTPException(status_code=503, detail="Models not initialized")
        
        # A/B assignment
        user_hash = hash(request.user_id) % 100
        if user_hash < 50:
            group = "control"
            model = CONTROL_MODEL
            model_version = "random_forest_v2.0"
        else:
            group = "treatment"
            model = TREATMENT_MODEL
            model_version = "gradient_boost_v2.0"
        
        # Prediction
        features = np.array([request.features])
        features_scaled = SCALER.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Risk score calculation
        risk_score = min((features[0, 0] * 0.3 + features[0, 3] * 0.7) * 2, 100)
        
        logger.info(f"Phase 2 Prediction: User {request.user_id} -> {group}, Risk: {probability:.1%}")
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=model_version,
            experiment_group=group,
            risk_score=float(risk_score),
            drift_status="monitoring_active"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/comprehensive-analysis")
async def drift_analysis(request: DriftCheckRequest):
    """Phase 2: Comprehensive drift detection"""
    try:
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'feature_drift': {},
            'summary': {
                'total_features': len(request.feature_data),
                'features_with_drift': 0,
                'any_drift_detected': False,
                'drift_alert_level': 'GREEN'
            }
        }
        
        # Analyze each feature
        for feature_name, data in request.feature_data.items():
            if 'baseline' not in data or 'current' not in data:
                continue
                
            drift_result = simple_drift_detection(data['baseline'], data['current'])
            drift_result['feature_name'] = feature_name
            
            results['feature_drift'][feature_name] = drift_result
            
            if drift_result['drift_detected']:
                results['summary']['features_with_drift'] += 1
                results['summary']['any_drift_detected'] = True
        
        # Set alert level
        if results['summary']['features_with_drift'] > 2:
            results['summary']['drift_alert_level'] = 'RED'
        elif results['summary']['any_drift_detected']:
            results['summary']['drift_alert_level'] = 'YELLOW'
        
        logger.info(f"Drift Analysis: {results['summary']['features_with_drift']}/{results['summary']['total_features']} features with drift")
        
        return results
        
    except Exception as e:
        logger.error(f"Drift analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/status")
async def drift_status():
    """Phase 2: Drift monitoring status"""
    return {
        "status": "monitoring_active",
        "alert_level": "GREEN",
        "last_24h_measurements": 0,
        "features_with_drift": 0,
        "drift_types": ["feature", "prediction", "concept"],
        "phase": "Phase 2 - Simplified Implementation"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "components": {
            "ab_testing": MODEL_LOADED,
            "drift_detection": True,
            "models": {
                "control": "RandomForestClassifier",
                "treatment": "GradientBoostingClassifier"
            }
        },
        "phase": "Phase 2 - Drift Detection Integrated"
    }

@app.get("/")
async def root():
    return {
        "message": " Phase 2: A/B Testing + Drift Detection",
        "phase": "Phase 2 Implementation",
        "status": "operational",
        "features": [
            "A/B Traffic Splitting",
            "Drift Detection Engine", 
            "Statistical Testing",
            "Real-time Monitoring"
        ],
        "endpoints": {
            "predict": "/predict",
            "drift_analysis": "/drift/comprehensive-analysis",
            "drift_status": "/drift/status",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
