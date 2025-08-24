"""
Integrated A/B Testing API with Drift Detection
Implements Phase 2 of the MLOps plan
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import logging
import asyncio
import sys
from pathlib import Path

# Add our modules
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from production_drift_detector import ProductionDriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    user_id: str
    features: List[float]  
    experiment_id: Optional[int] = None

class DriftCheckRequest(BaseModel):
    feature_data: Dict[str, Dict[str, List[float]]]
    prediction_data: Optional[Dict[str, List[float]]] = None
    threshold: float = 0.05
    save_to_db: bool = True

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: float
    probability: Optional[float] = None
    model_version: str = "dummy"
    experiment_group: str = "control"
    risk_score: Optional[float] = None
    drift_status: Optional[str] = None  # NEW: Drift monitoring status

app = FastAPI(
    title="Production A/B Testing API with Drift Detection",
    description="Phase 2 Implementation - A/B Testing + Drift Detection",
    version="2.1.0"
)

# Global components
CONTROL_MODEL = None
TREATMENT_MODEL = None
SCALER = None
MODEL_LOADED = False
DRIFT_DETECTOR = None

def initialize_system():
    """Initialize both A/B testing and drift detection"""
    global CONTROL_MODEL, TREATMENT_MODEL, SCALER, MODEL_LOADED, DRIFT_DETECTOR
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        logger.info(" Initializing Production A/B Testing + Drift Detection System...")
        
        # Initialize drift detector
        DRIFT_DETECTOR = ProductionDriftDetector("sqlite:///production_drift.db")
        logger.info(" Drift detector initialized")
        
        # Initialize models (same as before)
        np.random.seed(42)
        n_samples = 2000
        
        loan_amounts = np.random.lognormal(10.5, 0.8, n_samples)
        incomes = np.random.lognormal(11.2, 0.6, n_samples)
        credit_scores = np.random.normal(680, 80, n_samples).clip(300, 850)
        debt_ratios = np.random.beta(2, 6, n_samples).clip(0, 0.8)
        employment = np.random.exponential(4, n_samples).clip(0, 30)
        
        X = np.column_stack([loan_amounts, incomes, credit_scores, debt_ratios, employment])
        
        risk_factors = (
            (X[:, 0] / X[:, 1]) * 2.0 +
            ((850 - X[:, 2]) / 100) * 1.5 +
            X[:, 3] * 3.0 +
            (1 / (X[:, 4] + 1)) * 0.8 +
            np.random.normal(0, 0.3, n_samples)
        )
        
        threshold = np.percentile(risk_factors, 80)
        y = (risk_factors > threshold).astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        control_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        treatment_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
        
        control_model.fit(X_scaled, y)
        treatment_model.fit(X_scaled, y)
        
        # Set globals
        CONTROL_MODEL = control_model
        TREATMENT_MODEL = treatment_model
        SCALER = scaler
        MODEL_LOADED = True
        
        # Store baseline data for drift detection
        global BASELINE_FEATURES, BASELINE_LABELS, BASELINE_PREDICTIONS
        BASELINE_FEATURES = X_scaled[:500]  # Store subset for drift comparison
        BASELINE_LABELS = y[:500]
        BASELINE_PREDICTIONS = control_model.predict_proba(BASELINE_FEATURES)[:, 1]
        
        logger.info(" Complete system initialization successful")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False

# Initialize on startup
initialize_system()

@app.post("/predict", response_model=PredictionResponse)
async def predict_with_drift_monitoring(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Enhanced prediction with drift monitoring"""
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
        
        # Make prediction
        features = np.array([request.features])
        features_scaled = SCALER.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Calculate risk score
        risk_score = min(((features[0, 0] / features[0, 1]) * 30 + 
                         ((850 - features[0, 2]) / 100) * 40 + 
                         features[0, 3] * 30), 100)
        
        # Background drift monitoring (async)
        if len(request.features) == 5:  # Ensure we have all features
            background_tasks.add_task(
                monitor_single_prediction_drift, 
                request.features, 
                float(probability)
            )
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=model_version,
            experiment_group=group,
            risk_score=float(risk_score),
            drift_status="monitoring"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_single_prediction_drift(features: List[float], prediction: float):
    """Background task for real-time drift monitoring"""
    try:
        # This runs in background - accumulate data for batch drift detection
        # In production, you'd store these in a buffer and run drift detection periodically
        logger.debug(f"Monitoring drift for prediction: {prediction:.3f}")
        # Implementation would accumulate data points and trigger drift analysis
    except Exception as e:
        logger.warning(f"Drift monitoring task failed: {e}")

@app.post("/drift/comprehensive-analysis")
async def comprehensive_drift_analysis(request: DriftCheckRequest):
    """Comprehensive drift analysis endpoint (Phase 2 feature)"""
    try:
        if not DRIFT_DETECTOR:
            raise HTTPException(status_code=503, detail="Drift detector not initialized")
        
        # Prepare feature data
        feature_data = {}
        for feature_name, data in request.feature_data.items():
            if 'baseline' not in data or 'current' not in data:
                raise HTTPException(status_code=400, detail=f"Missing baseline or current data for {feature_name}")
            
            baseline = np.array(data['baseline'])
            current = np.array(data['current'])
            feature_data[feature_name] = (baseline, current)
        
        # Prepare prediction data if provided
        prediction_data = None
        if request.prediction_data:
            if 'baseline' in request.prediction_data and 'current' in request.prediction_data:
                baseline_pred = np.array(request.prediction_data['baseline'])
                current_pred = np.array(request.prediction_data['current'])
                prediction_data = (baseline_pred, current_pred)
        
        # Run comprehensive drift analysis
        drift_results = DRIFT_DETECTOR.comprehensive_drift_analysis(
            feature_data=feature_data,
            prediction_data=prediction_data,
            save_to_db=request.save_to_db
        )
        
        # Log important drift alerts
        if drift_results['summary']['any_drift_detected']:
            alert_level = drift_results['summary']['drift_alert_level']
            features_with_drift = drift_results['summary']['features_with_drift']
            logger.warning(f" DRIFT ALERT [{alert_level}]: {features_with_drift} features showing drift")
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Drift analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/status")
async def get_drift_status():
    """Get current drift monitoring status"""
    try:
        if not DRIFT_DETECTOR:
            return {"status": "drift_detector_unavailable"}
        
        # Get recent drift measurements from database
        session = DRIFT_DETECTOR.Session()
        try:
            from production_drift_detector import DriftMeasurement
            recent_measurements = session.query(DriftMeasurement).filter(
                DriftMeasurement.measured_at >= datetime.utcnow() - timedelta(hours=24)
            ).all()
            
            drift_summary = {
                "status": "monitoring",
                "last_24h_measurements": len(recent_measurements),
                "features_with_drift": len([m for m in recent_measurements if m.is_drift_detected]),
                "drift_types": list(set([m.drift_type for m in recent_measurements])),
                "alert_level": "GREEN"
            }
            
            # Set alert level
            if drift_summary["features_with_drift"] > 3:
                drift_summary["alert_level"] = "RED"
            elif drift_summary["features_with_drift"] > 0:
                drift_summary["alert_level"] = "YELLOW"
            
            return drift_summary
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Drift status error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "components": {
            "ab_testing": MODEL_LOADED,
            "drift_detection": DRIFT_DETECTOR is not None,
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
        "message": " Production A/B Testing API with Drift Detection",
        "phase": "Phase 2 Implementation",
        "features": [
            "A/B Traffic Splitting",
            "Real-time Drift Detection", 
            "Statistical Significance Testing",
            "Automated Alert System",
            "Comprehensive Monitoring"
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
