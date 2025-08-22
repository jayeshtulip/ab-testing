from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from datetime import datetime
import numpy as np
import json
import os
import logging
import asyncio
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ka-MLOps Multi-Pod PostgreSQL API", version="3.0.0")

class PredictionRequest(BaseModel):
    features: list = []

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_value: int

# Prometheus metrics
prediction_counter = Counter("ka_predictions_total", "Total predictions made")
prediction_accuracy = Gauge("ka_model_accuracy", "Current model accuracy")
prediction_f1_score = Gauge("ka_model_f1_score", "Current model F1 score")
retraining_trigger = Counter("ka_retraining_triggered_total", "Number of retraining triggers")

# Configuration
f1_threshold = 0.65
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://mlops_user:mlops_password@postgres:5432/mlops")

class DatabasePerformanceMonitor:
    def __init__(self):
        self.db_url = DATABASE_URL
        self.init_database()
        
    def get_connection(self):
        """Get database connection with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(self.db_url)
                return conn
            except psycopg2.OperationalError as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id VARCHAR(100) PRIMARY KEY,
                    prediction INTEGER NOT NULL,
                    actual_value INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create performance_metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    accuracy FLOAT,
                    f1_score FLOAT,
                    precision_score FLOAT,
                    recall_score FLOAT,
                    sample_count INTEGER,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create retraining_triggers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retraining_triggers (
                    id SERIAL PRIMARY KEY,
                    reason VARCHAR(200),
                    f1_score FLOAT,
                    accuracy FLOAT,
                    trigger_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'triggered'
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info(" Database initialized successfully")
            
        except Exception as e:
            logger.error(f" Database initialization failed: {e}")
    
    def add_prediction(self, prediction: int, prediction_id: str, actual: int = None):
        """Store prediction in database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (id, prediction, actual_value)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                actual_value = EXCLUDED.actual_value
            """, (prediction_id, prediction, actual))
            
            conn.commit()
            conn.close()
            
            logger.info(f" Stored prediction: {prediction_id}")
            
        except Exception as e:
            logger.error(f" Failed to store prediction: {e}")
    
    def update_feedback(self, prediction_id: str, actual_value: int):
        """Update prediction with actual value"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE predictions 
                SET actual_value = %s 
                WHERE id = %s
            """, (actual_value, prediction_id))
            
            if cursor.rowcount == 0:
                conn.close()
                raise HTTPException(status_code=404, detail="Prediction not found")
            
            conn.commit()
            conn.close()
            
            logger.info(f" Updated prediction {prediction_id} with feedback {actual_value}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to update feedback: {e}")
            return False
    
    def get_prediction_counts(self):
        """Get prediction counts from database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_value IS NOT NULL")
            evaluated_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {"total": total_count, "evaluated": evaluated_count}
            
        except Exception as e:
            logger.error(f" Failed to get counts: {e}")
            return {"total": 0, "evaluated": 0}
    
    def calculate_current_performance(self):
        """Calculate performance from database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get recent evaluated predictions (last 1000)
            cursor.execute("""
                SELECT prediction, actual_value 
                FROM predictions 
                WHERE actual_value IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 1000
            """)
            
            evaluated = cursor.fetchall()
            conn.close()
            
            if len(evaluated) < 10:
                logger.warning(f" Not enough evaluated predictions: {len(evaluated)}")
                return None
            
            # Calculate metrics
            correct = sum(1 for p in evaluated if p['prediction'] == p['actual_value'])
            accuracy = correct / len(evaluated)
            
            # F1 calculation
            tp = sum(1 for p in evaluated if p['prediction'] == 1 and p['actual_value'] == 1)
            fp = sum(1 for p in evaluated if p['prediction'] == 1 and p['actual_value'] == 0)
            fn = sum(1 for p in evaluated if p['prediction'] == 0 and p['actual_value'] == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            performance = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "samples": len(evaluated)
            }
            
            # Store performance metrics
            self.store_performance_metrics(performance)
            
            logger.info(f" Performance calculated: F1={f1:.3f}, Accuracy={accuracy:.3f}")
            return performance
            
        except Exception as e:
            logger.error(f" Performance calculation failed: {e}")
            return None
    
    def store_performance_metrics(self, performance: dict):
        """Store performance metrics in database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics 
                (accuracy, f1_score, precision_score, recall_score, sample_count)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                performance["accuracy"],
                performance["f1_score"], 
                performance["precision"],
                performance["recall"],
                performance["samples"]
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f" Failed to store performance metrics: {e}")
    
    def check_performance_degradation(self):
        """Check if retraining is needed"""
        current_perf = self.calculate_current_performance()
        if not current_perf:
            return False
        
        # Update Prometheus metrics
        prediction_accuracy.set(current_perf["accuracy"])
        prediction_f1_score.set(current_perf["f1_score"])
        
        # Check degradation
        if current_perf["f1_score"] < f1_threshold:
            logger.warning(f" Performance degradation detected: F1 {current_perf['f1_score']:.3f} < {f1_threshold}")
            self.log_retraining_trigger(current_perf)
            return True
        
        return False
    
    def log_retraining_trigger(self, performance: dict):
        """Log retraining trigger to database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO retraining_triggers 
                (reason, f1_score, accuracy)
                VALUES (%s, %s, %s)
            """, (
                "performance_degradation",
                performance["f1_score"],
                performance["accuracy"]
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(" Retraining trigger logged to database")
            
        except Exception as e:
            logger.error(f" Failed to log retraining trigger: {e}")

# Global monitor
monitor = DatabasePerformanceMonitor()

@app.get("/")
async def root():
    return {"message": "Ka-MLOps Multi-Pod PostgreSQL API", "version": "3.0.0", "status": "healthy"}

@app.get("/health")
async def health():
    try:
        counts = monitor.get_prediction_counts()
        return {
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(), 
            "predictions_count": counts["total"],
            "evaluated_count": counts["evaluated"],
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "database": "disconnected"
        }

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        prediction_counter.inc()
        prediction = int(np.random.choice([0, 1]))
        pred_id = f"pred_{datetime.now().timestamp()}"
        
        # Store in database
        monitor.add_prediction(prediction, pred_id)
        
        result = {
            "prediction": prediction,
            "prediction_id": pred_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f" Made prediction: {pred_id}")
        return result
        
    except Exception as e:
        logger.error(f" Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        success = monitor.update_feedback(request.prediction_id, request.actual_value)
        
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {"status": "feedback_received", "prediction_id": request.prediction_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/check")
async def check_performance():
    try:
        current_perf = monitor.calculate_current_performance()
        degradation = monitor.check_performance_degradation()
        
        if degradation:
            await trigger_retraining()
        
        counts = monitor.get_prediction_counts()
        
        return {
            "current_performance": current_perf,
            "degradation_detected": degradation,
            "last_check": datetime.now().isoformat(),
            "retraining_threshold": f1_threshold,
            "predictions_count": counts["total"]
        }
    except Exception as e:
        logger.error(f" Performance check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/history")
async def get_performance_history():
    counts = monitor.get_prediction_counts()
    return {
        "predictions_count": counts["total"],
        "evaluated_count": counts["evaluated"],
        "current_performance": monitor.calculate_current_performance(),
        "last_updated": datetime.now().isoformat()
    }

async def trigger_retraining():
    try:
        retraining_trigger.inc()
        logger.info(" Triggering automated retraining...")
        
        trigger_data = {
            "trigger_time": datetime.now().isoformat(),
            "reason": "performance_degradation",
            "current_performance": monitor.calculate_current_performance()
        }
        
        with open("/tmp/retrain_trigger", "w") as f:
            json.dump(trigger_data, f)
        
        logger.info(" Retraining trigger file created")
        
    except Exception as e:
        logger.error(f" Error triggering retraining: {e}")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/database/status")
async def database_status():
    """Check database connection and table status"""
    try:
        conn = monitor.get_connection()
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get table sizes
        table_sizes = {}
        for table in ['predictions', 'performance_metrics', 'retraining_triggers']:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_sizes[table] = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "connected",
            "tables": tables,
            "table_sizes": table_sizes,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
