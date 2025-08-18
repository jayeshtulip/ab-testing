"""API request schemas for loan default prediction."""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class PredictionRequest(BaseModel):
    """Single prediction request schema."""
    features: Dict[str, Any] = Field(
        ...,
        description="Feature values for prediction",
        json_schema_extra={
            "example": {
                "Attribute1": "A11",
                "Attribute2": "24", 
                "Attribute3": "A32",
                "Attribute5": "3500"
            }
        }
    )
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "features": {
                    "Attribute1": "A11",
                    "Attribute2": "24",
                    "Attribute3": "A32", 
                    "Attribute5": "3500"
                }
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    instances: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries for batch prediction",
        min_length=1,
        max_length=100
    )
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "instances": [
                    {
                        "Attribute1": "A11",
                        "Attribute2": "24",
                        "Attribute3": "A32",
                        "Attribute5": "3500"
                    },
                    {
                        "Attribute1": "A12", 
                        "Attribute2": "36",
                        "Attribute3": "A43",
                        "Attribute5": "5000"
                    }
                ]
            }
        }
    }


class HealthCheckRequest(BaseModel):
    """Health check request schema."""
    check_model: Optional[bool] = Field(
        default=False,
        description="Whether to check model loading status"
    )
    
    model_config = {"protected_namespaces": ()}