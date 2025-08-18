"""API response schemas for loan default prediction."""
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime


class PredictionResponse(BaseModel):
    """Single prediction response schema."""
    prediction: str = Field(
        ...,
        description="Predicted class (good/bad)",
        json_schema_extra={"example": "good"}
    )
    probability: float = Field(
        ...,
        description="Prediction probability",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.75}
    )
    model_version: str = Field(
        ...,
        description="Model version used for prediction",
        json_schema_extra={"example": "v1.0.0"}
    )
    timestamp: float = Field(
        ...,
        description="Prediction timestamp",
        json_schema_extra={"example": 1692284567.123}
    )
    
    model_config = {"protected_namespaces": ()}


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each instance"
    )
    batch_id: str = Field(
        ...,
        description="Unique identifier for this batch",
        json_schema_extra={"example": "batch_12345"}
    )
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made",
        json_schema_extra={"example": 10}
    )
    
    model_config = {"protected_namespaces": ()}


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(
        ...,
        description="Service health status",
        json_schema_extra={"example": "healthy"}
    )
    timestamp: float = Field(
        ...,
        description="Health check timestamp",
        json_schema_extra={"example": 1692284567.123}
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded",
        json_schema_extra={"example": True}
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Currently loaded model version",
        json_schema_extra={"example": "v1.0.0"}
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds",
        json_schema_extra={"example": 3600.5}
    )
    
    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(
        ...,
        description="Error message",
        json_schema_extra={"example": "Invalid input data"}
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information",
        json_schema_extra={"example": "Missing required field: Attribute1"}
    )
    timestamp: float = Field(
        ...,
        description="Error timestamp",
        json_schema_extra={"example": 1692284567.123}
    )
    
    model_config = {"protected_namespaces": ()}