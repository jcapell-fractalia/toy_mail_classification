"""
Pydantic schemas for the email classification API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmailRequest(BaseModel):
    """Request schema for email classification."""

    text: str = Field(..., description="The email text to classify", min_length=1, max_length=10000)

    class Config:
        schema_extra = {"example": {"text": "Dear customer, congratulations! You have won $1000000. Click here to claim your prize now!"}}


class ClassificationResponse(BaseModel):
    """Response schema for email classification."""

    predicted_class: str = Field(..., description="The predicted class label")

    confidence: float = Field(..., description="Confidence score for the prediction", ge=0.0, le=1.0)

    probabilities: Dict[str, float] = Field(..., description="Probability scores for all classes")

    class Config:
        schema_extra = {"example": {"predicted_class": "spam", "confidence": 0.95, "probabilities": {"spam": 0.95, "ham": 0.05}}}


class BatchEmailRequest(BaseModel):
    """Request schema for batch email classification."""

    emails: List[str] = Field(..., description="List of email texts to classify", min_items=1, max_items=100)

    class Config:
        schema_extra = {
            "example": {
                "emails": [
                    "Dear customer, congratulations! You have won $1000000.",
                    "Meeting scheduled for tomorrow at 2 PM in conference room A.",
                    "Your package has been delivered to your address.",
                ]
            }
        }


class BatchClassificationResponse(BaseModel):
    """Response schema for batch email classification."""

    results: List[ClassificationResponse] = Field(..., description="List of classification results")

    summary: Dict[str, Any] = Field(..., description="Summary statistics of the batch prediction")


class ModelInfo(BaseModel):
    """Schema for model information."""

    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    classes: List[str] = Field(..., description="List of possible classes")
    features: int = Field(..., description="Number of input features")
    accuracy: Optional[float] = Field(None, description="Model accuracy on test set")

    class Config:
        schema_extra = {"example": {"model_name": "EnsembleClassifier", "version": "1.0.0", "classes": ["spam", "ham"], "features": 5000, "accuracy": 0.95}}


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")

    class Config:
        schema_extra = {"example": {"status": "healthy", "timestamp": "2025-01-01T12:00:00Z", "model_loaded": True}}
