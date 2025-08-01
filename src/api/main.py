"""
FastAPI application for email classification service.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..data.preprocessing import EmailPreprocessor
from ..features.text_features import TextFeatureExtractor
from .schemas import (
    BatchClassificationResponse,
    BatchEmailRequest,
    ClassificationResponse,
    EmailRequest,
    HealthResponse,
    ModelInfo,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
model = None
preprocessor = None
feature_extractor = None
model_info = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting up email classification service...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down email classification service...")


# Create FastAPI app
app = FastAPI(title="Email Classification API", description="A REST API for classifying emails using machine learning", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    """Load the trained model and preprocessing components."""
    global model, preprocessor, feature_extractor, model_info

    try:
        # Define model paths
        model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
        preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
        feature_extractor_path = os.getenv("FEATURE_EXTRACTOR_PATH", "models/feature_extractor.pkl")

        # Load model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # Create a dummy model for demonstration
            from ..models.ensemble import EnsembleClassifier

            model = EnsembleClassifier()
            logger.info("Created dummy model for demonstration")

        # Load preprocessor
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            preprocessor = EmailPreprocessor()
            logger.info("Created default preprocessor")

        # Load feature extractor
        if os.path.exists(feature_extractor_path):
            feature_extractor = joblib.load(feature_extractor_path)
            logger.info(f"Feature extractor loaded from {feature_extractor_path}")
        else:
            feature_extractor = TextFeatureExtractor(include_statistical=True, include_tfidf=True, include_embeddings=False)
            logger.info("Created default feature extractor")

        # Set model info
        model_info = {
            "model_name": getattr(model, "model_name", "Unknown"),
            "version": "1.0.0",
            "classes": getattr(model, "classes_", ["spam", "ham"]).tolist() if hasattr(model, "classes_") else ["spam", "ham"],
            "features": getattr(feature_extractor, "n_features_", 0),
            "accuracy": None,  # Would be loaded from model metadata
        }

    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise


def get_model():
    """Dependency to get the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def preprocess_text(text: str) -> np.ndarray:
    """Preprocess text and extract features."""
    # Clean text
    cleaned_text = preprocessor.clean_text(text)

    # Extract features
    features = feature_extractor.transform([cleaned_text])

    return features


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {"message": "Email Classification API", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.utcnow().isoformat() + "Z", model_loaded=model is not None)


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(current_model=Depends(get_model)):
    """Get information about the loaded model."""
    return ModelInfo(**model_info)


@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest, current_model=Depends(get_model)):
    """
    Classify a single email.

    Args:
        request: Email classification request
        current_model: The loaded model (dependency)

    Returns:
        Classification result with predicted class and probabilities
    """
    try:
        # Preprocess the text
        features = preprocess_text(request.text)

        # Make prediction
        prediction = current_model.predict(features)[0]

        # Get probabilities if available
        try:
            probabilities = current_model.predict_proba(features)[0]
            prob_dict = {class_name: float(prob) for class_name, prob in zip(model_info["classes"], probabilities)}
            confidence = float(max(probabilities))
        except AttributeError:
            # Model doesn't support predict_proba
            prob_dict = {prediction: 1.0}
            confidence = 1.0

        return ClassificationResponse(predicted_class=str(prediction), confidence=confidence, probabilities=prob_dict)

    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_emails_batch(request: BatchEmailRequest, current_model=Depends(get_model)):
    """
    Classify multiple emails in a batch.

    Args:
        request: Batch email classification request
        current_model: The loaded model (dependency)

    Returns:
        Batch classification results with summary statistics
    """
    try:
        results = []
        class_counts = {}

        for email_text in request.emails:
            # Preprocess the text
            features = preprocess_text(email_text)

            # Make prediction
            prediction = current_model.predict(features)[0]

            # Get probabilities if available
            try:
                probabilities = current_model.predict_proba(features)[0]
                prob_dict = {class_name: float(prob) for class_name, prob in zip(model_info["classes"], probabilities)}
                confidence = float(max(probabilities))
            except AttributeError:
                # Model doesn't support predict_proba
                prob_dict = {prediction: 1.0}
                confidence = 1.0

            # Add to results
            result = ClassificationResponse(predicted_class=str(prediction), confidence=confidence, probabilities=prob_dict)
            results.append(result)

            # Count classes
            class_counts[str(prediction)] = class_counts.get(str(prediction), 0) + 1

        # Calculate summary
        total_emails = len(request.emails)
        avg_confidence = np.mean([r.confidence for r in results])

        summary = {"total_emails": total_emails, "class_distribution": class_counts, "average_confidence": float(avg_confidence)}

        return BatchClassificationResponse(results=results, summary=summary)

    except Exception as e:
        logger.error(f"Error during batch classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
