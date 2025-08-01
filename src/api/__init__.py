"""FastAPI application for email classification."""

from .main import app
from .schemas import ClassificationResponse, EmailRequest

__all__ = ["app", "EmailRequest", "ClassificationResponse"]
