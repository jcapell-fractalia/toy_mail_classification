"""
Email Classification Package

A comprehensive machine learning package for email classification using
advanced NLP techniques and ensemble learning methods.
"""

__version__ = "0.1.0"
__author__ = "Joan Capell"
__email__ = "joan.capell@example.com"

from .data import preprocessing
from .evaluation import metrics
from .features import text_features
from .models import BaseModel

__all__ = [
    "BaseModel",
    "preprocessing",
    "text_features",
    "metrics",
]
