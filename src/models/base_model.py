"""
Base model interface for email classification models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseModel(ABC, BaseEstimator, ClassifierMixin):
    """
    Abstract base class for email classification models.
    
    This class defines the interface that all email classification models
    should implement, ensuring consistency across different model types.
    """
    
    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize the base model.
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.is_fitted = False
        self.classes_ = None
        self.feature_names_ = None
        
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self: The fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        return joblib.load(filepath)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, return parameters for sub-estimators too
            
        Returns:
            Parameter names mapped to their values
        """
        return {"model_name": self.model_name}
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The model instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Mean accuracy score
        """
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None if not available
        """
        return None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', fitted={self.is_fitted})"
