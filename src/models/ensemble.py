"""
Ensemble model implementations for email classification.
"""

from typing import Any, Dict, List, Optional, Union

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from .base_model import BaseModel


class EnsembleClassifier(BaseModel):
    """
    Ensemble classifier combining multiple models for email classification.

    This class implements ensemble methods including voting and stacking
    to combine predictions from multiple base models.
    """

    def __init__(
        self,
        ensemble_type: str = "voting",
        voting: str = "soft",
        use_catboost: bool = True,
        use_lightgbm: bool = True,
        use_random_forest: bool = True,
        model_name: str = "EnsembleClassifier",
    ):
        """
        Initialize the ensemble classifier.

        Args:
            ensemble_type: Type of ensemble ('voting' or 'stacking')
            voting: Voting method for VotingClassifier ('hard' or 'soft')
            use_catboost: Whether to include CatBoost in the ensemble
            use_lightgbm: Whether to include LightGBM in the ensemble
            use_random_forest: Whether to include Random Forest in the ensemble
            model_name: Name identifier for the model
        """
        super().__init__(model_name)
        self.ensemble_type = ensemble_type
        self.voting = voting
        self.use_catboost = use_catboost
        self.use_lightgbm = use_lightgbm
        self.use_random_forest = use_random_forest

        # Initialize base models
        self.base_models = self._create_base_models()
        self.ensemble_model = None

    def _create_base_models(self) -> List[tuple]:
        """
        Create base models for the ensemble.

        Returns:
            List of (name, model) tuples
        """
        models = []

        if self.use_catboost:
            catboost_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=False, random_seed=42)
            models.append(("catboost", catboost_model))

        if self.use_lightgbm:
            lightgbm_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)
            models.append(("lightgbm", lightgbm_model))

        if self.use_random_forest:
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            models.append(("random_forest", rf_model))

        return models

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "EnsembleClassifier":
        """
        Fit the ensemble model to training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            self: The fitted model instance
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Store classes and feature information
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        # Create ensemble model
        if self.ensemble_type == "voting":
            self.ensemble_model = VotingClassifier(estimators=self.base_models, voting=self.voting, n_jobs=-1)
        elif self.ensemble_type == "stacking":
            # Use logistic regression as meta-learner
            meta_learner = LogisticRegression(random_state=42)
            self.ensemble_model = StackingClassifier(estimators=self.base_models, final_estimator=meta_learner, cv=5, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported ensemble_type: {self.ensemble_type}")

        # Fit the ensemble model
        self.ensemble_model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.ensemble_model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.ensemble_model.predict_proba(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get average feature importance from base models that support it.

        Returns:
            Feature importance array or None if not available
        """
        if not self.is_fitted:
            return None

        importances = []

        # Get feature importances from base models
        if self.ensemble_type == "voting":
            estimators = self.ensemble_model.estimators_
        else:  # stacking
            estimators = [est for name, est in self.ensemble_model.estimators_]

        for estimator in estimators:
            if hasattr(estimator, "feature_importances_"):
                importances.append(estimator.feature_importances_)

        if importances:
            # Return average importance
            return np.mean(importances, axis=0)
        else:
            return None

    def evaluate_base_models(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], cv: int = 5) -> Dict[str, float]:
        """
        Evaluate individual base models using cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds

        Returns:
            Dictionary of model names and their CV scores
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        results = {}

        for name, model in self.base_models:
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                results[name] = {"mean_cv_score": np.mean(cv_scores), "std_cv_score": np.std(cv_scores), "cv_scores": cv_scores.tolist()}
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = {"error": str(e)}

        return results

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep: If True, return parameters for sub-estimators too

        Returns:
            Parameter names mapped to their values
        """
        params = super().get_params(deep)
        params.update(
            {
                "ensemble_type": self.ensemble_type,
                "voting": self.voting,
                "use_catboost": self.use_catboost,
                "use_lightgbm": self.use_lightgbm,
                "use_random_forest": self.use_random_forest,
            }
        )
        return params

    def set_params(self, **params) -> "EnsembleClassifier":
        """
        Set the parameters of this estimator.

        Args:
            **params: Parameters to set

        Returns:
            self: The model instance
        """
        # Handle ensemble-specific parameters
        if "ensemble_type" in params:
            self.ensemble_type = params.pop("ensemble_type")
        if "voting" in params:
            self.voting = params.pop("voting")
        if "use_catboost" in params:
            self.use_catboost = params.pop("use_catboost")
        if "use_lightgbm" in params:
            self.use_lightgbm = params.pop("use_lightgbm")
        if "use_random_forest" in params:
            self.use_random_forest = params.pop("use_random_forest")

        # Recreate base models if model selection changed
        self.base_models = self._create_base_models()

        # Call parent set_params for remaining parameters
        return super().set_params(**params)
