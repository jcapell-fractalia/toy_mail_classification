"""
Tests for model functionality.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.base_model import BaseModel
from src.models.ensemble import EnsembleClassifier


class MockModel(BaseModel):
    """Mock model implementation for testing BaseModel."""

    def __init__(self):
        super().__init__("MockModel")
        self.mock_classes = np.array([0, 1])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.is_fitted = True
        return self

    def predict(self, X):
        # Return random predictions for testing
        return np.random.choice(self.mock_classes, size=X.shape[0])

    def predict_proba(self, X):
        # Return random probabilities for testing
        n_samples = X.shape[0]
        n_classes = len(self.mock_classes)
        proba = np.random.random((n_samples, n_classes))
        return proba / proba.sum(axis=1, keepdims=True)


class TestBaseModel:
    """Test cases for BaseModel abstract class."""

    def test_base_model_instantiation(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()

    def test_mock_model_instantiation(self):
        """Test MockModel instantiation."""
        model = MockModel()
        assert model.model_name == "MockModel"
        assert model.is_fitted is False
        assert model.classes_ is None

    def test_mock_model_fit_predict(self):
        """Test MockModel fit and predict methods."""
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

        model = MockModel()

        # Test fitting
        model.fit(X, y)
        assert model.is_fitted is True
        assert len(model.classes_) == 2

        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_mock_model_predict_proba(self):
        """Test MockModel predict_proba method."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

        model = MockModel()
        model.fit(X, y)

        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2)

        # Check that probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_get_set_params(self):
        """Test parameter getting and setting."""
        model = MockModel()

        # Test get_params
        params = model.get_params()
        assert "model_name" in params
        assert params["model_name"] == "MockModel"

        # Test set_params
        model.set_params(model_name="NewName")
        assert model.model_name == "NewName"

    def test_score_method(self):
        """Test the score method."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = MockModel()
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestEnsembleClassifier:
    """Test cases for EnsembleClassifier."""

    def test_ensemble_classifier_init_default(self):
        """Test EnsembleClassifier initialization with default parameters."""
        model = EnsembleClassifier()

        assert model.ensemble_type == "voting"
        assert model.voting == "soft"
        assert model.use_catboost is True
        assert model.use_lightgbm is True
        assert model.use_random_forest is True
        assert model.model_name == "EnsembleClassifier"

    def test_ensemble_classifier_init_custom(self):
        """Test EnsembleClassifier initialization with custom parameters."""
        model = EnsembleClassifier(
            ensemble_type="stacking", voting="hard", use_catboost=False, use_lightgbm=True, use_random_forest=True, model_name="CustomEnsemble"
        )

        assert model.ensemble_type == "stacking"
        assert model.voting == "hard"
        assert model.use_catboost is False
        assert model.use_lightgbm is True
        assert model.use_random_forest is True
        assert model.model_name == "CustomEnsemble"

    def test_create_base_models(self):
        """Test base model creation."""
        model = EnsembleClassifier()
        base_models = model._create_base_models()

        assert isinstance(base_models, list)
        assert len(base_models) > 0

        # Check that each base model is a tuple of (name, estimator)
        for name, estimator in base_models:
            assert isinstance(name, str)
            assert hasattr(estimator, "fit")
            assert hasattr(estimator, "predict")

    def test_ensemble_classifier_fit_predict_voting(self):
        """Test EnsembleClassifier fit and predict with voting ensemble."""
        # Create sample data
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Use only lightweight models for testing
        model = EnsembleClassifier(
            ensemble_type="voting",
            use_catboost=False,  # Skip CatBoost for faster testing
            use_lightgbm=False,  # Skip LightGBM for faster testing
            use_random_forest=True,
        )

        # Fit the model
        model.fit(X_train, y_train)
        assert model.is_fitted is True
        assert len(model.classes_) == 2

        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

        # Test predict_proba
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_ensemble_classifier_fit_predict_stacking(self):
        """Test EnsembleClassifier fit and predict with stacking ensemble."""
        # Create sample data
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Use only lightweight models for testing
        model = EnsembleClassifier(
            ensemble_type="stacking",
            use_catboost=False,  # Skip CatBoost for faster testing
            use_lightgbm=False,  # Skip LightGBM for faster testing
            use_random_forest=True,
        )

        # Fit the model
        model.fit(X_train, y_train)
        assert model.is_fitted is True

        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

        # Test predict_proba
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)

    def test_ensemble_classifier_invalid_ensemble_type(self):
        """Test EnsembleClassifier with invalid ensemble type."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

        model = EnsembleClassifier(ensemble_type="invalid_type")

        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_ensemble_classifier_predict_without_fit(self):
        """Test prediction without fitting the model."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

        model = EnsembleClassifier()

        with pytest.raises(ValueError):
            model.predict(X)

        with pytest.raises(ValueError):
            model.predict_proba(X)

    def test_evaluate_base_models(self):
        """Test base model evaluation functionality."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

        # Use only lightweight models for testing
        model = EnsembleClassifier(use_catboost=False, use_lightgbm=False, use_random_forest=True)

        results = model.evaluate_base_models(X, y, cv=3)

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that results contain expected keys
        for model_name, result in results.items():
            if "error" not in result:
                assert "mean_cv_score" in result
                assert "std_cv_score" in result
                assert "cv_scores" in result

    def test_get_set_params(self):
        """Test parameter getting and setting for EnsembleClassifier."""
        model = EnsembleClassifier()

        # Test get_params
        params = model.get_params()
        assert "ensemble_type" in params
        assert "voting" in params
        assert "use_catboost" in params
        assert "use_lightgbm" in params
        assert "use_random_forest" in params

        # Test set_params
        model.set_params(ensemble_type="stacking", use_catboost=False)
        assert model.ensemble_type == "stacking"
        assert model.use_catboost is False
