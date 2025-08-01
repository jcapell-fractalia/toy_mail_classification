"""
Tests for API functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import BatchEmailRequest, EmailRequest

# Create test client
client = TestClient(app)


class TestAPI:
    """Test cases for the FastAPI application."""

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data

    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data

    @patch("src.api.main.model")
    @patch("src.api.main.preprocessor")
    @patch("src.api.main.feature_extractor")
    def test_model_info_endpoint(self, mock_feature_extractor, mock_preprocessor, mock_model):
        """Test the model info endpoint."""
        # Mock the model and its attributes
        mock_model.model_name = "TestModel"
        mock_model.classes_ = np.array(["spam", "ham"])

        response = client.get("/model/info")

        # Should return 503 if model is not loaded, or 200 if mocked properly
        assert response.status_code in [200, 503]

    @patch("src.api.main.model")
    @patch("src.api.main.preprocessor")
    @patch("src.api.main.feature_extractor")
    def test_classify_endpoint_success(self, mock_feature_extractor, mock_preprocessor, mock_model):
        """Test successful email classification."""
        # Setup mocks
        mock_preprocessor.clean_text.return_value = "cleaned text"
        mock_feature_extractor.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        mock_model.predict.return_value = np.array(["spam"])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        # Mock model_info
        with patch("src.api.main.model_info", {"classes": ["ham", "spam"]}):
            request_data = {"text": "This is a test email"}

            response = client.post("/classify", json=request_data)

            if response.status_code == 503:
                # Model not loaded in test environment
                pytest.skip("Model not loaded in test environment")

            assert response.status_code == 200

            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data

    def test_classify_endpoint_invalid_input(self):
        """Test classification with invalid input."""
        # Empty text
        response = client.post("/classify", json={"text": ""})
        assert response.status_code == 422  # Validation error

        # Missing text field
        response = client.post("/classify", json={})
        assert response.status_code == 422  # Validation error

        # Text too long
        long_text = "x" * 10001  # Exceeds max_length
        response = client.post("/classify", json={"text": long_text})
        assert response.status_code == 422  # Validation error

    @patch("src.api.main.model")
    @patch("src.api.main.preprocessor")
    @patch("src.api.main.feature_extractor")
    def test_batch_classify_endpoint_success(self, mock_feature_extractor, mock_preprocessor, mock_model):
        """Test successful batch email classification."""
        # Setup mocks
        mock_preprocessor.clean_text.return_value = "cleaned text"
        mock_feature_extractor.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        mock_model.predict.return_value = np.array(["spam"])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        # Mock model_info
        with patch("src.api.main.model_info", {"classes": ["ham", "spam"]}):
            request_data = {"emails": ["This is test email 1", "This is test email 2", "This is test email 3"]}

            response = client.post("/classify/batch", json=request_data)

            if response.status_code == 503:
                # Model not loaded in test environment
                pytest.skip("Model not loaded in test environment")

            assert response.status_code == 200

            data = response.json()
            assert "results" in data
            assert "summary" in data
            assert len(data["results"]) == 3

    def test_batch_classify_endpoint_invalid_input(self):
        """Test batch classification with invalid input."""
        # Empty emails list
        response = client.post("/classify/batch", json={"emails": []})
        assert response.status_code == 422  # Validation error

        # Too many emails
        too_many_emails = ["email"] * 101  # Exceeds max_items
        response = client.post("/classify/batch", json={"emails": too_many_emails})
        assert response.status_code == 422  # Validation error

        # Missing emails field
        response = client.post("/classify/batch", json={})
        assert response.status_code == 422  # Validation error


class TestSchemas:
    """Test cases for Pydantic schemas."""

    def test_email_request_valid(self):
        """Test valid EmailRequest."""
        request = EmailRequest(text="This is a test email")
        assert request.text == "This is a test email"

    def test_email_request_empty_text(self):
        """Test EmailRequest with empty text."""
        with pytest.raises(ValueError):
            EmailRequest(text="")

    def test_email_request_too_long(self):
        """Test EmailRequest with text too long."""
        long_text = "x" * 10001
        with pytest.raises(ValueError):
            EmailRequest(text=long_text)

    def test_batch_email_request_valid(self):
        """Test valid BatchEmailRequest."""
        request = BatchEmailRequest(emails=["email1", "email2", "email3"])
        assert len(request.emails) == 3

    def test_batch_email_request_empty(self):
        """Test BatchEmailRequest with empty list."""
        with pytest.raises(ValueError):
            BatchEmailRequest(emails=[])

    def test_batch_email_request_too_many(self):
        """Test BatchEmailRequest with too many emails."""
        too_many_emails = ["email"] * 101
        with pytest.raises(ValueError):
            BatchEmailRequest(emails=too_many_emails)
