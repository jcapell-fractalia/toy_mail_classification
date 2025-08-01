"""
Tests for data preprocessing functionality.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import EmailPreprocessor, clean_text, extract_features


class TestEmailPreprocessor:
    """Test cases for EmailPreprocessor class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        preprocessor = EmailPreprocessor()
        assert preprocessor.language == "english"
        assert preprocessor.use_stemming is False
        assert preprocessor.use_lemmatization is True
        assert preprocessor.remove_stopwords is True
        assert preprocessor.min_word_length == 2

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        preprocessor = EmailPreprocessor(language="spanish", use_stemming=True, use_lemmatization=False, remove_stopwords=False, min_word_length=3)
        assert preprocessor.language == "spanish"
        assert preprocessor.use_stemming is True
        assert preprocessor.use_lemmatization is False
        assert preprocessor.remove_stopwords is False
        assert preprocessor.min_word_length == 3

    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        preprocessor = EmailPreprocessor()

        # Test with normal text
        text = "Hello World! This is a test email."
        cleaned = preprocessor.clean_text(text)
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0

    def test_clean_text_with_email_addresses(self):
        """Test cleaning text with email addresses."""
        preprocessor = EmailPreprocessor()

        text = "Contact us at support@example.com for help."
        cleaned = preprocessor.clean_text(text)

        # Email should be removed
        assert "@example.com" not in cleaned

    def test_clean_text_with_urls(self):
        """Test cleaning text with URLs."""
        preprocessor = EmailPreprocessor()

        text = "Visit our website at https://www.example.com for more info."
        cleaned = preprocessor.clean_text(text)

        # URL should be removed
        assert "https://www.example.com" not in cleaned

    def test_clean_text_empty_input(self):
        """Test cleaning empty or None input."""
        preprocessor = EmailPreprocessor()

        # Test empty string
        assert preprocessor.clean_text("") == ""

        # Test None
        assert preprocessor.clean_text(None) == ""

        # Test non-string input
        assert preprocessor.clean_text(123) == ""

    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        preprocessor = EmailPreprocessor()

        text = "Hello! This is a test email with 123 numbers."
        features = preprocessor.extract_basic_features(text)

        # Check that all expected features are present
        expected_features = [
            "char_count",
            "word_count",
            "sentence_count",
            "avg_word_length",
            "uppercase_ratio",
            "punctuation_count",
            "digit_count",
            "email_count",
            "url_count",
        ]

        for feature in expected_features:
            assert feature in features

        # Check some basic feature values
        assert features["char_count"] > 0
        assert features["word_count"] > 0
        assert features["digit_count"] == 3  # "123" has 3 digits

    def test_preprocess_dataframe(self):
        """Test DataFrame preprocessing."""
        preprocessor = EmailPreprocessor()

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "email_text": ["Hello, this is a test email!", "Another email with different content.", "Third email for testing purposes."],
                "label": ["ham", "spam", "ham"],
            }
        )

        result_df = preprocessor.preprocess_dataframe(df, "email_text")

        # Check that new columns are added
        assert "email_text_clean" in result_df.columns
        assert "char_count" in result_df.columns
        assert "word_count" in result_df.columns

        # Check that original data is preserved
        assert len(result_df) == len(df)
        assert "label" in result_df.columns


def test_clean_text_function():
    """Test the standalone clean_text function."""
    text = "Hello World! This is a test."
    cleaned = clean_text(text)

    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_extract_features_tfidf():
    """Test TF-IDF feature extraction."""
    texts = ["This is the first document.", "This document is the second document.", "And this is the third one."]

    features = extract_features(texts, feature_type="tfidf", max_features=10)

    # Check output shape
    assert features.shape[0] == len(texts)
    assert features.shape[1] <= 10


def test_extract_features_count():
    """Test count vectorizer feature extraction."""
    texts = ["This is the first document.", "This document is the second document.", "And this is the third one."]

    features = extract_features(texts, feature_type="count", max_features=10)

    # Check output shape
    assert features.shape[0] == len(texts)
    assert features.shape[1] <= 10


def test_extract_features_invalid_type():
    """Test extract_features with invalid feature type."""
    texts = ["Test text"]

    with pytest.raises(ValueError):
        extract_features(texts, feature_type="invalid_type")
