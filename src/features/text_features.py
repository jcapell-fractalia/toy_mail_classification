"""
Text feature extraction utilities for email classification.
"""

import re
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Comprehensive text feature extractor for email classification.

    This class extracts various types of features from text data including
    statistical features, TF-IDF features, and semantic embeddings.
    """

    def __init__(
        self,
        include_statistical: bool = True,
        include_tfidf: bool = True,
        include_embeddings: bool = False,
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the text feature extractor.

        Args:
            include_statistical: Whether to include statistical text features
            include_tfidf: Whether to include TF-IDF features
            include_embeddings: Whether to include sentence embeddings
            tfidf_max_features: Maximum number of TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF features
            embedding_model: Name of the sentence transformer model
        """
        self.include_statistical = include_statistical
        self.include_tfidf = include_tfidf
        self.include_embeddings = include_embeddings
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.embedding_model = embedding_model

        # Initialize extractors
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        self.feature_names_ = []
        self.is_fitted = False

    def _extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract statistical features from texts.

        Args:
            texts: List of text documents

        Returns:
            Array of statistical features
        """
        features = []

        for text in texts:
            if not isinstance(text, str):
                text = ""

            # Basic length features
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r"[.!?]+", text))

            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0

            # Character type ratios
            if char_count > 0:
                uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count
                lowercase_ratio = sum(1 for c in text if c.islower()) / char_count
                digit_ratio = sum(1 for c in text if c.isdigit()) / char_count
                punct_ratio = sum(1 for c in text if c in string.punctuation) / char_count
                space_ratio = sum(1 for c in text if c.isspace()) / char_count
            else:
                uppercase_ratio = lowercase_ratio = digit_ratio = punct_ratio = space_ratio = 0

            # Special character counts
            email_count = len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text))
            url_count = len(re.findall(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", text))
            phone_count = len(re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text))

            # Exclamation and question marks
            exclamation_count = text.count("!")
            question_count = text.count("?")

            # Average sentence length
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

            # Feature vector for this text
            text_features = [
                char_count,
                word_count,
                sentence_count,
                avg_word_length,
                avg_sentence_length,
                uppercase_ratio,
                lowercase_ratio,
                digit_ratio,
                punct_ratio,
                space_ratio,
                email_count,
                url_count,
                phone_count,
                exclamation_count,
                question_count,
            ]

            features.append(text_features)

        return np.array(features)

    def _get_statistical_feature_names(self) -> List[str]:
        """Get names for statistical features."""
        return [
            "char_count",
            "word_count",
            "sentence_count",
            "avg_word_length",
            "avg_sentence_length",
            "uppercase_ratio",
            "lowercase_ratio",
            "digit_ratio",
            "punct_ratio",
            "space_ratio",
            "email_count",
            "url_count",
            "phone_count",
            "exclamation_count",
            "question_count",
        ]

    def fit(self, X: Union[List[str], pd.Series], y=None):
        """
        Fit the feature extractor to the data.

        Args:
            X: Text data to fit on
            y: Target values (ignored)

        Returns:
            self: The fitted extractor
        """
        # Convert to list if necessary
        if isinstance(X, pd.Series):
            X = X.tolist()

        # Initialize feature names
        self.feature_names_ = []

        # Fit TF-IDF vectorizer
        if self.include_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features, ngram_range=self.tfidf_ngram_range, stop_words="english", lowercase=True, strip_accents="unicode"
            )
            self.tfidf_vectorizer.fit(X)
            tfidf_feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
            self.feature_names_.extend(tfidf_feature_names)

        # Initialize sentence transformer
        if self.include_embeddings:
            self.sentence_transformer = SentenceTransformer(self.embedding_model)
            # Get embedding dimension
            sample_embedding = self.sentence_transformer.encode(["sample text"])
            embedding_dim = sample_embedding.shape[1]
            embedding_feature_names = [f"embedding_{i}" for i in range(embedding_dim)]
            self.feature_names_.extend(embedding_feature_names)

        # Add statistical feature names
        if self.include_statistical:
            statistical_feature_names = self._get_statistical_feature_names()
            self.feature_names_.extend(statistical_feature_names)

        self.is_fitted = True
        return self

    def transform(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform text data into feature vectors.

        Args:
            X: Text data to transform

        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transforming")

        # Convert to list if necessary
        if isinstance(X, pd.Series):
            X = X.tolist()

        feature_matrices = []

        # Extract TF-IDF features
        if self.include_tfidf and self.tfidf_vectorizer:
            tfidf_features = self.tfidf_vectorizer.transform(X).toarray()
            feature_matrices.append(tfidf_features)

        # Extract embedding features
        if self.include_embeddings and self.sentence_transformer:
            embedding_features = self.sentence_transformer.encode(X)
            feature_matrices.append(embedding_features)

        # Extract statistical features
        if self.include_statistical:
            statistical_features = self._extract_statistical_features(X)
            feature_matrices.append(statistical_features)

        # Combine all features
        if feature_matrices:
            combined_features = np.hstack(feature_matrices)
        else:
            combined_features = np.array([]).reshape(len(X), 0)

        return combined_features

    def fit_transform(self, X: Union[List[str], pd.Series], y=None) -> np.ndarray:
        """
        Fit the extractor and transform the data.

        Args:
            X: Text data
            y: Target values (ignored)

        Returns:
            Feature matrix
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.

        Args:
            input_features: Ignored

        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before getting feature names")
        return self.feature_names_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep: If True, return parameters for sub-estimators too

        Returns:
            Parameter names mapped to their values
        """
        return {
            "include_statistical": self.include_statistical,
            "include_tfidf": self.include_tfidf,
            "include_embeddings": self.include_embeddings,
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_ngram_range": self.tfidf_ngram_range,
            "embedding_model": self.embedding_model,
        }

    def set_params(self, **params) -> "TextFeatureExtractor":
        """
        Set the parameters of this estimator.

        Args:
            **params: Parameters to set

        Returns:
            self: The extractor instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
