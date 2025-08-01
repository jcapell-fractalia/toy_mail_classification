"""
Email preprocessing utilities for text cleaning and feature extraction.
"""

import re
import string
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class EmailPreprocessor:
    """
    Comprehensive email preprocessing class for text cleaning and feature extraction.

    This class provides methods for cleaning email text, extracting features,
    and preparing data for machine learning models.
    """

    def __init__(
        self, language: str = "english", use_stemming: bool = False, use_lemmatization: bool = True, remove_stopwords: bool = True, min_word_length: int = 2
    ):
        """
        Initialize the EmailPreprocessor.

        Args:
            language: Language for stopwords removal
            use_stemming: Whether to apply stemming
            use_lemmatization: Whether to apply lemmatization
            remove_stopwords: Whether to remove stopwords
            min_word_length: Minimum word length to keep
        """
        self.language = language
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length

        # Download required NLTK data
        self._download_nltk_data()

        # Initialize NLTK tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.language))
        if self.use_stemming:
            self.stemmer = PorterStemmer()
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()

    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        required_data = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
        for data in required_data:
            try:
                nltk.data.find(f"tokenizers/{data}")
            except LookupError:
                nltk.download(data, quiet=True)

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess email text.

        Args:
            text: Raw email text

        Returns:
            Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove email addresses
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # Remove URLs
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove extra whitespace and special characters
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_word_length:
                continue

            # Remove stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Apply stemming or lemmatization
            if self.use_stemming:
                token = self.stemmer.stem(token)
            elif self.use_lemmatization:
                token = self.lemmatizer.lemmatize(token)

            filtered_tokens.append(token)

        return " ".join(filtered_tokens)

    def extract_basic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract basic statistical features from email text.

        Args:
            text: Email text

        Returns:
            Dictionary of extracted features
        """
        if not isinstance(text, str):
            text = ""

        features = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r"[.!?]+", text)),
            "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "punctuation_count": sum(1 for c in text if c in string.punctuation),
            "digit_count": sum(1 for c in text if c.isdigit()),
            "email_count": len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)),
            "url_count": len(re.findall(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", text)),
        }

        return features

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocess a DataFrame containing email data.

        Args:
            df: DataFrame with email data
            text_column: Name of the column containing email text

        Returns:
            DataFrame with preprocessed text and extracted features
        """
        df = df.copy()

        # Clean text
        df[f"{text_column}_clean"] = df[text_column].apply(self.clean_text)

        # Extract basic features
        basic_features = df[text_column].apply(self.extract_basic_features)
        feature_df = pd.DataFrame(basic_features.tolist())

        # Combine with original data
        result_df = pd.concat([df, feature_df], axis=1)

        return result_df


def clean_text(text: str, remove_stopwords: bool = True, use_lemmatization: bool = True) -> str:
    """
    Simple text cleaning function.

    Args:
        text: Input text to clean
        remove_stopwords: Whether to remove stopwords
        use_lemmatization: Whether to apply lemmatization

    Returns:
        Cleaned text
    """
    preprocessor = EmailPreprocessor(remove_stopwords=remove_stopwords, use_lemmatization=use_lemmatization)
    return preprocessor.clean_text(text)


def extract_features(texts: List[str], feature_type: str = "tfidf", max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
    """
    Extract features from a list of texts.

    Args:
        texts: List of text documents
        feature_type: Type of features ('tfidf', 'count', 'embeddings')
        max_features: Maximum number of features
        ngram_range: N-gram range for text features

    Returns:
        Feature matrix
    """
    if feature_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
        return vectorizer.fit_transform(texts).toarray()

    elif feature_type == "count":
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
        return vectorizer.fit_transform(texts).toarray()

    elif feature_type == "embeddings":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts)

    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
