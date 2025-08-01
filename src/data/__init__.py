"""Data processing utilities for email classification."""

from .preprocessing import EmailPreprocessor, clean_text, extract_features

__all__ = ["EmailPreprocessor", "clean_text", "extract_features"]
