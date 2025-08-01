"""
Email Classification Training Script

This script trains an ensemble model for email classification using various ML algorithms.
It supports configuration-based training, hyperparameter optimization, and comprehensive evaluation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessing import EmailPreprocessor
from src.evaluation.metrics import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.features.text_features import TextFeatureExtractor
from src.models.ensemble import EnsembleClassifier


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get("logging", {})

    # Create logs directory
    log_file = log_config.get("file", "logs/training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, str, str]:
    """Load and validate data."""
    data_config = config["data"]

    # Load data
    data_path = data_config["train_path"]
    logging.info(f"Loading data from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Validate columns
    text_column = data_config["text_column"]
    label_column = data_config["label_column"]

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in data")

    logging.info(f"Loaded {len(df)} samples with {len(df[label_column].unique())} classes")
    logging.info(f"Class distribution:\n{df[label_column].value_counts()}")

    return df, text_column, label_column


def create_preprocessor(config: Dict[str, Any]) -> EmailPreprocessor:
    """Create and configure preprocessor."""
    preprocess_config = config.get("preprocessing", {})

    preprocessor = EmailPreprocessor(
        language=preprocess_config.get("language", "english"),
        use_stemming=preprocess_config.get("use_stemming", False),
        use_lemmatization=preprocess_config.get("use_lemmatization", True),
        remove_stopwords=preprocess_config.get("remove_stopwords", True),
        min_word_length=preprocess_config.get("min_word_length", 2),
    )

    logging.info("Created email preprocessor")
    return preprocessor


def create_feature_extractor(config: Dict[str, Any]) -> TextFeatureExtractor:
    """Create and configure feature extractor."""
    features_config = config.get("features", {})

    extractor = TextFeatureExtractor(
        include_statistical=features_config.get("include_statistical", True),
        include_tfidf=features_config.get("include_tfidf", True),
        include_embeddings=features_config.get("include_embeddings", False),
        tfidf_max_features=features_config.get("tfidf_max_features", 5000),
        tfidf_ngram_range=tuple(features_config.get("tfidf_ngram_range", [1, 2])),
        embedding_model=features_config.get("embedding_model", "all-MiniLM-L6-v2"),
    )

    logging.info("Created text feature extractor")
    return extractor


def create_model(config: Dict[str, Any]) -> EnsembleClassifier:
    """Create and configure ensemble model."""
    model_config = config.get("model", {})

    model = EnsembleClassifier(
        ensemble_type=model_config.get("ensemble_type", "voting"),
        voting=model_config.get("voting", "soft"),
        use_catboost=model_config.get("use_catboost", True),
        use_lightgbm=model_config.get("use_lightgbm", True),
        use_random_forest=model_config.get("use_random_forest", True),
        model_name="EmailEnsembleClassifier",
    )

    logging.info(f"Created ensemble model: {model_config.get('ensemble_type', 'voting')}")
    return model


def prepare_data(
    df: pd.DataFrame, text_column: str, label_column: str, preprocessor: EmailPreprocessor, feature_extractor: TextFeatureExtractor, config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for training."""
    data_config = config["data"]

    # Extract text and labels
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].values

    # Preprocess texts
    logging.info("Preprocessing texts...")
    cleaned_texts = [preprocessor.clean_text(text) for text in texts]

    # Extract features
    logging.info("Extracting features...")
    X = feature_extractor.fit_transform(cleaned_texts)

    logging.info(f"Feature matrix shape: {X.shape}")

    # Split data
    test_size = data_config.get("test_size", 0.2)
    random_state = data_config.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state, stratify=labels)

    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def train_model(model: EnsembleClassifier, X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any]) -> EnsembleClassifier:
    """Train the ensemble model."""
    training_config = config.get("training", {})

    # Cross-validation evaluation before training
    if training_config.get("cross_validation", True):
        cv_folds = training_config.get("cv_folds", 5)
        logging.info(f"Evaluating base models with {cv_folds}-fold cross-validation...")

        cv_results = model.evaluate_base_models(X_train, y_train, cv=cv_folds)

        for model_name, result in cv_results.items():
            if "error" not in result:
                logging.info(f"{model_name}: CV Score = {result['mean_cv_score']:.4f} ± {result['std_cv_score']:.4f}")
            else:
                logging.warning(f"{model_name}: Error during evaluation - {result['error']}")

    # Train the ensemble model
    logging.info("Training ensemble model...")
    model.fit(X_train, y_train)

    logging.info("Model training completed")
    return model


def evaluate_and_save_results(
    model: EnsembleClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: EmailPreprocessor,
    feature_extractor: TextFeatureExtractor,
    config: Dict[str, Any],
) -> None:
    """Evaluate model and save results."""
    output_config = config.get("output", {})
    eval_config = config.get("evaluation", {})

    # Create output directories
    model_dir = output_config.get("model_dir", "models")
    plots_dir = output_config.get("plots_dir", "plots")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Evaluate model
    logging.info("Evaluating model...")
    results = evaluate_model(model, X_test, y_test, verbose=True)

    # Plot confusion matrix
    if eval_config.get("plot_confusion_matrix", True):
        fig = plot_confusion_matrix(results["confusion_matrix"], title="Email Classification Confusion Matrix")
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("Confusion matrix plot saved")

    # Plot ROC curve for binary classification
    if eval_config.get("plot_roc_curve", True) and "roc_auc" in results:
        fig = plot_roc_curves(results, title="Email Classification ROC Curve")
        plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("ROC curve plot saved")

    # Plot feature importance
    if eval_config.get("plot_feature_importance", True):
        importance = model.get_feature_importance()
        if importance is not None:
            feature_names = feature_extractor.get_feature_names_out()

            # Get top features
            top_n = 20
            top_indices = np.argsort(importance)[-top_n:]

            plt.figure(figsize=(12, 8))
            plt.barh(range(top_n), importance[top_indices])
            plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
            plt.xlabel("Feature Importance")
            plt.title("Top Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logging.info("Feature importance plot saved")

    # Save model and components
    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

    if output_config.get("save_preprocessor", True):
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Preprocessor saved to {preprocessor_path}")

    if output_config.get("save_feature_extractor", True):
        extractor_path = os.path.join(model_dir, "feature_extractor.pkl")
        joblib.dump(feature_extractor, extractor_path)
        logging.info(f"Feature extractor saved to {extractor_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train email classification model")
    parser.add_argument("--config", type=str, default="config/train_config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Setup logging
        setup_logging(config)
        logging.info("Starting email classification training")

        # Load data
        df, text_column, label_column = load_data(config)

        # Create components
        preprocessor = create_preprocessor(config)
        feature_extractor = create_feature_extractor(config)
        model = create_model(config)

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df, text_column, label_column, preprocessor, feature_extractor, config)

        # Train model
        trained_model = train_model(model, X_train, y_train, config)

        # Evaluate and save results
        evaluate_and_save_results(trained_model, X_test, y_test, preprocessor, feature_extractor, config)

        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
