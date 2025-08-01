"""
Email Classification Evaluation Script

This script evaluates trained models on test data with comprehensive metrics and visualizations.
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
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessing import EmailPreprocessor
from src.evaluation.metrics import (
    compare_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curves,
)
from src.features.text_features import TextFeatureExtractor
from src.models.base_model import BaseModel


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get("logging", {})

    # Create logs directory
    log_file = log_config.get("file", "logs/evaluation.log")
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


def load_test_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, str, str]:
    """Load and validate test data."""
    data_config = config["data"]

    # Load data
    data_path = data_config["test_path"]
    logging.info(f"Loading test data from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Validate columns
    text_column = data_config["text_column"]
    label_column = data_config["label_column"]

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in test data")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in test data")

    logging.info(f"Loaded {len(df)} test samples")
    logging.info(f"Test class distribution:\n{df[label_column].value_counts()}")

    return df, text_column, label_column


def load_model_components(config: Dict[str, Any]) -> Tuple[BaseModel, EmailPreprocessor, TextFeatureExtractor]:
    """Load trained model and preprocessing components."""
    model_config = config["model"]

    # Load model
    model_path = model_config["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    logging.info(f"Loaded model from {model_path}")

    # Load preprocessor
    preprocessor_path = model_config["preprocessor_path"]
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

    preprocessor = joblib.load(preprocessor_path)
    logging.info(f"Loaded preprocessor from {preprocessor_path}")

    # Load feature extractor
    extractor_path = model_config["feature_extractor_path"]
    if not os.path.exists(extractor_path):
        raise FileNotFoundError(f"Feature extractor file not found: {extractor_path}")

    feature_extractor = joblib.load(extractor_path)
    logging.info(f"Loaded feature extractor from {extractor_path}")

    return model, preprocessor, feature_extractor


def prepare_test_data(
    df: pd.DataFrame, text_column: str, label_column: str, preprocessor: EmailPreprocessor, feature_extractor: TextFeatureExtractor
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare test data for evaluation."""
    # Extract text and labels
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].values

    # Preprocess texts
    logging.info("Preprocessing test texts...")
    cleaned_texts = [preprocessor.clean_text(text) for text in texts]

    # Extract features
    logging.info("Extracting features from test data...")
    X_test = feature_extractor.transform(cleaned_texts)

    logging.info(f"Test feature matrix shape: {X_test.shape}")

    return X_test, labels


def comprehensive_evaluation(model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive model evaluation."""
    eval_config = config.get("evaluation", {})

    # Get class names
    class_names = eval_config.get("class_names")
    if class_names is None and hasattr(model, "classes_"):
        class_names = model.classes_.tolist()

    # Evaluate model
    logging.info("Performing comprehensive evaluation...")
    results = evaluate_model(model, X_test, y_test, class_names=class_names, verbose=True)

    return results


def create_visualizations(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Create and save evaluation visualizations."""
    eval_config = config.get("evaluation", {})
    plots_config = eval_config.get("plots", {})

    # Create plots directory
    plots_dir = os.path.dirname(plots_config.get("confusion_matrix", {}).get("save_path", "plots/"))
    os.makedirs(plots_dir, exist_ok=True)

    # Confusion Matrix
    cm_config = plots_config.get("confusion_matrix", {})
    if cm_config.get("enabled", True):
        fig = plot_confusion_matrix(
            results["confusion_matrix"],
            normalize=cm_config.get("normalize", False),
            title="Test Set Confusion Matrix",
            figsize=tuple(cm_config.get("figsize", [10, 8])),
            save_path=cm_config.get("save_path", "plots/confusion_matrix.png"),
        )
        plt.close(fig)
        logging.info("Confusion matrix plot created")

    # ROC Curve (for binary classification)
    roc_config = plots_config.get("roc_curve", {})
    if roc_config.get("enabled", True) and "fpr" in results:
        fig = plot_roc_curves(
            results, title="Test Set ROC Curve", figsize=tuple(roc_config.get("figsize", [10, 8])), save_path=roc_config.get("save_path", "plots/roc_curve.png")
        )
        plt.close(fig)
        logging.info("ROC curve plot created")

    # Precision-Recall Curve (for binary classification)
    pr_config = plots_config.get("precision_recall_curve", {})
    if pr_config.get("enabled", True) and "precision_curve" in results:
        fig = plot_precision_recall_curve(
            results,
            title="Test Set Precision-Recall Curve",
            figsize=tuple(pr_config.get("figsize", [10, 8])),
            save_path=pr_config.get("save_path", "plots/pr_curve.png"),
        )
        plt.close(fig)
        logging.info("Precision-Recall curve plot created")


def save_results(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save evaluation results to file."""
    output_config = config.get("output", {})

    # Create results directory
    results_dir = output_config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Prepare results for saving (convert numpy arrays to lists)
    save_results_dict = {
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1_score": float(results["f1_score"]),
        "classification_report": results["classification_report"],
    }

    # Add ROC AUC if available
    if "roc_auc" in results:
        save_results_dict["roc_auc"] = float(results["roc_auc"])
    if "pr_auc" in results:
        save_results_dict["pr_auc"] = float(results["pr_auc"])

    # Save predictions if requested
    if output_config.get("save_predictions", True):
        save_results_dict["predictions"] = results["predictions"].tolist()

    # Save probabilities if requested and available
    if output_config.get("save_probabilities", True) and "predicted_probabilities" in results:
        save_results_dict["predicted_probabilities"] = results["predicted_probabilities"].tolist()

    # Save in requested format
    report_format = output_config.get("report_format", "json")

    if report_format == "json":
        import json

        results_file = os.path.join(results_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(save_results_dict, f, indent=2)

    elif report_format == "yaml":
        results_file = os.path.join(results_dir, "evaluation_results.yaml")
        with open(results_file, "w") as f:
            yaml.dump(save_results_dict, f, default_flow_style=False)

    elif report_format == "txt":
        results_file = os.path.join(results_dir, "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write("Email Classification Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Accuracy: {save_results_dict['accuracy']:.4f}\n")
            f.write(f"Precision: {save_results_dict['precision']:.4f}\n")
            f.write(f"Recall: {save_results_dict['recall']:.4f}\n")
            f.write(f"F1-Score: {save_results_dict['f1_score']:.4f}\n")

            if "roc_auc" in save_results_dict:
                f.write(f"ROC AUC: {save_results_dict['roc_auc']:.4f}\n")
            if "pr_auc" in save_results_dict:
                f.write(f"PR AUC: {save_results_dict['pr_auc']:.4f}\n")

    logging.info(f"Results saved to {results_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate email classification model")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml", help="Path to evaluation configuration file")
    parser.add_argument("--model-path", type=str, help="Path to trained model (overrides config)")
    parser.add_argument("--data-path", type=str, help="Path to test data (overrides config)")
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override config with command line arguments
        if args.model_path:
            config["model"]["model_path"] = args.model_path
        if args.data_path:
            config["data"]["test_path"] = args.data_path

        # Setup logging
        setup_logging(config)
        logging.info("Starting email classification evaluation")

        # Load test data
        df, text_column, label_column = load_test_data(config)

        # Load model components
        model, preprocessor, feature_extractor = load_model_components(config)

        # Prepare test data
        X_test, y_test = prepare_test_data(df, text_column, label_column, preprocessor, feature_extractor)

        # Perform evaluation
        results = comprehensive_evaluation(model, X_test, y_test, config)

        # Create visualizations
        create_visualizations(results, config)

        # Save results
        save_results(results, config)

        logging.info("Evaluation completed successfully")

    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
