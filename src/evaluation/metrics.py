"""
Model evaluation metrics and visualization utilities.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model(
    model: BaseEstimator,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with multiple metrics.

    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test labels
        class_names: Names of the classes
        verbose: Whether to print results

    Returns:
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Try to get prediction probabilities
    try:
        y_pred_proba = model.predict_proba(X_test)
        has_proba = True
    except AttributeError:
        y_pred_proba = None
        has_proba = False
        warnings.warn("Model does not support predict_proba, some metrics will be unavailable")

    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Handle multiclass metrics
    unique_classes = np.unique(y_test)
    is_binary = len(unique_classes) == 2

    if is_binary:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    else:
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Prepare results dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": class_report,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "true_labels": y_test,
    }

    # Add probability-based metrics if available
    if has_proba and y_pred_proba is not None:
        results["predicted_probabilities"] = y_pred_proba

        if is_binary:
            # ROC AUC for binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            results["roc_auc"] = roc_auc
            results["fpr"] = fpr
            results["tpr"] = tpr

            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
            pr_auc = average_precision_score(y_test, y_pred_proba[:, 1])
            results["pr_auc"] = pr_auc
            results["precision_curve"] = precision_curve
            results["recall_curve"] = recall_curve

    # Print results if verbose
    if verbose:
        print("=== Model Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        if has_proba and is_binary and "roc_auc" in results:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
            print(f"PR AUC: {results['pr_auc']:.4f}")

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=class_names))

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix with customization options.

    Args:
        cm: Confusion matrix array
        class_names: Names of the classes
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax,
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(results: Dict[str, Any], title: str = "ROC Curves", figsize: tuple = (10, 8), save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for binary classification.

    Args:
        results: Results dictionary from evaluate_model
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    if "fpr" not in results or "tpr" not in results:
        raise ValueError("ROC curve data not available in results")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(results["fpr"], results["tpr"], color="darkorange", lw=2, label=f'ROC curve (AUC = {results["roc_auc"]:.4f})')

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    results: Dict[str, Any], title: str = "Precision-Recall Curve", figsize: tuple = (10, 8), save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve for binary classification.

    Args:
        results: Results dictionary from evaluate_model
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    if "precision_curve" not in results or "recall_curve" not in results:
        raise ValueError("Precision-Recall curve data not available in results")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot PR curve
    ax.plot(results["recall_curve"], results["precision_curve"], color="darkgreen", lw=2, label=f'PR curve (AUC = {results["pr_auc"]:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compare_models(
    models_results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ["accuracy", "precision", "recall", "f1_score"],
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare multiple models using bar plots.

    Args:
        models_results: Dictionary of model names and their evaluation results
        metrics: List of metrics to compare
        figsize: Figure size
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    # Prepare data for plotting
    model_names = list(models_results.keys())
    metric_values = {metric: [] for metric in metrics}

    for model_name in model_names:
        results = models_results[model_name]
        for metric in metrics:
            if metric in results:
                metric_values[metric].append(results[metric])
            else:
                metric_values[metric].append(0)

    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_names, metric_values[metric], alpha=0.7)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bar, value in zip(bars, metric_values[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

        # Rotate x-axis labels if too long
        if max(len(name) for name in model_names) > 10:
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
