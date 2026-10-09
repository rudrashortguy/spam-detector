"""Evaluation module for spam detector."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    classifier,
    X_test,
    y_test,
) -> dict:
    """
    Evaluate classifier performance.

    Args:
        classifier: Trained classifier pipeline.
        X_test: Test texts.
        y_test: True labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = classifier.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


def print_evaluation_report(metrics: dict) -> None:
    """Print formatted evaluation report."""
    print("=" * 50)
    print("SPAM DETECTOR EVALUATION REPORT")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 50)
