"""Model training and prediction module."""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_classifier() -> Pipeline:
    """
    Build a TF-IDF + Naive Bayes classifier pipeline.

    Returns:
        Scikit-learn Pipeline.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('classifier', MultinomialNB(alpha=1.0))
    ])


def train_classifier(
    classifier: Pipeline,
    X_train,
    y_train,
) -> Pipeline:
    """
    Train the classifier on the provided data.

    Args:
        classifier: Pipeline to train.
        X_train: Training texts.
        y_train: Training labels.

    Returns:
        Trained classifier pipeline.
    """
    classifier.fit(X_train, y_train)
    return classifier


def predict(classifier: Pipeline, texts) -> np.ndarray:
    """
    Predict labels for given texts.

    Args:
        classifier: Trained classifier pipeline.
        texts: List of text strings.

    Returns:
        Array of predicted labels (0=ham, 1=spam).
    """
    return classifier.predict(texts)


def predict_proba(classifier: Pipeline, texts) -> np.ndarray:
    """
    Predict class probabilities.

    Args:
        classifier: Trained classifier pipeline.
        texts: List of text strings.

    Returns:
        Array of shape (n_samples, 2) with probabilities.
    """
    return classifier.predict_proba(texts)


def save_classifier(classifier: Pipeline, filepath: str) -> None:
    """Save classifier to disk."""
    joblib.dump(classifier, filepath)


def load_classifier(filepath: str) -> Pipeline:
    """Load classifier from disk."""
    return joblib.load(filepath)
