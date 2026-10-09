"""Tests for spam detector."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from spam_detector.data import load_sms_data, preprocess_text
from spam_detector.model import build_classifier, train_classifier, predict
from spam_detector.evaluator import evaluate_model, print_evaluation_report


class TestPreprocessing:
    def test_preprocess_text_lowercase(self):
        """Test that text is converted to lowercase."""
        text = "Hello WORLD"
        result = preprocess_text(text)
        assert result == "hello world"

    def test_preprocess_text_removes_special_chars(self):
        """Test that special characters are removed."""
        text = "Hello, World! @#$%"
        result = preprocess_text(text)
        assert result == "hello world"

    def test_preprocess_text_removes_extra_whitespace(self):
        """Test that extra whitespace is collapsed."""
        text = "Hello   World"
        result = preprocess_text(text)
        assert result == "hello world"


class TestDataLoading:
    def test_load_sms_data_returns_tuple(self):
        """Test that load_sms_data returns a tuple of 4 elements."""
        X_train, X_test, y_train, y_test = load_sms_data()
        assert isinstance(X_train, pd.Series)
        assert isinstance(X_test, pd.Series)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_load_sms_data_missing_column_raises(self):
        """Test that missing column raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("text,category\nhello,spam\nworld,ham\n")
            path = f.name
        with pytest.raises(ValueError, match="label"):
            load_sms_data(filepath=path)
        os.unlink(path)


class TestModel:
    def test_build_classifier(self):
        """Test classifier building."""
        classifier = build_classifier()
        assert classifier is not None
        assert hasattr(classifier, "fit")
        assert hasattr(classifier, "predict")

    def test_train_classifier(self):
        """Test classifier training."""
        X_train = pd.Series(["free win", "hello friend", "win free phone"])
        y_train = pd.Series([1, 0, 1])
        classifier = build_classifier()
        trained = train_classifier(classifier, X_train, y_train)
        assert trained is not None

    def test_predict(self):
        """Test prediction."""
        X_train = pd.Series(["free win", "hello friend"])
        y_train = pd.Series([1, 0])
        classifier = train_classifier(build_classifier(), X_train, y_train)
        predictions = predict(classifier, ["win free"])
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]


class TestEvaluator:
    def test_evaluate_model_returns_metrics(self):
        """Test that evaluate_model returns a dictionary with expected keys."""
        X_train = pd.Series(["free win", "hello friend"] * 50)
        y_train = pd.Series([1, 0] * 50)
        X_test = pd.Series(["win free", "hello"] * 10)
        y_test = pd.Series([1, 0] * 10)

        classifier = train_classifier(build_classifier(), X_train, y_train)
        metrics = evaluate_model(classifier, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics

    def test_evaluate_accuracy_range(self):
        """Test that accuracy is between 0 and 1."""
        X_train = pd.Series(["free win", "hello friend"] * 50)
        y_train = pd.Series([1, 0] * 50)
        X_test = pd.Series(["win free", "hello"] * 10)
        y_test = pd.Series([1, 0] * 10)

        classifier = train_classifier(build_classifier(), X_train, y_train)
        metrics = evaluate_model(classifier, X_test, y_test)
        assert 0 <= metrics["accuracy"] <= 1

    def test_print_evaluation_report(self, capsys):
        """Test that print_evaluation_report works."""
        X_train = pd.Series(["free win", "hello friend"] * 50)
        y_train = pd.Series([1, 0] * 50)
        X_test = pd.Series(["win free", "hello"] * 10)
        y_test = pd.Series([1, 0] * 10)

        classifier = train_classifier(build_classifier(), X_train, y_train)
        metrics = evaluate_model(classifier, X_test, y_test)
        print_evaluation_report(metrics)
        captured = capsys.readouterr()
        assert "SPAM DETECTOR EVALUATION REPORT" in captured.out


class TestIntegration:
    def test_end_to_end_pipeline(self):
        """Full pipeline test."""
        X_train, X_test, y_train, y_test = load_sms_data(test_size=0.2)
        classifier = train_classifier(build_classifier(), X_train, y_train)
        metrics = evaluate_model(classifier, X_test, y_test)
        assert metrics["accuracy"] >= 0
        assert metrics["accuracy"] <= 1
