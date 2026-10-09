"""Spam Detector - Naive Bayes SMS classification."""

from .data import load_sms_data, preprocess_text
from .model import build_classifier, predict
from .evaluator import evaluate_model

__version__ = "1.0.0"
__all__ = [
    "load_sms_data",
    "preprocess_text",
    "build_classifier",
    "predict",
    "evaluate_model",
]
