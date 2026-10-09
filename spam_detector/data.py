"""Data loading and preprocessing module."""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess SMS text.

    Args:
        text: Raw text string.

    Returns:
        Preprocessed text string.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_sms_data(
    filepath: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load and preprocess SMS spam dataset.

    Args:
        filepath: Path to CSV file with 'text' and 'label' columns.
            If None, uses bundled sample data.
        test_size: Proportion of data for testing.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if filepath is None:
        filepath = _get_sample_data_path()

    df = pd.read_csv(filepath)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    # Preprocess texts
    df['text'] = df['text'].apply(preprocess_text)

    # Convert labels to binary (spam=1, ham=0)
    df['label'] = df['label'].map(lambda x: 1 if x == 'spam' or x == 1 else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )

    return X_train, X_test, y_train, y_test


def _get_sample_data_path() -> str:
    """Return path to bundled sample SMS data."""
    import tempfile
    sample_data = """text,label
"Hi dear, how are you?",ham
"Win a free iPhone! Click here!",spam
"Hey, are we still meeting tomorrow?",ham
"URGENT: You've won $1000! Call now!",spam
"Can you pick up groceries on the way?",ham
"FREE entry to win a prize! Text YES",spam
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(sample_data)
        return f.name
