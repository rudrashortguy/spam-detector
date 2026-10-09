# Spam Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Naive Bayes-based SMS spam detector using scikit-learn.

## Features

- SMS text preprocessing and feature extraction
- Naive Bayes classifier for spam detection
- Train/test split and cross-validation
- Model evaluation with precision, recall, F1-score
- Prediction of new messages
- Command-line interface
- Comprehensive test suite
- GitHub Actions CI

## Installation

```bash
git clone https://github.com/rudrashortguy/spam-detector.git
cd spam-detector
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Training a model

```bash
spam-detector train --input data/sms_spam.csv --output models/spam_classifier.pkl
```

### Predicting spam

```bash
spam-detector predict --model models/spam_classifier.pkl --text "Win a free iPhone!"
```

### Python API

```python
from spam_detector.data import load_sms_data
from spam_detector.model import build_classifier, predict
from spam_detector.evaluator import evaluate_model

# Load data
X_train, X_test, y_train, y_test = load_sms_data()

# Build and train classifier
classifier = build_classifier()
classifier.fit(X_train, y_train)

# Evaluate
metrics = evaluate_model(classifier, X_test, y_test)
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Predict
prediction = predict(classifier, ["Free entry!"])
print(f"Predicted: {'spam' if prediction[0] == 1 else 'ham'}")
```

## Project Structure

```
spam-detector/
├── data/                  # Data storage (not tracked)
├── spam_detector/         # Source code
│   ├── __init__.py
│   ├── data.py            # Data loading and preprocessing
│   ├── model.py           # Classifier implementation
│   └── evaluator.py       # Evaluation metrics
├── tests/                 # Unit tests
├── models/                # Trained models (not tracked)
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml
└── .github/workflows/tests.yml
```

## Tech Stack

- **Language**: Python 3.8+
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## Roadmap

- [x] Naive Bayes classifier implementation
- [x] SMS data preprocessing
- [x] Model training and evaluation
- [ ] Support for custom datasets
- [ ] Web interface for real-time classification
- [ ] Advanced NLP techniques (TF-IDF, word embeddings)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.