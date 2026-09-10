# PhishNet AI 🎯

A machine-learning phishing email detector that classifies emails as **phishing** or **legitimate** with **99%+ accuracy**, built end-to-end in Python using object-oriented design principles.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

PhishNet AI trains a supervised classification model on a real-world Kaggle dataset of 5,728 emails, converting raw email text into numerical features with `CountVectorizer` and classifying them with a Naive Bayes model. The project is built entirely around custom Python classes (encapsulation, composition, custom exceptions) rather than loose scripts, and is usable from both a terminal CLI and a Jupyter notebook.

## Features

- 🔍 **Text classification** — detects phishing emails from raw email content using `CountVectorizer` + `MultinomialNB`
- 🏗️ **Object-oriented design** — `EmailDataset` and `PhishingDetector` classes with encapsulation, custom exceptions, and dunder methods
- 💾 **Model persistence** — save/load trained models with `Joblib`
- 🖥️ **Terminal CLI** — menu-driven app to train, test, and save models interactively
- 📊 **Jupyter notebook** — exploratory data analysis with visualizations (class distribution, email length, confusion matrix)
- ✅ **Unit tested** — 10-test `unittest` suite covering data validation, training, and prediction
- 🐚 **Bash automation** — one-command setup, testing, and execution via shell scripts

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML / Data | Scikit-learn, Pandas, Joblib |
| Visualization | Matplotlib, Jupyter Notebook |
| Testing | unittest |
| Automation | Bash |
| Dataset | [Kaggle: Spam/Phishing Email Dataset](https://www.kaggle.com/) (5,728 emails) |

## Project Structure

```
PhishNetAI/
├── main.py                       # Terminal CLI (menu-driven app)
├── email_dataset.py              # EmailDataset class — loads & validates data
├── phishing_detector.py          # PhishingDetector class — ML pipeline
├── exceptions.py                 # Custom exception classes
├── test_phishing_detector.py     # unittest suite
├── PhishNet_AI_Analysis.ipynb    # Jupyter notebook (EDA + training)
├── emails.csv                    # Kaggle dataset
├── phishnet_model.joblib         # Pre-trained saved model
├── requirements.txt              # Python dependencies
├── setup.sh                      # Creates venv & installs dependencies
├── run.sh                        # Runs the CLI app
└── test.sh                       # Runs the unit test suite
```


## How It Works

**ML Pipeline:**
```
Raw email text
↓
CountVectorizer → numeric feature matrix (X) (bag-of-words)
↓
MultinomialNB classifier → trained on labeled data
↓
Prediction: "phishing" or "legitimate" + confidence score
```


**Class design:**

- **`EmailDataset`** — wraps the CSV in a class with private attributes and getters. Validates that the file exists, isn't empty, and has the right columns; raises a custom `InvalidDatasetError` otherwise. Implements `__len__`, `__getitem__`, and `__str__` so the dataset behaves like a first-class Python object.

- **`PhishingDetector`** — *composes* a `CountVectorizer` and a `MultinomialNB` classifier internally. Exposes `train()`, `predict()`, `predict_batch()`, `save_model()`, and `load_model()`. Raises a custom `ModelNotTrainedError` if you try to predict or save before training.

- **`exceptions.py`** — `InvalidDatasetError` and `ModelNotTrainedError`, both subclassing `Exception`, each carrying extra context (e.g. the file path) about what went wrong.

This design keeps the machine-learning details fully encapsulated — `main.py` never touches `CountVectorizer` or `MultinomialNB` directly, only the public methods of `PhishingDetector`.

## Installation

**Prerequisites:** Python 3.9+, Bash (macOS/Linux Terminal, or Git Bash/WSL on Windows)

```bash
# 1. Clone the repository
git clone https://github.com/sahassan321/PhishNetAI.git
cd PhishNetAI

# 2. Make the shell scripts executable
chmod +x setup.sh run.sh test.sh

# 3. Create a virtual environment & install dependencies
./setup.sh
```

`setup.sh` creates an isolated `venv/` and installs everything in `requirements.txt` (pandas, scikit-learn, joblib, matplotlib, jupyter) — nothing touches your system Python.

## Usage

### Terminal CLI

```bash
./run.sh
```


### Jupyter Notebook

```bash
source venv/bin/activate
jupyter notebook PhishNet_AI_Analysis.ipynb
```

The notebook reuses the same `EmailDataset` and `PhishingDetector` classes as the CLI, and adds:
- Class distribution chart (phishing vs. legitimate)
- Email length histogram
- Confusion matrix heatmap
- Live predictions on sample emails

### Using the classes directly

```python
from email_dataset import EmailDataset
from phishing_detector import PhishingDetector

dataset = EmailDataset('emails.csv')
detector = PhishingDetector()
accuracy = detector.train(dataset)

label, confidence = detector.predict("Congratulations! You've won a prize, click here!")
print(label, confidence)   # phishing 0.998
```

## Testing

```bash
./test.sh
```

10 tests covering:
- Dataset loading and validation (`InvalidDatasetError` on bad input)
- Model training and accuracy threshold (regression guard: fails if accuracy drops below 90%)
- Prediction correctness on obvious phishing text
- Guard behavior (`ModelNotTrainedError` when predicting before training)

## Results

| Metric | Score |
|---|---|
| Test accuracy | 99.3% |
| Precision (phishing) | 0.97 |
| Recall (phishing) | 1.00 |
| F1-score (phishing) | 0.98 |
| Dataset size | 5,728 emails (1,368 phishing / 4,360 legitimate) |

Evaluated on a stratified 80/20 train/test split.
