"""
model.py

Trains and evaluates a baseline logistic regression model for
Home Credit default risk prediction.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_baseline_model(X_train, y_train):
    """
    Scale features and train a baseline logistic regression model.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Tuple of (fitted model, fitted scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    print("Model trained.")
    print(f"Number of features: {X_train.shape[1]}")

    return model, scaler

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate a trained model on the test set using AUC-ROC and classification metrics.

    Args:
        model: Fitted classifier.
        scaler: Fitted scaler (must match the one used in training).
        X_test: Test features.
        y_test: Test target.

    Returns:
        Dict with 'auc', 'y_pred', 'y_pred_proba'.
    """
    X_test_scaled = scaler.transform(X_test)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {"auc": auc, "y_pred": y_pred, "y_pred_proba": y_pred_proba}

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def tune_threshold(y_test, y_pred_proba, thresholds=None):
    """
    Evaluate precision, recall, and F1 across a range of classification
    thresholds to understand the tradeoff, rather than relying on the
    default 0.5 cutoff.

    Args:
        y_test: True labels.
        y_pred_proba: Predicted probabilities (from evaluate_model's output).
        thresholds: Array of thresholds to test. Defaults to 0.1 to 0.9.

    Returns:
        DataFrame with columns ['threshold', 'precision', 'recall', 'f1'].
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    results = []
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        precision = precision_score(y_test, y_pred_t, zero_division=0)
        recall = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        results.append({
            "threshold": round(t, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    return results_df

def predict_with_threshold(y_pred_proba, threshold=0.65):
    """
    Apply a specific classification threshold to predicted probabilities.

    Args:
        y_pred_proba: Predicted probabilities of default.
        threshold: Classification cutoff (default 0.65, chosen via F1 tuning).

    Returns:
        Array of binary predictions.
    """
    return (y_pred_proba >= threshold).astype(int)
