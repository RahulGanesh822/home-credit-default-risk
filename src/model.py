"""
model.py

Trains and evaluates a baseline logistic regression model for
Home Credit default risk prediction.
"""

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
