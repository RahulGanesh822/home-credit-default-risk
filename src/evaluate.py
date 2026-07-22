"""
evaluate.py

Persists model outputs: saved model files, prediction CSVs, and
evaluation charts (ROC curve, feature importance).
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score


def save_model(model, path: str, scaler=None):
    """
    Save a trained model (and optionally its scaler) to disk.

    Args:
        model: Fitted model object.
        path: File path to save to (e.g. 'models/xgboost_model.pkl').
        scaler: Optional fitted scaler to save alongside the model.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if scaler is not None:
        joblib.dump({"model": model, "scaler": scaler}, path)
    else:
        joblib.dump({"model": model, "scaler": None}, path)

    print(f"Model saved to {path}")


def generate_predictions_csv(df_original, X_test, y_test, y_pred_proba, path: str, id_col: str = "SK_ID_CURR"):
    """
    Save a CSV of test set predictions: applicant ID, actual label,
    predicted probability of default.

    Args:
        df_original: The DataFrame that still contains id_col (pre-split).
        X_test: Test features (used only for its index, to recover IDs).
        y_test: True test labels.
        y_pred_proba: Predicted probabilities of default.
        path: File path to save to.
        id_col: Name of the ID column to recover.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    ids = df_original.loc[X_test.index, id_col]

    results = pd.DataFrame({
        id_col: ids.values,
        "actual_default": y_test.values,
        "predicted_probability": y_pred_proba
    })

    results.to_csv(path, index=False)
    print(f"Predictions saved to {path} ({len(results)} rows)")

    return results


def plot_roc_curve(y_test, model_predictions: dict, path: str):
    """
    Plot ROC curves for one or more models on the same test set.

    Args:
        y_test: True test labels.
        model_predictions: Dict of {model_name: y_pred_proba}.
        path: File path to save the chart to.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for name, y_pred_proba in model_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Home Credit Default Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"ROC curve saved to {path}")


def plot_feature_importance_chart(importance_df: pd.DataFrame, value_col: str, title: str, path: str, top_n: int = 15):
    """
    Plot a horizontal bar chart of top feature importances.

    Args:
        importance_df: DataFrame from get_logistic_feature_importance or
                        get_xgboost_feature_importance.
        value_col: Column to plot ('abs_coefficient' or 'importance').
        title: Chart title.
        path: File path to save the chart to.
        top_n: Number of top features to show.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plot_data = importance_df.head(top_n).sort_values(value_col)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_data["feature"], plot_data[value_col])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Feature importance chart saved to {path}")
