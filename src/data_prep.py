"""
data_prep.py

Handles loading and initial cleaning of the Home Credit application data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_application_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load application_train.csv from the raw data directory.

    Args:
        data_dir: Path to the folder containing raw CSVs.

    Returns:
        Raw application data as a DataFrame.
    """
    path = Path(data_dir) / "application_train.csv"
    df = pd.read_csv(path)
    return df

def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the count and percentage of missing values per column,
    sorted from most to least missing.

    Args:
        df: Input DataFrame to audit.

    Returns:
        DataFrame with columns ['null_count', 'null_pct'], indexed by
        original column name, sorted descending by null_pct.
    """
    null_count = df.isnull().sum()
    null_pct = (null_count / len(df)) * 100

    summary = pd.DataFrame({
        "null_count": null_count,
        "null_pct": null_pct
    })

    summary = summary[summary["null_count"] > 0]
    summary = summary.sort_values("null_pct", ascending=False)

    return summary


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """
    Drop columns whose missing-value percentage exceeds the given threshold.

    Args:
        df: Input DataFrame.
        threshold: Null percentage above which a column is dropped (default 60%).

    Returns:
        DataFrame with high-null columns removed.
    """
    nulls = null_summary(df)
    cols_to_drop = nulls[nulls["null_pct"] > threshold].index.tolist()

    print(f"Dropping {len(cols_to_drop)} columns with >{threshold}% missing:")
    for col in cols_to_drop:
        print(f"  - {col}")

    df_clean = df.drop(columns=cols_to_drop)
    return df_clean