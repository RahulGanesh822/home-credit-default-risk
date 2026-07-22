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


def drop_high_null_columns(
    df: pd.DataFrame,
    threshold: float = 60.0,
    protect: list = None
) -> pd.DataFrame:
    """
    Drop columns whose missing-value percentage exceeds the given threshold,
    except columns explicitly protected regardless of missingness.

    Args:
        df: Input DataFrame.
        threshold: Null percentage above which a column is dropped (default 60%).
        protect: List of column names to keep regardless of missing percentage.

    Returns:
        DataFrame with high-null columns removed.
    """
    if protect is None:
        protect = []

    nulls = null_summary(df)
    cols_to_drop = nulls[nulls["null_pct"] > threshold].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c not in protect]

    print(f"Dropping {len(cols_to_drop)} columns with >{threshold}% missing:")
    for col in cols_to_drop:
        print(f"  - {col}")

    if protect:
        protected_and_high_null = [c for c in protect if c in nulls.index and nulls.loc[c, "null_pct"] > threshold]
        if protected_and_high_null:
            print(f"\nProtected despite exceeding threshold: {protected_and_high_null}")

    df_clean = df.drop(columns=cols_to_drop)
    return df_clean

def get_column_types(df: pd.DataFrame) -> dict:
    """
    Split DataFrame columns into numeric and categorical groups based on dtype.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with keys 'numeric' and 'categorical', each a list of column names.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    return {"numeric": numeric_cols, "categorical": categorical_cols}


def impute_missing_values(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """
    Fill missing values: numeric columns with median, categorical columns
    with the literal string 'Missing'.

    Args:
        df: Input DataFrame (post column-dropping).
        col_types: Dict with 'numeric' and 'categorical' column lists,
                   as returned by get_column_types().

    Returns:
        DataFrame with no remaining nulls.
    """
    df_imputed = df.copy()

    numeric_cols = [c for c in col_types["numeric"] if c in df_imputed.columns]
    categorical_cols = [c for c in col_types["categorical"] if c in df_imputed.columns]

    for col in numeric_cols:
        if df_imputed[col].isnull().sum() > 0:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)

    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            df_imputed[col] = df_imputed[col].fillna("Missing")

    remaining_nulls = df_imputed.isnull().sum().sum()
    print(f"Remaining nulls after imputation: {remaining_nulls}")

    return df_imputed

def encode_categorical_columns(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    One-hot encode categorical columns.

    Args:
        df: Input DataFrame (post imputation, no nulls).
        categorical_cols: List of categorical column names to encode.

    Returns:
        DataFrame with categorical columns replaced by one-hot encoded columns.
    """
    cols_to_encode = [c for c in categorical_cols if c in df.columns]

    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    print(f"Columns before encoding: {df.shape[1]}")
    print(f"Columns after encoding: {df_encoded.shape[1]}")

    return df_encoded


from sklearn.model_selection import train_test_split


def split_features_target(df: pd.DataFrame, target_col: str = "TARGET"):
    """
    Split into train/test sets, stratified on the target to preserve
    class balance in both sets.

    Args:
        df: Fully encoded, imputed DataFrame including the target column.
        target_col: Name of the target column.

    Returns:
        X_train, X_test, y_train, y_test
    """
    id_cols = [c for c in ["SK_ID_CURR"] if c in df.columns]

    X = df.drop(columns=[target_col] + id_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.4f}")
    print(f"Test default rate: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test

def fix_days_employed_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the DAYS_EMPLOYED anomaly: a placeholder value of 365243 is used
    to represent 'not currently employed' (e.g. retirees), rather than
    a real day count. This creates a flag column and replaces the
    placeholder with NaN so it gets handled by normal imputation.

    Args:
        df: Input DataFrame containing DAYS_EMPLOYED.

    Returns:
        DataFrame with DAYS_EMPLOYED_ANOMALY flag added and placeholder
        values replaced with NaN.
    """
    df_fixed = df.copy()

    anomaly_value = 365243
    anomaly_count = (df_fixed["DAYS_EMPLOYED"] == anomaly_value).sum()

    df_fixed["DAYS_EMPLOYED_ANOMALY"] = (df_fixed["DAYS_EMPLOYED"] == anomaly_value).astype(int)
    df_fixed.loc[df_fixed["DAYS_EMPLOYED"] == anomaly_value, "DAYS_EMPLOYED"] = np.nan

    print(f"DAYS_EMPLOYED anomaly (365243) found in {anomaly_count} rows — flagged and set to NaN.")

    return df_fixed


def load_and_aggregate_bureau(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load bureau.csv and aggregate from one-row-per-credit to
    one-row-per-client (SK_ID_CURR), producing summary features
    about each client's previous credit history from other lenders.

    Args:
        data_dir: Path to the folder containing raw CSVs.

    Returns:
        DataFrame with one row per SK_ID_CURR and aggregated bureau features.
    """
    path = Path(data_dir) / "bureau.csv"
    bureau = pd.read_csv(path)

    bureau["IS_OVERDUE"] = (bureau["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_OVERDUE_COUNT=("IS_OVERDUE", "sum"),
        BUREAU_DAYS_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AMT_OVERDUE_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_CNT_PROLONGED_TOTAL=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
    ).reset_index()

    print(f"Aggregated bureau data: {agg.shape[0]} clients, {agg.shape[1]} features")

    return agg

def merge_bureau_features(df: pd.DataFrame, bureau_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Merge aggregated bureau features onto the main application DataFrame.
    Handles clients with no bureau history explicitly: count/sum columns
    are filled with 0 (factually correct — no history means zero previous
    credits), while BUREAU_DAYS_CREDIT_MEAN is left as NaN for standard
    median imputation later, since 0 would be a false value for "average
    recency." A HAS_BUREAU_HISTORY flag captures the missingness itself.

    Args:
        df: Main application DataFrame (must contain SK_ID_CURR).
        bureau_agg: Aggregated bureau features, as returned by
                    load_and_aggregate_bureau().

    Returns:
        DataFrame with bureau features merged in.
    """
    df_merged = df.merge(bureau_agg, on="SK_ID_CURR", how="left").copy()

    df_merged["HAS_BUREAU_HISTORY"] = df_merged["BUREAU_CREDIT_COUNT"].notnull().astype(int)

    

    zero_fill_cols = [
        "BUREAU_CREDIT_COUNT",
        "BUREAU_ACTIVE_COUNT",
        "BUREAU_OVERDUE_COUNT",
        "BUREAU_DAYS_OVERDUE_MAX",
        "BUREAU_AMT_CREDIT_SUM_TOTAL",
        "BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL",
        "BUREAU_AMT_OVERDUE_TOTAL",
        "BUREAU_CNT_PROLONGED_TOTAL",
    ]
    for col in zero_fill_cols:
        df_merged[col] = df_merged[col].fillna(0)

    no_history_count = (df_merged["HAS_BUREAU_HISTORY"] == 0).sum()
    print(f"Clients with no bureau history: {no_history_count} ({no_history_count / len(df_merged) * 100:.2f}%)")
    print(f"Shape after merge: {df_merged.shape}")

    return df_merged
