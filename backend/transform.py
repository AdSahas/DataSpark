import pandas as pd
from typing import List


def encode_discrete_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Encodes categorical columns into numeric labels using factorization.
    Safe: only affects specified columns.
    """
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            continue

        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            df[col], _ = pd.factorize(df[col])

        # handle boolean explicitly
        elif df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    return df


def normalize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            std = df[c].std()
            df[c] = (df[c] - df[c].mean()) / (std if std > 0 else 1)
    return df
