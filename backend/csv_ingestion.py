import pandas as pd
import warnings


def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # try parsing as date
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(series.dropna().head(20))
        return "date"
    except:
        return "categorical"


def compute_stats(series: pd.Series, col_type: str) -> dict:
    null_count = int(series.isna().sum())

    if col_type == "numeric":
        return {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "null_count": null_count,
            "unique_count": int(series.nunique()),
        }

    if col_type == "categorical":
        top = series.value_counts().head(5).to_dict()
        return {
            "unique_count": int(series.nunique()),
            "top_values": [{"val": str(k), "count": int(v)} for k, v in top.items()],
            "null_count": null_count,
        }

    return {"null_count": null_count, "unique_count": int(series.nunique())}


def build_data_summary(df: pd.DataFrame) -> dict:
    column_profiles = []
    for col in df.columns:
        col_type = infer_column_type(df[col])
        stats = compute_stats(df[col], col_type)
        column_profiles.append({"name": col, "type": col_type, **stats})

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": column_profiles,
        "sample": df.head(5).to_dict(orient="records"),
    }
