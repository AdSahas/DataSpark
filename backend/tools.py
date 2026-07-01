from typing import Literal, Any, List, Dict, TypedDict, Callable, Tuple
from functools import wraps

import numpy as np
import duckdb
import pandas as pd
from langchain_core.tools import tool
from backend.csv_ingestion import compute_stats, infer_column_type
from scipy import stats
import statsmodels.api as sm
import pymannkendall as mk
from backend.transform import encode_discrete_columns, normalize_columns


# ─────────────────────────────────────────────────────────────
# RESULT SCHEMA
# ─────────────────────────────────────────────────────────────

class ResultSchema(TypedDict, total=False):
    status: Literal["ok", "error"]
    data: Any
    message: str
    diagnostics: Dict[str, Any]


def _ok(data: Any = None, message: str = "", diagnostics: Dict = None) -> ResultSchema:
    return {
        "status": "ok",
        "data": data,
        "message": message,
        "diagnostics": diagnostics or {}
    }


def _err(message: str, diagnostics: Dict = None) -> ResultSchema:
    return {
        "status": "error",
        "data": None,
        "message": message,
        "diagnostics": diagnostics or {}
    }


# ─────────────────────────────────────────────────────────────
# STRICT SCHEMA ENFORCEMENT DECORATOR
# ─────────────────────────────────────────────────────────────

def enforce_schema(fn: Callable) -> Callable:
    """
    Guarantees every tool returns ResultSchema.

    Rules:
    - dict with status → pass-through
    - raw return → wrapped in _ok
    - exceptions → _err
    """

    @wraps(fn)
    def wrapper(*args, **kwargs) -> ResultSchema:
        try:
            result = fn(*args, **kwargs)

            if isinstance(result, dict) and "status" in result:
                return result

            return _ok(result)

        except Exception as e:
            return _err(
                str(e),
                diagnostics={
                    "tool": fn.__name__,
                    "error_type": type(e).__name__
                }
            )

    return wrapper


# ─────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────

_df: pd.DataFrame = None


def set_dataframe(df: pd.DataFrame):
    global _df
    _df = df


def _require_df():
    if _df is None:
        raise RuntimeError("No dataframe loaded. Call set_dataframe() first.")


# ─────────────────────────────────────────────────────────────
# VALIDATION ENGINE
# ─────────────────────────────────────────────────────────────

def _prepare_stats(
    y_col: str,
    x_cols: List[str],
    test_type: Literal["linear", "logistic", "pearson", "spearman", "chi2"],
    normalize: bool = False,
):
    _require_df()

    cols = list(dict.fromkeys([y_col] + x_cols))

    for c in cols:
        if c not in _df.columns:
            return _err(f"Column '{c}' not found")

    # drop nulls and encode categorical columns if one is called for here.
    data = _df[cols].dropna()
    data = encode_discrete_columns(data, cols)

    # normalize if linear or logistic regression
    if normalize and test_type in ("linear", "logistic"):
        data = normalize_columns(data, cols)

    if len(data) < 10:
        return _err("Not enough data (min 10 rows required)")

    y = data[y_col]
    X = data[x_cols].copy()

    # ── linear regression
    if test_type == "linear":
        if not pd.api.types.is_numeric_dtype(y):
            return _err("Linear regression requires numeric target")

    # ── logistic regression
    if test_type == "logistic":
        if y.nunique() != 2:
            return _err("Logistic regression requires binary target")

        uniq = sorted(y.unique())
        if set(uniq) != {0, 1}:
            y = y.map({uniq[0]: 0, uniq[1]: 1})

    # ── correlations
    if test_type in ("pearson", "spearman"):
        for c in cols:
            if not pd.api.types.is_numeric_dtype(data[c]):
                return _err(f"Column '{c}' must be numeric")

    # ── chi-square
    if test_type == "chi2":
        for c in cols:
            if pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() > 10:
                return _err("Chi-square requires categorical columns")

    # ── regression checks
    if test_type in ("linear", "logistic"):
        for c in x_cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                return _err(f"Predictor '{c}' must be numeric")

        if len(x_cols) > 1:
            corr_matrix = X.corr().abs().mask(np.eye(len(x_cols), dtype=bool))
            max_corr = float(corr_matrix.max().max())
            if max_corr > 0.90:
                high_pairs = [
                    (c1, c2, round(corr_matrix.loc[c1, c2], 3))
                    for c1 in corr_matrix.columns
                    for c2 in corr_matrix.columns
                    if c1 < c2 and corr_matrix.loc[c1, c2] > 0.90
                ]
                return {
                    "error": (
                        f"Severe multicollinearity detected (max r={max_corr:.2f}). "
                        f"Highly correlated pairs: {high_pairs}. Remove or combine them."
                    )
                }

    return X, y, data


'''DATA ANALYSIS TOOLS'''


@tool
@enforce_schema
def get_schema():
    """
    Get column names and inferred types for the loaded dataset.
    """
    _require_df()
    return {
        "row_count": len(_df),
        "columns": [
            {"name": col, "type": infer_column_type(_df[col])}
            for col in _df.columns
        ],
    }


@tool
@enforce_schema
def get_sample(n: int = 5):
    """
    Get n random rows from the dataset.
    """
    _require_df()
    return _df.sample(min(n, len(_df))).to_dict(orient="records")


@tool
@enforce_schema
def get_column_stats(column: str):
    """
    Get descriptive statistics for a single column (min, max, mean, std, nulls).
    """
    _require_df()
    if column not in _df.columns:
        return _err(f"Column '{column}' not found")

    col_type = infer_column_type(_df[column])
    return {
        "column": column,
        "type": col_type,
        **compute_stats(_df[column], col_type),
    }


@tool
@enforce_schema
def get_value_counts(column: str, top_n: int = 10):
    """
    Get frequency distribution of values in a categorical column.
    """
    _require_df()

    counts = _df[column].value_counts().head(top_n).to_dict()
    return {
        "column": column,
        "value_counts": {str(k): int(v) for k, v in counts.items()},
    }


@tool
@enforce_schema
def detect_outliers(column: str):
    """
    Detect outliers in a numeric column using mean ± 2 std rule.
    """
    _require_df()

    series = _df[column]

    if not pd.api.types.is_numeric_dtype(series):
        return _err(f"Column '{column}' is not numeric")

    mean, std = series.mean(), series.std()
    outliers = _df[(series < mean - 2 * std) | (series > mean + 2 * std)]

    return {
        "column": column,
        "mean": float(mean),
        "std": float(std),
        "outlier_count": len(outliers),
        "outlier_rows": outliers.head(10).to_dict(orient="records"),
    }


@tool
@enforce_schema
def classify_dataset():
    """
    Produces a structural profile of the dataset.
    """
    _require_df()
    return {
        "shape": _df.shape,
        "columns": list(_df.columns),
    }


@tool
@enforce_schema
def sql_query(query: str):
    """
    Execute SQL query using DuckDB (table name: df).
    """
    _require_df()

    con = duckdb.connect(":memory:")
    con.register("df", _df)
    res = con.execute(query).fetchdf()

    return res.to_dict(orient="records")


'''Statistical tools'''


@tool
@enforce_schema
def linear_regression(y_col: str, x_col: List[str]):
    """
    Multiple linear regression (OLS).
    """
    prep = _prepare_stats(y_col, x_col, "linear", normalize=True)
    if isinstance(prep, dict):
        return prep

    X, y, _ = prep
    model = sm.OLS(y, sm.add_constant(X)).fit()

    return {
        "r2": float(model.rsquared),
        "coefficients": model.params.to_dict(),
    }


@tool
@enforce_schema
def logistic_regression(y_col: str, x_col: List[str]):
    """
    Multiple logistic regression.
    """
    prep = _prepare_stats(y_col, x_col, "logistic", normalize=True)
    if isinstance(prep, dict):
        return prep

    X, y, _ = prep
    model = sm.Logit(y, sm.add_constant(X)).fit(disp=False)

    return {
        "llf": float(model.llf),
        "coefficients": model.params.to_dict(),
    }


@tool
@enforce_schema
def run_pearson_correlation(col_x: str, col_y: str):
    """
    Pearson correlation coefficient.
    """
    prep = _prepare_stats(col_x, [col_y], "pearson")
    if isinstance(prep, dict):
        return prep

    _, _, data = prep
    r, p = stats.pearsonr(data[col_x], data[col_y])

    return {"r": float(r), "p": float(p)}


@tool
@enforce_schema
def run_spearman_correlation(col_x: str, col_y: str):
    """
    Spearman rank correlation.
    """
    prep = _prepare_stats(col_x, [col_y], "spearman")
    if isinstance(prep, dict):
        return prep

    _, _, data = prep
    r, p = stats.spearmanr(data[col_x], data[col_y])

    return {"rho": float(r), "p": float(p)}


@tool
@enforce_schema
def run_chi_squared(col_a: str, col_b: str):
    """
    Chi-squared test of independence.
    """
    prep = _prepare_stats(col_a, [col_b], "chi2")
    if isinstance(prep, dict):
        return prep

    _, _, data = prep
    table = pd.crosstab(data[col_a], data[col_b])
    chi2, p, _, _ = stats.chi2_contingency(table)

    return {
        "chi2": float(chi2),
        "p": float(p),
    }


@tool
@enforce_schema
def detect_trends(x_col: str, y_col: str):
    """
    Mann-Kendall trend detection.
    """
    _require_df()

    d = _df[[x_col, y_col]].dropna().sort_values(x_col)

    if len(d) < 3:
        return _err("Not enough data")

    res = mk.original_test(d[y_col].to_numpy())

    return {
        "trend": res.trend,
        "p": float(res.p),
        "tau": float(res.Tau),
    }


@tool
@enforce_schema
def fit_curves(col_a: str, col_b: str, degree: int = 4):
    """
    Polynomial curve fitting with R² comparison across degrees.
    """
    _require_df()

    if col_a not in _df.columns or col_b not in _df.columns:
        return _err("Columns not found")

    data = _df[[col_a, col_b]].dropna()
    x = data[col_a].values
    y = data[col_b].values

    results = {}

    for d in range(2, degree + 1):
        coeffs = np.polyfit(x, y, d)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - ss_res / ss_tot

        results[d] = {
            "coefficients": coeffs.tolist(),
            "r2": float(r2),
        }

    best = max(results.items(), key=lambda x: x[1]["r2"])

    return _ok({
        "variable_x": col_a,
        "variable_y": col_b,
        "models": results,
        "best_model": {
            "degree": best[0],
            "r2": best[1]["r2"],
        },
        "interpretation": f"Best fit degree {best[0]} with R²={best[1]['r2']:.4f}",
    })
