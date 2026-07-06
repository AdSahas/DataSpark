from typing import Literal, Any, List, Dict, TypedDict, Callable
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


# ── RESULT SCHEMA ──────────────────────────────────────────────────────

class ResultSchema(TypedDict, total=False):
    status: Literal["ok", "error"]
    data: Any
    message: str
    diagnostics: Dict[str, Any]


def _ok(data: Any = None, message: str = "", diagnostics: Dict = None) -> ResultSchema:
    return {"status": "ok", "data": data, "message": message, "diagnostics": diagnostics or {}}


def _err(message: str, diagnostics: Dict = None) -> ResultSchema:
    return {"status": "error", "data": None, "message": message, "diagnostics": diagnostics or {}}


def enforce_schema(fn: Callable) -> Callable:
    """Guarantees every tool returns ResultSchema."""
    @wraps(fn)
    def wrapper(*args, **kwargs) -> ResultSchema:
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, dict) and "status" in result:
                return result
            return _ok(result)
        except Exception as e:
            return _err(str(e), diagnostics={"tool": fn.__name__, "error_type": type(e).__name__})
    return wrapper


# ── SHARED STATE ───────────────────────────────────────────────────────

_df: pd.DataFrame = None


def set_dataframe(df: pd.DataFrame):
    global _df, _df_version
    _df = df


def _require_df():
    if _df is None:
        raise RuntimeError("No dataframe loaded. Call set_dataframe() first.")


# ── INTERNAL HELPERS ───────────────────────────────────────────────────

def _shapiro(series: pd.Series) -> Dict[str, Any]:
    """Shapiro-Wilk normality test. Caps at 5000 samples."""
    s = series.dropna()
    if len(s) < 3:
        return {"normal": True, "p_value": None, "note": "Too few samples to test normality"}
    if len(s) > 5000:
        s = s.sample(5000, random_state=0)
    stat, p = stats.shapiro(s)
    return {"statistic": float(stat), "p_value": float(p), "normal": bool(p > 0.05)}


def _levene(groups: List[np.ndarray]) -> Dict[str, Any]:
    """Levene's test for equal variances."""
    stat, p = stats.levene(*groups)
    return {"statistic": float(stat), "p_value": float(p), "equal_variance": bool(p > 0.05)}


def _validate_groups(data: pd.DataFrame, value_col: str, group_col: str,
                     min_groups: int = 2, max_groups: int = None, min_size: int = 2):
    """Validate group structure. Returns (groups_series, error_or_None)."""
    for c in (value_col, group_col):
        if c not in data.columns:
            return None, _err(f"Column '{c}' not found")
    if not pd.api.types.is_numeric_dtype(data[value_col]):
        return None, _err(f"Value column '{value_col}' must be numeric")

    groups = data.groupby(group_col)[value_col].apply(
        lambda s: s.dropna().values)
    groups = groups[groups.apply(len) >= min_size]

    n = len(groups)
    if n < min_groups:
        return None, _err(f"Need at least {min_groups} groups with >= {min_size} obs each (found {n})")
    if max_groups and n > max_groups:
        return None, _err(f"This test requires at most {max_groups} groups (found {n})")
    return groups, None


def _interpret_effect(value: float, thresholds, labels) -> str:
    """Map an effect-size magnitude to a label given ascending thresholds."""
    v = abs(value)
    for t, lab in zip(thresholds, labels):
        if v < t:
            return lab
    return labels[-1]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0:
        return {"name": "Cohen's d", "value": None, "magnitude": "undefined (zero variance)"}
    d = (np.mean(a) - np.mean(b)) / pooled
    return {"name": "Cohen's d", "value": float(d),
            "magnitude": _interpret_effect(d, [0.2, 0.5, 0.8], ["negligible", "small", "medium", "large"])}


def _eta_squared(groups: pd.Series) -> Dict[str, Any]:
    all_vals = np.concatenate(groups.values)
    grand = all_vals.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values)
    ss_total = np.sum((all_vals - grand) ** 2)
    if ss_total == 0:
        return {"name": "Eta-squared", "value": None, "magnitude": "undefined (zero variance)"}
    eta2 = ss_between / ss_total
    return {"name": "Eta-squared", "value": float(eta2),
            "magnitude": _interpret_effect(eta2, [0.01, 0.06], ["small", "medium", "large"])}


def _epsilon_squared(h: float, n: int, k: int) -> Dict[str, Any]:
    """Effect size for Kruskal-Wallis."""
    if n - k == 0:
        return {"name": "Epsilon-squared", "value": None, "magnitude": "undefined"}
    eps2 = max(0.0, (h - k + 1) / (n - k))
    return {"name": "Epsilon-squared", "value": float(eps2),
            "magnitude": _interpret_effect(eps2, [0.01, 0.08], ["small", "medium", "large"])}


def _cramers_v(table: pd.DataFrame, chi2: float) -> Dict[str, Any]:
    n = table.values.sum()
    r, k = table.shape
    denom = n * (min(r - 1, k - 1))
    if denom == 0:
        return {"name": "Cramér's V", "value": None, "magnitude": "undefined"}
    v = np.sqrt(chi2 / denom)
    return {"name": "Cramér's V", "value": float(v),
            "magnitude": _interpret_effect(v, [0.1, 0.3, 0.5], ["negligible", "small", "medium", "large"])}


def _vif_scores(X: pd.DataFrame) -> Dict[str, float]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = sm.add_constant(X)
    return {col: float(variance_inflation_factor(Xc.values, i))
            for i, col in enumerate(Xc.columns) if col != "const"}


# ── CORE STATS PREPARATION ─────────────────────────────────────────────

def _prepare_stats(y_col, x_cols, test_type, normalize=False):
    """Prepare + validate data. Returns _ok((X, y, data)) or _err(...)."""
    _require_df()

    cols = list(dict.fromkeys([y_col] + x_cols))
    for c in cols:
        if c not in _df.columns:
            return _err(f"Column '{c}' not found")

    data = _df[cols].dropna()
    data = encode_discrete_columns(data, cols)
    if normalize and test_type in ("linear", "logistic"):
        data = normalize_columns(data, cols)
    if len(data) < 10:
        return _err("Not enough data (min 10 rows required)")

    y = data[y_col]
    X = data[x_cols].copy() if x_cols else None

    if test_type == "linear" and not pd.api.types.is_numeric_dtype(y):
        return _err("Linear regression requires numeric target")

    if test_type == "logistic":
        if y.nunique() != 2:
            return _err("Logistic regression requires binary target")
        uniq = sorted(y.unique())
        if set(uniq) != {0, 1}:
            y = y.map({uniq[0]: 0, uniq[1]: 1})

    if test_type in ("pearson", "spearman", "ttest_paired", "wilcoxon"):
        for c in cols:
            if not pd.api.types.is_numeric_dtype(data[c]):
                return _err(f"Column '{c}' must be numeric for this test")

    if test_type == "chi2":
        for c in cols:
            if pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() > 10:
                return _err("Chi-square requires categorical columns")

    if test_type in ("linear", "logistic") and X is not None and len(x_cols) > 1:
        corr = X.corr().abs().mask(np.eye(len(x_cols), dtype=bool))
        max_corr = float(corr.max().max())
        if max_corr > 0.90:
            high = [(c1, c2, round(corr.loc[c1, c2], 3))
                    for c1 in corr.columns for c2 in corr.columns
                    if c1 < c2 and corr.loc[c1, c2] > 0.90]
            return _err(f"Severe multicollinearity (max r={max_corr:.2f}). Pairs: {high}")

    result = _ok((
        X.copy(deep=True),
        y.copy(deep=True),
        data.copy(deep=True),
    ))
    return result


# ── DATA ANALYSIS TOOLS ────────────────────────────────────────────────

@tool
@enforce_schema
def get_schema():
    """Get column names and inferred types for the loaded dataset."""
    _require_df()
    return {"row_count": len(_df),
            "columns": [{"name": c, "type": infer_column_type(_df[c])} for c in _df.columns]}


@tool
@enforce_schema
def get_sample(n: int = 5):
    """Get n random rows from the dataset."""
    _require_df()
    return _df.sample(min(n, len(_df))).to_dict(orient="records")


@tool
@enforce_schema
def get_column_stats(column: str):
    """Get descriptive statistics for a single column."""
    _require_df()
    if column not in _df.columns:
        return _err(f"Column '{column}' not found")
    col_type = infer_column_type(_df[column])
    return {"column": column, "type": col_type, **compute_stats(_df[column], col_type)}


@tool
@enforce_schema
def get_value_counts(column: str, top_n: int = 10):
    """Get frequency distribution of values in a categorical column."""
    _require_df()
    counts = _df[column].value_counts().head(top_n).to_dict()
    return {"column": column, "value_counts": {str(k): int(v) for k, v in counts.items()}}


@tool
@enforce_schema
def detect_outliers(column: str):
    """Detect outliers using mean ± 2 std rule."""
    _require_df()
    series = _df[column]
    if not pd.api.types.is_numeric_dtype(series):
        return _err(f"Column '{column}' is not numeric")
    mean, std = series.mean(), series.std()
    outliers = _df[(series < mean - 2 * std) | (series > mean + 2 * std)]
    return {"column": column, "mean": float(mean), "std": float(std),
            "outlier_count": len(outliers),
            "outlier_rows": outliers.head(10).to_dict(orient="records")}


@tool
@enforce_schema
def classify_dataset():
    """Produces a structural profile of the dataset."""
    _require_df()
    return {"shape": _df.shape, "columns": list(_df.columns)}


@tool
@enforce_schema
def sql_query(query: str):
    """Execute SQL query using DuckDB (table name: df)."""
    _require_df()
    con = duckdb.connect(":memory:")
    con.register("df", _df)
    return con.execute(query).fetchdf().to_dict(orient="records")


# ── CORRELATION PIPELINE ───────────────────────────────────────────────

@tool
@enforce_schema
def run_correlation(col_x: str, col_y: str):
    """
    Correlation pipeline. Auto-selects Pearson (if both vars normal)
    or Spearman (if not), reports strength, and interprets the result.
    """
    prep = _prepare_stats(col_x, [col_y], "pearson")
    if prep["status"] == "error":
        return prep
    _, _, data = prep["data"]

    nx, ny = _shapiro(data[col_x]), _shapiro(data[col_y])
    both_normal = nx["normal"] and ny["normal"]

    if both_normal:
        test, r, p = "Pearson", *stats.pearsonr(data[col_x], data[col_y])
        coef_name = "r"
    else:
        test, r, p = "Spearman", *stats.spearmanr(data[col_x], data[col_y])
        coef_name = "rho"

    strength = _interpret_effect(r, [0.1, 0.3, 0.5, 0.7],
                                 ["negligible", "weak", "moderate", "strong", "very strong"])
    direction = "positive" if r > 0 else "negative"
    sig = p < 0.05

    return {
        "Test Used": test,
        "Reason": "Both variables normal" if both_normal else "Normality violated -> Spearman",
        "Variables": [col_x, col_y],
        "Assumptions": {"Normality X": nx, "Normality Y": ny},
        "Statistics": {coef_name: float(r), "P-Value": float(p)},
        "Significant": bool(sig),
        "Interpretation": (
            f"{strength.capitalize()} {direction} {'and significant' if sig else 'but not significant'} "
            f"association ({coef_name}={r:.3f}, p={p:.4f})."
        ),
    }


# ── INDEPENDENT TWO-GROUP PIPELINES ────────────────────────────────────

@tool
@enforce_schema
def run_ttest_independent(value_col: str, group_col: str):
    """
    Independent samples t-test pipeline. Checks normality + equal variance,
    then auto-selects Student's t or Welch's t. Reports Cohen's d.
    Returns an error if normality is violated — use run_mann_whitney instead.
    """
    _require_df()
    data = _df[[value_col, group_col]].dropna()
    groups, err = _validate_groups(
        data, value_col, group_col, min_groups=2, max_groups=2)
    if err:
        return err

    names = list(groups.index)
    g1, g2 = groups.iloc[0], groups.iloc[1]
    norm1, norm2 = _shapiro(pd.Series(g1)), _shapiro(pd.Series(g2))
    both_normal = norm1["normal"] and norm2["normal"]

    if not both_normal:
        return _err(
            "Normality assumption violated. Use run_mann_whitney instead.",
            diagnostics={"normality_g1": norm1, "normality_g2": norm2}
        )

    lev = _levene([g1, g2])
    if lev["equal_variance"]:
        test, reason = "Student's t-test", "Equal variances"
        stat, p = stats.ttest_ind(g1, g2, equal_var=True)
    else:
        test, reason = "Welch's t-test", "Unequal variances -> Welch correction"
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)

    effect = _cohens_d(g1, g2)
    sig = p < 0.05
    return {
        "Test Used": test,
        "Reason": reason,
        "Groups": names,
        "Group Sizes": [len(g1), len(g2)],
        "Means": [float(np.mean(g1)), float(np.mean(g2))],
        "Assumptions": {"Normality": [norm1, norm2], "Equal Variance": lev},
        "Statistics": {"T Statistic": float(stat), "P-Value": float(p)},
        "Effect Size": effect,
        "Significant": bool(sig),
        "Interpretation": (
            f"{names[0]} (M={np.mean(g1):.2f}) vs {names[1]} (M={np.mean(g2):.2f}): "
            f"{'significant' if sig else 'no significant'} difference (p={p:.4f}), "
            f"{effect['magnitude']} effect (d={effect['value']:.3f})."
        ),
    }


@tool
@enforce_schema
def run_mann_whitney(value_col: str, group_col: str):
    """
    Mann-Whitney U test pipeline (nonparametric two-group comparison).
    Use when normality is violated. Reports rank-biserial r as effect size.
    """
    _require_df()
    data = _df[[value_col, group_col]].dropna()
    groups, err = _validate_groups(
        data, value_col, group_col, min_groups=2, max_groups=2)
    if err:
        return err

    names = list(groups.index)
    g1, g2 = groups.iloc[0], groups.iloc[1]
    norm1, norm2 = _shapiro(pd.Series(g1)), _shapiro(pd.Series(g2))

    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    n1, n2 = len(g1), len(g2)
    rb_r = 1 - (2 * stat) / (n1 * n2)
    effect = {
        "name": "Rank-biserial r",
        "value": float(rb_r),
        "magnitude": _interpret_effect(rb_r, [0.1, 0.3, 0.5],
                                       ["negligible", "small", "medium", "large"])
    }

    sig = p < 0.05
    return {
        "Test Used": "Mann-Whitney U",
        "Groups": names,
        "Group Sizes": [len(g1), len(g2)],
        "Medians": [float(np.median(g1)), float(np.median(g2))],
        "Assumptions": {"Normality G1": norm1, "Normality G2": norm2},
        "Statistics": {"U Statistic": float(stat), "P-Value": float(p)},
        "Effect Size": effect,
        "Significant": bool(sig),
        "Interpretation": (
            f"{names[0]} (Mdn={np.median(g1):.2f}) vs {names[1]} (Mdn={np.median(g2):.2f}): "
            f"{'significant' if sig else 'no significant'} difference (p={p:.4f}), "
            f"{effect['magnitude']} effect (r={rb_r:.3f})."
        ),
    }


# ── PAIRED PIPELINE ────────────────────────────────────────────────────

@tool
@enforce_schema
def run_ttest_paired(col_a: str, col_b: str):
    """
    Paired comparison pipeline. Checks normality of differences, then
    auto-selects paired t-test or Wilcoxon signed-rank. Reports Cohen's dz.
    """
    prep = _prepare_stats(col_a, [col_b], "ttest_paired")
    if prep["status"] == "error":
        return prep
    _, _, data = prep["data"]
    if len(data) < 3:
        return _err("Need at least 3 pairs")

    diff = (data[col_a] - data[col_b]).values
    norm = _shapiro(pd.Series(diff))

    if norm["normal"]:
        test, reason = "Paired t-test", "Differences normal"
        stat, p = stats.ttest_rel(data[col_a], data[col_b])
        stat_name = "t_statistic"
    else:
        test, reason = "Wilcoxon signed-rank", "Differences non-normal -> nonparametric"
        stat, p = stats.wilcoxon(data[col_a], data[col_b])
        stat_name = "w_statistic"

    sd = np.std(diff, ddof=1)
    dz = float(np.mean(diff) / sd) if sd else None
    sig = p < 0.05
    return {
        "Test Used": test,
        "Reason": reason,
        "Variables": [col_a, col_b],
        "N Pairs": len(data),
        "Mean Diff": float(np.mean(diff)),
        "Assumptions": {"Normality of Differences": norm},
        "Statistics": {stat_name: float(stat), "P-Value": float(p)},
        "Effect Size": {"name": "Cohen's dz", "value": dz,
                        "magnitude": _interpret_effect(dz or 0, [0.2, 0.5, 0.8],
                                                       ["negligible", "small", "medium", "large"])},
        "Significant": bool(sig),
        "Interpretation": (
            f"Mean difference {np.mean(diff):.3f}: "
            f"{'significant' if sig else 'no significant'} change (p={p:.4f})."
        ),
    }


@tool
@enforce_schema
def run_ttest_onesample(column: str, test_value: float = 0):
    """One-sample test pipeline. Auto-selects t-test or Wilcoxon vs a fixed value."""
    _require_df()
    if column not in _df.columns:
        return _err(f"Column '{column}' not found")
    series = _df[column].dropna()
    if len(series) < 3:
        return _err("Need at least 3 samples")

    norm = _shapiro(series)
    if norm["normal"]:
        test, reason = "One-sample t-test", "Data normal"
        stat, p = stats.ttest_1samp(series, test_value)
        stat_name = "t_statistic"
    else:
        test, reason = "Wilcoxon signed-rank", "Non-normal -> nonparametric"
        stat, p = stats.wilcoxon(series - test_value)
        stat_name = "w_statistic"

    sig = p < 0.05
    return {
        "Test Used": test, "Reason": reason, "Column": column, "Test Value": test_value,
        "Sample Mean": float(series.mean()), "N": len(series),
        "Assumptions": {"Normality": norm},
        "Statistics": {stat_name: float(stat), "P-Value": float(p)},
        "Significant": bool(sig),
        "Interpretation": (
            f"Sample mean {series.mean():.3f} vs {test_value}: "
            f"{'significant' if sig else 'no significant'} difference (p={p:.4f})."
        ),
    }


# ── MULTI-GROUP PIPELINE (ANOVA / KRUSKAL) ─────────────────────────────

@tool
@enforce_schema
def run_anova(value_col: str, group_col: str):
    """
    Multi-group mean comparison pipeline (3+ groups). Checks normality of
    every group + equal variance, then auto-selects one-way ANOVA or
    Kruskal-Wallis. Reports eta-squared / epsilon-squared and interprets.
    """
    _require_df()
    data = _df[[value_col, group_col]].dropna()
    groups, err = _validate_groups(
        data, value_col, group_col, min_groups=3, min_size=2)
    if err:
        return err

    normality = {str(name): _shapiro(pd.Series(vals))
                 for name, vals in groups.items()}
    all_normal = all(r["normal"] for r in normality.values())
    lev = _levene(list(groups.values))
    n_total = sum(len(g) for g in groups.values)
    k = len(groups)

    if all_normal and lev["equal_variance"]:
        test, reason = "One-way ANOVA", "Assumptions met"
        stat, p = stats.f_oneway(*groups.values)
        stat_name = "f_statistic"
        effect = _eta_squared(groups)
    else:
        test = "Kruskal-Wallis H"
        reason = "Normality/variance assumptions violated -> nonparametric"
        stat, p = stats.kruskal(*groups.values)
        stat_name = "h_statistic"
        effect = _epsilon_squared(stat, n_total, k)

    sig = p < 0.05
    return {
        "Test Used": test,
        "Reason": reason,
        "# of groups": k,
        "Group Sizes": {str(n): len(v) for n, v in groups.items()},
        "Group Means": {str(n): float(np.mean(v)) for n, v in groups.items()},
        "Assumptions": {"Normality": normality, "Equal Variance": lev},
        "Statistics": {stat_name: float(stat), "P-Value": float(p)},
        "Effect Size": effect,
        "Significant": bool(sig),
        "Interpretation": (
            f"{'Significant' if sig else 'No significant'} differences across {k} groups "
            f"(p={p:.4f}), {effect['magnitude']} effect."
        ),
    }


# ── CATEGORICAL ASSOCIATION PIPELINE ───────────────────────────────────

@tool
@enforce_schema
def run_chi_squared(col_a: str, col_b: str):
    """
    Categorical association pipeline. Builds the contingency table, checks
    the expected-frequency assumption, and auto-falls back to Fisher's Exact
    for 2x2 tables when cells are sparse. Reports Cramér's V.
    """
    prep = _prepare_stats(col_a, [col_b], "chi2")
    if prep["status"] == "error":
        return prep
    _, _, data = prep["data"]

    table = pd.crosstab(data[col_a], data[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    small_cells = int((expected < 5).sum())
    assumption_ok = small_cells == 0

    if not assumption_ok and table.shape == (2, 2):
        odds, p = stats.fisher_exact(table.values)
        test, reason = "Fisher's Exact", "Sparse cells in 2x2 -> exact test"
        stat_block = {"odds_ratio": float(odds), "p_value": float(p)}
    else:
        test = "Chi-squared (independence)"
        reason = ("Assumption met" if assumption_ok
                  else f"{small_cells} cells with expected<5 (interpret with caution)")
        stat_block = {"chi2_statistic": float(chi2), "p_value": float(p),
                      "degrees_of_freedom": int(dof)}

    effect = _cramers_v(table, chi2)
    sig = p < 0.05
    return {
        "Test Used": test,
        "Reason": reason,
        "Variables": [col_a, col_b],
        "Assumptions": {"Cells Expected < 5": small_cells, "Assumption Met": assumption_ok},
        "Statistics": stat_block,
        "Effect Size": effect,
        "Significant": bool(sig),
        "Interpretation": (
            f"{'Significant' if sig else 'No significant'} association (p={p:.4f}), "
            f"{effect['magnitude']} strength."
        ),
    }


@tool
@enforce_schema
def run_chi_squared_gof(column: str, expected_freq: Dict[str, float] = None):
    """Chi-squared goodness-of-fit pipeline (observed vs expected distribution)."""
    _require_df()
    if column not in _df.columns:
        return _err(f"Column '{column}' not found")

    observed = _df[column].value_counts().sort_index()
    if expected_freq is None:
        expected = np.array([len(_df) / len(observed)] * len(observed))
    else:
        expected = np.array([expected_freq.get(str(cat), 0)
                            for cat in observed.index], dtype=float)
        if expected.sum() == 0:
            return _err("Provided expected frequencies sum to zero")
        expected = expected / expected.sum() * observed.sum()

    small_cells = int((expected < 5).sum())
    chi2, p = stats.chisquare(observed, expected)
    sig = p < 0.05
    return {
        "Test Used": "Chi-squared (goodness-of-fit)",
        "Column": column,
        "# of categories": len(observed),
        "Assumptions": {"cells_expected_lt_5": small_cells},
        "Statistics": {"chi2_statistic": float(chi2), "p_value": float(p)},
        "Significant": bool(sig),
        "Interpretation": (
            f"Observed distribution {'differs' if sig else 'does not differ'} "
            f"from expected (p={p:.4f})."
        ),
    }


# ── REGRESSION PIPELINES ───────────────────────────────────────────────

@tool
@enforce_schema
def linear_regression(y_col: str, x_col: List[str]):
    """
    OLS regression pipeline. Normalizes predictors, guards against
    multicollinearity, fits the model, and returns VIF, residual normality
    (Shapiro), and heteroscedasticity (Breusch-Pagan) diagnostics.
    """
    prep = _prepare_stats(y_col, x_col, "linear", normalize=True)
    if prep["status"] == "error":
        return prep
    X, y, _ = prep["data"]

    model = sm.OLS(y, sm.add_constant(X)).fit()
    resid = model.resid

    diagnostics = {"VIF": _vif_scores(X) if len(x_col) > 1 else "N/A (single predictor)",
                   "Residual Normality": _shapiro(pd.Series(resid))}
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp = het_breuschpagan(resid, sm.add_constant(X))
        diagnostics["Heteroscedasticity"] = {"LM P-Value": float(bp[1]),
                                             "Homoscedastic": bool(bp[1] > 0.05)}
    except Exception as e:
        diagnostics["Heteroscedasticity"] = f"unavailable: {e}"

    return _ok({
        "R²": float(model.rsquared),
        "Adj R²": float(model.rsquared_adj),
        "F stat": float(model.fvalue),
        "F P-value": float(model.f_pvalue),
        "Coefficients": model.params.to_dict(),
        "P-Values": model.pvalues.to_dict(),
        "Significant": bool(model.f_pvalue < 0.05),
        "Interpretation": (
            f"Model explains {model.rsquared:.1%} of variance "
            f"({'significant' if model.f_pvalue < 0.05 else 'not significant'}, "
            f"F p={model.f_pvalue:.4f})."
        ),
    }, diagnostics=diagnostics)


@tool
@enforce_schema
def logistic_regression(y_col: str, x_col: List[str]):
    """
    Logistic regression pipeline. Validates/encodes a binary target,
    normalizes predictors, guards multicollinearity, and returns odds
    ratios, VIF, and class balance.
    """
    prep = _prepare_stats(y_col, x_col, "logistic", normalize=True)
    if prep["status"] == "error":
        return prep
    X, y, _ = prep["data"]

    model = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
    diagnostics = {"VIF": _vif_scores(X) if len(x_col) > 1 else "N/A (single predictor)",
                   "Class Balance": y.value_counts(normalize=True).round(3).to_dict()}

    return _ok({
        "LLF": float(model.llf),
        "AIC": float(model.aic),
        "Pseudo R²": float(model.prsquared),
        "Coefficients": model.params.to_dict(),
        "Odds Ratios": np.exp(model.params).round(4).to_dict(),
        "P-Values": model.pvalues.to_dict(),
        "Interpretation": f"Pseudo R²={model.prsquared:.3f}, AIC={model.aic:.1f}.",
    }, diagnostics=diagnostics)


# ── TREND & CURVE PIPELINES ────────────────────────────────────────────

@tool
@enforce_schema
def detect_trends(x_col: str, y_col: str):
    """Mann-Kendall monotonic trend pipeline (sorts by x, tests y for trend)."""
    _require_df()
    d = _df[[x_col, y_col]].dropna().sort_values(x_col)
    if len(d) < 3:
        return _err("Not enough data (min 3 rows)")

    res = mk.original_test(d[y_col].to_numpy())
    return {
        "Trend": res.trend,
        "Significant": bool(res.p < 0.05),
        "P value": float(res.p),
        "Tau": float(res.Tau),
        "Slope": float(res.slope),
        "Interpretation": (
            f"{'Significant' if res.p < 0.05 else 'No significant'} "
            f"{res.trend} monotonic trend (tau={res.Tau:.3f}, p={res.p:.4f})."
        ),
    }


@tool
@enforce_schema
def fit_curves(col_a: str, col_b: str, degree: int = 4):
    """
    Polynomial curve-fitting pipeline. Fits degrees 1..N, compares by
    adjusted R², and returns the best model plus a shape interpretation.
    """
    _require_df()
    if col_a not in _df.columns or col_b not in _df.columns:
        return _err("Columns not found")

    data = _df[[col_a, col_b]].dropna().sort_values(col_a)
    x, y = data[col_a].values.astype(float), data[col_b].values.astype(float)
    n = len(x)
    if n < 5:
        return _err("Need at least 5 points to fit curves")

    xs = (x - x.mean()) / (x.std() or 1)
    results = {}
    for d in range(1, degree + 1):
        if n <= d + 1:
            break
        coeffs = np.polyfit(xs, y, d)
        y_pred = np.poly1d(coeffs)(xs)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - d - 1) if n - d - 1 > 0 else r2
        results[d] = {"coefficients": coeffs.tolist(
        ), "r2": float(r2), "adj_r2": float(adj_r2)}

    if not results:
        return _err("Not enough data to fit any model")

    best_deg, best = max(results.items(), key=lambda kv: kv[1]["adj_r2"])
    shape = {1: "linear", 2: "quadratic (single bend)", 3: "cubic (S-shaped)"}.get(
        best_deg, f"degree-{best_deg} (multiple bends)")

    return {
        "Variable X": col_a,
        "Variable Y": col_b,
        "Models": results,
        "Best Model": {"degree": best_deg, "r2": best["r2"], "adj_r2": best["adj_r2"]},
        "Interpretation": (
            f"Best fit is a {shape} relationship "
            f"(adj R²={best['adj_r2']:.4f}, R²={best['r2']:.4f})."
        ),
    }
