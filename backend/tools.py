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


_df: pd.DataFrame = None


def set_dataframe(df: pd.DataFrame):
    global _df, _df_version
    _df = df


def _require_df():
    if _df is None:
        raise RuntimeError("No dataframe loaded. Call set_dataframe() first.")


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


@tool
@enforce_schema
def compare_two_groups(value_col: str, group_col: str) -> ResultSchema:
    """
    Compares a continuous numeric variable across exactly two distinct groups.
    Automatically checks normality (Shapiro-Wilk) and variance equality (Levene), 
    then runs the mathematically optimal test: Student's t, Welch's t, or Mann-Whitney U.

    Args:
        value_col: The continuous/numeric column to measure (e.g., 'mean_area').
        group_col: The categorical/binary column defining the 2 groups (e.g., 'diagnosis').
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
    lev = _levene([g1, g2])

    if both_normal:
        if lev["equal_variance"]:
            test, reason = "Student's t-test", "Both groups normal + equal variances (Parametric)"
            stat, p = stats.ttest_ind(g1, g2, equal_var=True)
        else:
            test, reason = "Welch's t-test", "Both groups normal but unequal variances -> Welch correction"
            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        effect = _cohens_d(g1, g2)
        stat_name = "T Statistic"
        metric_name, metric_vals = "Means", [
            float(np.mean(g1)), float(np.mean(g2))]
        effect_str = f"{effect['magnitude']} effect (d={effect['value']:.3f})"
    else:
        test, reason = "Mann-Whitney U", "Normality assumption violated -> Swapped to Nonparametric"
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        n1, n2 = len(g1), len(g2)
        rb_r = 1 - (2 * stat) / (n1 * n2)
        effect = {
            "name": "Rank-biserial r",
            "value": float(rb_r),
            "magnitude": _interpret_effect(rb_r, [0.1, 0.3, 0.5], ["negligible", "small", "medium", "large"])
        }
        stat_name = "U Statistic"
        metric_name, metric_vals = "Medians", [
            float(np.median(g1)), float(np.median(g2))]
        effect_str = f"{effect['magnitude']} effect (r={effect['value']:.3f})"

    sig = p < 0.05
    return _ok({
        "Test Used": test,
        "Routing Logic": reason,
        "Groups": names,
        "Group Sizes": [len(g1), len(g2)],
        metric_name: metric_vals,
        "Statistics": {stat_name: float(stat), "P-Value": float(p)},
        "Effect Size": effect,
        "Significant": bool(sig),
        "Interpretation": f"{names[0]} vs {names[1]} ({metric_name.lower()}): {'significant' if sig else 'no significant'} difference (p={p:.4f}), matching a {effect_str}."
    }, diagnostics={"normality": [norm1, norm2], "equal_variance": lev})


@tool
@enforce_schema
def compare_multi_groups(value_col: str, group_col: str) -> ResultSchema:
    """
    Compares a continuous numeric variable across 3 or more distinct groups.
    Automatically verifies parametric assumptions across all groups, then routes
    to One-way ANOVA or a nonparametric Kruskal-Wallis H test.

    Args:
        value_col: The continuous/numeric metrics column to analyze.
        group_col: The categorical column containing 3+ group identifiers.
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
        test, reason = "One-way ANOVA", "Parametric assumptions met across all groups."
        stat, p = stats.f_oneway(*groups.values)
        stat_name = "F Statistic"
        effect = _eta_squared(groups)
    else:
        test, reason = "Kruskal-Wallis H", "Normality or variance thresholds violated -> Nonparametric fallback."
        stat, p = stats.kruskal(*groups.values)
        stat_name = "H Statistic"
        effect = _epsilon_squared(stat, n_total, k)

    sig = p < 0.05
    return _ok({
        "Test Used": test,
        "Routing Logic": reason,
        "Number of Groups": k,
        "Group Sizes": {str(n): len(v) for n, v in groups.items()},
        "Group Means": {str(n): float(np.mean(v)) for n, v in groups.items()},
        "Statistics": {stat_name: float(stat), "P-Value": float(p)},
        "Effect Size": effect,
        "Significant": bool(sig),
        "Interpretation": f"{'Significant' if sig else 'No significant'} global differences found across the {k} groups (p={p:.4f}), yielding a {effect['magnitude']} effect size."
    }, diagnostics={"normality": normality, "equal_variance": lev})


@tool
@enforce_schema
def compare_categorical_association(col_a: str, col_b: str) -> ResultSchema:
    """
    Analyzes the statistical association between two discrete categorical columns.
    Builds a contingency matrix and safely drops back to Fisher's Exact Test 
    if cell counts are sparse inside a 2x2 grid.
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
        test, reason = "Fisher's Exact Test", "Sparse frequencies detected inside a 2x2 table grid."
        stat_block = {"Odds Ratio": float(odds), "P-Value": float(p)}
    else:
        test = "Chi-squared Test of Independence"
        reason = "Standard structural cell criteria satisfied." if assumption_ok else f"Warning: {small_cells} matrix blocks with expected values < 5."
        stat_block = {"Chi2 Statistic": float(
            chi2), "P-Value": float(p), "Degrees of Freedom": int(dof)}

    effect = _cramers_v(table, chi2)
    sig = p < 0.05

    return _ok({
        "Test Used": test,
        "Routing Logic": reason,
        "Variables": [col_a, col_b],
        "Contingency Table": table.to_dict(orient="index"),
        "Statistics": stat_block,
        "Effect Size": effect,
        "Significant": bool(sig),
        "Interpretation": f"{'Significant' if sig else 'No significant'} association found between columns (p={p:.4f}, Cramér's V strength: {effect['magnitude']})."
    }, diagnostics={"cells_expected_under_5": small_cells, "assumptions_passed": assumption_ok})


@tool
@enforce_schema
def run_regression(y_col: str, x_cols: List[str]) -> ResultSchema:
    """
    Unified regression pipeline. Automatically detects the type of target variable (y_col)
    and routes to the correct model structure:
    - Linear Regression (OLS): For continuous numeric target variables.
    - Logistic Regression: For binary categorical target variables.

    Automatically handles data preparation, feature normalization, multicollinearity checking,
    and calculates model diagnostics (VIF, residual normality, and homoscedasticity).

    Args:
        y_col: The target/dependent variable to predict.
        x_cols: List of predictor/independent variables.
    """
    _require_df()

    if y_col not in _df.columns:
        return _err(f"Target column '{y_col}' not found")

    y_series = _df[y_col].dropna()
    distinct_count = y_series.nunique()

    if distinct_count == 2:
        test_type = "logistic"
        reason = "Target variable is binary (categorical/discrete). Automated routing to Logistic Regression."
    elif pd.api.types.is_numeric_dtype(y_series):
        test_type = "linear"
        reason = "Target variable is continuous numeric data. Automated routing to Linear Regression (OLS)."
    else:
        return _err(f"Target column '{y_col}' must be either numeric (for Linear) or binary (for Logistic). Found type: {y_series.dtype}")

    prep = _prepare_stats(y_col, x_cols, test_type, normalize=True)
    if prep["status"] == "error":
        return prep
    X, y, _ = prep["data"]

    if test_type == "linear":
        model = sm.OLS(y, sm.add_constant(X)).fit()
        resid = model.resid

        diagnostics = {
            "Model Type Determined": "Linear (OLS)",
            "VIF": _vif_scores(X) if len(x_cols) > 1 else "N/A (single predictor)",
            "Residual Normality": _shapiro(pd.Series(resid))
        }
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp = het_breuschpagan(resid, sm.add_constant(X))
            diagnostics["Heteroscedasticity"] = {
                "LM P-Value": float(bp[1]), "Homoscedastic": bool(bp[1] > 0.05)}
        except Exception as e:
            diagnostics["Heteroscedasticity"] = f"Unavailable: {e}"

        return _ok({
            "Test Used": "Linear Regression (OLS)",
            "Routing Logic": reason,
            "R²": float(model.rsquared),
            "Adj R²": float(model.rsquared_adj),
            "F stat": float(model.fvalue),
            "F P-value": float(model.f_pvalue),
            "Coefficients": model.params.to_dict(),
            "P-Values": model.pvalues.to_dict(),
            "Significant": bool(model.f_pvalue < 0.05),
            "Interpretation": f"Model explains {model.rsquared:.1%} of variance ({'significant' if model.f_pvalue < 0.05 else 'not significant'}, F p={model.f_pvalue:.4f})."
        }, diagnostics=diagnostics)

    else:
        model = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
        diagnostics = {
            "Model Type Determined": "Logistic (Logit)",
            "VIF": _vif_scores(X) if len(x_cols) > 1 else "N/A (single predictor)",
            "Class Balance": y.value_counts(normalize=True).round(3).to_dict()
        }

        return _ok({
            "Test Used": "Logistic Regression",
            "Routing Logic": reason,
            "LLF": float(model.llf),
            "AIC": float(model.aic),
            "Pseudo R²": float(model.prsquared),
            "Coefficients": model.params.to_dict(),
            "Odds Ratios": np.exp(model.params).round(4).to_dict(),
            "P-Values": model.pvalues.to_dict(),
            "Interpretation": f"Logistic fit achieved. Pseudo R²={model.prsquared:.3f}, AIC={model.aic:.1f}."
        }, diagnostics=diagnostics)


@tool
@enforce_schema
def analyze_trend_and_curve(x_col: str, y_col: str, max_degree: int = 4) -> ResultSchema:
    """
    Unified trend and curve-fitting pipeline. Investigates the relationship between 
    an independent variable (x_col) and a dependent variable (y_col) by running:
    1. A Mann-Kendall monotonic trend test (to identify overall directional drift).
    2. A polynomial curve-fitting optimizer (to detect linear, quadratic, cubic, or complex shapes).

    Automatically handles sorting, handles missing data dropping, and selects the optimal curve model 
    based on the highest Adjusted R² score.
    """
    _require_df()

    if x_col not in _df.columns or y_col not in _df.columns:
        return _err(f"Columns '{x_col}' or '{y_col}' not found in dataset")

    data = _df[[x_col, y_col]].dropna().sort_values(x_col)
    x = data[x_col].values.astype(float)
    y = data[y_col].values.astype(float)
    n = len(x)

    if n < 5:
        return _err(f"Not enough data points (found {n}, min 5 required for reliable trend/curve fitting)")

    try:
        mk_res = mk.original_test(y)
        trend_analysis = {
            "Detected Trend": mk_res.trend,
            "Significant": bool(mk_res.p < 0.05),
            "P-Value": float(mk_res.p),
            "Tau (Direction Strength)": float(mk_res.Tau),
            "Theil-Sen Slope": float(mk_res.slope)
        }
    except Exception as e:
        trend_analysis = {"Error evaluating monotonic trend": str(e)}

    xs = (x - x.mean()) / (x.std() or 1)
    curve_results = {}

    for d in range(1, max_degree + 1):
        if n <= d + 1:
            break
        coeffs = np.polyfit(xs, y, d)
        y_pred = np.poly1d(coeffs)(xs)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - d - 1) if n - d - 1 > 0 else r2

        curve_results[d] = {
            "coefficients": coeffs.tolist(),
            "r2": float(r2),
            "adj_r2": float(adj_r2)
        }

    if not curve_results:
        return _err("Data structure insufficient to fit polynomial vectors.")

    best_deg, best = max(curve_results.items(), key=lambda kv: kv[1]["adj_r2"])
    shape_desc = {1: "linear", 2: "quadratic (single bend/parabola)", 3: "cubic (S-shaped curve)"}.get(
        best_deg, f"degree-{best_deg} polynomial (multiple complex bends)"
    )

    sig_trend = trend_analysis.get("Significant", False)
    trend_str = f"a significant {trend_analysis.get('Detected Trend')} monotonic trend (p={trend_analysis.get('P-Value'):.4f})" if sig_trend else "no consistent monotonic trend"

    interpretation = (
        f"Analysis shows {trend_str}. When evaluating structural shape, the data is best modeled by a "
        f"{shape_desc} relationship (Best Adjusted R² = {best['adj_r2']:.4f}, raw R² = {best['r2']:.4f})."
    )

    return _ok({
        "Test Purpose": "Trend Directionality & Geometric Curve Estimation",
        "Data Points Evaluated": n,
        "Monotonic Trend Summary": trend_analysis,
        "Tested Curve Models": curve_results,
        "Optimal Curve Choice": {
            "Best Degree": best_deg,
            "Mathematical Shape": shape_desc,
            "R²": best["r2"],
            "Adjusted R²": best["adj_r2"]
        },
        "Interpretation": interpretation
    })
