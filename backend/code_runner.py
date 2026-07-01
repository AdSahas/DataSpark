from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass

import pandas as pd

from scipy import stats

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

OpType = Literal[
    "describe",
    "correlation",
    "t_test_ind",
    "t_test_paired",
    "chi_square",
    "mann_whitney",
    "linear_regression",
    "logistic_regression",
    "kmeans",
    "pca"
]


@dataclass
class Operation:
    op: OpType
    a: Optional[str] = None
    b: Optional[str] = None
    features: Optional[List[str]] = None
    target: Optional[str] = None
    k: Optional[int] = None


class CodeRunner:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cols = set(df.columns)

    def validate(self, op: Operation) -> None:
        fields = [op.a, op.b, op.target]

        if op.features:
            fields += op.features

        for field in fields:
            if field and field not in self.cols:
                raise ValueError(
                    f"Column '{field}' not found in DataFrame. Available columns: {self.cols}")

    def _execute(self, op: Operation):
        self.validate(op)

        if op.op == "describe":
            return self.df.describe(include="all").to_dict()

        if op.op == "correlation":
            return self.df.corr(numeric_only=True).to_dict()

        if op.op == "t_test_ind":
            stat, p = stats.ttest_ind(
                self.df[op.a], self.df[op.b], nan_policy='omit')
            return {"statistic": stat, "p_value": p}

        if op.op == "t_test_paired":
            stat, p = stats.ttest_rel(
                self.df[op.a], self.df[op.b], nan_policy='omit')
            return {"statistic": stat, "p_value": p}

        if op.op == "mann_whitney":
            stat, p = stats.mannwhitneyu(
                self.df[op.a], self.df[op.b], alternative='two-sided')
            return {"statistic": float(stat), "p_value": p}

        if op.op == "chi-square":
            table = pd.crosstab(self.df[op.a], self.df[op.b])
            stat, p, dof, expected = stats.chi2_contingency(table)
            return {"statistic": float(stat), "p_value": p, "degrees_of_freedom": dof, "expected_freq": expected.tolist()}

        if op.op == "linear_regression":
            model = LinearRegression()
            X = self.df[op.features]
            y = self.df[op.target]
            model.fit(X, y)
            return {"coefficients": model.coef_.tolist(), "intercept": model.intercept_, "r_squared": model.score(X, y)}

        if op.op == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
            X = self.df[op.features]
            y = self.df[op.target]
            model.fit(X, y)
            return {"coefficients": model.coef_.tolist(), "intercept": model.intercept_, "accuracy": model.score(X, y)}

        if op.op == "kmeans":
            model = KMeans(n_clusters=op.k)
            X = self.df[op.features]
            model.fit(X)
            return {"labels": model.labels_.tolist(), "inertia": model.inertia_, "centers": model.cluster_centers_.tolist()
                    }

        if op.op == "pca":
            model = PCA(n_components=op.k)
            X = self.df[op.features]
            model.fit(X)
            return {"explained_variance_ratio": model.explained_variance_ratio_.tolist(), "components": model.components_.tolist()}

        else:
            raise ValueError(f"Unsupported operation: {op.op}")

    def run(self, ops: List[Operation]) -> List[Any]:
        return [self._execute(op) for op in ops]
