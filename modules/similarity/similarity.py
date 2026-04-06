import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Callable


class SimilarityAnalyzer:
    """
    Computes pairwise similarity or distance between every pair of rows
    in the numeric portion of the DataFrame.

    This is a terminal analysis module — it produces a pairwise comparison
    table and does not transform the original data. It is registered via
    register_analysis() and runs through pipeline.analyze().

    Supported metrics
    -----------------
    euclidean  : Straight-line distance (L2 norm). Lower means more similar.
    cosine     : Cosine of the angle between two vectors. 1 = identical, -1 = opposite.
    manhattan  : City-block distance (L1 norm). Lower means more similar.
    minkowski  : Generalised Lp norm (p configurable, default 3).
    pearson    : Linear correlation as a similarity score.

    The returned DataFrame contains one row per pair with columns:
        point_i          – index of the anchor row
        point_j          – index of the comparison row
        similarity_score – computed metric value
        metric           – name of the metric used

    Parameters
    ----------
    method : str
        One of the keys in METHODS.
    columns : list[str] | None
        Numeric columns to use. If None all numeric columns are used.
    max_pairs : int
        Upper limit on pairs returned to avoid memory blow-up on large datasets.
    p : int
        Power parameter for Minkowski distance. Ignored by other metrics.
    """

    METHODS = {
        "Euclidean Distance": "euclidean",
        "Cosine Similarity": "cosine",
        "Manhattan Distance": "manhattan",
        "Minkowski Distance": "minkowski",
        "Pearson Correlation": "pearson",
    }

    def __init__(
        self,
        method: str = "euclidean",
        columns: list[str] | None = None,
        max_pairs: int = 500,
        p: int = 3,
    ) -> None:
        self.method = method
        self.columns = columns
        self.max_pairs = max_pairs
        self.p = p
        self.report_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes pairwise similarity scores and returns a results DataFrame.

        The original data is not modified. The returned DataFrame contains
        one row per pair — it is a summary result, not a transformed version
        of the input. This method is intentionally terminal.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Pairwise similarity results with columns:
            point_i, point_j, similarity_score, metric.
        """
        numeric_cols = self._select_numeric_columns(df)

        if not numeric_cols:
            self.report_ = {
                "method": self.method,
                "status": "No numeric columns available.",
                "pairs_computed": 0,
            }
            return df

        matrix = self._build_matrix(df, numeric_cols)
        pairs_df = self._compute_pairs(matrix, df.index.tolist())

        self.report_ = {
            "method": self.method,
            "columns_used": ", ".join(numeric_cols),
            "num_columns": len(numeric_cols),
            "rows_analysed": len(matrix),
            "pairs_computed": len(pairs_df),
            "max_pairs_limit": self.max_pairs,
            "score_min": round(float(pairs_df["similarity_score"].min()), 6),
            "score_max": round(float(pairs_df["similarity_score"].max()), 6),
            "score_mean": round(float(pairs_df["similarity_score"].mean()), 6),
            **({"p": self.p} if self.method == "minkowski" else {}),
        }

        return pairs_df

    def get_report(self) -> dict:
        return self.report_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Returns the numeric columns requested, falling back to all numeric."""
        all_numeric = df.select_dtypes(include="number").columns.tolist()
        if self.columns:
            valid = [c for c in self.columns if c in all_numeric]
            return valid if valid else all_numeric
        return all_numeric

    def _build_matrix(self, df: pd.DataFrame, cols: list[str]) -> NDArray:
        """
        Extracts a clean 2D float array from the selected columns.
        Rows with any NaN are dropped so metrics do not crash.
        """
        return df[cols].dropna().to_numpy(dtype=float)

    def _compute_pairs(
        self, matrix: NDArray, original_index: list
    ) -> pd.DataFrame:
        """
        Iterates over all (i, j) pairs with i < j, computes the chosen
        metric, and returns a DataFrame limited to max_pairs rows.
        """
        records: list[dict] = []
        metric_fn = self._get_metric_fn()
        n = len(matrix)

        for i in range(n):
            for j in range(i + 1, n):
                score = metric_fn(matrix[i], matrix[j])
                records.append(
                    {
                        "point_i": original_index[i] if i < len(original_index) else i,
                        "point_j": original_index[j] if j < len(original_index) else j,
                        "similarity_score": round(score, 6),
                        "metric": self.method,
                    }
                )
                if len(records) >= self.max_pairs:
                    break
            if len(records) >= self.max_pairs:
                break

        return pd.DataFrame(records)

    def _get_metric_fn(self) -> Callable[[NDArray, NDArray], float]:
        """Returns the metric function corresponding to self.method."""
        dispatch: dict[str, Callable[[NDArray, NDArray], float]] = {
            "euclidean": self._euclidean,
            "cosine": self._cosine,
            "manhattan": self._manhattan,
            "minkowski": self._minkowski,
            "pearson": self._pearson,
        }
        fn = dispatch.get(self.method)
        if fn is None:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Valid options: {list(dispatch.keys())}"
            )
        return fn

    # ------------------------------------------------------------------
    # Distance and similarity implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean(a: NDArray, b: NDArray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))

    @staticmethod
    def _cosine(a: NDArray, b: NDArray) -> float:
        """Returns 0.0 for zero vectors to avoid division by zero."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _manhattan(a: NDArray, b: NDArray) -> float:
        return float(np.sum(np.abs(a - b)))

    def _minkowski(self, a: NDArray, b: NDArray) -> float:
        p = max(self.p, 1)
        return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))

    @staticmethod
    def _pearson(a: NDArray, b: NDArray) -> float:
        """Returns 0.0 when either vector is constant (undefined correlation)."""
        if a.std() == 0.0 or b.std() == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
