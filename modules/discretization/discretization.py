import numpy as np
import pandas as pd
from math import log2


class Discretization:
    """
    Discretizes continuous numeric attributes into labeled intervals.

    Supports two methods:
        - Entropy-based: supervised top-down recursive split using information gain.
          Requires a class label column in the DataFrame.
        - 3-4-5 Rule: unsupervised natural partitioning based on the most
          significant digit of the data range.

    METHODS registry allows the frontend to dynamically discover available options
    without hardcoding them.

    Parameters
    ----------
    method : str
        Key from METHODS dict. Defaults to 'entropy'.
    column : str
        The numeric column to discretize.
    class_column : str, optional
        Required only for entropy-based method.
    max_intervals : int
        Stopping criterion for entropy method. Defaults to 5.
    gain_threshold : float
        Minimum information gain to continue splitting. Defaults to 0.01.
    """

    METHODS = {
        "Entropy-Based (Information Gain)": "entropy",
        "3-4-5 Rule (Natural Partitioning)": "partition_345",
    }

    def __init__(
        self,
        method: str = "entropy",
        column: str = None,
        class_column: str = None,
        max_intervals: int = 5,
        gain_threshold: float = 0.01,
    ):
        self.method = method
        self.column = column
        self.class_column = class_column
        self.max_intervals = max_intervals
        self.gain_threshold = gain_threshold

        self.boundaries_ = []
        self.report_ = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies discretization to the specified column and returns a new DataFrame
        with an additional column containing the discretized interval labels.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Original DataFrame with a new column '{column}_discretized'.
        """
        if self.column is None or self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")

        df = df.copy()
        series = df[self.column].dropna()

        if self.method == "entropy":
            if self.class_column is None or self.class_column not in df.columns:
                raise ValueError("Entropy method requires a valid 'class_column'.")
            boundaries = self._entropy_split(
                series.values,
                df.loc[series.index, self.class_column].values,
                depth=0,
            )
        elif self.method == "partition_345":
            boundaries = self._partition_345(series.values)
        else:
            raise ValueError(f"Unknown method '{self.method}'.")

        boundaries = sorted(set(boundaries))
        self.boundaries_ = boundaries

        bins = [-np.inf] + boundaries + [np.inf]
        labels = [f"({bins[i]:.2f}, {bins[i+1]:.2f}]" for i in range(len(bins) - 1)]
        labels[0] = f"(-inf, {boundaries[0]:.2f}]"
        labels[-1] = f"({boundaries[-1]:.2f}, inf)"

        df[f"{self.column}_discretized"] = pd.cut(
            df[self.column], bins=bins, labels=labels, include_lowest=True
        )

        self.report_ = {
            "method": self.method,
            "column": self.column,
            "boundaries": boundaries,
            "num_intervals": len(labels),
        }

        return df

    def get_report(self) -> dict:
        """Returns a summary of the discretization that was applied."""
        return self.report_

    # ------------------------------------------------------------------
    # Entropy-Based Discretization
    # ------------------------------------------------------------------

    def _entropy(self, class_labels: np.ndarray) -> float:
        n = len(class_labels)
        if n == 0:
            return 0.0
        _, counts = np.unique(class_labels, return_counts=True)
        probs = counts / n
        return -sum(p * log2(p) for p in probs if p > 0)

    def _weighted_entropy(
        self,
        left_labels: np.ndarray,
        right_labels: np.ndarray,
    ) -> float:
        total = len(left_labels) + len(right_labels)
        return (len(left_labels) / total) * self._entropy(left_labels) + (
            len(right_labels) / total
        ) * self._entropy(right_labels)

    def _best_split(
        self, values: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        sorted_idx = np.argsort(values)
        values, labels = values[sorted_idx], labels[sorted_idx]

        best_gain, best_threshold = -np.inf, None

        for i in range(1, len(values)):
            if values[i] == values[i - 1]:
                continue
            threshold = (values[i] + values[i - 1]) / 2
            gain = self._entropy(labels) - self._weighted_entropy(
                labels[:i], labels[i:]
            )
            if gain > best_gain:
                best_gain, best_threshold = gain, threshold

        return best_threshold, best_gain

    def _entropy_split(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        depth: int,
        boundaries: list = None,
    ) -> list:
        if boundaries is None:
            boundaries = []

        if len(np.unique(labels)) <= 1:
            return boundaries

        intervals_so_far = len(boundaries) + 1
        if intervals_so_far >= self.max_intervals:
            return boundaries

        threshold, gain = self._best_split(values, labels)

        if threshold is None or gain < self.gain_threshold:
            return boundaries

        boundaries.append(threshold)

        left_mask = values <= threshold
        right_mask = ~left_mask

        self._entropy_split(values[left_mask], labels[left_mask], depth + 1, boundaries)
        self._entropy_split(values[right_mask], labels[right_mask], depth + 1, boundaries)

        return boundaries

    # ------------------------------------------------------------------
    # 3-4-5 Rule (Segmentation by Natural Partitioning)
    # ------------------------------------------------------------------

    def _round_down_to_msd(self, value: float, msd: float) -> float:
        return np.floor(value / msd) * msd

    def _round_up_to_msd(self, value: float, msd: float) -> float:
        return np.ceil(value / msd) * msd

    def _get_msd(self, low: float, high: float) -> float:
        magnitude = 10 ** np.floor(np.log10(abs(high - low))) if high != low else 1
        return magnitude

    def _num_partitions(self, distinct_count: int) -> int:
        if distinct_count in (3, 6, 7, 9):
            return 3
        elif distinct_count in (2, 4, 8):
            return 4
        elif distinct_count in (1, 5, 10):
            return 5
        else:
            return 3

    def _partition_345(self, values: np.ndarray) -> list:
        low_pct = np.percentile(values, 5)
        high_pct = np.percentile(values, 95)
        minimum, maximum = values.min(), values.max()

        msd = self._get_msd(low_pct, high_pct)

        low_rounded = self._round_down_to_msd(low_pct, msd)
        high_rounded = self._round_up_to_msd(high_pct, msd)

        distinct = round((high_rounded - low_rounded) / msd)
        n_parts = self._num_partitions(distinct)

        step = (high_rounded - low_rounded) / n_parts

        boundaries = [low_rounded + step * i for i in range(1, n_parts)]

        left_boundary = self._round_down_to_msd(minimum, msd / 10)
        right_boundary = self._round_up_to_msd(maximum, msd / 10)

        if left_boundary < low_rounded:
            boundaries = [low_rounded] + boundaries
        if right_boundary > high_rounded:
            boundaries = boundaries + [high_rounded]

        return boundaries
