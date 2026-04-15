import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

class DataSmoother:
    """
    Applies noise reduction and smoothing techniques to a specific column.
    Supports Equal-Width/Depth Binning, Linear Regression, and K-Means Clustering.
    """

    RANDOM_SEED = 42
    N_INIT = 10

    # 1. Non-negotiable METHODS dict for the dashboard dropdown
    METHODS = {
        "Binning (Equal Width)": "bin_width",
        "Binning (Equal Depth)": "bin_depth",
        "Regression (Linear)": "regression",
        "Clustering (K-Means)": "clustering",
    }

    # 2. Parameters go in the constructor with type hints
    def __init__(
        self,
        method: str = "bin_width",
        column: str | None = None,
        num_bins: int = 3,
        smoothing_strategy: str = "mean",
        n_clusters: int = 3
    ):
        self.method = method
        self.column = column
        self.num_bins = num_bins
        self.smoothing_strategy = smoothing_strategy
        self.n_clusters = n_clusters
        self.report_: dict = {}

    # 3. Accept a DataFrame and return a DataFrame
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Always work on a copy
        df_copy = df.copy()

        if not self.column or self.column not in df_copy.columns:
            self.report_ = {"error": f"Column '{self.column}' not found or not specified."}
            return df_copy

        # Handle NaN values gracefully by isolating valid data
        valid_mask = df_copy[self.column].notna()
        if not valid_mask.any():
            self.report_ = {"error": "Column contains only NaN values."}
            return df_copy

        # Use .to_numpy() per guidelines
        values: NDArray = df_copy.loc[valid_mask, self.column].to_numpy()

        # Route to appropriate private helper method
        if self.method in ("bin_width", "bin_depth"):
            smoothed_values = self._apply_binning(values)
        elif self.method == "regression":
            smoothed_values = self._apply_regression(values)
        elif self.method == "clustering":
            smoothed_values = self._apply_clustering(values)
        else:
            smoothed_values = values

        # Apply smoothed values back to the copied dataframe
        df_copy.loc[valid_mask, self.column] = np.round(smoothed_values, 2)

        # Generate the flat report dict
        self.report_ = {
            "method": self.method,
            "column": self.column,
            "rows_modified": int(valid_mask.sum()),
            "parameter_value": self.num_bins if "bin" in self.method else self.n_clusters
        }

        return df_copy

    # 4. Return flat report
    def get_report(self) -> dict:
        return self.report_

    # --- Private Helper Methods ---

    def _apply_binning(self, values: NDArray) -> NDArray:
        """Sorts, bins, smooths, and restores original order."""
        sort_indices = np.argsort(values)
        sorted_vals = values[sort_indices]
        smoothed_sorted = np.zeros_like(sorted_vals, dtype=float)

        if self.method == "bin_depth":
            chunks = np.array_split(sorted_vals, self.num_bins)
            indices_chunks = np.array_split(np.arange(len(sorted_vals)), self.num_bins)
        else:
            # Equal width
            labels = pd.cut(sorted_vals, bins=self.num_bins, labels=False, include_lowest=True)
            chunks = [sorted_vals[labels == i] for i in range(self.num_bins)]
            indices_chunks = [np.where(labels == i)[0] for i in range(self.num_bins)]

        for chunk, idxs in zip(chunks, indices_chunks):
            if len(chunk) == 0:
                continue

            if self.smoothing_strategy == "median":
                val = np.median(chunk)
            elif self.smoothing_strategy == "mode":
                val = pd.Series(chunk).mode().iloc[0]
            else:
                val = np.mean(chunk)

            smoothed_sorted[idxs] = val

        # Map back to original unsorted order so we don't scramble the dataframe rows
        original_order = np.zeros_like(smoothed_sorted)
        original_order[sort_indices] = smoothed_sorted
        return original_order

    def _apply_regression(self, values: NDArray) -> NDArray:
        """Fits values to a linear trendline to remove localized noise."""
        X = np.arange(len(values)).reshape(-1, 1)
        model = LinearRegression().fit(X, values)
        return model.predict(X)

    def _apply_clustering(self, values: NDArray) -> NDArray:
        """Replaces values with the centroid of their assigned K-Means cluster."""
        X = values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.N_INIT, random_state=self.RANDOM_SEED)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_[kmeans.labels_]
        return centroids.flatten()
