import pandas as pd
import numpy as np


class DataReduction:
    """
    Module for performing data reduction techniques.
    Includes numerosity reduction (sampling), feature selection, and dimensionality reduction (PCA).

    Parameters
    ----------
    method : str
        Key from METHODS dict.
    column : str | None
        Required for stratified sampling.
    sample_size : int
        Number of rows for simple random sampling.
    replacement : bool
        Whether to sample with replacement.
    sample_fraction : float
        Fraction of rows to sample per group for stratified sampling.
    variance_threshold : float
        Columns with variance below this are dropped during feature selection.
    drop_cols : list[str] | None
        Columns to explicitly drop.
    n_components : int
        Number of principal components to keep for PCA.
    """

    METHODS = {
        "Simple Random Sampling": "simple_random",
        "Stratified Sampling": "stratified",
        "Heuristic Feature Selection": "feature_selection",
        "Drop Columns": "drop_columns",
        "PCA (Principal Components)": "pca",
    }

    RANDOM_SEED = 42

    def __init__(
        self,
        method: str = "simple_random",
        column: str | None = None,
        sample_size: int = 1000,
        replacement: bool = False,
        sample_fraction: float = 0.1,
        variance_threshold: float = 0.01,
        drop_cols: list[str] | None = None,
        n_components: int = 2,
    ):
        self.method = method
        self.column = column
        self.sample_size = sample_size
        self.replacement = replacement
        self.sample_fraction = sample_fraction
        self.variance_threshold = variance_threshold
        self.drop_cols = drop_cols or []
        self.n_components = n_components

        self.report_: dict = {"status": "No transformation applied yet"}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the selected data reduction method to a copy of the DataFrame."""
        df = df.copy()

        if self.method == "simple_random":
            return self._simple_random(df)
        elif self.method == "stratified":
            return self._stratified(df)
        elif self.method == "feature_selection":
            return self._feature_selection(df)
        elif self.method == "drop_columns":
            return self._drop_columns(df)
        elif self.method == "pca":
            return self._pca(df)
        else:
            self.report_ = {"error": f"Unknown method key: {self.method}"}
            return df

    def get_report(self) -> dict:
        return self.report_

    def _simple_random(self, df: pd.DataFrame) -> pd.DataFrame:
        total_rows = len(df)
        actual_size = self.sample_size

        if self.sample_size > total_rows and not self.replacement:
            actual_size = total_rows

        reduced_df = df.sample(
            n=actual_size, replace=self.replacement, random_state=self.RANDOM_SEED
        )

        self.report_ = {
            "method_applied": "Simple Random Sampling",
            "replacement": self.replacement,
            "original_rows": total_rows,
            "final_rows": len(reduced_df),
        }
        return reduced_df

    def _stratified(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.column or self.column not in df.columns:
            self.report_ = {
                "error": f"Stratified column '{self.column}' not found or not provided."
            }
            return df

        stratified_df = df.groupby(self.column, group_keys=False).sample(
            frac=self.sample_fraction, random_state=self.RANDOM_SEED
        )

        self.report_ = {
            "method_applied": "Stratified Sampling",
            "grouping_column": self.column,
            "sample_fraction": self.sample_fraction,
            "original_rows": len(df),
            "final_rows": len(stratified_df),
        }
        return stratified_df

    def _feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            self.report_ = {"error": "No numeric columns found for variance thresholding."}
            return df

        variances = numeric_df.var()
        features_to_keep = variances[variances > self.variance_threshold].index.tolist()
        features_to_drop = variances[variances <= self.variance_threshold].index.tolist()

        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
        reduced_df = df[non_numeric_cols + features_to_keep].copy()

        self.report_ = {
            "method_applied": "Heuristic Feature Selection",
            "variance_threshold": self.variance_threshold,
            "features_dropped": len(features_to_drop),
            "dropped_columns": ", ".join(features_to_drop) if features_to_drop else "None",
        }
        return reduced_df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        reduced_df = df.drop(columns=cols_to_drop)
        
        self.report_ = {
            "method_applied": "Drop Columns",
            "columns_dropped": ", ".join(cols_to_drop) if cols_to_drop else "None",
            "total_dropped": len(cols_to_drop)
        }
        return reduced_df

    def _pca(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            self.report_ = {"error": "No numeric columns available for PCA."}
            return df
            
        # PCA requires data without NaNs. We impute with mean temporarily.
        clean_numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Scaling is essential for PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_numeric_df)
        
        n_comps = min(self.n_components, scaled_data.shape[1])
        pca = PCA(n_components=n_comps, random_state=self.RANDOM_SEED)
        
        principal_components = pca.fit_transform(scaled_data)
        
        pca_cols = [f"PC{i+1}" for i in range(n_comps)]
        pca_df = pd.DataFrame(principal_components, columns=pca_cols, index=df.index)
        
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
        reduced_df = pd.concat([df[non_numeric_cols], pca_df], axis=1)
        
        self.report_ = {
            "method_applied": "PCA (Dimensionality Reduction)",
            "components_kept": n_comps,
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "total_variance_explained": round(float(sum(pca.explained_variance_ratio_)), 4)
        }
        return reduced_df


class Histogram:
    """
    Terminal analysis module that summarizes a numeric column into
    equal-width buckets and returns a frequency table.

    This module does not transform the pipeline's DataFrame — it is
    registered via register_analysis() and runs through pipeline.analyze()
    rather than pipeline.apply().

    Parameters
    ----------
    method : str
        Key from METHODS dict.
    column : str | None
        The numeric column to bucket.
    num_buckets : int
        Number of equal-width buckets to create.
    allow_negatives : bool
        If False, rows with negative values in the column are excluded.
    """

    METHODS = {
        "Equal-Width Histogram": "histogram",
    }

    def __init__(
        self,
        method: str = "histogram",
        column: str | None = None,
        num_buckets: int = 10,
        allow_negatives: bool = True,
    ):
        self.method = method
        self.column = column
        self.num_buckets = num_buckets
        self.allow_negatives = allow_negatives

        self.report_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a frequency table DataFrame, not the original data.
        This is intentional — this module is terminal by design.
        """
        if not self.column or self.column not in df.columns:
            self.report_ = {"error": f"Column '{self.column}' not found."}
            return df

        df = df.copy()

        if not self.allow_negatives:
            dropped = int((df[self.column] < 0).sum())
            df = df[df[self.column] >= 0]
        else:
            dropped = 0

        hist_df = (
            pd.cut(df[self.column], bins=self.num_buckets, right=False)
            .value_counts()
            .reset_index()
        )
        hist_df.columns = ["Bucket_Range", "Frequency_Count"]
        hist_df = hist_df.sort_values("Bucket_Range").reset_index(drop=True)
        hist_df["Bucket_Range"] = hist_df["Bucket_Range"].astype(str)

        self.report_ = {
            "method_applied": "Equal-Width Histogram",
            "column": self.column,
            "buckets_created": self.num_buckets,
            "negatives_dropped": dropped,
        }
        return hist_df

    def get_report(self) -> dict:
        return self.report_
