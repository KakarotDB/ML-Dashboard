import pandas as pd

class DataReduction:
    """
    Module for performing data reduction techniques.
    Includes numerosity reduction (sampling, histograms) and dimensionality reduction (feature selection).
    """

    METHODS = {
        "Simple Random Sampling": "simple_random",
        "Stratified Sampling": "stratified",
        "Equal-Width Histogram": "histogram",
        "Heuristic Feature Selection": "feature_selection",
    }

    RANDOM_SEED = 42

    def __init__(
        self,
        method: str = "simple_random",
        column: str | None = None,
        sample_size: int = 1000,
        replacement: bool = False,
        sample_fraction: float = 0.1,
        num_buckets: int = 10,
        allow_negatives: bool = True,
        variance_threshold: float = 0.01
    ):
        self.method = method
        self.column = column
        self.sample_size = sample_size
        self.replacement = replacement
        self.sample_fraction = sample_fraction
        self.num_buckets = num_buckets
        self.allow_negatives = allow_negatives
        self.variance_threshold = variance_threshold
        
        self.report_: dict = {"status": "No transformation applied yet"}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the selected data reduction method to a copy of the DataFrame."""
        df = df.copy()

        if self.method == "simple_random":
            return self._simple_random(df)
        elif self.method == "stratified":
            return self._stratified(df)
        elif self.method == "histogram":
            return self._histogram(df)
        elif self.method == "feature_selection":
            return self._feature_selection(df)
        else:
            self.report_ = {"error": f"Unknown method key: {self.method}"}
            return df

    def get_report(self) -> dict:
        return self.report_

    # Private helper methods:
    
    def _simple_random(self, df: pd.DataFrame) -> pd.DataFrame:
        total_rows = len(df)
        actual_size = self.sample_size
        
        if self.sample_size > total_rows and not self.replacement:
            actual_size = total_rows
            
        reduced_df = df.sample(n=actual_size, replace=self.replacement, random_state=self.RANDOM_SEED)
        
        self.report_ = {
            "method_applied": "Simple Random Sampling",
            "replacement": self.replacement,
            "original_rows": total_rows,
            "final_rows": len(reduced_df)
        }
        return reduced_df

    def _stratified(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.column or self.column not in df.columns:
            self.report_ = {"error": f"Stratified column '{self.column}' not found or not provided."}
            return df
            
        stratified_df = df.groupby(self.column, group_keys=False).sample(
            frac=self.sample_fraction, 
            random_state=self.RANDOM_SEED
        )
        
        self.report_ = {
            "method_applied": "Stratified Sampling",
            "grouping_column": self.column,
            "sample_fraction": self.sample_fraction,
            "original_rows": len(df),
            "final_rows": len(stratified_df)
        }
        return stratified_df

    def _histogram(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.column or self.column not in df.columns:
            self.report_ = {"error": f"Histogram column '{self.column}' not found or not provided."}
            return df
            
        if not self.allow_negatives:
            clean_df = df[df[self.column] >= 0].copy()
            dropped_rows = len(df) - len(clean_df)
        else:
            clean_df = df.copy()
            dropped_rows = 0
            
        hist_df = pd.cut(clean_df[self.column], bins=self.num_buckets, right=False).value_counts().reset_index()
        hist_df.columns = ['Bucket_Range', 'Frequency_Count']
        hist_df = hist_df.sort_values(by='Bucket_Range').reset_index(drop=True)
        
        hist_df['Bucket_Range'] = hist_df['Bucket_Range'].astype(str)
        
        self.report_ = {
            "method_applied": "Equal-Width Histogram",
            "column": self.column,
            "buckets_created": self.num_buckets,
            "negatives_dropped": dropped_rows
        }
        return hist_df

    def _feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=['number', 'float', 'int'])
        
        if numeric_df.empty:
            self.report_ = {"error": "No numeric columns found for variance thresholding."}
            return df
            
        variances = numeric_df.var()
        
        features_to_keep = variances[variances > self.variance_threshold].index.tolist()
        features_to_drop = variances[variances <= self.variance_threshold].index.tolist()
        
        non_numeric_cols = df.select_dtypes(exclude=['number', 'float', 'int']).columns.tolist()
        
        final_columns = non_numeric_cols + features_to_keep
        reduced_df = df[final_columns].copy()
        
        self.report_ = {
            "method_applied": "Heuristic Feature Selection",
            "variance_threshold": self.variance_threshold,
            "features_dropped": len(features_to_drop),
            "dropped_columns": ", ".join(features_to_drop) if features_to_drop else "None"
        }
        return reduced_df