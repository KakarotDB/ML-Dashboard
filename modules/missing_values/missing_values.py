import pandas as pd
import numpy as np
from typing import Any
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class MissingValueEstimator:
    """Handles missing data via basic, statistical, and ML-based imputation techniques."""

    N_ESTIMATORS = 10
    RANDOM_SEED = 42

    METHODS = {
        "Mean Imputation": "mean",
        "Median Imputation": "median",
        "Mode Imputation": "mode",
        "Arbitrary Value (0)": "zero",
        "Constant Value ('Unknown')": "constant",
        "KNN Imputation": "knn",
        "Regression Imputation": "regression",
        "Random Forest Imputation": "rf_impute",
    }

    def __init__(
        self,
        method: str = "mean",
        column: str | None = None,
        n_neighbors: int = 5,
        constant_val: Any = "Unknown",
    ):
        self.method = method
        self.column = column
        self.n_neighbors = n_neighbors
        self.constant_val = constant_val
        self.report_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes the DataFrame and fills missing values in the target column."""
        df = df.copy()

        if not self.column or self.column not in df.columns:
            self.report_ = {"status": "Error", "message": "No valid column selected"}
            return df

        initial_missing = int(df[self.column].isnull().sum())

        if self.method in ["mean", "median", "mode"]:
            self._apply_simple_imputer(df)
        elif self.method == "zero":
            df[self.column] = df[self.column].fillna(0)
        elif self.method == "constant":
            df[self.column] = df[self.column].fillna(self.constant_val)
        elif self.method == "knn":
            self._apply_knn(df)
        elif self.method == "regression":
            self._apply_regression(df)
        elif self.method == "rf_impute":
            self._apply_ml_imputer(df)

        self.report_ = {
            "strategy": self.method,
            "target_column": self.column,
            "nulls_fixed": initial_missing,
            "remaining_nulls": int(df[self.column].isnull().sum()),
        }

        return df

    def get_report(self) -> dict:
        """Returns the results of the last fit_transform call."""
        return self.report_

    # --- Private Helper Methods ---

    def _apply_simple_imputer(self, df: pd.DataFrame) -> None:
        """Uses Scikit-Learn SimpleImputer for mean, median, and mode."""
        strategy = self.method if self.method != "mode" else "most_frequent"
        imputer = SimpleImputer(strategy=strategy)
        col_data: NDArray = df[[self.column]].to_numpy()
        df[self.column] = imputer.fit_transform(col_data)

    def _apply_knn(self, df: pd.DataFrame) -> None:
        """Fills missing values using the average of K-nearest neighbors."""
        numeric_df = df.select_dtypes(include=[np.number])

        if self.column not in numeric_df.columns:
            return

        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        imputed_data = imputer.fit_transform(numeric_df)

        col_idx = list(numeric_df.columns).index(self.column)
        df[self.column] = imputed_data[:, col_idx]

    def _apply_regression(self, df: pd.DataFrame) -> None:
        """Predicts missing values using a linear regression model."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        clean_df = df[numeric_cols].dropna(subset=numeric_cols.difference([self.column]))

        train_df = clean_df[clean_df[self.column].notnull()]
        test_df = clean_df[clean_df[self.column].isnull()]

        if not train_df.empty and not test_df.empty:
            model = LinearRegression()
            X_train = train_df.drop(columns=[self.column])
            y_train = train_df[self.column]
            X_test = test_df.drop(columns=[self.column])

            model.fit(X_train, y_train)
            df.loc[df[self.column].isnull(), self.column] = model.predict(X_test)

    def _apply_ml_imputer(self, df: pd.DataFrame) -> None:
        """Predicts missing values using a random forest model."""
        is_numeric = pd.api.types.is_numeric_dtype(df[self.column])

        model = (
            RandomForestRegressor(n_estimators=self.N_ESTIMATORS, random_state=self.RANDOM_SEED)
            if is_numeric
            else RandomForestClassifier(n_estimators=self.N_ESTIMATORS, random_state=self.RANDOM_SEED)
        )

        feature_df = df.select_dtypes(include=[np.number]).fillna(0)

        train_mask = df[self.column].notnull()
        test_mask = df[self.column].isnull()

        if test_mask.any() and train_mask.any():
            X_train = feature_df[train_mask].drop(columns=[self.column], errors="ignore")
            y_train = df.loc[train_mask, self.column]
            X_test = feature_df[test_mask].drop(columns=[self.column], errors="ignore")

            model.fit(X_train, y_train)
            df.loc[test_mask, self.column] = model.predict(X_test)
