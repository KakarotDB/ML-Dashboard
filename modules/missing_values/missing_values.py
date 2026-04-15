import pandas as pd
import numpy as np
from typing import Any
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class MissingValueEstimator:
    """
    A module for the ML-Dashboard to handle missing data using 
    Basic, Statistical, and Machine Learning techniques.

    Handling missig_values-

    1-Basic Methods
          1-mean/med`ian/mode imputation
          2-arbitary value imputation
          3-constant value imputation

    2-statistical Methods
           1-knn imputation
           2-regression imputation

    3-machine learning based methods
           1-Random forest imputation
          
    
    """

    # Dashboard uses this to build the selection UI automatically
    METHODS = {
        "Mean Imputation": "mean",
        "Median Imputation": "median",
        "Mode Imputation": "mode",
        "Arbitrary Value (0)": "zero",
        "Constant Value ('Unknown')": "constant",
        "KNN Imputation": "knn",
        "Regression Imputation": "regression",
        "Random Forest Imputation": "rf_impute"
    }

    def __init__(
        self, 
        method: str = "mean", 
        column: str | None = None, 
        n_neighbors: int = 5,
        constant_val: Any = "Unknown"
    ):
        """
        Initializes the imputer with user-defined parameters.
        """
        self.method = method
        self.column = column
        self.n_neighbors = n_neighbors
        self.constant_val = constant_val
        self.report_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the DataFrame and fills missing values in the target column.
        Always returns a copy to avoid mutating original data.
        """
        # RULE: Always work on a copy
        df = df.copy()
        
        # Validation: Ensure a column is selected and exists
        if not self.column or self.column not in df.columns:
            self.report_ = {"status": "Error", "message": "No valid column selected"}
            return df

        # Store count of missing values before processing for the report
        initial_missing = int(df[self.column].isnull().sum())
        
        # Branching logic based on selected method
        if self.method in ["mean", "median", "mode"]:
            self._apply_simple_imputer(df)
        
        elif self.method == "zero":
            # Basic Technique (ii): Arbitrary Value
            df[self.column] = df[self.column].fillna(0)
            
        elif self.method == "constant":
            # Basic Technique (iii): Constant Value (Domain Knowledge)
            df[self.column] = df[self.column].fillna(self.constant_val)
            
        elif self.method == "knn":
            # Statistical Method (i): K-Nearest Neighbors
            self._apply_knn(df)
            
        elif self.method == "regression":
            # Statistical Method (ii): Regression Imputation
            self._apply_regression(df)
            
        elif self.method == "rf_impute":
            # ML Method (i): Random Forest Imputation
            self._apply_ml_imputer(df)

        # RULE: Update the flat report dictionary
        self.report_ = {
            "strategy": self.method,
            "target_column": self.column,
            "nulls_fixed": initial_missing,
            "remaining_nulls": int(df[self.column].isnull().sum())
        }
        
        return df

    def get_report(self) -> dict:
        """Returns the results of the last fit_transform call."""
        return self.report_

    # --- Private Helper Methods (Prefixed with _) ---

    def _apply_simple_imputer(self, df: pd.DataFrame) -> None:
        """Uses Scikit-Learn SimpleImputer for Mean, Median, and Mode."""
        strategy = self.method if self.method != "mode" else "most_frequent"
        imputer = SimpleImputer(strategy=strategy)
        
        # RULE: Use .to_numpy() and reshape for Sklearn compatibility
        col_data: NDArray = df[[self.column]].to_numpy()
        df[self.column] = imputer.fit_transform(col_data)

    def _apply_knn(self, df: pd.DataFrame) -> None:
        """Fills missing values using the average of K-nearest neighbors."""
        # KNN requires numeric features to calculate distance
        numeric_df = df.select_dtypes(include=[np.number])
        
        if self.column not in numeric_df.columns:
            return # Cannot apply KNN to categorical via this simple logic

        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        # We impute based on all available numeric columns
        imputed_data = imputer.fit_transform(numeric_df)
        
        col_idx = list(numeric_df.columns).index(self.column)
        df[self.column] = imputed_data[:, col_idx]

    def _apply_regression(self, df: pd.DataFrame) -> None:
        """Predicts missing values using a Linear Regression model."""
        # We only use numeric columns as features for the regression
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Drop rows where OTHER columns have NaNs so the model can train
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
        """Predicts missing values using Random Forest (ML-based method)."""
        is_numeric = pd.api.types.is_numeric_dtype(df[self.column])
        
        # Choose Regressor for numbers, Classifier for categories/classes
        model = RandomForestRegressor(n_estimators=10) if is_numeric else RandomForestClassifier(n_estimators=10)
        
        # Prepare a simple feature set (Numeric only, filling NaNs with 0 for safety)
        feature_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Ensure target column is included in the processing pool
        train_mask = df[self.column].notnull()
        test_mask = df[self.column].isnull()
        
        if test_mask.any() and train_mask.any():
            X_train = feature_df[train_mask].drop(columns=[self.column], errors='ignore')
            y_train = df.loc[train_mask, self.column]
            X_test = feature_df[test_mask].drop(columns=[self.column], errors='ignore')
            
            model.fit(X_train, y_train)
            df.loc[test_mask, self.column] = model.predict(X_test)