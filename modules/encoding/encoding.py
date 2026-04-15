import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CategoricalEncoder:
    """
    Encodes categorical (text) columns into numerical values so they can be used in ML models.
    Supports Label Encoding and One-Hot Encoding.
    """

    METHODS = {
        "Label Encoding": "label",
        "One-Hot Encoding": "onehot",
    }

    def __init__(
        self,
        method: str = "label",
        column: str | None = None,
        apply_to_all: bool = False
    ):
        self.method = method
        self.column = column
        self.apply_to_all = apply_to_all
        self.report_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        # Determine which columns to process
        if self.apply_to_all:
            # Select object and category dtypes
            target_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()
            if not target_cols:
                self.report_ = {"error": "No categorical columns found in the dataset."}
                return df_copy
        else:
            if not self.column or self.column not in df_copy.columns:
                self.report_ = {"error": f"Column '{self.column}' not found or not specified."}
                return df_copy
            
            # Check if the specific column is actually categorical
            if pd.api.types.is_numeric_dtype(df_copy[self.column]):
                self.report_ = {"error": f"Column '{self.column}' is already numeric."}
                return df_copy
                
            target_cols = [self.column]

        columns_processed = 0
        new_columns_created = 0

        for col in target_cols:
            # Handle NaNs before encoding: temporarily fill with a placeholder if any exist
            # For simplicity, we convert to string, fillna with "Missing", encode, then optionally restore NaNs?
            # Sklearn LabelEncoder fails on NaNs. pd.get_dummies handles them gracefully (ignores or creates a dummy).
            
            if self.method == "label":
                # LabelEncoder cannot handle NaNs. We need to handle them.
                # A robust way is to fill NaNs with a string, encode, and then put NaNs back,
                # or just encode non-NaN values.
                le = LabelEncoder()
                
                # Get mask of valid values
                valid_mask = df_copy[col].notna()
                if valid_mask.any():
                    # Fit and transform only non-NaN values
                    encoded_vals = le.fit_transform(df_copy.loc[valid_mask, col].astype(str))
                    
                    # Bypass strict StringDtype or CategoricalDtype constraints by casting to object
                    df_copy[col] = df_copy[col].astype(object)
                    
                    df_copy.loc[valid_mask, col] = encoded_vals
                    
                    # Convert to float to allow NaNs to remain if there were any
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    columns_processed += 1

            elif self.method == "onehot":
                # pd.get_dummies handles NaNs (ignores them by default, so all dummy cols are 0 for that row)
                # dummy_na=False is default
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=False, dummy_na=False)
                
                # Convert booleans to integers (0/1) for better compatibility
                dummies = dummies.astype(int)
                
                # Drop original column and concatenate dummies
                df_copy = df_copy.drop(columns=[col])
                df_copy = pd.concat([df_copy, dummies], axis=1)
                
                columns_processed += 1
                new_columns_created += len(dummies.columns) - 1 # -1 because we removed the original

        self.report_ = {
            "method": self.method,
            "applied_to": "All Categorical" if self.apply_to_all else self.column,
            "columns_processed": columns_processed,
        }
        
        if self.method == "onehot":
            self.report_["net_new_columns"] = new_columns_created

        return df_copy

    def get_report(self) -> dict:
        return self.report_
