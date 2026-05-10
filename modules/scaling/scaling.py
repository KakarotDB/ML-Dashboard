# modules/scaling/scaling.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

class DataScaler:
    """
    Scales or standardizes numeric attributes.
    """
    METHODS = {
        "Min-Max Scaling": "minmax",
        "Z-Score Standardization": "standardization",
        "Robust Scaling": "robust",
        "MaxAbs Scaling": "maxabs"
    }

    def __init__(
        self,
        method: str = "standardization",
        columns: list[str] | str = "all_numeric"
    ):
        self.method = method
        self.columns = columns
        self.report_: dict = {}
        
    def get_report(self) -> dict:
        return self.report_

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Column Selection
        if self.columns == "all_numeric":
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(self.columns, list):
            target_cols = [col for col in self.columns if col in df.columns]
        else:
            raise ValueError("columns must be 'all_numeric' or a list of column names")
            
        if not target_cols:
            self.report_ = {"method": self.method, "columns_scaled": [], "num_columns_scaled": 0}
            return df
            
        # 2. Scaler Selection
        if self.method == "standardization":
            scaler = StandardScaler()
        elif self.method == "minmax":
            scaler = MinMaxScaler()
        elif self.method == "robust":
            scaler = RobustScaler()
        elif self.method == "maxabs":
            scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown scaling method '{self.method}'")
            
        # 3. Scaling Execution
        # scikit-learn scalers ignore np.nan during fit and return nan during transform
        subset_df = df[target_cols]
        scaled_data = scaler.fit_transform(subset_df)
        
        # 4. Data Assembly
        scaled_col_names = [f"{c}_scaled" for c in target_cols]
        df[scaled_col_names] = scaled_data
        
        # 5. Reporting
        self.report_ = {
            "method": self.method,
            "columns_scaled": target_cols,
            "num_columns_scaled": len(target_cols)
        }
        
        return df