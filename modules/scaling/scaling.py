# modules/scaling/scaling.py
import pandas as pd
import numpy as np

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