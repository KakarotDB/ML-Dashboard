import pandas as pd
import numpy as np

class HistogramDiscretizer:

    METHODS = {
        "Equal Width Binning": "equal_width",
        "Equal Frequency Binning": "equal_freq"
    }

    def __init__(self, method: str = "equal_width", column: str | None = None, bins: int = 5):
        self.method = method
        self.column = column
        self.bins = bins
        self.report_ = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.column is None or self.column not in df.columns:
            return df

        values = df[self.column].to_numpy()

        if self.method == "equal_width":
            df[self.column] = self._equal_width(values)

        elif self.method == "equal_freq":
            df[self.column] = self._equal_freq(values)

        self.report_ = {
            "method": self.method,
            "column": self.column,
            "bins": self.bins,
            "total_values": len(values)
        }

        return df

    def _equal_width(self, values):
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)

        bins = np.linspace(min_val, max_val, self.bins + 1)
        return np.digitize(values, bins)

    def _equal_freq(self, values):
        quantiles = np.linspace(0, 1, self.bins + 1)
        bins = np.nanquantile(values, quantiles)
        return np.digitize(values, bins)

    def get_report(self) -> dict:
        return self.report_
