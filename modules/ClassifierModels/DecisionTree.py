import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModule:
    """
    Wrapper for Scikit-Learn's DecisionTreeClassifier.
    """
    
    # 1. Non-negotiable METHODS dict for the dashboard dropdown
    METHODS = {
        "Gini Impurity": "gini",
        "Entropy": "entropy",
        "Log Loss": "log_loss"
    }

    def __init__(
        self,
        method: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = 42
    ):
        self.method = method
        self.max_depth = max_depth if max_depth and max_depth > 0 else None
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.model = DecisionTreeClassifier(
            criterion=self.method,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        self.report_: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the Decision Tree model."""
        self.model.fit(X, y)
        
        # Populate report after training
        self.report_ = {
            "Model": "Decision Tree Classifier",
            "Criterion": self.method.capitalize(),
            "Max Depth": "None" if self.max_depth is None else self.max_depth,
            "Min Samples Split": self.min_samples_split,
            "Tree Depth Achieved": self.model.get_depth(),
            "Leaves Count": self.model.get_n_leaves(),
            "Classes": list(self.model.classes_)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predictions."""
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities."""
        return self.model.predict_proba(X)

    def get_report(self) -> dict:
        """Returns the training parameters and tree statistics."""
        return self.report_
