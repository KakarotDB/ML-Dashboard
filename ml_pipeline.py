import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split


class MLPipeline:
    """
    Central orchestrator for machine learning operations.
    Maintains a registry of ML models and handles training/evaluation orchestration.
    """

    _model_registry: dict[str, type] = {}

    def __init__(self):
        self.last_model_ = None
        self.last_split_: dict[str, Any] | None = None

    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """Registers a machine learning model class."""
        cls._validate_model(model_class)
        cls._model_registry[name] = model_class

    @classmethod
    def get_registered_models(cls) -> list[str]:
        """Returns names of all registered ML models."""
        return list(cls._model_registry.keys())

    @classmethod
    def get_methods_for_model(cls, model_name: str) -> dict:
        """Returns the METHODS dict of a registered model."""
        if model_name not in cls._model_registry:
            raise ValueError(
                f"'{model_name}' is not registered. "
                f"Available: {cls.get_registered_models()}"
            )
        return cls._model_registry[model_name].METHODS

    @classmethod
    def _validate_model(cls, model_class: type) -> None:
        required = ["fit", "predict", "get_report", "METHODS"]
        missing = [attr for attr in required if not hasattr(model_class, attr)]
        if missing:
            raise TypeError(
                f"'{model_class.__name__}' is missing required attributes: "
                f"{missing}. All ML models must implement "
                f"fit(), predict(), get_report(), and a METHODS dict."
            )

    def train(
        self,
        df: pd.DataFrame,
        model_name: str,
        target_col: str,
        feature_cols: list[str],
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ) -> None:
        """
        Prepares the data and trains the specified model.
        """
        if model_name not in self._model_registry:
            raise ValueError(f"'{model_name}' is not registered.")
            
        if not target_col or not feature_cols:
            raise ValueError("Target column and at least one feature column must be selected.")

        # Instantiate the model with hyperparameters
        module_class = self._model_registry[model_name]
        model_instance = module_class(**kwargs)

        # Prepare data (drop NAs in used columns)
        clean_df = df.dropna(subset=feature_cols + [target_col])
        
        if clean_df.empty:
            raise ValueError("Data is empty after dropping missing values in selected columns.")

        X = clean_df[feature_cols]
        y = clean_df[target_col]

        # Determine if stratification is possible (for classification)
        # Check if y has enough samples per class for stratification
        stratify = y if y.nunique() > 1 and len(y) > y.nunique() else None
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
        except ValueError:
            # Fallback if stratify fails (e.g., class with only 1 sample)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Train model
        model_instance.fit(X_train, y_train)

        # Save state
        self.last_model_ = model_instance
        self.last_split_ = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "target_col": target_col,
        }
