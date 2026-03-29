import pandas as pd
from dataclasses import dataclass


@dataclass
class PipelineStep:
    """Represents a single applied preprocessing step with its before/after state."""
    step_name: str
    method: str
    report: dict
    before: pd.DataFrame
    after: pd.DataFrame


class PreprocessingPipeline:
    """
    Central orchestrator for all preprocessing and ML operations.

    Maintains two separate registries:
        - _registry: transformation modules that modify current_df via apply()
        - _analysis_registry: terminal modules that summarize data via analyze()

    Modules are registered externally via registry.py. The pipeline never
    imports or hardcodes any module directly. Adding a new module only
    requires changes to registry.py, nothing here.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset loaded from a CSV upload or sample file.

    Usage
    -----
    From registry.py:
        PreprocessingPipeline.register("Discretization", Discretization)
        PreprocessingPipeline.register_analysis("Histogram", Histogram)

    From app.py or tests:
        pipeline.apply("Discretization", method="entropy", column="Age", ...)
        pipeline.analyze("Histogram", column="Fare", num_buckets=10)
    """

    _registry: dict[str, type] = {}
    _analysis_registry: dict[str, type] = {}

    def __init__(self, df: pd.DataFrame):
        self.original_df: pd.DataFrame = df.copy()
        self.current_df: pd.DataFrame = df.copy()
        self.history: list[PipelineStep] = []
        self.last_analysis_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Transformation Registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, module_class: type) -> None:
        """
        Registers a transformation module that modifies the pipeline DataFrame.

        The module must implement fit_transform(df), get_report(), and a METHODS
        dict — these are validated at registration time.

        Parameters
        ----------
        name : str
            Display name shown in the dashboard (e.g. "Discretization").
        module_class : type
            The module class to register.
        """
        cls._validate_module(module_class)
        cls._registry[name] = module_class

    @classmethod
    def get_registered_modules(cls) -> list[str]:
        """Returns names of all registered transformation modules."""
        return list(cls._registry.keys())

    @classmethod
    def get_methods_for(cls, module_name: str) -> dict:
        """
        Returns the METHODS dict of a registered transformation module.
        Used by the dashboard to dynamically build method dropdowns.
        """
        if module_name not in cls._registry:
            raise ValueError(
                f"'{module_name}' is not registered. "
                f"Available: {cls.get_registered_modules()}"
            )
        return cls._registry[module_name].METHODS

    # ------------------------------------------------------------------
    # Analysis Registry
    # ------------------------------------------------------------------

    @classmethod
    def register_analysis(cls, name: str, module_class: type) -> None:
        """
        Registers a terminal analysis module that summarizes data without
        modifying the pipeline's current DataFrame or history.

        Parameters
        ----------
        name : str
            Display name shown in the dashboard (e.g. "Histogram").
        module_class : type
            The module class to register.
        """
        cls._validate_module(module_class)
        cls._analysis_registry[name] = module_class

    @classmethod
    def get_registered_analysis_modules(cls) -> list[str]:
        """Returns names of all registered analysis modules."""
        return list(cls._analysis_registry.keys())

    @classmethod
    def get_methods_for_analysis(cls, module_name: str) -> dict:
        """
        Returns the METHODS dict of a registered analysis module.
        Used by the dashboard to dynamically build method dropdowns.
        """
        if module_name not in cls._analysis_registry:
            raise ValueError(
                f"'{module_name}' is not a registered analysis module. "
                f"Available: {cls.get_registered_analysis_modules()}"
            )
        return cls._analysis_registry[module_name].METHODS

    # ------------------------------------------------------------------
    # Shared Validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_module(cls, module_class: type) -> None:
        required = ["fit_transform", "get_report", "METHODS"]
        missing = [attr for attr in required if not hasattr(module_class, attr)]
        if missing:
            raise TypeError(
                f"'{module_class.__name__}' is missing required attributes: "
                f"{missing}. All pipeline modules must implement "
                f"fit_transform(), get_report(), and a METHODS dict."
            )

    # ------------------------------------------------------------------
    # Core Apply
    # ------------------------------------------------------------------

    def apply(self, module_name: str, **kwargs) -> pd.DataFrame:
        """
        Instantiates and runs a registered transformation module on the
        current DataFrame. Updates current_df and appends to history.

        Parameters
        ----------
        module_name : str
            Must match a key in the transformation registry.
        **kwargs
            Passed directly to the module's constructor.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame after the module is applied.
        """
        if module_name not in self._registry:
            raise ValueError(
                f"'{module_name}' is not registered. "
                f"Available: {self.get_registered_modules()}"
            )

        before = self.current_df.copy()
        module = self._registry[module_name](**kwargs)
        self.current_df = module.fit_transform(self.current_df)

        self.history.append(
            PipelineStep(
                step_name=module_name,
                method=kwargs.get("method", ""),
                report=module.get_report(),
                before=before,
                after=self.current_df.copy(),
            )
        )

        return self.current_df

    # ------------------------------------------------------------------
    # Core Analyze
    # ------------------------------------------------------------------

    def analyze(self, module_name: str, **kwargs) -> pd.DataFrame:
        """
        Runs a terminal analysis module on the current DataFrame.

        Unlike apply(), this does not update current_df or history.
        The result is stored in last_analysis_ and returned.

        Parameters
        ----------
        module_name : str
            Must match a key in the analysis registry.
        **kwargs
            Passed directly to the module's constructor.

        Returns
        -------
        pd.DataFrame
            The analysis result, e.g. a frequency table.
        """
        if module_name not in self._analysis_registry:
            raise ValueError(
                f"'{module_name}' is not a registered analysis module. "
                f"Available: {self.get_registered_analysis_modules()}"
            )

        module = self._analysis_registry[module_name](**kwargs)
        result = module.fit_transform(self.current_df)
        self.last_analysis_ = result
        return result

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Resets the pipeline back to the original unmodified DataFrame."""
        self.current_df = self.original_df.copy()
        self.history = []
        self.last_analysis_ = None

    def undo(self) -> pd.DataFrame:
        """
        Removes the last applied step and rolls the DataFrame back to
        the state before that step was applied.

        Returns
        -------
        pd.DataFrame
            The restored DataFrame.
        """
        if not self.history:
            return self.current_df

        last_step = self.history.pop()
        self.current_df = last_step.before.copy()
        return self.current_df

    def get_step(self, index: int) -> PipelineStep:
        """Returns a specific step from history by its index."""
        if index < 0 or index >= len(self.history):
            raise IndexError(f"No step at index {index}.")
        return self.history[index]

    def get_history_summary(self) -> list[dict]:
        """
        Returns a lightweight summary of all applied steps suitable
        for rendering in the dashboard's step log panel.
        """
        return [
            {
                "step": i + 1,
                "name": s.step_name,
                "method": s.method,
                "report": s.report,
            }
            for i, s in enumerate(self.history)
        ]
