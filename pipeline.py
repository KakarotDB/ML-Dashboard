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

    Modules are registered externally via registry.py — the pipeline never
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

    From app.py or tests:
        pipeline.apply("Discretization", method="entropy", column="Age", ...)
    """

    _registry: dict[str, type] = {}

    def __init__(self, df: pd.DataFrame):
        self.original_df: pd.DataFrame = df.copy()
        self.current_df: pd.DataFrame = df.copy()
        self.history: list[PipelineStep] = []

    # Registry

    @classmethod
    def register(cls, name: str, module_class: type) -> None:
        """
        Registers a module class under a given display name.

        The module must implement fit_transform(df), get_report(), and
        expose a METHODS dict — these are validated at registration time.

        Parameters
        ----------
        name : str
            Display name shown in the dashboard (e.g. "Discretization").
        module_class : type
            The module class to register.
        """
        required = ["fit_transform", "get_report", "METHODS"]
        missing = [attr for attr in required if not hasattr(module_class, attr)]

        if missing:
            raise TypeError(
                f"'{module_class.__name__}' is missing required attributes: "
                f"{missing}. All pipeline modules must implement "
                f"fit_transform(), get_report(), and a METHODS dict."
            )

        cls._registry[name] = module_class

    @classmethod
    def get_registered_modules(cls) -> list[str]:
        """Returns names of all currently registered modules."""
        return list(cls._registry.keys())

    @classmethod
    def get_methods_for(cls, module_name: str) -> dict:
        """
        Returns the METHODS dict of a registered module.
        Used by the dashboard to dynamically build method dropdowns.
        """
        if module_name not in cls._registry:
            raise ValueError(
                f"Module '{module_name}' is not registered. "
                f"Available: {cls.get_registered_modules()}"
            )
        return cls._registry[module_name].METHODS

    # Core Apply

    def apply(self, module_name: str, **kwargs) -> pd.DataFrame:
        """
        Instantiates and runs a registered module on the current DataFrame.

        Parameters
        ----------
        module_name : str
            Must match a key in the registry.
        **kwargs
            Passed directly to the module's constructor.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame after the module is applied.
        """
        if module_name not in self._registry:
            raise ValueError(
                f"Module '{module_name}' is not registered. "
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

    # State Management

    def reset(self) -> None:
        """Resets the pipeline back to the original unmodified DataFrame."""
        self.current_df = self.original_df.copy()
        self.history = []

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
