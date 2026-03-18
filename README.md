# ML Dashboard

A data preprocessing and machine learning dashboard built with Python and Streamlit.
This is our mini project for the Data Mining course. The goal is to have a single
interactive dashboard where we can apply all our preprocessing techniques to a dataset
and visualize what each step does in real time.

---

## Project Structure

```
ML-Dashboard/
├── app.py                          # Streamlit dashboard (do not modify)
├── pipeline.py                     # Central pipeline orchestrator (do not modify)
├── registry.py                     # Wire your module in here when ready
├── test_discretization.py          # Example of how to test your module (do not include in commits)
├── requirements.txt
├── datasets/
│   └── titanic.csv                 # Default sample dataset
└── modules/
    ├── discretization/
    │   └── discretization.py
    ├── missing_values/
    │   └── missing_values.py
    ├── smoothing/
    │   └── smoothing.py
    ├── data_reduction/
    │   └── data_reduction.py
    ├── histogram_discretization/
    │   └── histogram_discretization.py
    └── similarity/
        └── similarity.py
```

The only files you need to touch are:

- Your own module file inside `modules/your_folder/`
- `registry.py` when your module is ready to be integrated

Do not modify `pipeline.py` or `app.py`.

---

## Setting Up Your Environment

Make sure you have Python 3.10 or above. Then run the following inside the project folder:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Once your venv is active you should see `(.venv)` at the start of your terminal line.
Always make sure this is active before running anything.

To start the dashboard:

```powershell
streamlit run app.py
```

---

## How to Write Your Module

This is the most important part so please read it carefully.

Every module must follow the same structure. The pipeline does not care what your
algorithm does internally, it only cares that your class looks like this:

```python
class YourModuleName:

    METHODS = {
        "Display Name For Dropdown": "internal_key",
        "Another Method": "another_key",
    }

    def __init__(self, method: str = "default_key", column: str | None = None, ...):
        self.method = method
        self.column = column
        # your other params
        self.report_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # do your processing here
        # always work on a copy, never mutate the input
        df = df.copy()
        # ...
        return df

    def get_report(self) -> dict:
        return self.report_
```

Three things are non-negotiable:

1. `METHODS` dict must exist as a class-level attribute
2. `fit_transform(df)` must accept a DataFrame and return a DataFrame
3. `get_report()` must return a dict describing what was done

That is all the pipeline needs from you. You can add as many internal helper methods
as you want, structure your logic however makes sense, use whatever libraries you need.
As long as those three things are there, your module will plug in without any issues.

---

## The METHODS Dict

The dashboard reads your `METHODS` dict to automatically build the method dropdown for
your module. So whatever you put in there is what the user sees.

```python
METHODS = {
    "Mean Imputation": "mean",
    "Median Imputation": "median",
    "KNN Imputation": "knn",
}
```

The key is the display label, the value is what gets passed to your constructor as
`method`. Keep the display names clear and readable since the professor will see them.

---

## Parameters and the Constructor

All your configurable options go in `__init__`. The dashboard passes whatever parameters
are relevant when calling your module, so name them clearly.

Some general guidelines:

- Use `str | None = None` for optional string parameters, not `str = None`
- Use type hints for everything, we are on Python 3.14
- Use `NDArray` from `numpy.typing` for numpy arrays, not bare `np.ndarray`
- Default values should make the module usable out of the box without tweaking

---

## Working With DataFrames

A few rules to keep things consistent across everyone's code:

Always copy the input DataFrame at the start of `fit_transform`:

```python
def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ...
```

Never hardcode a dataset path anywhere in your module. Your module receives a DataFrame,
it does not know or care where that DataFrame came from.

Use `.to_numpy()` instead of `.values` when converting Series or DataFrame columns to
numpy arrays. The type hints work out cleaner and it is the recommended way in newer
pandas versions:

```python
values = df["Age"].to_numpy()
```

Handle NaN values gracefully. Either skip them or fill them, but do not let your code
crash on a column that has missing values. The missing value module will usually run
before others in the pipeline, but we cannot guarantee the order the user applies things.

---

## The get_report Method

This is what gets displayed to the user after your module runs, so make it informative.
Return a flat dict with whatever is most relevant about what your algorithm did:

```python
def get_report(self) -> dict:
    return {
        "method": self.method,
        "column": self.column,
        "values_filled": 12,
        "strategy": "mean",
    }
```

Avoid nested dicts or lists inside the report, keep it flat so it renders cleanly
in the dashboard.

---

## Registering Your Module

Once your module is working correctly, open `registry.py` and:

```python
from modules.missing_values.missing_values import MissingValueEstimator
PreprocessingPipeline.register("Missing Values", MissingValueEstimator)
```

That is all you need to do. The dashboard will automatically pick up your module,
add it to the module dropdown, and build the UI for it.

---

## Testing Your Module Before Integrating

Before you register anything, test your module in isolation. Create a file in the
root of the project called `test_yourmodule.py` and do something like this:

```python
import pandas as pd
from modules.your_folder.your_file import YourClass

df = pd.read_csv("datasets/titanic.csv")

module = YourClass(method="your_method", column="Age")
result = module.fit_transform(df)

print(result.head(20))
print(module.get_report())
```

Check that:

- The output DataFrame has the changes you expected
- NaN rows are handled and do not crash anything
- `get_report()` returns something sensible
- Running it twice on the same DataFrame gives the same result

Only integrate into `registry.py` once all of this is working cleanly.

---

## Code Style

Try to follow these basics so the
codebase stays readable for everyone:

- Keep functions short. If a function is getting long, break it into smaller private
  methods prefixed with an underscore like `_calculate_entropy()`
- Private helper methods go below the public ones in the class
- Write a docstring for your class and for `fit_transform`. You do not need docstrings
  on every single internal helper but the public interface should be documented
- No print statements anywhere. Use `get_report()` to surface information
- No hardcoded magic numbers sitting in the middle of logic. Give them a name:

  ```python
  # bad
  boundaries = np.percentile(values, 5), np.percentile(values, 95)

  # good
  LOWER_PERCENTILE = 5
  UPPER_PERCENTILE = 95
  boundaries = np.percentile(values, LOWER_PERCENTILE), np.percentile(values, UPPER_PERCENTILE)
  ```

---

## Git Practices

When your module is done and tested, open a pull request

Commit messages should say what you actually did:

```
# bad
git commit -m "changes"
git commit -m "fixed stuff"

# good
git commit -m "[ADDED] add median and KNN imputation strategies"
git commit -m "[FIXED] fix NaN handling in fit_transform for object columns"
```

Please do not commit your `.venv` folder, the `.gitignore` already handles this
but double check before pushing.

---
