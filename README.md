# ML Dashboard

A data preprocessing and machine learning dashboard built with Python and Streamlit.
This is our mini project for the Data Mining course. The goal is to have a single
interactive dashboard where we can apply all our preprocessing techniques to a dataset,
visualize what each step does in real time, and immediately train and evaluate Machine Learning Classification Models on that data.

---

## Project Structure

```
ML-Dashboard/
├── app.py                          # Main Streamlit dashboard UI (do not modify core logic)
├── pipeline.py                     # Central preprocessing orchestrator (Data Engineering)
├── ml_pipeline.py                  # Central machine learning orchestrator (Model Training)
├── registry.py                     # Wire your new modules and models here when ready
├── requirements.txt
├── datasets/
│   └── titanic.csv                 # Default sample dataset
└── modules/                        # Preprocessing and ML Modules live here
    ├── discretization/
    │   └── discretization.py
    ├── missing_values/
    │   └── missing_values.py
    ├── smoothing/
    │   └── smoothing.py
    ├── data_reduction/
    │   └── data_reduction.py
    ├── encoding/
    │   └── encoding.py
    ├── histogram_disc/
    │   └── histogram_discretization.py
    ├── similarity/
    │   └── similarity.py
    └── ClassifierModels/
        └── DecisionTree.py
```

The primary files you need to touch when adding new features are:

- Your own module file inside `modules/your_folder/`
- `registry.py` (to register your module to the correct pipeline)
- `app.py` (only to add UI inputs for your module's parameters inside `render_module_params` or `render_model_params`)

---

## Setting Up Your Environment

Make sure you have Python 3.10 or above. Then run the following inside the project folder:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

*(Note: Adjust the python command depending on your OS/version, e.g., `python3 -m venv .venv`)*

Once your venv is active you should see `(.venv)` at the start of your terminal line.
Always make sure this is active before running anything.

To start the dashboard locally:

```powershell
streamlit run app.py
```
This will automatically open the interactive dashboard in your web browser.

---

## How the Dashboard Works

The dashboard is split into two primary modes (toggled via the Sidebar navigation):

### 1. Data Preprocessing Pipeline
This mode allows you to load raw data, apply transformations sequentially, and visualize the results.
It maintains a strict history (Undo/Reset functionality) and displays Before/After comparisons. 
- **Transformation Modules** (e.g., Missing Values, Encoding): These actually change the DataFrame and are appended to history.
- **Analysis Modules** (e.g., Similarity, Histogram): These summarize or analyze data *without* changing the actual DataFrame state.

### 2. Classifier Models
This mode allows you to pick the cleaned dataset from the Preprocessing phase and train a Machine Learning classification model on it. 
It handles the Train/Test splitting securely, dropping NAs automatically for selected features. It displays:
- **Training Summary:** Hyperparameters, tree statistics, split sizes.
- **Evaluation:** Scikit-Learn Classification Report (Accuracy, Precision, Recall) and an interactive Plotly Confusion Matrix.

*(Note: Categorical features must be converted using the `Encoding` module in the Preprocessing tab before they can be used in the Classifier tab).*

---

## How to Write Your Preprocessing Module

Every transformation module must follow the same duck-typed structure. The pipeline does not care what your algorithm does internally, it only cares that your class looks like this:

```python
class YourModuleName:

    # 1. Non-negotiable dict mapping Dropdown labels -> internal keys
    METHODS = {
        "Display Name For Dropdown": "internal_key",
    }

    def __init__(self, method: str = "default_key", column: str | None = None, ...):
        self.method = method
        self.column = column
        # your other params
        self.report_: dict = {}

    # 2. Must accept and return a DataFrame
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy() # always copy!
        # do your processing here...
        return df

    # 3. Must return a flat dictionary
    def get_report(self) -> dict:
        return self.report_
```

---

## How to Write Your Machine Learning Model

ML Models follow a slightly different contract and are handled by `MLPipeline`.

```python
class YourModelModule:

    METHODS = {
        "Gini Impurity": "gini",
        "Entropy": "entropy",
    }

    def __init__(self, method: str = "gini", **kwargs):
        self.method = method
        # instantiate your sklearn model here
        self.model = ...
        self.report_: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.report_ = {"Model": "Name", "Hyperparam": self.method}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_report(self) -> dict:
        return self.report_
```

---

## Registering Your Module

Once your module is working correctly, open `registry.py` and register it:

**For a Preprocessing Module:**
```python
from modules.your_folder.your_file import YourModule
PreprocessingPipeline.register("Your Feature", YourModule)
```

**For an ML Model:**
```python
from modules.ClassifierModels.YourModel import YourModel
MLPipeline.register_model("Your Model", YourModel)
```

Finally, open `app.py` and add the UI sliders/dropdowns for your specific parameters inside `render_module_params(module_name)` or `render_model_params(model_name)`.

---

## Code Style & DataFrames

- **Copying Data:** Always copy the input DataFrame at the start of `fit_transform` (`df = df.copy()`).
- **Numpy:** Use `.to_numpy()` instead of `.values` when converting pandas Series to arrays. Use `NDArray` from `numpy.typing` for type hints.
- **NaN Handling:** Handle NaN values gracefully in preprocessing. Either skip them or fill them, but do not let your code crash. (The ML pipeline automatically drops NaNs before fitting the model, but preprocessing modules must be defensive).
- **No Hardcoded Paths:** Your module receives a DataFrame, it does not know or care where that DataFrame came from.
- **Flat Reports:** `get_report()` must return a flat dictionary (no nested objects).
- **Docstrings:** Write docstrings for your class and public methods. Keep internal helpers private with an underscore (e.g., `_my_helper()`).

---

## Git Practices

When your module is done and tested, commit it with a clear, descriptive message:

```
# bad
git commit -m "changes"

# good
git commit -m "[ADDED] add median and KNN imputation strategies"
git commit -m "[FIXED] fix NaN handling in fit_transform for object columns"
```

Please do not commit your `.venv` folder or test scratchpad files.

---
