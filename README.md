# TabEva

**TabEva** is a modular evaluation framework for synthetic tabular data.

## Project Structure

```
TabEva/
├── tabeva/                        # Core library
│   ├── __init__.py                # Exports TabEva class and data_preprocess
│   ├── preprocessing.py           # Label-encoding + MinMax scaling
│   ├── evaluator.py               # TabEva class – main evaluation interface
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── univariate.py          # KS, Mann-Whitney, Levene, Chi-square, KL/JS/EM
│   │   ├── bivariate.py           # Cramer's V, Theil's U, correlation ratio, heatmaps
│   │   ├── multivariate.py        # MMD, PRDC, ML utility, feature importance, PCA plot
│   │   ├── cluster.py             # KMeans cluster distribution comparison
│   │   └── sample.py             # Propensity MSE detection, 1-NN / centroid distances + SHAP
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py         # load_data() helper
├── experiments/
│   └── adult_experiment.py        # Example end-to-end experiment script
├── TabEva_1_0.ipynb               # Original prototype notebook
└── requirements.txt
```

## Metric Modules

| Module | Metrics |
|---|---|
| `univariate` | KS test, Mann-Whitney, Levene, Chi-square, KL/JS divergence, Wasserstein distance |
| `bivariate` | Cramer's V, Theil's U, Correlation Ratio, Matthews CC, pairwise association heatmaps |
| `multivariate` | MMD (linear/RBF/poly), Density & Coverage (PRDC), ML utility (LGBM/RF/LR/DT/MLP), PCA/t-SNE plots |
| `cluster` | KMeans cluster distribution comparison (KS test) |
| `sample` | Propensity score MSE, 1-NN distance, mean distance, centroid distance, SHAP explainability |

## Quick Start

```python
from tabeva import TabEva
from tabeva.utils import load_data

loaded = load_data("adult", "/path/to/data/", ["ctgan.csv", "tabddpm.csv"])

evaluator = TabEva(
    real=loaded["real"],
    fake=loaded["fake"]["ctgan"],
    df_test=loaded["test"],
    cat_cols=["workclass", "education", "income"],
    target_col="income",
    target_type="class",
    synthesizer_name="CTGAN",
)

results, abs_diff, distances = evaluator.evaluate()
```

## Installation

```bash
pip install -r requirements.txt
```

## Using `uv` and `pyproject.toml`

I created a `pyproject.toml` listing the project's dependencies and a `.venv` virtual environment in the project root.

- Install the `uv` CLI (if not already):

```powershell
python -m pip install --user uv
```

- Create the venv (already created by this script):

```powershell
python -m venv .venv
```

- Activate the venv:

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

CMD:

```cmd
.venv\Scripts\activate.bat
```

- Install dependencies into the venv (from `pyproject.toml`):

```powershell
.venv\\Scripts\\python.exe -m pip install -U pip setuptools wheel
.venv\\Scripts\\python.exe -m pip install .
```

Note: installation of `pandas` may fail to build on Python 3.12 if a compatible binary wheel is not available. If you see a build error, either use Python 3.11 or install the required build tools (MSVC) or adjust the `pandas` version in `pyproject.toml`.

