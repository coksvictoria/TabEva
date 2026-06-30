# TabEva

TabEva is a concise evaluation toolkit for synthetic tabular data. It computes
univariate, bivariate, multivariate, cluster and sample-level metrics and
produces plots and serialized outputs for downstream analysis.

Quick highlights
- Evaluate synthetic data quality (KS, JS, MMD, PRDC, ML utility).
- Per-record distances: 1-NN, mean and centroid distances.
- Example experiment script: `experiments/adult_experiment.py`.
- Simple plotting helper: `scripts/plot_adult_distances_simple.py`.

Install (recommended in a venv)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # or .venv/bin/activate on Unix
pip install -e .
```

Quick usage
```python
from tabeva import TabEva
from tabeva.utils import load_data
loaded = load_data("adult", "synthetic", ["ctgan.csv"])
ev = TabEva(loaded['real'], loaded['fake']['ctgan'], loaded['test'],
            cat_cols=['workclass','education','income'],
            target_col='income', target_type='class', synthesizer_name='CTGAN')
results, abs_diff, distances = ev.evaluate()
```

Run the example experiment
```bash
python experiments/adult_experiment.py
```

Plot saved distances (per the example)
```bash
python scripts/plot_adult_distances_simple.py
```

Notes
- Plots and pickles are written under `output/` by default.
- Core dependencies are listed in `pyproject.toml`.


