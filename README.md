# TabEva

TabEva is a concise evaluation toolkit for synthetic tabular data. It computes
univariate, bivariate, multivariate, cluster and sample-level metrics and
produces plots and serialized outputs for downstream analysis.

Quick highlights

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

# TabEva

Lightweight evaluation tools for synthetic tabular data. TabEva computes
univariate, bivariate, multivariate, clustering and sample-level metrics,
produces diagnostic plots, and saves serialized results for downstream use.

Highlights
- Metrics: KS, JS, MMD, PRDC and ML-utility style scores.
- Sample-level distances: 1-NN, centroid and mean distances.
- Example experiments: see `experiments/` for runnable examples.

Install (recommended inside a virtual environment)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# or: source .venv/bin/activate  # Unix
pip install -e .
```

Basic usage

```python
from tabeva import TabEva
from tabeva.utils import load_data

loaded = load_data("adult", "synthetic", ["ctgan.csv"])
real = loaded['real']
fake = loaded['fake']['ctgan']
test = loaded['test']

ev = TabEva(real=real,
                        fake=fake,
                        df_test=test,
                        cat_cols=['workclass','education','income'],
                        target_col='income',
                        target_type='class',
                        synthesizer_name='CTGAN',
                        dir_output='output/adult/')

results, abs_diff, distances = ev.evaluate()
```

Run an example experiment

```bash
python experiments/adult_experiment.py
```

Notes
- Results and plots are written to `output/` by default; consider adding
    `output/` to `.gitignore` for repo cleanliness.
- Dependencies are declared in `pyproject.toml`.

If you want, I can also:
- prune the repo further for publication,
- prepare a minimal `requirements.txt` and CI workflow, or
- create a fresh GitHub repo and push the cleaned code.

**Publication**

TabEva: Unified Tabular Data Synthesis Evaluation Framework — Accepted in Applied Soft Computing.

Authors: Alex X. Wang; Colin R. Simpson; Binh P. Nguyen (corresponding author: binh.p.nguyen@vuw.ac.nz)

Abstract: Generative AI has made significant progress, especially in computer vision and natural language processing. Now, sophisticated algorithms and architectures are expanding into tabular data synthesis, driven by the practical benefits of tabular data's widespread presence in daily life. However, synthesizing tabular data presents unique challenges due to its heterogeneous feature types, necessitating tailored evaluation models. Yet, there is a critical gap in unified and effective evaluation methods specifically designed for synthetic tabular data. To address this gap, we introduce TabEva, a unified and extensible evaluation framework for synthetic tabular data. Rather than proposing entirely new individual metrics, TabEva integrates established univariate, bivariate, multivariate, cluster-level, utility, and unit-record diagnostics into a consistent evaluation pipeline with standardized preprocessing, reporting, and visualization. The framework separates marginal distribution assessment, pairwise dependency preservation, global distributional similarity, support coverage, downstream utility, and post-hoc disclosure-risk diagnostics. In addition, TabEva includes framework-level refinements, such as a modified cluster tendency score, sign-aware association discrepancy handling, and structured visual diagnostics. Through experiments on diverse tabular datasets and synthesis algorithms, we demonstrate that TabEva provides a practical platform for comparing, ranking, and diagnosing synthetic tabular data generators. The results show that TabEva can reveal complementary strengths and weaknesses across different model families, supporting more informed selection and development of tabular data synthesis methods.

**Cite this work**

If you use TabEva in a publication or report, please cite:

Wang, A. X., Simpson, C. R., & Nguyen, B. P. (2026). TabEva: Unified Tabular Data Synthesis Evaluation Framework. Applied Soft Computing. Accepted.

BibTeX entry is available in `CITATION.bib`.




