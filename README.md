# TabEva: Unified Tabular Data Synthesis Evaluation Framework

#### Alex X. Wang (https://people.wgtn.ac.nz/alex.wang); Colin R. Simpson; Binh P. Nguyen

## Abstract
Generative AI has made significant progress, especially in computer vision and natural language processing. Now, sophisticated algorithms and architectures are expanding into tabular data synthesis, driven by the practical benefits of tabular data's widespread presence in daily life. However, synthesizing tabular data presents unique challenges due to its heterogeneous feature types, necessitating tailored evaluation models. Yet, there is a critical gap in unified and effective evaluation methods specifically designed for synthetic tabular data. To address this gap, we introduce TabEva, a unified and extensible evaluation framework for synthetic tabular data. Rather than proposing entirely new individual metrics, TabEva integrates established univariate, bivariate, multivariate, cluster-level, utility, and unit-record diagnostics into a consistent evaluation pipeline with standardized preprocessing, reporting, and visualization. The framework separates marginal distribution assessment, pairwise dependency preservation, global distributional similarity, support coverage, downstream utility, and post-hoc disclosure-risk diagnostics. In addition, TabEva includes framework-level refinements, such as a modified cluster tendency score, sign-aware association discrepancy handling, and structured visual diagnostics. Through experiments on diverse tabular datasets and synthesis algorithms, we demonstrate that TabEva provides a practical platform for comparing, ranking, and diagnosing synthetic tabular data generators. The results show that TabEva can reveal complementary strengths and weaknesses across different model families, supporting more informed selection and development of tabular data synthesis methods.



#### Install (recommended in a venv)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # or .venv/bin/activate on Unix
pip install -e .
```

#### Quick usage

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

#### Run the example experiment
```bash
python experiments/adult_experiment.py
```

Notes
- Results and plots are written to `output/` by default; consider adding
    `output/` to `.gitignore` for repo cleanliness.
- Dependencies are declared in `pyproject.toml`.


## Reference
We appreciate your citations if you find this repository useful to your research!
```
@article{wang2026tabeva,
  title={TabEva: Unified Tabular Data Synthesis Evaluation Framework},
  author={Alex X. Wang, Colin R. Simpson and Binh P. Nguyen},
  journal={Applied Soft Computing},
  year={2026},
  publisher={Elsevier}
}
```




