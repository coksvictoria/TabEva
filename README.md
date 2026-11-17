# TabEva

TabEva is a Python framework for evaluating synthetic tabular data. It provides
univariate, bivariate, multivariate, cluster, utility, and record-level
diagnostics.

## Install

Requires Python 3.12 or newer and Git. Install the latest version from GitHub:

```bash
python -m pip install git+https://github.com/coksvictoria/TabEva.git
```

To work from a local checkout instead:

```bash
git clone https://github.com/coksvictoria/TabEva.git
cd TabEva
python -m pip install -e .
```

## Quick start

The repository includes Adult example data in `synthetic/adult/`:

```text
real.csv
test.csv
SMOTE.csv
ctabgan.csv
tabddpm.csv
```

From the repository root, evaluate one included synthetic dataset:

```python
from tabeva import TabEva
from tabeva.utils import load_data

data = load_data("adult", "synthetic", ["SMOTE.csv"])
evaluator = TabEva(
    real=data["real"],
    fake=data["fake"]["SMOTE"],
    df_test=data["test"],
    cat_cols=[
        "workclass", "education", "marital.status", "occupation",
        "relationship", "race", "sex", "native.country", "income",
    ],
    target_col="income",
    target_type="class",
    synthesizer_name="SMOTE",
    dir_output="output/adult/",
)
results, abs_diffs, distances = evaluator.evaluate()
```

`load_data(data_name, data_root, synthesizer_files)` expects each dataset folder
to contain `real.csv`, `test.csv`, and the synthetic CSV files you want to load.
For your own data, pass its root directory as `data_root`.

## Run the included experiment

The root-level runner evaluates every synthetic CSV in the selected dataset
folder and saves reports under `output/<dataset>/`:

```bash
python run_experiments.py --datasets adult
```

Edit `DATASETS_TO_EVALUATE` in `run_experiments.py` to change the default
dataset selection. Dataset-specific target and categorical-column settings are
defined in `DATASET_CONFIG`.

## Citation

If you use TabEva in your research, please cite:

```bibtex
@article{wang2026tabeva,
  title={TabEva: Unified tabular data synthesis evaluation framework},
  author={Wang, Alex X and Simpson, Colin R and Nguyen, Binh P},
  journal={Applied Soft Computing},
  pages={116553},
  year={2026},
  publisher={Elsevier}
}
```
