"""
Example experiment script: evaluate synthetic generators on the Adult dataset.

Usage
-----
    python experiments/adult_experiment.py

Set DATA and GMs below (or pass them via environment variables / CLI args).
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no screen display
import matplotlib.pyplot as plt
import pandas as pd

from tabeva import TabEva
from tabeva.utils import load_data
from tabeva.metrics.bivariate import bivariate_plots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA = os.environ.get("TABEVA_DATA", os.path.join("synthetic"))
OUTPUT = os.environ.get("TABEVA_OUTPUT", "output/adult/")

DATA_FOLDER = "adult"
TARGET_COL = "income"
TARGET_TYPE = "class"

CAT_COLS = [
    "workclass", "education", "marital.status", "occupation",
    "relationship", "race", "sex", "native.country", "income",
]

GMs = [
    "real.csv",
    "simulation.csv",
    # "ADASYN.csv",
    "copula.csv",
    # "copulagan.csv",
    # "ctabgan.csv",
    # "ctgan.csv",
    # "SMOTE.csv",
    # "SMOTENC.csv",
    # "SMOTETomek.csv",
    # "stasy.csv",
    # "synthpop.csv",
    # "tabddpm.csv",
    # "tabsyn.csv",
    # "ttvae.csv",
    # "tvae.csv",
    "twae.csv",
]

os.makedirs(OUTPUT, exist_ok=True)

# Map generator filenames to human-friendly synthesizer names used in reports
SYN_NAME_MAP = {
    "real.csv": "Real",
    "simulation.csv": "Simulation",
    # "ADASYN.csv": "ADASYN",
    "copula.csv": "Copula",
    # "copulagan.csv": "CopulaGAN",
    # "ctabgan.csv": "CTABGAN",
    # "ctgan.csv": "CTGAN",
    # "SMOTE.csv": "SMOTE",
    # "SMOTENC.csv": "SMOTENC",
    # "SMOTETomek.csv": "SMOTE-Tomek",
    # "stasy.csv": "STasy",
    # "synthpop.csv": "synthpop",
    # "tabddpm.csv": "TabDDPM",
    # "tabsyn.csv": "TabSyn",
    # "ttvae.csv": "TTVAE",
    # "tvae.csv": "TVAE",
    "twae.csv": "TWAE",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

loaded = load_data(DATA_FOLDER, DATA, GMs)
real = loaded["real"]
df_test = loaded["test"]

# ---------------------------------------------------------------------------
# Evaluate each synthesiser
# ---------------------------------------------------------------------------

results: dict = {}
abs_diffs: dict = {}
distances: list = []

for fname, fake in loaded["fake"].items():
    print(f"\n{'='*60}")
    # convert filename into a human-friendly synthesizer name
    syn_name = SYN_NAME_MAP.get(fname, os.path.splitext(fname)[0])
    print(f"Evaluating: {syn_name}")
    print(f"{'='*60}")

    evs = TabEva(
        real=real,
        fake=fake,
        df_test=df_test,
        cat_cols=CAT_COLS,
        target_col=TARGET_COL,
        target_type=TARGET_TYPE,
        synthesizer_name=syn_name,
        dir_output=OUTPUT,
    )

    rs, b, d = evs.evaluate()
    results[syn_name] = rs
    abs_diffs[syn_name] = b
    distances.append(d)

# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------

pickle.dump(results, open(os.path.join(OUTPUT, "results.pickle"), "wb"))
pickle.dump(abs_diffs, open(os.path.join(OUTPUT, "abs_diffs.pickle"), "wb"))
pickle.dump(distances, open(os.path.join(OUTPUT, "distances.pickle"), "wb"))

print("\nAll results saved to", OUTPUT)

# ---------------------------------------------------------------------------
# Summary bivariate plot
# ---------------------------------------------------------------------------

bivariate_plots(
    abs_diffs,
    num_col=3,
    figsize=(12, 12),
    filename=os.path.join(OUTPUT, "bivariate", "biv_plot.pdf"),
)

# ---------------------------------------------------------------------------
# Summary results table
# ---------------------------------------------------------------------------

summary = {}
for name, v in results.items():
    summary[name] = pd.DataFrame.from_dict(v, orient="index").ffill().iloc[-1]


summary_df = pd.DataFrame(summary).T.round(2)

print("\n=== Summary ===")
print(summary_df.to_string())
