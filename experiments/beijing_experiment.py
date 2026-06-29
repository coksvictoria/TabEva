"""
Example experiment script: evaluate synthetic generators on the Beijing dataset.

Usage
-----
    python experiments/beijing_experiment.py

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
OUTPUT = os.environ.get("TABEVA_OUTPUT", "output/beijing/")

DATA_FOLDER = "beijing"
TARGET_COL = "pm2.5"
TARGET_TYPE = "regr"

# Categorical columns for Beijing (wind direction is categorical)
CAT_COLS = [
    "cbwd",
]

GMs = [
    "SMOTE.csv",
    "tvae.csv",
    "copulagan.csv",
    "ctabgan.csv",
    "copula.csv",
    "tabddpm.csv",
    "SMOTENC.csv",
    "ttvae.csv",
    "tabsyn.csv",
]

os.makedirs(OUTPUT, exist_ok=True)

# Map generator filenames to human-friendly synthesizer names used in reports
SYN_NAME_MAP = {
    "copula": "DP-CTGAN",
    "copulagan": "CTGAN",
    "ctabgan": "CTABGAN",
    "SMOTE": "SMOTE",
    "SMOTENC": "GReaT",
    "tabddpm": "TabDDPM",
    "tabsyn": "TabSyn",
    "ttvae": "Tabula",
    "tvae": "TVAE",
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

summary_df = pd.DataFrame(summary).T.round(4)

out_path = os.path.join(OUTPUT, "summary.csv")
summary_df.to_csv(out_path, float_format="%.4f")  # saves with 4 decimal places
print("Saved summary to", out_path)



print("\n=== Summary ===")
print(summary_df.to_string())
