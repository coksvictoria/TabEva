"""Controlled fault-injection experiments for TabEva metrics.

Generates corrupted synthetic datasets from the Adult real data and runs a
set of TabEva metrics to verify that each fault is detected by the
appropriate metric family (univariate, bivariate, multivariate, PRDC,
and ML utility when a binary target exists).

Outputs per-case `abs_diff` CSVs and a summary CSV in `output/controlled_bivariate/`.
"""
from copy import deepcopy
import os
import pickle

import random
from typing import Dict

import numpy as np
import pandas as pd

from tabeva import TabEva
from tabeva.utils import load_data
from tabeva.metrics.bivariate import bivariate_plots


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REAL_PATH = os.path.join(ROOT, 'synthetic/adult', 'real.csv')
TEST_PATH = os.path.join(ROOT, 'synthetic/adult', 'test.csv')
OUTPUT = os.environ.get("TABEVA_OUTPUT", "output/adult/controlled/")
OUT_DIR = os.path.join(ROOT, 'output', 'controlled')
TARGET_COL = "income"
TARGET_TYPE = "class"

CAT_COLS = [
    "workclass", "education", "marital.status", "occupation",
    "relationship", "race", "sex", "native.country", "income",
]

def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def detect_cat_cols(df: pd.DataFrame):
    return [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() <= 20]


def make_random_data(real: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(index=range(len(real)))
    for c in real.columns:
        if pd.api.types.is_numeric_dtype(real[c]):
            mu, sigma = real[c].mean(), real[c].std()
            sigma = float(sigma) if not np.isnan(sigma) and sigma > 0 else 1.0
            df[c] = rng.normal(loc=float(mu), scale=sigma, size=len(real))
        else:
            vals = real[c].dropna().unique().tolist()
            if len(vals) == 0:
                df[c] = [''] * len(real)
            else:
                df[c] = [random.choice(vals) for _ in range(len(real))]
    return df


def make_shuffled_columns(real: pd.DataFrame) -> pd.DataFrame:
    df = real.copy().reset_index(drop=True)
    # Use a single reproducible RNG to generate a different permutation
    # for each column (so columns are shuffled independently).
    rng = np.random.default_rng(1)
    n = len(df)
    for c in df.columns:
        perm = rng.permutation(n)
        df[c] = df[c].iloc[perm].reset_index(drop=True)
    return df


def destroy_correlations(real: pd.DataFrame) -> pd.DataFrame:
    df = real.copy().reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return make_shuffled_columns(real)
    for col in num_cols:
        df[col] = df[col].sample(frac=1.0, random_state=2 + hash(col) % 1000).reset_index(drop=True)
    return df


def missing_minority_class(real: pd.DataFrame) -> pd.DataFrame:
    df = real.copy().reset_index(drop=True)
    cat_cols = CAT_COLS
    if not cat_cols:
        return df.sample(frac=0.8, random_state=3).reset_index(drop=True)
    candidates = [c for c in cat_cols if df[c].nunique() > 2]
    col = candidates[0] if candidates else cat_cols[0]
    counts = df[col].value_counts()
    drop_val = counts.idxmin()
    fake = df[df[col] != drop_val].reset_index(drop=True)
    if len(fake) < len(df):
        fake = fake.sample(n=len(df), replace=True, random_state=4).reset_index(drop=True)
    return fake


def memorised_records(real: pd.DataFrame) -> pd.DataFrame:
    n = len(real)
    df = real.copy().reset_index(drop=True)
    k = max(1, int(0.01 * n))
    subset = df.sample(n=k, random_state=5)
    fake = df.sample(n=n - 5 * k, replace=True, random_state=6).reset_index(drop=True)
    extras = pd.concat([subset] * 5, ignore_index=True)
    fake = pd.concat([fake, extras], ignore_index=True).sample(frac=1.0, random_state=7).reset_index(drop=True)
    return fake


def mode_collapse(real: pd.DataFrame) -> pd.DataFrame:
    n = len(real)
    df = real.copy().reset_index(drop=True)
    k = max(1, int(0.005 * n))
    modes = df.sample(n=k, random_state=8)
    reps = int(np.ceil(n / k))
    fake = pd.concat([modes] * reps, ignore_index=True).iloc[:n].reset_index(drop=True)
    return fake


def identical(real: pd.DataFrame) -> pd.DataFrame:
    return real.copy().reset_index(drop=True)


def outlier_inflation(real: pd.DataFrame) -> pd.DataFrame:
    df = real.copy().reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df
    col = random.choice(num_cols)
    # inject extreme values in 5% of rows
    n = len(df)
    idx = np.random.choice(n, size=max(1, n // 20), replace=False)
    df.loc[idx, col] = df[col].mean() + 10 * (df[col].std() if df[col].std() > 0 else 1.0)
    return df


# def category_corruption(real: pd.DataFrame) -> pd.DataFrame:
#     df = real.copy().reset_index(drop=True)
#     cat_cols = CAT_COLS
#     if not cat_cols:
#         return df
#     col = random.choice(cat_cols)
#     # replace 10% of entries with an unseen rare token
#     n = len(df)
#     idx = np.random.choice(n, size=max(1, n // 10), replace=False)
#     df.loc[idx, col] = '__RARE__'
#     return df


def run_experiments(real_path: str = REAL_PATH, test_path: str = TEST_PATH):
    ensure_out()
    real = pd.read_csv(real_path)
    df_test = pd.read_csv(test_path)
    cat_cols = CAT_COLS


    cases = {
        # 'identical': identical,
        # 'destroyed_correlations': destroy_correlations,
        # 'shuffled_columns': make_shuffled_columns,
        # 'missing_minority_class': missing_minority_class,
        # 'memorisation': memorised_records,
        # 'mode_collapse': mode_collapse,
        'outlier_inflation': outlier_inflation
        # 'random_rows': make_random_data,
    }

    results: dict = {}
    abs_diffs: dict = {}
    distances: list = []
    for name, fn in cases.items():
        print(f'Running case: {name}')
        fake = fn(real)

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        evs = TabEva(
            real=real,
            fake=fake,
            df_test=df_test,
            cat_cols=cat_cols,
            target_col=TARGET_COL,
            target_type=TARGET_TYPE,
            synthesizer_name=name,
            dir_output=OUTPUT,
        )

        rs, b, d = evs.evaluate()
        results[name] = rs
        abs_diffs[name] = b
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


if __name__ == '__main__':
    run_experiments()
