"""
Univariate metrics: statistical tests and distribution distance measures
for comparing real vs. synthetic tabular data column-by-column.
"""

import math
import warnings
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import spatial, stats
from scipy.special import kl_div


# ---------------------------------------------------------------------------
# Statistical tests – numerical columns
# ---------------------------------------------------------------------------

def kolmogorov_smirnov_test(col_name: str, real_col, fake_col) -> Dict:
    """Two-sample KS test: checks whether real and fake follow the same distribution."""
    statistic, p_value = stats.ks_2samp(real_col, fake_col)
    equality = "identical" if p_value > 0.01 else "different"
    return {"col_name": col_name, "ks_statistic": statistic, "ks_p-value": p_value, "equal distribution": equality}


# Note: some statistical-test helpers (mood_test, mannwhitneyu) were removed
# because they are no longer used by the evaluator. Keep this file focused
# on the tests and divergence measures currently consumed by `TabEva`.


def num_statistics_df(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    stats_func: Callable,
    numerical_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply a statistical test function over all numerical columns in parallel."""
    assert real.columns.tolist() == fake.columns.tolist(), "Columns are not identical between `real` and `fake`."
    real_iter = real[numerical_columns].items()
    fake_iter = fake[numerical_columns].items()
    distances = Parallel(n_jobs=-1)(
        delayed(stats_func)(colname, real_col, fake_col)
        for (colname, real_col), (_, fake_col) in zip(real_iter, fake_iter)
    )
    distances_df = pd.DataFrame(distances).set_index("col_name")
    distances_df.loc["mean"] = distances_df.mean(numeric_only=True)
    return distances_df


# ---------------------------------------------------------------------------
# Statistical tests – categorical columns
# ---------------------------------------------------------------------------

def get_frequencies(real, synthetic, percent: bool = True):
    """Return observed and expected frequency lists for chi-square tests."""
    f_obs, f_exp = [], []
    real_counter, synthetic_counter = Counter(real), Counter(synthetic)
    for value in synthetic_counter:
        if value not in real_counter:
            warnings.warn(f"Unexpected value {value} in synthetic data.")
            real_counter[value] += 1e-6  # regularisation to prevent NaN

    if percent:
        for value in real_counter:
            f_obs.append(synthetic_counter[value] / sum(synthetic_counter.values()))
            f_exp.append(real_counter[value] / sum(real_counter.values()))
    else:
        for value in real_counter:
            f_obs.append(synthetic_counter[value])
            f_exp.append(real_counter[value])
    return f_obs, f_exp


def chisquare_test(col_name: str, real_col, fake_col) -> Dict:
    """Total variation distance between categorical distributions.

    Replaces Chi-square with TVD = 0.5 * sum |p_synth - p_real| (range [0,1]).
    """
    f_obs, f_exp = get_frequencies(real_col, fake_col)
    # f_obs/f_exp are percentages (sums to 1). TVD = 0.5 * L1 distance
    tvd = 0.5 * sum(abs(np.asarray(f_obs) - np.asarray(f_exp)))
    equality = "identical" if tvd < 0.01 else "different"
    return {"col_name": col_name, "ct_statistic": float(tvd), "ct_p-value": np.nan, "equal frequency": equality}


def cat_statistics_df(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    stats_func: Callable,
    categorical_columns: List[str],
) -> pd.DataFrame:
    """Apply a statistical test function over all categorical columns in parallel."""
    assert real.columns.tolist() == fake.columns.tolist(), "Columns are not identical between `real` and `fake`."
    real_iter = real[categorical_columns].items()
    fake_iter = fake[categorical_columns].items()
    distances = Parallel(n_jobs=-1)(
        delayed(stats_func)(colname, real_col, fake_col)
        for (colname, real_col), (_, fake_col) in zip(real_iter, fake_iter)
    )
    distances_df = pd.DataFrame(distances).set_index("col_name")
    distances_df.loc["mean"] = distances_df.mean(numeric_only=True)
    return distances_df


# ---------------------------------------------------------------------------
# Distribution distance measures
# ---------------------------------------------------------------------------

def get_frequency(X_gt: pd.DataFrame, X_synth: pd.DataFrame, c_col: List[str], n_histogram_bins: int = 10) -> dict:
    """
    Compute percentual frequency distributions for each column.
    Categorical columns use value counts; numerical columns use histogram bins.
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if col in c_col:
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        # Align keys deterministically (sorted union) so returned probability
        # vectors correspond elementwise for both real and synthetic.
        eps = 1e-11
        all_keys = sorted(set(list(gt.keys()) + list(synth.keys())))
        gt_list = [gt.get(k, eps) if gt.get(k, 0) != 0 else eps for k in all_keys]
        synth_list = [synth.get(k, eps) if synth.get(k, 0) != 0 else eps for k in all_keys]
        res[col] = (gt_list, synth_list)
    return res


def js_divergence(colname: str, x, y) -> Dict:
    """Jensen-Shannon distance between two probability distributions."""
    js = spatial.distance.jensenshannon(x, y)
    return {"col_name": colname, "js_distance": js}


def num_divergence_df(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    stats_func: Callable,
    cat_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply a divergence function over all columns using histogram-based frequencies."""
    freqs = get_frequency(real, fake, cat_columns or [])
    res = {}
    for col in real.columns:
        real_freq, fake_freq = freqs[col]
        res[col] = stats_func(col, real_freq, fake_freq)
    distances_df = pd.DataFrame(res).T.set_index("col_name")
    distances_df = distances_df.apply(pd.to_numeric, errors="coerce")
    return distances_df


# `num_distance_df` removed — numeric divergences are computed via histogram
# frequencies with `num_divergence_df` and `js_divergence`.


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def univariate_num_plot(real: pd.DataFrame, fake: pd.DataFrame, n_col: List[str], num_col: int, top_n: int = 10, filename: str = ""):
    """Side-by-side histogram comparison for numerical columns."""
    real, fake = real.copy(), fake.copy()
    num_row = math.ceil(len(n_col) / num_col)
    fig, ax = plt.subplots(num_row, num_col, figsize=(4.5 * num_col, 3.5 * num_row))

    for i in range(len(n_col)):
        ridx, cidx = i // num_col, i % num_col
        col_name = n_col[i]
        frq, edges = np.histogram(real[col_name], bins=top_n)
        frq2, edges2 = np.histogram(fake[col_name], bins=edges)
        ax[ridx][cidx].bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", color="#0077b6", align="edge", alpha=0.6)
        ax[ridx][cidx].bar(edges2[:-1], frq2, width=np.diff(edges2), edgecolor="white", align="edge", color="#caf0f8", alpha=0.8)
        ax[ridx][cidx].set_xlabel(col_name, fontsize=11)
        ax[ridx][cidx].set_ylabel(None)
        ax[ridx][cidx].tick_params(axis="both", labelsize=10)

    ax[0][num_col - 1].legend(["real", "fake"], prop={"size": 12})
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    return plt


def univariate_cat_plot(real: pd.DataFrame, fake: pd.DataFrame, c_col: List[str], num_col: int, top_n: int = 10, filename: str = ""):
    """Side-by-side bar chart comparison for categorical columns."""
    real, fake = real.copy(), fake.copy()
    num_row = math.ceil(len(c_col) / num_col)
    fig, ax = plt.subplots(num_row, num_col, figsize=(4.5 * num_col, 3.5 * num_row))

    for i in range(len(c_col)):
        ridx, cidx = i // num_col, i % num_col
        order = real[c_col[i]].value_counts().iloc[:top_n].index
        sns.countplot(real, x=c_col[i], alpha=0.6, color="#0077b6", edgecolor="black", ax=ax[ridx][cidx], order=order)
        sns.countplot(fake, x=c_col[i], alpha=0.8, color="#caf0f8", edgecolor="white", ax=ax[ridx][cidx], order=order)
        ax[ridx][cidx].set_xlabel(c_col[i], fontsize=11)
        ax[ridx][cidx].set(xticklabels=[])
        ax[ridx][cidx].set_ylabel(None)

    ax[0][num_col - 1].legend(["real", "fake"], prop={"size": 12})
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return plt
