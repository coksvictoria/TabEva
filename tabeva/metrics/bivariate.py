"""
Bivariate metrics: association measures and correlation-matrix comparison
for real vs. synthetic tabular data.
"""

import math
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.metrics import matthews_corrcoef


# ---------------------------------------------------------------------------
# Association helpers
# ---------------------------------------------------------------------------

def conditional_entropy(x, y) -> float:
    """Entropy of x given y."""
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter:
        p_xy = xy_counter[xy] / total
        p_y = y_counter[xy[1]] / total
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theils_u(x, y) -> float:
    """
    Theil's U (Uncertainty Coefficient).
    Asymmetric, range [0, 1]. Measures how much knowing x reduces uncertainty about y.
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total = sum(x_counter.values())
    p_x = [n / total for n in x_counter.values()]
    s_x = stats.entropy(p_x)
    return 1.0 if s_x == 0 else (s_x - s_xy) / s_x


def cramers_v(x, y) -> float:
    """
    Cramer's V: symmetric categorical association, range [0, 1].
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    v = np.sqrt(phi2 / min(k - 1, r - 1))
    if -1e-4 <= v < 1e-4:
        return 0.0
    if 1.0 - 1e-4 < v <= 1.0 + 1e-4:
        return 1.0
    return v


def correlation_ratio(categories, measurements) -> float:
    """
    Correlation Ratio (Eta): categorical–continuous association, range [0, 1].
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(y_avg_array * n_array) / np.sum(n_array)
    numerator = np.sum(n_array * np.power(y_avg_array - y_total_avg, 2))
    denominator = np.sum(np.power(measurements - y_total_avg, 2))
    if numerator == 0:
        return 0.0
    eta = np.sqrt(numerator / denominator)
    if 1.0 < eta <= 1.0 + 1e-4:
        warnings.warn(f"Rounded eta={eta} to 1.0 due to floating point precision.", RuntimeWarning)
        return 1.0
    return eta


# ---------------------------------------------------------------------------
# Pairwise association matrix
# ---------------------------------------------------------------------------

def column_associations(real: pd.DataFrame, c_col: List[str], theil_u: bool = False) -> Tuple[pd.DataFrame, List[float]]:
    """
    Compute a pairwise association matrix across all column pairs.
    - Cat-Cat  : Cramer's V (or Theil's U) / Matthews CC for binary pairs
    - Cat-Num  : Correlation Ratio (or Point-Biserial for binary cat)
    - Num-Num  : Pearson r
    """
    corr = pd.DataFrame(index=real.columns, columns=real.columns, dtype=float)
    b_col = [i for i in c_col if real[i].nunique() == 2]
    cor_coefficients: List[float] = []

    for i, ac in enumerate(corr.columns):
        for j, bc in enumerate(corr.columns):
            if i > j:
                continue
            if ac in c_col and bc in c_col:
                if ac in b_col and bc in b_col:
                    c = matthews_corrcoef(real[ac].values, real[bc].values)
                elif theil_u:
                    c = theils_u(real[ac].values, real[bc].values)
                else:
                    c = cramers_v(real[ac].values, real[bc].values)
            elif ac in b_col or bc in b_col:
                c, _ = stats.pointbiserialr(real[ac].values, real[bc].values)
            elif ac in c_col or bc in c_col:
                c = correlation_ratio(real[ac].values, real[bc].values)
            else:
                c, _ = stats.pearsonr(real[ac].sort_values(), real[bc].sort_values())

            corr.loc[ac, bc] = corr.loc[bc, ac] = c
            cor_coefficients.append(c)

    return corr, cor_coefficients


# ---------------------------------------------------------------------------
# Bivariate test
# ---------------------------------------------------------------------------

def bivariate_test(real: pd.DataFrame, fake: pd.DataFrame, c_col: List[str]) -> Tuple[float, pd.DataFrame]:
    """
    Compare pairwise association matrices of real and fake.
    Returns KS p-value and absolute difference matrix.
    """
    real_corr, r_ce = column_associations(real, c_col, theil_u=False)
    fake_corr, f_ce = column_associations(fake, c_col, theil_u=False)
    _, p = stats.ks_2samp(r_ce, f_ce)

    abs_diff = (real_corr.astype(float) - fake_corr.astype(float)).abs()
    # Mark sign-flipped correlations as maximally different
    sign_flip = (real_corr.astype(float) * fake_corr.astype(float)) < 0
    abs_diff[sign_flip] = 1.0

    return round(p, 4), abs_diff


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def bivariate_plot(abs_diff: pd.DataFrame, figsize: Tuple = (10, 10), filename: str = "") -> None:
    """Heatmap of absolute pairwise association differences."""
    fig, ax = plt.subplots(figsize=figsize)
    rc_mask = np.tril(np.ones_like(abs_diff, dtype=bool))
    fc_mask = np.triu(np.ones_like(abs_diff, dtype=bool))

    sns.heatmap(abs_diff.fillna(np.nan), mask=fc_mask, cmap="Blues", xticklabels=False,
                vmin=0, vmax=1, linewidths=0.5, cbar=False, annot=True, fmt=".2f", annot_kws={"fontsize": 10})
    sns.heatmap(abs_diff.fillna(np.nan), mask=rc_mask, cmap=ListedColormap(["white"]),
                xticklabels=False, cbar=False)
    if filename:
        fig.savefig(filename)
    plt.close(fig)


def bivariate_plots(abs_diffs: Dict[str, pd.DataFrame], num_col: int = 2, figsize: Tuple = (10, 4), filename: str = "") -> None:
    """Multi-panel heatmap comparison across multiple synthesisers."""
    num_row = math.ceil(len(abs_diffs) / num_col)
    fig, axs = plt.subplots(num_row, num_col, figsize=figsize, constrained_layout=True, sharey=True)
    axs = np.array(axs).reshape(num_row, num_col)

    for i, (syn_name, abs_diff) in enumerate(abs_diffs.items()):
        ridx, cidx = i // num_col, i % num_col
        rc_mask = np.tril(np.ones_like(abs_diff, dtype=bool))
        fc_mask = np.triu(np.ones_like(abs_diff, dtype=bool))

        sns.heatmap(abs_diff.fillna(np.nan), mask=fc_mask, cmap="Blues", xticklabels=False,
                    vmin=0, vmax=1, linewidths=0.5, cbar=False, ax=axs[ridx][cidx],
                    annot=False, fmt=".2f", annot_kws={"fontsize": 6})
        sns.heatmap(abs_diff.fillna(np.nan), mask=rc_mask, cmap=ListedColormap(["white"]),
                    ax=axs[ridx][cidx], xticklabels=False, cbar=False)
        axs[ridx][cidx].tick_params(axis="y", labelsize=12)
        axs[ridx][cidx].set_title(syn_name, fontsize=12, loc="center", pad=10)

    if filename:
        fig.savefig(filename, dpi=500, bbox_inches="tight")
    plt.close(fig)
