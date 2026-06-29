"""Run Friedman/Nemenyi per metric and create a combined CD diagram.

Reads `output/all_ranks.csv` (rows: metric x synthesizer, columns: datasets)
and produces `output/cd_all_metrics.png` containing a subplot per metric.

Usage:
  python experiments/run_cd_all_metrics.py
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from statsmodels.stats.libqsturng import qsturng


def run_friedman_nemenyi(data: pd.DataFrame, alpha: float = 0.05, higher_is_better: bool = True):
    """Run Friedman test and Nemenyi post-hoc (local copy).

    Returns dict with keys: friedman_stat, friedman_p, avg_ranks (Series), nemenyi_pvals (DataFrame or None), cd (float)
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame with columns as methods")

    k = data.shape[1]
    N = data.shape[0]

    if N <= 0 or k <= 0:
        return {
            "friedman_stat": float('nan'),
            "friedman_p": float('nan'),
            "avg_ranks": pd.Series(dtype=float),
            "nemenyi_pvals": None,
            "cd": float('nan'),
        }

    arrays = [data[col].values for col in data.columns]
    try:
        stat, p_value = friedmanchisquare(*arrays)
    except Exception:
        stat, p_value = float('nan'), float('nan')

    ranks = data.rank(axis=1, method='average', ascending=not higher_is_better)
    avg_ranks = ranks.mean().sort_values()

    nemenyi_p = None
    if N > 1 and k > 1 and not np.isnan(p_value):
        try:
            nemenyi_res = sp.posthoc_nemenyi_friedman(data.values)
            nemenyi_p = pd.DataFrame(nemenyi_res, index=data.columns, columns=data.columns)
        except Exception:
            try:
                nemenyi_res = sp.posthoc_nemenyi_friedman(ranks.values)
                nemenyi_p = pd.DataFrame(nemenyi_res, index=data.columns, columns=data.columns)
            except Exception:
                nemenyi_p = None

    try:
        if N > 0 and k > 0:
            q_alpha = float(qsturng(1.0 - alpha, k, np.inf))
            cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * N))
        else:
            cd = float('nan')
    except Exception:
        cd = float('nan')

    return {
        "friedman_stat": float(stat) if not np.isnan(stat) else float('nan'),
        "friedman_p": float(p_value) if not np.isnan(p_value) else float('nan'),
        "avg_ranks": avg_ranks,
        "nemenyi_pvals": nemenyi_p,
        "cd": float(cd),
    }


def draw_cd_on_ax(ax, avg_ranks: pd.Series, cd: float, title: str = None):
    # avg_ranks: Series indexed by method, lower is better
    sorted_sr = avg_ranks.sort_values()
    names = list(sorted_sr.index)
    ranks = sorted_sr.values
    k = len(ranks)

    ys = np.arange(k)
    for y, (nm, r) in zip(ys, zip(names, ranks)):
        ax.plot(r, y, 'o', color='black')
        ax.text(r, y + 0.15, nm, ha='center', va='bottom', fontsize=7, rotation=30)

    ax.set_yticks([])
    ax.set_xlabel('Average Rank')
    if k == 0:
        return

    if cd is None or not np.isfinite(cd):
        ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.8)
    else:
        ax.set_xlim(min(ranks) - 0.5, max(ranks) + cd + 0.8)
        x_start = max(ranks) + 0.2
        y_line = -0.6
        ax.plot([x_start, x_start + cd], [y_line, y_line], color='black', lw=2)
        ax.plot([x_start, x_start], [y_line - 0.05, y_line + 0.05], color='black', lw=2)
        ax.plot([x_start + cd, x_start + cd], [y_line - 0.05, y_line + 0.05], color='black', lw=2)
        ax.text(x_start + cd / 2, y_line - 0.12, f'CD={cd:.2f}', ha='center', fontsize=8)

    if title:
        ax.set_title(title, fontsize=9)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run Friedman/Nemenyi per metric and plot CD diagrams')
    parser.add_argument('--src', default='output/all_ranks.csv')
    parser.add_argument('--out', default='output/cd_all_metrics.png')
    parser.add_argument('--single', action='store_true', help='Create one combined plot overlaying all metrics')
    args = parser.parse_args()

    src = args.src
    out_png = args.out

    df = pd.read_csv(src)
    # detect dataset columns (everything after Synthesizer)
    cols = list(df.columns)
    if 'Metrics' not in cols or 'Synthesizer' not in cols:
        raise SystemExit('Expected columns Metrics and Synthesizer in all_ranks.csv')

    data_cols = [c for c in cols if c not in ('Metrics', 'Synthesizer')]

    metrics = df['Metrics'].unique().tolist()

    results: Dict[str, Tuple[pd.Series, float, float]] = {}
    # compute stats per metric
    for m in metrics:
        sub = df[df['Metrics'] == m].set_index('Synthesizer')
        # columns are datasets, rows are methods; transpose to datasets x methods
        data = sub[data_cols].transpose()
        data = data.apply(pd.to_numeric)
        res = run_friedman_nemenyi(data, higher_is_better=False)
        results[m] = (res['avg_ranks'], res['cd'], res['friedman_p'])

    if not args.single:
        # plotting as separate subplots (existing behaviour)
        n = len(metrics)
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, m in zip(axes, metrics):
            avg_ranks, cd, pval = results[m]
            draw_cd_on_ax(ax, avg_ranks, cd, title=f'{m} (p={pval:.3f})')

        # hide unused axes
        for ax in axes[len(metrics):]:
            ax.axis('off')

        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        print(f'Wrote combined CD diagram to {out_png}')
        return

    # single combined plot: y-axis is methods, x-axis is average rank; each metric plotted with color
    # collect all methods (synthesizers)
    all_methods = list(df['Synthesizer'].unique())
    method_to_y = {m: i for i, m in enumerate(reversed(all_methods))}

    fig_width = 10.5
    fig_height = max(4, len(all_methods) * 0.53)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap('tab10')
    colors = {m: cmap(i % 10) for i, m in enumerate(metrics)}

    # plot points per metric
    for i, m in enumerate(metrics):
        avg_ranks, cd, pval = results[m]
        for method, rank in avg_ranks.items():
            if method not in method_to_y:
                continue
            y = method_to_y[method]
            ax.plot(rank, y, 'o', color=colors[m], alpha=0.9, markersize=6)

    # draw method labels on y-axis
    y_ticks = list(method_to_y.values())
    y_labels = list(reversed(all_methods))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Average Rank')
    # ax.set_title('Average ranks per method across metrics')

    # compute x-limits from data but cap so range does not exceed 10
    all_vals = []
    for avg_ranks, cd, pval in results.values():
        try:
            vals = np.array(list(avg_ranks.values()), dtype=float)
            all_vals.append(vals)
        except Exception:
            continue
    if len(all_vals) > 0:
        all_vals = np.concatenate(all_vals)
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size > 0:
            left = max(0.5, finite.min() - 0.5)
            right = min(10.0, finite.max() + 0.8)
            # ensure a sensible minimum width
            if right - left < 3:
                mid = (left + right) / 2
                left = max(0.5, mid - 4)
                right = min(10.0, mid + 4)
            ax.set_xlim(left, right)

    # add legend for metrics including Friedman p-value
    handles = []
    for m in metrics:
        _, _, pval = results[m]
        label = f'{m} (p={pval:.3f})'
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[m], markersize=6, label=label))

    # place legend centered at bottom with 6 columns to create two rows
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=6, fontsize=9, frameon=False)

    # tighten layout but leave space at bottom for the legend
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_single = out_png.replace('.png', '_single.png')
    fig.savefig(out_single, dpi=300)
    print(f'Wrote single combined plot to {out_single}')


if __name__ == '__main__':
    main()
