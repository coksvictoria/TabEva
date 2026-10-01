"""
Plot how the `education` vs `education_num` relationship is captured
in each synthetic Adult dataset.

Produces `output/adult/education_relationship.png` containing one subplot
per synthesizer with real vs synthetic mean `education_num` per
`education` category.

Usage:
    python experiments/plot_education_relationship.py
"""
import os
import math
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tabeva.utils import load_data


def list_synth_files(data_folder: str) -> List[str]:
    files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    # exclude real/test
    files = [f for f in files if f not in ('real.csv', 'test.csv')]
    return sorted(files)


def main(data_root: str = os.path.join('synthetic'),
         data_folder_name: str = 'adult',
         output_path: str = os.path.join('output', 'adult', 'education_relationship.png')) -> None:

    data_folder = os.path.join(data_root, data_folder_name)
    synth_files = list_synth_files(data_folder)
    if not synth_files:
        raise FileNotFoundError(f'No synthetic CSV files found in {data_folder}')

    loaded = load_data(data_folder_name, data_root, synth_files)
    real = loaded['real']
    fakes = loaded['fake']

    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine numeric column corresponding to education level (e.g. 'education.num', 'education_num')
    def find_education_num_col(df: pd.DataFrame) -> str:
        candidates = [c for c in df.columns if 'education' in c.lower() and c.lower() != 'education']
        # prefer exact matches
        for pref in ('education_num', 'education.num', 'education-num'):
            if pref in df.columns:
                return pref
        # otherwise pick first candidate that is numeric
        for c in candidates:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        raise KeyError('Could not find numeric education column (education_num) in dataframe')

    edu_num_col = find_education_num_col(real)
    # Determine category order based on real data means
    real_group = real.groupby('education')[edu_num_col].mean()
    order = real_group.sort_values().index.tolist()

    names = list(fakes.keys())
    n = len(names)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = list(axes.flatten()) if hasattr(axes, 'flatten') else [axes]

    for ax_idx, name in enumerate(names):
        ax = axes[ax_idx]
        fake = fakes[name]

        # compute means per education category; align to `order`
        fake_group = fake.groupby('education')[edu_num_col].mean().reindex(order)
        real_vals = real_group.reindex(order)

        x = list(range(len(order)))
        ax.plot(x, real_vals.values, marker='o', color='black', label='Real', linestyle='-')
        ax.plot(x, fake_group.values, marker='o', color='#0077b6', label=name, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('mean education_num')
        ax.set_title(name)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    # hide unused axes
    for j in range(len(names), len(axes)):
        try:
            axes[j].axis('off')
        except Exception:
            pass

    fig.suptitle('Education vs education_num: real vs synthetic (mean per category)')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Wrote plot to {output_path}')


if __name__ == '__main__':
    main()
