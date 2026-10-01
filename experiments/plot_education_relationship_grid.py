"""
Create a 2x5 grid plot showing how `education` vs `education_num`
relationship is captured by each specified synthesizer.

Mapping keys may be either filenames (with `.csv`) or synthesizer keys
used by `load_data` (without `.csv`). The script overlays real vs
synthetic means per `education` category; the `Real` panel shows only
the real data.

Usage:
    python experiments/plot_education_relationship_grid.py
"""
import os
import math
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tabeva.utils import load_data


SYN_NAME_MAP = {
    "real.csv": "Real",
    "SMOTE": "SMOTE",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "ctabgan": "CTABGAN",
    "copulagan": "TabDDPM",
    "tabsyn": "TabSyn",
    "SMOTENC": "TTVAE",
    "ttvae": "Tabula",
    "delta": "DELTA",
}


def prepare_synth_file_list(mapping: dict) -> List[str]:
    files = []
    for key in mapping.keys():
        if key.lower() == 'real.csv' or key.lower().endswith('.csv') and key.lower() == 'real.csv':
            continue
        fname = key if key.endswith('.csv') else f"{key}.csv"
        files.append(fname)
    return files


def find_education_num_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if 'education' in c.lower() and c.lower() != 'education']
    for pref in ('education_num', 'education.num', 'education-num'):
        if pref in df.columns:
            return pref
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise KeyError('Could not find numeric education column (education_num) in dataframe')


def main(data_root: str = os.path.join('synthetic'),
         data_folder_name: str = 'adult',
         output_path: str = os.path.join('output', 'adult', 'education_relationship_grid.png')) -> None:

    synth_files = prepare_synth_file_list(SYN_NAME_MAP)
    loaded = load_data(data_folder_name, data_root, synth_files)
    real = loaded['real']
    fakes = loaded['fake']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    edu_num_col = find_education_num_col(real)
    real_group = real.groupby('education')[edu_num_col].mean()
    order = real_group.sort_values().index.tolist()

    # exclude the explicit Real entry and limit to 9 synthesizers for a 3x3 grid
    keys = [k for k in SYN_NAME_MAP.keys() if k.lower() != 'real.csv']
    keys = keys[:9]
    labels = [SYN_NAME_MAP[k] for k in keys]

    # 3 rows x 3 cols layout
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        label = SYN_NAME_MAP[key]

        # key may be with or without .csv; loaded fake keys are without .csv
        fake_key = key[:-4] if key.endswith('.csv') else key
        fake = fakes.get(fake_key)
        if fake is None:
            # draw only real if synthetic missing
            ax.plot(range(len(order)), real_group.reindex(order).values, marker='o', color='black', label='Real')
        else:
            fake_group = fake.groupby('education')[edu_num_col].mean().reindex(order)
            ax.plot(range(len(order)), real_group.reindex(order).values, marker='o', color='black', label='Real')
            ax.plot(range(len(order)), fake_group.values, marker='o', color='#0077b6', linestyle='--', label=label)

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('mean education_num')
        ax.set_title(label)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(fontsize=8)

    # turn off unused axes (if any)
    for j in range(len(keys), len(axes)):
        axes[j].axis('off')

    # fig.suptitle('education vs education.num: Real vs Synthetic (mean per category)')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Wrote plot to {output_path}')


if __name__ == '__main__':
    main()
