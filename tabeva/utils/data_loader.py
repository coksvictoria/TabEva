"""
Data loading utilities for TabEva experiments.
"""

import os
from typing import Dict, List

import pandas as pd


def load_data(data_name: str, data_root: str, synthesizer_files: List[str]) -> Dict:
    """
    Load real, test, and synthetic datasets from a directory.

    Expected directory layout::

        <data_root>/<data_name>/
            real.csv
            test.csv
            <synthesizer_name>.csv   (one per entry in synthesizer_files)

    Parameters
    ----------
    data_name        : Sub-folder name (e.g. 'adult').
    data_root        : Root path that contains the data sub-folders.
    synthesizer_files: List of CSV filenames for synthetic datasets (e.g. ['ctgan.csv', 'tabddpm.csv']).

    Returns
    -------
    dict with keys:
        'name' : dataset name
        'real' : pd.DataFrame
        'test' : pd.DataFrame
        'fake' : dict[str -> pd.DataFrame]  (key = filename without .csv)
    """
    data_folder = os.path.join(data_root, data_name)

    result: Dict = {
        "name": data_name,
        "real": pd.read_csv(os.path.join(data_folder, "real.csv")),
        "test": pd.read_csv(os.path.join(data_folder, "test.csv")),
        "fake": {},
    }

    def _strip_leading_spaces(df: pd.DataFrame) -> pd.DataFrame:
        """Remove leading spaces from string-like (object/category) columns in-place.

        This preserves NaN values and leaves non-string values unchanged.
        """
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = df[col].apply(lambda x: x.lstrip() if isinstance(x, str) else x)
        return df

    # Clean real/test frames
    result["real"] = _strip_leading_spaces(result["real"])
    result["test"] = _strip_leading_spaces(result["test"])

    for filename in synthesizer_files:
        path = os.path.join(data_folder, filename)
        fake = pd.read_csv(path)
        fake = _strip_leading_spaces(fake)
        print(f"Loaded {filename}: {fake.shape}")
        result["fake"][filename[:-4]] = fake  # strip .csv for the key

    return result
