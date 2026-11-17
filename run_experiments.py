"""Run TabEva experiments for one or more configured datasets."""

import argparse
import os
import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "synthetic"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# Edit this list to choose which datasets run by default.
DATASETS_TO_EVALUATE = ["adult"]

DATASET_CONFIG = {
    "adult": {
        "target_col": "income",
        "target_type": "class",
        "cat_cols": [
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native.country",
            "income",
        ],
    },
}


def evaluate_dataset(dataset_name: str) -> None:
    config = DATASET_CONFIG[dataset_name]
    data_folder = DATA_ROOT / dataset_name

    if not data_folder.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_folder}")

    required_files = ("real.csv", "test.csv")
    missing_files = [name for name in required_files if not (data_folder / name).is_file()]
    if missing_files:
        missing = ", ".join(str(data_folder / name) for name in missing_files)
        raise FileNotFoundError(
            f"Cannot evaluate '{dataset_name}'; required data file(s) missing: {missing}"
        )

    synthesizer_files = sorted(
        path.name
        for path in data_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.name not in required_files
    )
    if not synthesizer_files:
        raise FileNotFoundError(
            f"No synthetic CSV files found in {data_folder}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from tabeva.evaluator import TabEva
    from tabeva.metrics.bivariate import bivariate_plots
    from tabeva.utils import load_data

    loaded = load_data(dataset_name, str(DATA_ROOT), synthesizer_files)
    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    abs_diffs = {}
    distances = []

    for synthesizer_name, fake in loaded["fake"].items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name} | Synthesizer: {synthesizer_name}")
        print(f"{'=' * 60}")

        evaluator = TabEva(
            real=loaded["real"],
            fake=fake,
            df_test=loaded["test"],
            cat_cols=config["cat_cols"],
            target_col=config["target_col"],
            target_type=config["target_type"],
            synthesizer_name=synthesizer_name,
            dir_output=str(output_dir),
        )
        dataset_results, abs_diff, distance = evaluator.evaluate()
        results[synthesizer_name] = dataset_results
        abs_diffs[synthesizer_name] = abs_diff
        distances.append(distance)

    with (output_dir / "results.pickle").open("wb") as file:
        pickle.dump(results, file)
    with (output_dir / "abs_diffs.pickle").open("wb") as file:
        pickle.dump(abs_diffs, file)
    with (output_dir / "distances.pickle").open("wb") as file:
        pickle.dump(distances, file)

    bivariate_plots(
        abs_diffs,
        num_col=3,
        figsize=(12, 12),
        filename=os.fspath(output_dir / "bivariate" / "biv_plot.pdf"),
    )

    summary = {
        name: pd.DataFrame.from_dict(values, orient="index").ffill().iloc[-1]
        for name, values in results.items()
    }
    summary_df = pd.DataFrame(summary).T.round(4)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, float_format="%.4f")

    print(f"\nAll results saved to {output_dir}")
    print(f"Saved summary to {summary_path}")
    print("\n=== Summary ===")
    print(summary_df.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic datasets with TabEva."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIG),
        help="Datasets to evaluate (overrides DATASETS_TO_EVALUATE in this file).",
    )
    args = parser.parse_args()
    selected_datasets = args.datasets or DATASETS_TO_EVALUATE

    if not selected_datasets:
        parser.error("Select at least one dataset in DATASETS_TO_EVALUATE or --datasets.")

    for dataset_name in selected_datasets:
        evaluate_dataset(dataset_name)


if __name__ == "__main__":
    main()
