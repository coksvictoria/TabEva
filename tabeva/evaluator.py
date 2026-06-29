"""
TabEva: Main evaluator class for synthetic tabular data.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from tabeva.preprocessing import data_preprocess
from tabeva.metrics.univariate import (
    kolmogorov_smirnov_test,
    chisquare_test,
    num_statistics_df,
    cat_statistics_df,
    js_divergence,
    num_divergence_df,
    univariate_num_plot,
    univariate_cat_plot,
)
from tabeva.metrics.bivariate import bivariate_test as _bivariate_test, bivariate_plot
from tabeva.metrics.multivariate import table_plot, mmd_kernel, compute_prdc, ml_evaluation
from tabeva.metrics.cluster import cluster_df
from tabeva.metrics.sample import ml_detection, record_df, plot_nnd


class TabEva:
    """
    Evaluator for synthetic tabular data.

    Wraps univariate, bivariate, multivariate, cluster, and sample-level metrics
    into a single convenient interface.

    Parameters
    ----------
    real            : Real (training) dataset.
    fake            : Synthetic dataset.
    df_test         : Held-out test set (used for ML utility evaluation).
    cat_cols        : Column names to treat as categorical.
    target_col      : Name of the prediction target column.
    target_type     : 'class' for classification, 'regr' for regression.
    synthesizer_name: Label for the synthesiser (used in output file names).
    dir_output      : Optional directory for saving output files.
    """

    def __init__(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        df_test: pd.DataFrame,
        cat_cols: List[str],
        target_col: str,
        target_type: str,
        synthesizer_name: str,
        dir_output: Optional[str] = None,
    ) -> None:
        self.real = real.copy()
        self.fake = fake.copy()
        self.df_test = df_test.copy()
        # Strip leading spaces in string-like categorical columns for all datasets
        def _strip_leading_spaces_df(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include=["object", "category"]).columns:
                df[col] = df[col].apply(lambda x: x.lstrip() if isinstance(x, str) else x)
            return df

        self.real = _strip_leading_spaces_df(self.real)
        self.fake = _strip_leading_spaces_df(self.fake)
        self.df_test = _strip_leading_spaces_df(self.df_test)
        self.categorical_columns = cat_cols
        self.numerical_columns = [col for col in self.real.columns if col not in self.categorical_columns]

        self.synthesizer_name = synthesizer_name
        self.target_col = target_col
        self.target_type = target_type
        self.dir_output = dir_output

        if dir_output:
            import os
            for subdir in ("univariate", "bivariate", "multivariate", "cluster", "sample"):
                os.makedirs(os.path.join(dir_output, subdir), exist_ok=True)

        r_r, r_c = self.real.shape
        f_r, f_c = self.fake.shape
        print("Datasets uploading-------------------")
        print(f"Real dataset      : {r_r} rows × {r_c} columns")
        print(f"  Unique records  : {len(self.real.drop_duplicates())}")
        print(f"Synthetic dataset : {f_r} rows × {f_c} columns")
        print(f"  Unique records  : {len(self.fake.drop_duplicates())}")

        # Align column order
        if len(real.columns) == len(fake.columns):
            self.fake = self.fake[self.real.columns.tolist()]
        assert self.real.columns.tolist() == self.fake.columns.tolist(), (
            "Columns in real and fake dataframe are not the same."
        )

        # Handle missing values
        self._fill_missing(self.real)
        self._fill_missing(self.fake)

        # Preprocessed versions: label-encoded + scaled (realb/fakeb)
        # and one-hot-encoded (realo/fakeo)
        self.realb, self.fakeb = data_preprocess(self.real, self.fake, self.categorical_columns)
        self.realo, self.fakeo = data_preprocess(self.real, self.fake, self.categorical_columns, onehot=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_missing(self, df: pd.DataFrame) -> None:
        """In-place missing value imputation."""
        na_count = [df[cc].isnull().sum() for cc in df.columns]
        if sum(na_count) > 0:
            na_df = pd.DataFrame({"column": df.columns, "na_count": na_count}).query("na_count > 0")
            print(na_df.sort_values("na_count", ascending=False).to_string(index=False))
            df.loc[:, self.categorical_columns] = (
                df.loc[:, self.categorical_columns].fillna("[NAN]").astype(str)
            )
            df.loc[:, self.numerical_columns] = (
                df.loc[:, self.numerical_columns].fillna(df[self.numerical_columns].mean())
            )

    def _output_path(self, filename: str, subdir: str = "") -> str:
        """Prepend dir_output (and optional subdir) if set."""
        if self.dir_output:
            import os
            base = os.path.join(self.dir_output, subdir) if subdir else self.dir_output
            return os.path.join(base, filename)
        return filename

    # ------------------------------------------------------------------
    # Metric methods
    # ------------------------------------------------------------------

    def univariate_test(self, num_col: Optional[int] = None) -> Dict:
        """
        Run univariate statistical tests and distance measures.

        Parameters
        ----------
        num_col : subplot column count for distribution plots (None = skip plots).
        """
        # Use Jensen-Shannon distance as the primary univariate metric.
        # For categorical columns: empirical category probabilities.
        # For numerical columns: bin values using bins fitted on the real data.
        output: Dict = {}
        # Compute KS D for numerical columns (aggregation: mean of ks_statistic)
        ks_df = num_statistics_df(self.real, self.fake, kolmogorov_smirnov_test, self.numerical_columns)
        output["ks"] = float(ks_df["ks_statistic"].mean())

        # Compute TVD for categorical columns (aggregation: mean of ct_statistic)
        ct_df = cat_statistics_df(self.real, self.fake, chisquare_test, self.categorical_columns)
        output["tvd"] = float(ct_df["ct_statistic"].mean())

        # Compute JS distance for all columns using histogram/probabilities.
        js_df = num_divergence_df(self.real, self.fake, js_divergence, self.categorical_columns)
        # js_df has a column 'js_distance' indexed by column name
        # Mean JS across numerical columns
        num_cols = [c for c in self.numerical_columns if c in js_df.index]
        cat_cols = [c for c in self.categorical_columns if c in js_df.index]
        # Overall mean JS across all columns
        output["js"] = float(js_df["js_distance"].mean())

        if num_col is not None:
            univariate_num_plot(
                self.real, self.fake, self.numerical_columns, num_col, top_n=10,
                filename=self._output_path(f"{self.synthesizer_name}_uni_n.pdf", "univariate"),
            )
            univariate_cat_plot(
                self.real, self.fake, self.categorical_columns, num_col, top_n=10,
                filename=self._output_path(f"{self.synthesizer_name}_uni_c.pdf", "univariate"),
            )
        return output

    def bivariate_test(self, figsize: Optional[Tuple] = None) -> Tuple[Dict, pd.DataFrame]:
        """
        Compare pairwise association matrices.

        Returns
        -------
        result : dict with KS p-value for correlation distributions
        abs_diff : DataFrame of absolute differences between association matrices
        """
        p, abs_diff = _bivariate_test(self.realb, self.fakeb, self.categorical_columns)
        if figsize is not None:
            bivariate_plot(
                abs_diff, figsize=figsize,
                filename=self._output_path(f"{self.synthesizer_name}_biv.pdf", "bivariate"),
            )
        return {"corr": p}, abs_diff

    def multivariate_test(self) -> Dict:
        """
        Dimensionality-reduction scatter plot (PCA).
        """
        # Dimensionality-reduction plot (PCA/t-SNE)
        table_plot(
            self.realb, self.fakeb,
            dimensionality_reduction="PCA",
            filename=self._output_path(self.synthesizer_name + "_table", "multivariate"),
        )

        output: Dict = {}

        # Maximum Mean Discrepancy (MMD) on the preprocessed (numeric) features
        mmd_value = mmd_kernel(self.realb.values, self.fakeb.values)
        output["mmd"] = mmd_value

        # Precision / Recall / Density / Coverage (use nearest_k=5)
        
        density, coverage = compute_prdc(self.realb.values, self.fakeb.values, nearest_k=5)
        output["density"] = float(density)
        output["coverage"] = float(coverage)

        # ML utility: train on synthetic, evaluate on real test set
        ml_metric = ml_evaluation(
            self.real,
            self.fake,
            self.df_test,
            self.categorical_columns,
            self.numerical_columns,
            self.target_col,
            self.target_type,
            model_name=self.synthesizer_name,
            filename=self._output_path(self.synthesizer_name + "_ml", "multivariate"),
        )
        # ml_evaluation returns a single aggregated float (avg F1 or avg RMSE)
        output["ml"] = float(ml_metric) if ml_metric is not None else float("nan")

        return output

    def cluster_test(
        self,
        n_cluster: int = 20,
        num_col: Optional[int] = 4,
        figsize: Tuple = (10, 12.5),
        filename: str = "",
    ) -> Dict:
        """
        KMeans-based cluster distribution comparison.

        Returns
        -------
        dict with KS p-value for cluster count distributions.
        """
        p = cluster_df(
            self.realo, self.fakeo,
            n_cluster=n_cluster,
            num_col=num_col,
            figsize=figsize,
            filename=filename or self._output_path(f"{self.synthesizer_name}_cluster.pdf", "cluster"),
        )
        return {"cluster": p}

    def record_test(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Sample-level: propensity MSE detection + nearest-neighbour distances.

        Returns
        -------
        record_output : dict of aggregate distance statistics
        distances     : DataFrame with per-record distance features
        """
        output: Dict = {}
        output["pmse"] = ml_detection(self.realo, self.fakeo)

        shap_path = self._output_path(f"{self.synthesizer_name}_shap.pdf", "sample")
        distances = record_df(self.realo, self.fakeo, self.fakeb, filename=shap_path)

        # Plot nearest-neighbour (d1nn) and centroid distances (cdis)
        try:
            plot_nnd(distances[["distance_1nn"]], filename=self._output_path(f"{self.synthesizer_name}_d1nn_hist.pdf", "sample"))
        except Exception:
            pass
        try:
            plot_nnd(distances[["distance_to_centroid"]], filename=self._output_path(f"{self.synthesizer_name}_cdis_hist.pdf", "sample"))
        except Exception:
            pass
        output["mdis"] = distances["distance_mean"].mean()
        output["d1nn"] = distances["distance_1nn"].mean()
        output["cdis"] = distances["distance_to_centroid"].mean()

        return output, distances

    def evaluate(self) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """
        Run the full evaluation pipeline.

        Returns
        -------
        output   : nested dict of all metric results
        abs_diff : bivariate association difference matrix
        distance : per-record distance DataFrame
        """
        output: Dict = {}

        print("---------- univariate ----------")
        output["univariate"] = self.univariate_test(num_col=3)

        print("---------- bivariate ----------")
        output["bivariate"], abs_diff = self.bivariate_test(figsize=(10, 10))

        print("---------- multivariate ----------")
        output["multivariate"] = self.multivariate_test()

        print("---------- cluster ----------")
        output["cluster"] = self.cluster_test()

        print("---------- record ----------")
        output["record"], distance = self.record_test()

        return output, abs_diff, distance
