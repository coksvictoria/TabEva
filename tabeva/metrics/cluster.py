"""
Cluster metrics: KMeans-based distribution comparison between real and synthetic data.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabeva.metrics.colors import REAL_COLOR, FAKE_COLOR, ALERT_COLOR


def cluster_df(
    reale: pd.DataFrame,
    fakee: pd.DataFrame,
    n_cluster: int = 20,
    num_col: int = 5,
    figsize: Tuple = (15, 12.5),
    filename: str = "",
    ratio_threshold: float = 0.2,
) -> float:
    """
    Fit KMeans on real data, predict clusters for both datasets, and compare
    cluster-count distributions with a KS test.

    Parameters
    ----------
    num_col : int or None
        Number of subplot columns for cluster visualisation. Pass None to skip plotting.

    Returns
    -------
    p_value : float
        KS test p-value comparing cluster count distributions of real vs. fake.
    """
    model = cluster.KMeans(n_clusters=n_cluster, random_state=0)
    data_r = reale.copy()
    data_f = fakee.copy()

    clt = model.fit(data_r)
    centroid = clt.cluster_centers_

    data_r["data_source"] = "real"
    data_f["data_source"] = "fake"
    df_rf_raw = pd.concat([data_r, data_f]).reset_index(drop=True)

    
    pre_clu = clt.predict(df_rf_raw.iloc[:, :-1])

    df_rf = df_rf_raw.copy()
    df_rf["cluster"] = pre_clu

    cluster_R = pd.crosstab(df_rf["cluster"], df_rf["data_source"])
    _, p_value = stats.ks_2samp(cluster_R["real"], cluster_R["fake"])

    cluster_plot(df_rf, df_rf_raw, centroid, num_col, figsize, filename, ratio_threshold=ratio_threshold)


    return p_value


def cluster_plot(
    df_rf: pd.DataFrame,
    df_rf_raw: pd.DataFrame,
    centroid: np.ndarray,
    num_col: int,
    figsize: Tuple = (12.5, 10),
    filename: Optional[str] = None,
    ratio_threshold: float = 0.2,
) -> None:
    """
    PCA scatter plots for each cluster, showing real vs. fake samples and the centroid.
    """
    n_cluster = df_rf["cluster"].nunique()
    print(f"Plotting {n_cluster} clusters with {num_col} columns per row.")

    num_row = max(1, n_cluster // num_col)
    fig, axs = plt.subplots(num_row, num_col, sharex=False, figsize=figsize)
    axs = np.array(axs).reshape(num_row, num_col)

    fig.supxlabel("Principal Component 2", fontsize=12)
    fig.supylabel("Principal Component 1", fontsize=12)

    for i in sorted(df_rf["cluster"].unique()):
        ridx, cidx = i // num_col, i % num_col
        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA())])

        treal = df_rf_raw[(df_rf["cluster"] == i) & (df_rf["data_source"] == "real")].drop(columns=["data_source"])
        tfake = df_rf_raw[(df_rf["cluster"] == i) & (df_rf["data_source"] == "fake")].drop(columns=["data_source"])

        if len(treal) == 0 or len(tfake) == 0:
            axs[ridx][cidx].set_visible(False)
            continue

        trealt = pipe.fit_transform(treal)
        tfaket = pipe.transform(tfake)
        centroid_df = pd.DataFrame(
            centroid[i].reshape(1, -1),
            columns=treal.columns
        )
        centroidt = pipe.transform(centroid_df)
        # centroidt = pipe.transform(centroid[i].reshape(1, -1))

        axs[ridx][cidx].scatter(trealt[:, 0], trealt[:, 1], s=30, c=[REAL_COLOR], label="Real", alpha=0.6, marker="^")
        axs[ridx][cidx].scatter(tfaket[:, 0], tfaket[:, 1], s=30, c=[FAKE_COLOR], label="Fake", alpha=0.8, marker=".")
        # color centroid star red when the ratio of real/fake in this cluster is very low
        real_count = len(treal)
        fake_count = len(tfake)
        try:
            ratio = min(real_count, fake_count) / max(real_count, fake_count)
        except ZeroDivisionError:
            ratio = 0.0

        star_color = ALERT_COLOR if ratio < ratio_threshold else "black"

        axs[ridx][cidx].scatter(centroidt[0][0], centroidt[0][1], c=star_color, s=120, marker="*", label=f"Centroid {i}")

    axs[0][num_col - 1].legend(["Real", "Fake"], prop={"size": 10})
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close(fig)
