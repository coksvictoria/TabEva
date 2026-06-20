"""
Sample-level metrics: ML detection (propensity score MSE) and
nearest-neighbour / centroid distance analysis with SHAP explainability.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn import cluster
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from tabeva.metrics.multivariate import compute_pairwise_distance


# ---------------------------------------------------------------------------
# ML Detection (Propensity Score MSE)
# ---------------------------------------------------------------------------

def ml_detection(real: pd.DataFrame, fake: pd.DataFrame, kfold: int = 5) -> float:
    """
    Train a logistic regression discriminator to distinguish real from fake.
    Returns the mean propensity score MSE across folds (lower = more similar).

    Parameters
    ----------
    real, fake : preprocessed (scaled/encoded) DataFrames
    kfold : number of stratified CV folds
    """
    real = real.copy()
    fake = fake.copy()
    real["flag"] = 1
    fake["flag"] = 0

    df = pd.concat([real, fake]).reset_index(drop=True)
    x = df.drop(["flag"], axis=1)
    y = df["flag"]

    np.random.seed(1)
    lr = LogisticRegression(solver="lbfgs", max_iter=10000, random_state=1)
    skf = StratifiedKFold(kfold, shuffle=True, random_state=1)

    propensity_mses = []
    for train_idx, test_idx in skf.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        lr.fit(x_train, y_train)
        real_proba = lr.predict_proba(x_test)[:, 1]
        mse = np.mean(((real_proba - 0.5) ** 2) / 0.25)
        propensity_mses.append(mse)

    return float(np.mean(propensity_mses))


# ---------------------------------------------------------------------------
# Record-level distance analysis
# ---------------------------------------------------------------------------

def record_df(reale: pd.DataFrame, fakee: pd.DataFrame, fakes: pd.DataFrame, exp_metric: str = "dc", filename: str = "") -> pd.DataFrame:
    """
    Compute per-record distance features for synthetic samples relative to real data.

    Parameters
    ----------
    reale, fakee : preprocessed (one-hot-encoded) DataFrames
    fakes       : preprocessed (label-encoded) DataFrame used for SHAP
    exp_metric  : {'dc', 'dm', 'd1nn'} — centroid distance, mean distance, or 1-NN distance

    Returns
    -------
    fakec : DataFrame with cluster, distance_mean, distance_1nn, distance_to_centroid columns
    """
    fakec = pd.DataFrame()
    kms = cluster.KMeans(n_clusters=20, random_state=0)
    clt = kms.fit(reale)

    fakec["cluster"] = clt.predict(fakee).tolist()

    distance_real_fake = compute_pairwise_distance(reale, fakee)
    fakec["distance_mean"] = distance_real_fake.mean(axis=0).tolist()
    fakec["distance_1nn"] = distance_real_fake.min(axis=0).tolist()
    fakec["distance_to_centroid"] = clt.transform(fakee).min(axis=1).tolist()

    record_plot(fakec, fakes, exp_metric, filename=filename)

    return fakec


def record_plot(fakec: pd.DataFrame, fakes: pd.DataFrame, exp_metric: str = "dc", filename: str = "") -> None:
    """
    Fit a LightGBM regressor on synthetic features to predict distance,
    then produce a SHAP bar plot to explain which features drive distance.
    """
    model_dc = lgb.LGBMRegressor(n_estimators=100, random_state=1)

    if exp_metric == "dc":
        distance = fakec["distance_to_centroid"]
    elif exp_metric == "dm":
        distance = fakec["distance_mean"]
    else:
        distance = fakec["distance_1nn"]

    model_dc.fit(fakes, distance)

    explainer = shap.TreeExplainer(model_dc)
    shap_values = explainer(fakes)
    shap.plots.bar(shap_values, show=False)
    if filename:
        plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.close()
