"""
Multivariate metrics: MMD, Precision/Recall/Density/Coverage,
ML utility evaluation, feature importance, and dimensionality-reduction plots.
"""

import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    pairwise_distances,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD)
# ---------------------------------------------------------------------------

def mmd_kernel(X: np.ndarray, Y: np.ndarray, kernel: str = "rbf", gamma: float = 1.0, degree: int = 2, coef0: float = 0.0) -> float:
    """
    Compute MMD between two feature matrices using linear, RBF, or polynomial kernel.

    Parameters
    ----------
    X : array [n_samples_1, dim]
    Y : array [n_samples_2, dim]
    kernel : {'linear', 'rbf', 'polynomial'}
    """
    if kernel == "linear":
        delta = X.mean(0) - Y.mean(0)
        return float(delta.dot(delta.T))

    if kernel == "rbf":
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)

    elif kernel == "polynomial":
        XX = polynomial_kernel(X, X, degree, gamma, coef0)
        YY = polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = polynomial_kernel(X, Y, degree, gamma, coef0)

    else:
        raise ValueError(f"Unsupported kernel '{kernel}'. Choose from 'linear', 'rbf', 'polynomial'.")

    return float(XX.mean() + YY.mean() - 2 * XY.mean())


# ---------------------------------------------------------------------------
# Precision / Recall / Density / Coverage (PRDC)
# ---------------------------------------------------------------------------

def compute_pairwise_distance(data_x: np.ndarray, data_y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute pairwise Euclidean distances between rows of data_x and data_y."""
    if data_y is None:
        data_y = data_x
    return pairwise_distances(data_x, data_y)


def get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Return the k-th smallest values along a given axis."""
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    return k_smallests.max(axis=axis)


def compute_nearest_neighbour_distances(input_features: np.ndarray, nearest_k: int) -> np.ndarray:
    """Compute distances to the k-th nearest neighbour for each sample."""
    distances = compute_pairwise_distance(input_features)
    return get_kth_value(distances, k=nearest_k + 1, axis=-1)


def compute_prdc(real: np.ndarray, fake: np.ndarray, nearest_k: int) -> Tuple[float, float]:
    """
    Compute Density and Coverage between real and fake manifolds.

    Returns
    -------
    density, coverage : float
    """
    real_radii = compute_nearest_neighbour_distances(real, nearest_k)
    distance_real_fake = compute_pairwise_distance(real, fake)

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_radii, axis=1)
    ).sum(axis=0).mean()

    coverage = (distance_real_fake.min(axis=1) < real_radii).mean()

    return float(density), float(coverage)


# ---------------------------------------------------------------------------
# ML Utility
# ---------------------------------------------------------------------------

def ml_evaluation(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    df_test: pd.DataFrame,
    c_col: List[str],
    n_col: List[str],
    target_col: str,
    target_type: str = "class",
    model_name: str = "synthetic",
    filename: str = "",
) -> pd.DataFrame:
    """
    Train classifiers/regressors on synthetic data, evaluate on real test set.

    Parameters
    ----------
    target_type : {'class', 'regr'}
    """
    real, fake, df_test = real.copy(), fake.copy(), df_test.copy()
    fake_x = fake.drop([target_col], axis=1)
    X_test = df_test.drop([target_col], axis=1)

    if target_type == "class":
        c_col = [i for i in c_col if i != target_col]
        le = LabelEncoder()
        fake_y = le.fit_transform(fake[target_col])
        y_test = le.transform(df_test[target_col])
    elif target_type == "regr":
        n_col = [i for i in n_col if i != target_col]
        fake_y = fake[target_col].values
        y_test = df_test[target_col].values
    else:
        raise ValueError("target_type must be 'class' or 'regr'.")

    imp_n = SimpleImputer(strategy="median")
    ss = MinMaxScaler()
    imp_c = SimpleImputer(strategy="constant", fill_value="U")
    c_encoder = OneHotEncoder(handle_unknown="ignore")

    cat_pipe = Pipeline([("imp_c", imp_c), ("enc_c", c_encoder)])
    num_pipe = Pipeline([("imp_n", imp_n), ("ss", ss)])
    col_transformer = ColumnTransformer(
        transformers=[("nums", num_pipe, n_col), ("cats", cat_pipe, c_col)],
        remainder="drop",
        n_jobs=-1,
    )

    R: List = []

    if target_type == "regr":
        estimators = [
            lgb.LGBMRegressor(n_estimators=100, random_state=1),
            RandomForestRegressor(n_estimators=100, random_state=1),
            Lasso(random_state=1),
            Ridge(alpha=1.0, random_state=1),
            ElasticNet(random_state=1),
        ]
        estimator_names = ["LGBM", "RF", "LS", "RD", "EN"]

        for est_name, est in zip(estimator_names, estimators):
            start = time.time()
            print(est_name)
            if len(np.unique(fake_y)) == 1:
                R.append([model_name, est_name, 0, 0, 0])
            else:
                model = Pipeline([("preprocess", col_transformer), ("regressor", est)])
                model.fit(fake_x, fake_y)
                y_pred = model.predict(X_test)
                R.append([model_name, est_name, r2_score(y_test, y_pred),
                          mean_absolute_error(y_test, y_pred),
                          mean_squared_error(y_test, y_pred, squared=False)])
            print(f"{est_name} took {round(time.time() - start, 2)}s")

        return pd.DataFrame(R, columns=["data_strategy", "reg_name", "r2", "mae", "rmse"]).sort_values("data_strategy")

    # Classification
    estimators = [
        lgb.LGBMClassifier(n_estimators=100, random_state=1),
        RandomForestClassifier(n_estimators=100, random_state=1),
        LogisticRegression(multi_class="auto", solver="lbfgs", max_iter=500, random_state=1),
        DecisionTreeClassifier(random_state=1),
        MLPClassifier([50, 50], solver="adam", activation="relu", learning_rate="adaptive", random_state=1),
    ]
    estimator_names = ["LGBM", "RF", "LR", "DT", "MLP"]

    for est_name, est in zip(estimator_names, estimators):
        start = time.time()
        print(est_name)
        if len(np.unique(fake_y)) == 1:
            R.append([model_name, est_name, 0, 0, 0, 0, 0, 0])
        else:
            try:
                model = Pipeline([("preprocess", col_transformer), ("classifier", est)])
                model.fit(fake_x, fake_y)
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                R.append([
                    model_name, est_name,
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred),
                    recall_score(y_test, y_pred),
                    f1_score(y_test, y_pred),
                    average_precision_score(y_test, y_proba),
                    roc_auc_score(y_test, y_proba),
                ])
            except Exception:
                pass
        print(f"{est_name} took {round(time.time() - start, 2)}s")

    return pd.DataFrame(R, columns=["data_strategy", "clf_name", "acc", "pre", "rec", "f1", "aucpr", "aucroc"]).sort_values("data_strategy")


def bar_comparison(
    vectors: List[np.ndarray],
    std: Optional[List[np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    tick_names=None,
    max_length: int = 5,
    save_name: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing feature importances across multiple models."""
    num_bars = len(vectors)
    indices = np.argsort(vectors[0])[::-1][:max_length]
    fig, ax = plt.subplots(figsize=(12, 4))
    tot_bar_width = 0.7
    width = tot_bar_width / num_bars
    x = np.arange(len(indices))
    colors = sns.color_palette("Blues", num_bars)

    if tick_names is None:
        tick_names = list(range(len(vectors[0])))
    if labels is None:
        labels = ["Original", "Synthetic", "Transfer"]

    for i, vec in enumerate(vectors):
        xbar = x - tot_bar_width / 2 + (i + 0.5) * width
        if std is not None:
            ax.bar(xbar, vec[indices], yerr=std[i][indices], width=width, label=labels[i], color=colors[i])
        else:
            ax.bar(xbar, vec[indices], width=width, label=labels[i], color=colors[i])

    ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    ticks = np.array(tick_names, dtype="object")[indices]
    plt.xticks(x, ticks, fontsize=12)
    plt.legend()
    plt.xlim([-1, len(indices)])
    if save_name is not None:
        fig.savefig(save_name + "feature_imp.pdf", bbox_inches="tight")
    plt.close(fig)


def ml_feature_importance(
    fakes: Dict,
    c_col: List[str],
    n_col: List[str],
    target_col: str,
    target_type: str = "class",
    filename: str = "",
) -> List[np.ndarray]:
    """
    Compute and plot LGBM feature importances for real data and each synthesiser.

    Parameters
    ----------
    fakes : dict with keys 'real', 'test', 'fake' (dict of name -> DataFrame)
    """
    real, df_test = fakes["real"], fakes["test"]
    real_x = real.drop([target_col], axis=1)

    if target_type == "class":
        c_col = [i for i in c_col if i != target_col]
        le = LabelEncoder()
        real_y = le.fit_transform(real[target_col])
    else:
        n_col = [i for i in n_col if i != target_col]
        real_y = real[target_col].values
        le = None

    imp_n = SimpleImputer(strategy="median")
    ss = MinMaxScaler()
    imp_c = SimpleImputer(strategy="constant", fill_value="U")
    c_encoder = OneHotEncoder(handle_unknown="ignore")
    cat_pipe = Pipeline([("imp_c", imp_c), ("enc_c", c_encoder)])
    num_pipe = Pipeline([("imp_n", imp_n), ("ss", ss)])
    col_transformer = ColumnTransformer(
        transformers=[("nums", num_pipe, n_col), ("cats", cat_pipe, c_col)],
        remainder="drop",
        n_jobs=-1,
    )

    est = lgb.LGBMRegressor(n_estimators=100, random_state=1) if target_type == "regr" else lgb.LGBMClassifier(n_estimators=100, random_state=1)

    importances: List[np.ndarray] = []
    method_names: List[str] = []

    model = Pipeline([("preprocess", col_transformer), ("classifier", est)])
    model.fit(real_x, real_y)
    feature_names = [e[6:] for e in model["preprocess"].get_feature_names_out()]
    method_names.append("real")
    importances.append(model["classifier"].feature_importances_)

    for syn_name, fake in fakes["fake"].items():
        print(syn_name)
        fake_x = fake.drop([target_col], axis=1)
        fake_y = le.transform(fake[target_col]) if le is not None else fake[target_col].values
        syn_model = Pipeline([("preprocess", col_transformer), ("classifier", est)])
        syn_model.fit(fake_x, fake_y)
        importances.append(syn_model["classifier"].feature_importances_)
        method_names.append(syn_name)

    bar_comparison(importances, std=None, labels=method_names, tick_names=feature_names, max_length=10, save_name=filename)
    return importances


# ---------------------------------------------------------------------------
# Dimensionality-reduction plot
# ---------------------------------------------------------------------------

def table_plot(reals: pd.DataFrame, fakes: pd.DataFrame, dimensionality_reduction: str = "PCA", filename: str = "") -> None:
    """
    Joint scatter plot of real vs. fake data projected to 2D via PCA or t-SNE.
    """
    if dimensionality_reduction == "PCA":
        from sklearn.decomposition import KernelPCA
        model = KernelPCA(n_components=2, kernel="rbf")
    elif dimensionality_reduction == "TSNE":
        from sklearn.manifold import TSNE
        model = TSNE(n_components=2, perplexity=10, random_state=1)
    else:
        raise ValueError(f"Unsupported reduction method: {dimensionality_reduction}")

    if reals.shape[0] > 10000:
        reals = reals.sample(10000)
        fakes = fakes.sample(10000)

    real_t = model.fit_transform(reals)
    fake_t = model.fit_transform(fakes)

    real_df = pd.DataFrame(real_t, columns=["Component 1", "Component 2"])
    fake_df = pd.DataFrame(fake_t, columns=["Component 1", "Component 2"])
    real_df["dataset"] = "real"
    fake_df["dataset"] = "fake"

    g = sns.jointplot(
        data=pd.concat([real_df, fake_df]),
        x="Component 1",
        y="Component 2",
        palette=["#2171B5", "#6BAED6"],
        joint_kws={"alpha": 0.8},
        hue="dataset",
    )
    g.fig.set_size_inches((5, 5))
    if filename:
        g.savefig(filename + ".pdf")
    plt.close(g.fig)
