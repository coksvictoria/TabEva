from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


def data_preprocess(reald: pd.DataFrame, faked: pd.DataFrame, cat_cols: list, onehot: bool = False):
    """
    Encode categorical columns with LabelEncoder and scale numeric columns with MinMaxScaler.
    Optionally apply one-hot encoding to multi-class categorical columns.

    Returns
    -------
    real_processed, fake_processed : pd.DataFrame
    """
    real = reald.copy().reset_index(drop=True)
    fake = faked.copy().reset_index(drop=True)

    col_names = real.columns.to_list()
    n_col = [s for s in col_names if s not in cat_cols]

    # Label-encode categorical columns (fit on real, transform fake)
    d: dict = defaultdict(LabelEncoder)
    real[cat_cols] = real[cat_cols].apply(lambda x: d[x.name].fit_transform(x))
    fake[cat_cols] = fake[cat_cols].apply(lambda x: d[x.name].transform(x))

    ss = MinMaxScaler()
    real_scaled = pd.DataFrame(ss.fit_transform(real[n_col]), columns=n_col)
    fake_scaled = pd.DataFrame(ss.transform(fake[n_col]), columns=n_col)

    if onehot:
        # Exclude binary columns from one-hot encoding
        b_cols = [i for i in col_names if real[i].nunique() == 2]
        ohe_cols = [i for i in cat_cols if i not in b_cols]

        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe_real = ohe.fit_transform(real[ohe_cols])
        ohe_fake = ohe.transform(fake[ohe_cols])

        real_encoded = pd.DataFrame(ohe_real.toarray(), columns=ohe.get_feature_names_out())
        fake_encoded = pd.DataFrame(ohe_fake.toarray(), columns=ohe.get_feature_names_out())

        real_processed = pd.concat([real_scaled, real_encoded, real[b_cols]], axis=1)
        fake_processed = pd.concat([fake_scaled, fake_encoded, fake[b_cols]], axis=1)
    else:
        real_processed = pd.concat([real_scaled, real[cat_cols]], axis=1)[col_names]
        fake_processed = pd.concat([fake_scaled, fake[cat_cols]], axis=1)[col_names]

    return real_processed, fake_processed
