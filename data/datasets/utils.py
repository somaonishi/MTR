import numpy as np
from sklearn.preprocessing import QuantileTransformer


def quantiletransform(df_cont, noise=1e-3, seed=42):
    df_cont = df_cont.copy()
    sc = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=max(min(df_cont.shape[0] // 30, 1000), 10),
        subsample=int(1e9),
        random_state=seed,
    )
    stds = np.std(df_cont.values, axis=0, keepdims=True)
    noise_std = noise / np.maximum(stds, noise)
    normal = np.random.default_rng(seed).standard_normal(df_cont.shape)
    df_cont += noise_std * normal
    sc.fit(df_cont)
    return sc
