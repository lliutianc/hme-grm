import numpy as np


def to_score(x_norm, n_level):
    _percent = np.linspace(0, 1, n_level + 1)[1:-1]
    quantiles = np.quantile(x_norm, _percent)
    y = np.zeros_like(x_norm)
    for level, quant in zip(range(1, n_level+1), quantiles):
        y[x_norm >= quant] = level
    return y


def create_data(n_sample, n_feature, n_level):
    x = np.random.uniform(size=(n_sample * n_feature),
                          low=-10, high=10).reshape(n_sample, n_feature)
    x_norm = np.linalg.norm(x, axis=1, keepdims=1, ord=1)
    y = to_score(x_norm, n_level)
    return x, y


