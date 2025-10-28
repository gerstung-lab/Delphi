import numpy as np
from scipy.stats import rankdata


def mann_whitney_auc(x1: np.ndarray, x2: np.ndarray) -> float:

    n1 = len(x1)
    n2 = len(x2)
    x12 = np.concatenate([x1, x2])
    ranks = rankdata(x12, method="average")

    R1 = ranks[:n1].sum()
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2
