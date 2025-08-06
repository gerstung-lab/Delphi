import numpy as np

from delphi.eval.auc import mann_whitney_auc


def ordinal_mann_whitney_auc(x1: np.ndarray, x2: np.ndarray) -> float:

    n1 = len(x1)
    n2 = len(x2)
    R1 = np.concatenate([x1, x2]).argsort().argsort()[:n1].sum() + n1
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def test_equivalence():

    rng = np.random.default_rng()
    x12 = rng.choice(20, size=10, replace=False)
    x1, x2 = x12[:7], x12[7:]

    old_auc = ordinal_mann_whitney_auc(x1=x1, x2=x2)
    new_auc = mann_whitney_auc(x1=x1, x2=x2)

    assert old_auc == new_auc
