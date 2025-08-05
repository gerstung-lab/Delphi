import numpy as np

from delphi.eval.auc import mann_whitney_auc

x1 = np.array([11, 12, 13])
x2 = np.array([1, 2])

mann_whitney_auc(x1=x1, x2=x2)
