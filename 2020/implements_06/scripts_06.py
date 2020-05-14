from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def probA_1(x, y):
    # only accepts points betwen -5 and +5 in both axes
    assert all(np.abs(x)<5) and all(np.abs(y)<5)
    assert x.shape==y.shape
    pA = stats.norm(0, 1).pdf(x)*stats.norm(0, 1).pdf(y) 
    pB = np.ones_like(pA)/200.
    return pA/(pA+pB)
