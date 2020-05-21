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

def probA_2(x, y):
    # only accepts points betwen -5 and +5 in both axes
    assert all(np.abs(x)<5) and all(np.abs(y)<5)
    assert x.shape==y.shape
    pA = (stats.norm(4, 0.5).pdf(x)*stats.norm(0, 1).pdf(y) 
          + stats.norm(3, 0.5).pdf(x)*stats.norm(-3, .5).pdf(y)
          + stats.norm(-5, 0.5).pdf(x)*stats.norm(-3, 1.2).pdf(y)
          + stats.norm(1, 0.5).pdf(x)*stats.norm(1, 0.5).pdf(y))
    pB = np.ones_like(pA)/200.
    return pA/(pA+pB)

def plot_points(X, labels, title=None, ax=None, polar=False):
    if not ax:
        if polar:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='polar')
        else:
            fig, ax = plt.subplots(figsize=(6,6))
    mask = labels=='A'
    A, = ax.plot(X[mask, 0], X[mask, 1], 'rx', markersize=7, label='A')
    B, = ax.plot(X[~mask, 0], X[~mask, 1], 'bo', markersize=3, label='B')
    
    fontdict={
        'fontsize': 15
    }
    
    if polar:
        try:
            fig.set_size_inches(6*np.sqrt(2), 6*np.sqrt(2))
        except:
            pass
    else:
        ax.set_xlabel('$X_0$', fontdict=fontdict)
        ax.set_ylabel('$X_1$', fontdict=fontdict)
        ax.set_xticks(np.arange(-5, 6))
        ax.set_yticks(np.arange(-5, 6))
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))

    ax.legend(handles=(A, B), 
              labels=('A', 'B'), 
              loc='upper right', 
              fancybox=True, 
              framealpha=1, 
              **fontdict)
    
    if title:
        ax.set_title(title, fontdict=fontdict)
        
    return ax

