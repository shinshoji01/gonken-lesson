import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

def rectangle(num, R, r, mode="in"):
    array = np.zeros((2, 1))
    while(True):
        xx, yy = np.meshgrid(np.arange(-R, R, 0.01),
                             np.arange(-R, R, 0.01))
        x, y = np.c_[xx.ravel(), yy.ravel()].T
        
        x_in = np.abs(x)<r
        y_in = np.abs(y)<r
        if mode=="out":
            new = np.array([x, y])[:,np.array(1-x_in * y_in, dtype=bool)]
        elif mode=="in":
            new = np.array([x, y])[:,x_in * y_in]
        if array.shape[1] == 1:
            array = np.concatenate([array, new], axis=1)[:,1:]
        else:
            array = np.concatenate([array, new], axis=1)
        if array.shape[1]>num:
            break
    array = array[:, np.random.choice(np.arange(array.shape[1]), size=array.shape[1], replace=False)]
    return array[:, :num]

def data_generation(r_max, r_min, num, mode="train", alpha=0):
    theta = np.linspace(0, 2*np.pi, num)
    if mode=="train":
        x1 = rectangle(num, r_max, r_min, "in")
        x2 = rectangle(num, r_max, r_min, "out")
    elif mode=="test":
        x1 = rectangle(num, r_max, r_min, "in")
        x2 = rectangle(int(num*(r_max/r_min)), r_max, r_min, "out")
    x1 = x1 + np.random.randn(*x1.shape)*alpha
    x2 = x2 + np.random.randn(*x2.shape)*alpha
    x1_target = np.concatenate([x1, np.zeros((1, x1.shape[1]))], axis=0)
    x2_target = np.concatenate([x2, np.ones((1, x2.shape[1]))], axis=0)
    data = pd.DataFrame(np.concatenate([x1_target, x2_target], axis=1).T, columns=["small", "large", "target"])
    return data

def train_test_generation(r, alpha, train_num=500, test_num=1000):
    train = data_generation(r*1.5, r, train_num, "train", alpha)
    test = data_generation(1+min(train["large"].max(), np.abs(train["large"].min())), r, test_num, "test", 0)
    return train, test

def get_poly(x, degree):
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    new_x = pf.fit_transform(x)
    return new_x

def get_decision_boundary(model, train, test, degree=1, include_test=False):
    X_train = train.drop("target", axis=1).values
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(get_poly(np.c_[xx.ravel(), yy.ravel()], degree))
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1)
    x1 = train[train.target==0].drop("target", axis=1).values
    x2 = train[train.target==1].drop("target", axis=1).values
    ax.scatter(x1[:, 0], x1[:, 1], label="small, 0", s=10)
    ax.scatter(x2[:, 0], x2[:, 1], label="large, 1", s=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.legend()
    ax.contour(xx, yy, Z, cmap=plt.cm.Paired)
    if include_test:
        ax = fig.add_subplot(1, 2, 2)
        x1 = test[test.target==0].drop("target", axis=1).values
        x2 = test[test.target==1].drop("target", axis=1).values
        ax.scatter(x1[:, 0], x1[:, 1], label="small, 0", s=3)
        ax.scatter(x2[:, 0], x2[:, 1], label="large, 1", s=3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.legend()
        ax.contour(xx, yy, Z, cmap=plt.cm.Paired)
    
    plt.show()
    return 