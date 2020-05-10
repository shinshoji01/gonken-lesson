import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def f_true(x):
    y = x**3-20*x**2
    return y

def data_generator(num, test_num=10, alpha=0, x_min=0, x_max=20, add=5):
    np.random.seed(42)
    x = np.random.random((num, 1))*(x_max-x_min)-x_min
    x_test = np.reshape(np.linspace(x_max, x_max+add, test_num), (test_num,1)) # generate random point in range [0, -20]
    y = f_true(x)
    y_test = f_true(x_test)
    np.random.seed(42)
    y = y + np.random.randn(*x.shape)*alpha
    return x, y, x_test, y_test

def mse(y_pred, y_true):
    d = y_pred - y_true
    return np.mean(d**2)

def f(x, model):
    coef = np.reshape(model.coef_.reshape(-1), (1, -1))
    a = np.tile(x, (1, coef.shape[1]))
    for i in range(coef.shape[1]):
        a[:, i] = a[:, i] ** (i+1)
    y = np.dot(a, coef.T)
    return y

def predict_polyreg(x, y, degree):
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    model = LinearRegression()
    x_ = pf.fit_transform(x)
    model.fit(x_, y)
    return model, pf

def get_errors(pf, model, x, y, x_test, y_test):
    x_ = pf.fit_transform(x)
    y_pred = model.predict(x_)
    train_error = mse(y_pred, y)
    x_ = pf.fit_transform(x_test)
    y_pred = model.predict(x_)
    test_error = mse(y_pred, y_test)
    return train_error, test_error

def coef_visualizer(model):
    coef = np.reshape(model.coef_.reshape(-1), (1, -1))
    for i in range(coef.shape[1]):
        print(f"Coefficient {i+1} = {round(coef[0,i], 6)}")