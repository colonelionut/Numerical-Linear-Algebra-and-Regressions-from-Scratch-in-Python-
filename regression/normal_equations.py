"""Linear regression in closed form: X^T X theta = X^T y."""

import numpy as np


def design_matrix(x):
	"""Column of ones (bias) followed by the feature."""
	X = np.ones((x.shape[0], 2))
	X[:, 1] = x
	return X


def fit_normal_equation(X, y):
	return np.linalg.solve(X.T @ X, X.T @ y)


def predict(X, theta):
	return X @ theta


def mse(X, y, theta):
	e = predict(X, theta) - y
	return np.dot(e, e) / y.shape[0]
