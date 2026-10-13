"""Three ways to fit least squares, and data that makes them disagree."""

import numpy as np


def fit_inv(X, y):
	"""Explicit inverse of X^T X."""
	return np.linalg.inv(X.T @ X) @ X.T @ y


def fit_solve(X, y):
	"""Solve X^T X theta = X^T y without forming the inverse."""
	return np.linalg.solve(X.T @ X, X.T @ y)


def fit_lstsq(X, y):
	"""SVD-based least squares on X directly; never forms X^T X."""
	return np.linalg.lstsq(X, y, rcond=None)[0]


def make_data(m, collinearity, rng):
	"""Two features; x2 = x1 + collinearity * noise (None = independent)."""
	x1 = rng.standard_normal(m)
	if collinearity is None:
		x2 = rng.standard_normal(m)
	else:
		x2 = x1 + collinearity * rng.standard_normal(m)
	X = np.column_stack([np.ones(m), x1, x2])
	theta_true = np.array([1.0, 2.0, -3.0])
	y = X @ theta_true + 0.01 * rng.standard_normal(m)
	return X, y, theta_true
