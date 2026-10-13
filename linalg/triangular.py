"""Diagonal and triangular solvers."""

import numpy as np


def solve_diagonal(A, b):
	n = b.shape[0]
	x = np.zeros(n)
	for i in range(n):
		x[i] = b[i] / A[i, i]
	return x


def forward_substitution(L, b):
	"""Solve L x = b for lower-triangular L."""
	n = b.shape[0]
	x = np.zeros(n)
	for i in range(n):
		s = np.dot(L[i, :i], x[:i])
		x[i] = (b[i] - s) / L[i, i]
	return x


def back_substitution(U, b):
	"""Solve U x = b for upper-triangular U."""
	n = b.shape[0]
	x = np.zeros(n)
	for i in range(n - 1, -1, -1):
		s = 0.0
		for j in range(i + 1, n):
			s = s + U[i, j] * x[j]
		x[i] = (b[i] - s) / U[i, i]
	return x
