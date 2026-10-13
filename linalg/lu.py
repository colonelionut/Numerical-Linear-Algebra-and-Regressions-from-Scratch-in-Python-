"""Crout LU factorization: A = L U, with unit diagonal on U."""

import numpy as np

from .triangular import back_substitution, forward_substitution


def crout(A):
	A = A.astype(float)
	n = len(A)
	L = np.zeros((n, n))
	U = np.eye(n)
	for p in range(n):
		for i in range(p):
			U[i, p] = (A[i, p] - np.dot(L[i, :p], U[:p, p])) / L[i, i]
		for i in range(p, n):
			L[i, p] = A[i, p] - np.dot(L[i, :p], U[:p, p])
	return L, U


def lu_solve(L, U, b):
	"""Solve A x = b from A = L U: L y = b, then U x = y."""
	y = forward_substitution(L, b.astype(float))
	return back_substitution(U, y)
