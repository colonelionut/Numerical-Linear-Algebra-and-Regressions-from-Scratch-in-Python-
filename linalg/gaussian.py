"""Gaussian elimination: naive and with complete pivoting."""

import numpy as np

from .triangular import back_substitution


def gaussian_elimination(A, b):
	"""Naive elimination (no pivoting). Fails on a zero or tiny pivot."""
	A = A.astype(float)
	b = b.astype(float)
	n = b.shape[0]
	for j in range(n - 1):
		for i in range(j + 1, n):
			m = A[i, j] / A[j, j]
			for k in range(j + 1, n):
				A[i, k] = A[i, k] - m * A[j, k]
			b[i] = b[i] - m * b[j]
	return back_substitution(A, b)


def gaussian_complete_pivoting(A, b):
	"""Elimination with complete pivoting: max |a_ij| of the trailing submatrix."""
	A = A.astype(float)
	b = b.astype(float)
	n = len(A)
	perm = np.arange(n)
	for p in range(n):
		sub = A[p:, p:]
		r, c = np.unravel_index(np.argmax(np.abs(sub)), sub.shape)
		r += p
		c += p
		A[[p, r], :] = A[[r, p], :]
		A[:, [p, c]] = A[:, [c, p]]
		b[[p, r]] = b[[r, p]]
		perm[[p, c]] = perm[[c, p]]
		for i in range(p + 1, n):
			m = A[i, p] / A[p, p]
			for k in range(p + 1, n):
				A[i, k] = A[i, k] - m * A[p, k]
			b[i] = b[i] - m * b[p]
	y = back_substitution(A, b)
	x = np.zeros(n)
	x[perm] = y
	return x
