"""Iterative refinement with convergence and stagnation detection."""

import numpy as np

from .gaussian import gaussian_complete_pivoting


def iterative_refinement(A, b, max_iter=50, tol=1e-14):
	"""Repeat: r = b - A x, solve A e = r, x = x + e.

	Stops when ||e||_inf < tol ("converged") or when ||e||_inf stops
	decreasing ("stalled"). Returns x, iterations used, status.
	"""
	A = A.astype(float)
	b = b.astype(float)
	x = np.zeros(len(A))
	e_prev = np.inf
	for k in range(1, max_iter + 1):
		r = b - A @ x
		e = gaussian_complete_pivoting(A, r)
		x = x + e
		e_norm = np.linalg.norm(e, np.inf)
		if e_norm < tol:
			return x, k, "converged"
		if e_norm >= e_prev:
			return x, k, "stalled"
		e_prev = e_norm
	return x, max_iter, "max_iter"
