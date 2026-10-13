"""Stationary iterative methods: Jacobi and Gauss-Seidel."""

import numpy as np


def jacobi_iteration_matrix(A):
	"""T = -D^{-1} (L + U)."""
	A = A.astype(float)
	D = np.diag(np.diag(A))
	return -np.linalg.inv(D) @ (A - D)


def spectral_radius(T):
	return np.max(np.abs(np.linalg.eigvals(T)))


def jacobi_step(A, b, x):
	n = len(A)
	x_new = np.zeros(n)
	for i in range(n):
		s = 0.0
		for j in range(n):
			if i != j:
				s = s + A[i, j] * x[j]
		x_new[i] = (b[i] - s) / A[i, i]
	return x_new


def jacobi(A, b, max_iter=500, tol=1e-10):
	A = A.astype(float)
	b = b.astype(float)
	x = np.zeros(len(A))
	for k in range(1, max_iter + 1):
		x_new = jacobi_step(A, b, x)
		if np.linalg.norm(x_new - x, np.inf) < tol:
			return x_new, k
		x = x_new
	return x, max_iter


def jacobi_history(A, b, x_true, n_iter):
	"""Error ||x_k - x_true||_inf for k = 1..n_iter."""
	A = A.astype(float)
	b = b.astype(float)
	x = np.zeros(len(A))
	errors = []
	for _ in range(n_iter):
		x = jacobi_step(A, b, x)
		errors.append(np.linalg.norm(x - x_true, np.inf))
	return np.array(errors)


def gauss_seidel(A, b, max_iter=500, tol=1e-10):
	"""Like Jacobi, but uses each new component as soon as it is computed."""
	A = A.astype(float)
	b = b.astype(float)
	n = len(A)
	x = np.zeros(n)
	for k in range(1, max_iter + 1):
		x_old = x.copy()
		for i in range(n):
			s_new = np.dot(A[i, :i], x[:i])
			s_old = np.dot(A[i, i + 1:], x_old[i + 1:])
			x[i] = (b[i] - s_new - s_old) / A[i, i]
		if np.linalg.norm(x - x_old, np.inf) < tol:
			return x, k
	return x, max_iter
