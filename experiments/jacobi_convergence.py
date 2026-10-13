"""Jacobi: convergence rate is set by the spectral radius of T = -D^{-1}(L+U)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linalg import jacobi, jacobi_history, jacobi_iteration_matrix, spectral_radius
from . import FIGURES


def make_diagonally_dominant(n, strength, rng):
	M = rng.standard_normal((n, n))
	for i in range(n):
		M[i, i] = strength * np.sum(np.abs(M[i])) + 1.0
	return M


def reordering_example():
	A = np.array([[2.0, 7.0, 1.0],
		      [4.0, 1.0, -1.0],
		      [1.0, -3.0, 12.0]])
	b = np.array([19.0, 3.0, 31.0])
	order = [1, 0, 2]
	for name, M, v in [("original", A, b), ("rows swapped", A[order], b[order])]:
		rho = spectral_radius(jacobi_iteration_matrix(M))
		x, k = jacobi(M, v, max_iter=100)
		print(f"  {name:13s} rho(T) = {rho:6.3f}  after {k:3d} it: x = {np.round(x, 6)}")


def main():
	print("Same system, two row orders (true solution [1, 2, 3]):")
	reordering_example()

	rng = np.random.default_rng(0)
	n, n_iter = 20, 60
	plt.figure(figsize=(8, 5))
	for strength in [3.0, 1.5, 1.0, 0.7]:
		A = make_diagonally_dominant(n, strength, rng)
		x_true = np.ones(n)
		errors = jacobi_history(A, A @ x_true, x_true, n_iter)
		rho = spectral_radius(jacobi_iteration_matrix(A))
		k = np.arange(1, n_iter + 1)
		line, = plt.semilogy(k, errors, label=f"strength {strength}, $\\rho(T)$ = {rho:.3f}")
		plt.semilogy(k, errors[0] * rho ** (k - 1), "--", color=line.get_color(), alpha=0.5)
		print(f"strength {strength}: rho(T) = {rho:.3f}")

	plt.ylim(1e-18, 1e2)
	plt.xlabel("iteration k")
	plt.ylabel(r"$\|x_k - x_{true}\|_\infty$")
	plt.title(r"Jacobi convergence vs diagonal dominance (dashed: $\rho(T)^k$ prediction)")
	plt.legend()
	plt.grid(True, which="both", alpha=0.3)
	plt.savefig(FIGURES / "jacobi_convergence.png", dpi=130, bbox_inches="tight")
	print("saved figures/jacobi_convergence.png")


if __name__ == "__main__":
	main()
