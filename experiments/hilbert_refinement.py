"""Iterative refinement on Hilbert matrices: the error floor follows kappa(A) * eps."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linalg import iterative_refinement
from . import FIGURES

EPS = np.finfo(float).eps


def hilbert(n):
	i = np.arange(n)
	return 1.0 / (i[:, None] + i[None, :] + 1)


def main():
	rows = []
	print(f"{'n':>3} {'cond(A)':>10} {'cond*eps':>10} {'error':>10} {'iters':>5}  status")
	for n in range(2, 13):
		A = hilbert(n)
		x_true = np.ones(n)
		x, k, status = iterative_refinement(A, A @ x_true, max_iter=50, tol=1e-15)
		kappa = np.linalg.cond(A)
		err = np.linalg.norm(x - x_true, np.inf)
		rows.append((n, kappa, err))
		print(f"{n:>3} {kappa:>10.2e} {kappa * EPS:>10.2e} {err:>10.2e} {k:>5}  {status}")

	n, kappa, err = map(np.array, zip(*rows))
	plt.figure(figsize=(7, 4.5))
	plt.loglog(kappa, err, "o-", label="error after refinement")
	plt.loglog(kappa, kappa * EPS, "--", label=r"$\kappa(A)\,\varepsilon_{mach}$")
	for ni, ki, ei in zip(n, kappa, err):
		plt.annotate(str(ni), (ki, ei), textcoords="offset points", xytext=(4, 4), fontsize=8)
	plt.xlabel(r"$\kappa(A)$")
	plt.ylabel(r"$\|x - x_{true}\|_\infty$")
	plt.title("Iterative refinement on Hilbert matrices (labels = n)")
	plt.grid(True, which="both", alpha=0.3)
	plt.legend()
	plt.savefig(FIGURES / "hilbert_refinement.png", dpi=130, bbox_inches="tight")
	print("saved figures/hilbert_refinement.png")


if __name__ == "__main__":
	main()
