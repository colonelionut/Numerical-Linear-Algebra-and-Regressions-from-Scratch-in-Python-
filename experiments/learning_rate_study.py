"""Gradient descent vs normal equations, and the learning-rate stability limit.

For the MSE the Hessian is H = (2/m) X^T X, constant. Gradient descent is a
linear iteration with matrix I - lr*H, so it converges iff lr < 2 / lambda_max(H).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from regression import gradient_descent as gd
from regression import normal_equations as ne
from . import FIGURES


def main():
	rng = np.random.default_rng(0)
	m = 100
	x = np.linspace(-3, 3, m)
	y = 2.0 * x - 1.0 + 0.5 * rng.standard_normal(m)

	X = ne.design_matrix(x)
	b_ne, w_ne = ne.fit_normal_equation(X, y)
	cost_ne = ne.mse(X, y, np.array([b_ne, w_ne]))
	print(f"normal equations : w = {w_ne:.4f}  b = {b_ne:.4f}  MSE = {cost_ne:.4f}")

	H = (2 / m) * X.T @ X
	lr_max = 2 / np.max(np.linalg.eigvalsh(H))
	print(f"stability limit  : lr < 2 / lambda_max(H) = {lr_max:.4f}\n")

	epochs = 200
	plt.figure(figsize=(8, 5))
	for lr in [0.001, 0.05, 0.3, 0.35]:
		w, b, hist = gd.train(x, y, lr=lr, epochs=epochs)
		tag = "converges" if lr < lr_max else "diverges"
		print(f"lr = {lr:<6} {tag:9s} w = {w:.4g}  b = {b:.4g}  final MSE = {hist[-1]:.4g}")
		plt.semilogy(np.arange(1, epochs + 1), np.maximum(hist - cost_ne, 1e-16), label=f"lr = {lr} ({tag})")

	plt.xlabel("epoch")
	plt.ylabel("MSE(epoch) - MSE(normal equations)")
	plt.title(f"Gradient descent: stable only for lr < {lr_max:.3f}")
	plt.legend()
	plt.grid(True, which="both", alpha=0.3)
	plt.savefig(FIGURES / "learning_rate_study.png", dpi=130, bbox_inches="tight")
	print("saved figures/learning_rate_study.png")


if __name__ == "__main__":
	main()
