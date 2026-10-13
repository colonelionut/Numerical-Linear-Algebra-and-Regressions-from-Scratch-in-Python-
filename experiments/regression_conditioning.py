"""inv vs solve vs lstsq as the features become collinear.

Forming X^T X squares the condition number: cond(X^T X) = cond(X)^2.
With x1 ~ x2 only w1 + w2 is determined by the data (true value -1); the
difference w1 - w2 lies along the smallest singular direction of X.
"""

import numpy as np

from regression.conditioning import fit_inv, fit_solve, fit_lstsq, make_data


def main():
	rng = np.random.default_rng(0)
	for label, c in [("independent", None), ("x2 = x1 + 1e-4 noise", 1e-4),
			 ("x2 = x1 + 1e-8 noise", 1e-8)]:
		X, y, theta_true = make_data(200, c, rng)
		kx = np.linalg.cond(X)
		print(f"\n{label}: cond(X) = {kx:.2e}, cond(X)^2 = {kx ** 2:.2e}, "
		      f"cond(X^T X) = {np.linalg.cond(X.T @ X):.2e}")
		for name, fit in [("inv", fit_inv), ("solve", fit_solve), ("lstsq", fit_lstsq)]:
			theta = fit(X, y)
			err = np.linalg.norm(theta - theta_true, np.inf)
			r = X @ theta - y
			print(f"  {name:6s} theta = {np.round(theta, 4)}  error = {err:.2e}  "
			      f"w1 + w2 = {theta[1] + theta[2]:+.4f}  MSE = {np.dot(r, r) / len(y):.2e}")


if __name__ == "__main__":
	main()
