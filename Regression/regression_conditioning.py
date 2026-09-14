import numpy as np


def fit_inv(X, y):
	# naive variant: direct inverse of X^T X
	# TODO: theta = inv(X^T X) @ (X^T y)  -- use np.linalg.inv
	return (np.linalg.inv(np.transpose(X) @ X)) @ np.transpose(X) @ y


def fit_solve(X, y):
	# middle variant: solve the system X^T X theta = X^T y (no inverse)
	A = X.T @ X
	c = X.T @ y
	theta = np.linalg.solve(A,c)
	return theta



def fit_lstsq(X, y):
	# best variant: least squares directly on X, y (SVD, never forms X^T X)
	# TODO: theta = least squares solution  -- np.linalg.lstsq(X, y, rcond=None)[0]
	return np.linalg.lstsq(X,y, rcond = None)[0]


def make_data(m, kind, rng):
	if kind == "well_conditioned":
		# two independent features, well conditioned
		x1 = rng.standard_normal(m)
		x2 = rng.standard_normal(m)
	else:
		# two ALMOST identical features -> near-collinear columns -> ill conditioned
		x1 = rng.standard_normal(m)
		x2 = x1 + 1e-4 * rng.standard_normal(m)   # x2 almost equal to x1
	X = np.column_stack([np.ones(m), x1, x2])
	theta_true = np.array([1.0, 2.0, -3.0])       # [bias, w1, w2]
	y = X @ theta_true + 0.01 * rng.standard_normal(m)
	return X, y, theta_true


def report(name, X, y, theta_true):
	kappa = np.linalg.cond(X.T @ X)
	print(f"\n=== {name} data  (cond(X^T X) = {kappa:.2e}) ===")
	print(f"{'method':8s} {'theta found':40s} {'error vs true':s}")
	for label, fn in [("inv", fit_inv), ("solve", fit_solve), ("lstsq", fit_lstsq)]:
		theta = fn(X, y)
		err = np.linalg.norm(theta - theta_true, np.inf)
		print(f"{label:8s} {str(np.round(theta, 4)):40s} {err:.2e}")


def main():
	rng = np.random.default_rng(0)
	m = 200
	for kind in ["well_conditioned", "ill_conditioned"]:
		X, y, theta_true = make_data(m, kind, rng)
		report(kind, X, y, theta_true)


if __name__ == "__main__":
	main()