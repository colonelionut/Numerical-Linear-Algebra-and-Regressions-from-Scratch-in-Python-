import numpy as np

from regression import conditioning, gradient_descent as gd, normal_equations as ne


def data():
	rng = np.random.default_rng(0)
	x = np.linspace(-3, 3, 100)
	y = 2.0 * x - 1.0 + 0.5 * rng.standard_normal(100)
	return x, y


def test_gradient_descent_matches_normal_equations():
	x, y = data()
	b, w = ne.fit_normal_equation(ne.design_matrix(x), y)
	w_gd, b_gd, hist = gd.train(x, y, lr=0.05, epochs=2000)
	assert np.isclose(w_gd, w) and np.isclose(b_gd, b)
	assert np.all(np.diff(hist) <= 1e-12)


def test_normal_equations_match_lstsq():
	x, y = data()
	X = ne.design_matrix(x)
	assert np.allclose(ne.fit_normal_equation(X, y), np.linalg.lstsq(X, y, rcond=None)[0])


def test_fits_agree_when_well_conditioned():
	X, y, _ = conditioning.make_data(200, None, np.random.default_rng(1))
	t = [f(X, y) for f in (conditioning.fit_inv, conditioning.fit_solve, conditioning.fit_lstsq)]
	assert np.allclose(t[0], t[2]) and np.allclose(t[1], t[2])


def test_condition_number_squares():
	X, _, _ = conditioning.make_data(200, 1e-3, np.random.default_rng(2))
	assert np.isclose(np.linalg.cond(X.T @ X), np.linalg.cond(X) ** 2, rtol=1e-3)
