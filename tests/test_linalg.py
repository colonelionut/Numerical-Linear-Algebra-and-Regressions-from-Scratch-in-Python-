import numpy as np
import pytest

from linalg import (back_substitution, crout, forward_substitution, gauss_seidel,
		    gaussian_complete_pivoting, gaussian_elimination, iterative_refinement,
		    jacobi, jacobi_iteration_matrix, lu_solve, solve_diagonal, spectral_radius)

rng = np.random.default_rng(42)


def well_conditioned(n):
	A = rng.standard_normal((n, n))
	return A + n * np.eye(n)


@pytest.mark.parametrize("n", [1, 3, 10, 30])
def test_triangular(n):
	A = well_conditioned(n)
	b = rng.standard_normal(n)
	L, U = np.tril(A), np.triu(A)
	assert np.allclose(L @ forward_substitution(L, b), b)
	assert np.allclose(U @ back_substitution(U, b), b)
	D = np.diag(np.diag(A))
	assert np.allclose(D @ solve_diagonal(D, b), b)


@pytest.mark.parametrize("solver", [gaussian_elimination, gaussian_complete_pivoting])
@pytest.mark.parametrize("n", [1, 4, 20])
def test_gaussian(solver, n):
	A = well_conditioned(n)
	b = rng.standard_normal(n)
	assert np.allclose(solver(A, b), np.linalg.solve(A, b))


def test_inputs_not_modified():
	A = well_conditioned(5)
	b = rng.standard_normal(5)
	A0, b0 = A.copy(), b.copy()
	gaussian_complete_pivoting(A, b)
	assert np.array_equal(A, A0) and np.array_equal(b, b0)


def test_complete_pivoting_handles_zero_pivot():
	A = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 3.0]])
	x = np.array([3.0, 2.0, 1.0])
	assert np.allclose(gaussian_complete_pivoting(A, A @ x), x)


@pytest.mark.parametrize("n", [2, 5, 15])
def test_crout(n):
	A = well_conditioned(n)
	L, U = crout(A)
	assert np.allclose(L @ U, A)
	assert np.allclose(L, np.tril(L))
	assert np.allclose(U, np.triu(U)) and np.allclose(np.diag(U), 1.0)
	b = rng.standard_normal(n)
	assert np.allclose(lu_solve(L, U, b), np.linalg.solve(A, b))


def test_refinement_converges_on_easy_system():
	A = well_conditioned(8)
	x_true = rng.standard_normal(8)
	x, _, status = iterative_refinement(A, A @ x_true, tol=1e-13)
	assert np.allclose(x, x_true)
	assert status in ("converged", "stalled")


def test_jacobi_and_gauss_seidel():
	A = np.array([[4.0, 1.0, -1.0], [2.0, 7.0, 1.0], [1.0, -3.0, 12.0]])
	b = A @ np.array([1.0, 2.0, 3.0])
	assert spectral_radius(jacobi_iteration_matrix(A)) < 1
	xj, kj = jacobi(A, b)
	xg, kg = gauss_seidel(A, b)
	assert np.allclose(xj, [1, 2, 3]) and np.allclose(xg, [1, 2, 3])
	assert kg < kj
