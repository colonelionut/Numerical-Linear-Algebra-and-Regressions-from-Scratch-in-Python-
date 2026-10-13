"""Where naive Gaussian elimination breaks and complete pivoting does not."""

import warnings

import numpy as np

from linalg import gaussian_elimination, gaussian_complete_pivoting


def compare(name, A, x_true):
	b = A @ x_true
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		x_naive = gaussian_elimination(A, b)
	x_piv = gaussian_complete_pivoting(A, b)
	print(f"\n{name}")
	print("  true        :", x_true)
	print("  naive       :", x_naive)
	print("  complete piv:", x_piv)


def main():
	A = np.array([[1.0, 1.0, 1.0],
		      [1.0, 1.0, 3.0],
		      [2.0, 5.0, 8.0]])
	compare("Zero pivot appears after the first step", A, np.array([1.0, 2.0, 3.0]))

	A = np.array([[1e-17, 1.0],
		      [1.0, 1.0]])
	compare("Tiny pivot (1e-17): no division by zero, still wrong", A, np.array([1.0, 1.0]))


if __name__ == "__main__":
	main()
