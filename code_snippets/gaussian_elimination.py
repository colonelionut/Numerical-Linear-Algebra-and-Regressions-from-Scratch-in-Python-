import numpy as np
from functions_for_linear_systems import solve_using_back_substitution 


def solve_with_gaussian_elimination(A,b):
    n = b.shape[0]

    # It is very important we only work on copies, because the function automatically destroys the originals
    # We also want to work with float values only to avoid conversion problems
    A = A.astype(float)
    b = b.astype(float)

    for j in range (0, n-1):
        for i in range (j+1, n):
            m = A[i,j] / A[j,j]
            for k in range (j+1, n):
                A[i,k] = A[i,k] - m * A[j,k]
            b[i] = b[i] - m * b[j]

    # We have performed the Gaussian elimination, now we can solve the system using back substitution
    x = solve_using_back_substitution(A,b)
    return x

def main():
	A = np.array([[2.0, 1.0, -1.0],
	              [-3.0, -1.0, 2.0],
	              [-2.0, 1.0, 2.0]])
	b = np.array([3.0, -11.0, -3.0])
	x = solve_with_gaussian_elimination(A, b)
	print("x      =", x)
	print("A @ x  =", A @ x)
	print("b      =", b)
	print("corect :", np.allclose(A @ x, b))


if __name__ == "__main__":
	main()
