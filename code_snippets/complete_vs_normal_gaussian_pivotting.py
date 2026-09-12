import numpy as np

from functions_for_linear_systems import complete_gaussian_pivoting
from functions_for_linear_systems import solve_with_gaussian_elimination


def main():
    A = np.array([[1.0, 1.0, 1.0],
              [1.0, 1.0, 3.0],
              [2.0, 5.0, 8.0]])
    real_solution = np.array([1.0, 2.0, 3.0])
    b = np.array([6.0, 12.0, 36.0])

    x_normal = solve_with_gaussian_elimination(A,b)
    x_complete = complete_gaussian_pivoting(A,b)

    print(x_normal)
    print(x_complete)

if __name__ == "__main__":
    main()
