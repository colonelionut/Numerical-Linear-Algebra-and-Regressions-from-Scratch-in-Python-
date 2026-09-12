import numpy as np
from functions_for_linear_systems import solve_using_back_substitution

def complete_gaussian_pivoting(A, b):

    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    perm = np.arange(n)

    # Since we have to interchange columns too, we can use slicing instead of another matrix!

    for p in range(n):
        # We look for the maximum value (absolute value) in the submatrix
        submatrix = A[p:, p:] 

        # We have to remember its coordinates as well, in order to interchange row and columns

        # To obtain the row number, we use argmax
        idx = np.argmax(np.abs(submatrix))


        # submatrix is treated as an independent matrix, unravel_index gives us the indices in submatrix, not in A!
        row_submatrix, column_submatrix = np.unravel_index(idx, submatrix.shape)

        # We now translate the indices into our matrix A
        maximum_row = p + row_submatrix
        maximum_column = p + column_submatrix

        # Now that we have obtained the indices for the maximum absolute value in the submatrix, we can move on to interchanging the columns and rows

        # Interchange rows
        A[[p, maximum_row], :] = A[[maximum_row, p], :]

        # Interchange columns
        A[:, [p, maximum_column]] = A[:, [maximum_column, p]]

        # Interchange rows in b

        b[[p,maximum_row]] = b[[maximum_row, p]]
        perm[p], perm[maximum_column] = perm[maximum_column], perm[p]

        # Now we can perform the normal Gaussian elimination
        for i in range(p+1, n):
            m = A[i,p] / A[p,p]
            for k in range (p+1, n):
                A[i,k] = A[i,k] - m * A[p,k]
            b[i] = b[i] - m * b[p]

    # We have now obtained the echelon form. We can now solve the system using back substitution
    y = solve_using_back_substitution(A,b)

    x = np.zeros(n)

    # Rearrange the solutions
    x[perm] = y 

    return x


def main():
    A = np.array([[0.0, 2.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [2.0, 1.0, 3.0]])
    x_true = np.array([3.0, 2.0, 1.0])
    b = A @ x_true              # b garantat coerent cu solutia
    A_orig = A.copy()
    b_orig = b.copy()
    x = complete_gaussian_pivoting(A, b)
    print("x adevarat :", x_true)
    print("x gasit    :", x)
    print("corect     :", np.allclose(A_orig @ x, b_orig))


if __name__ == "__main__":
    main()







