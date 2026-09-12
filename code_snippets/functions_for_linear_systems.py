import numpy as np

def solve_diagonal_system(A,b):
    n = b.shape[0]
    x = np.zeros(n)
    for i in range (0, n):
        x[i] = b[i]/A[i,i]
    return x

def solve_using_back_substitution(A,b):
	n = b.shape[0]
	x = np.zeros(n)
	for i in range(n-1, -1, -1):
		s = 0.0
		for j in range(i+1, n):
			s = s + A[i,j]*x[j]
		x[i] = (b[i] - s)/A[i,i]
		pass
	return x

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


def crout_method(A):

    A = A.astype(float)

    # We can initialise L as a zero matrix
    n = len(A)

    L = np.zeros((n,n))

    # We can initialise U as the identity matrix, since the main diagonal will remain 1

    U = np.eye(n)

    for p in range(n):
        for i in range(0,p):
            U[i,p] = (A[i,p] - np.dot(L[i, 0:p], U[0:p, p]))/ L[i,i]
        for i in range(p,n):
            L[i,p] = A[i,p] - np.dot(L[i, 0:p], U[0:p, p])


            
    return L, U



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
