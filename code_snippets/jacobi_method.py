import numpy as np 

def get_spectral_radius_jacobi(A):
    A = A.astype(float)
    D = np.diag(np.diag(A))
    LU = A - D
    T = -np.linalg.inv(D) @ LU
    eigenvalues = np.linalg.eigvals(T)      
    return np.max(np.abs(eigenvalues))       

def jacobi(A, b, max_iter, TOL):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    x = np.zeros(n)
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j:
                    s = s + A[i, j] * x[j]
            x_new[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < TOL:
            x = x_new
            break
        x = x_new
    return x

def main():
    A = np.array([[2.0, 7.0, 1.0],
                  [4.0, 1.0, -1.0],
                  [1.0, -3.0, 12.0]])
    b = np.array([19.0, 3.0, 31.0])
    x = jacobi(A,b, 10, 1e-10)
    real_solution = np.linalg.solve(A,b) 
    print("The spectral radius of the iteration matrix of A is ", get_spectral_radius_jacobi(A))
    
    # The solution will not be near correct, because A is not diagonally dominant.
    print("The Jacobi method solution is: ", x)
    print("The real solution is: ", real_solution)

    A1 = np.array([[4.0, 1.0, -1.0],
                  [2.0, 7.0, 1.0],
                  [1.0, -3.0, 12.0]])
    b1 = np.array([3.0, 19.0, 31.0])
    # The solution will be a little more accurate, but still off from the real result after 10 iterations, even after interchanging rows
    print("The spectral radius of the iteration matrix of A1 is ", get_spectral_radius_jacobi(A1))

    x2 = jacobi(A1, b1, 10, 1e-10)
    print("10 iterations: The Jacobi solution after interchanging rows is ", x2)

    x3 = jacobi(A1, b1, 20, 1e-10)
    print("20 iterations : The Jacobi solution after interchanging rows is ", x3)


if __name__ == "__main__":
    main()

