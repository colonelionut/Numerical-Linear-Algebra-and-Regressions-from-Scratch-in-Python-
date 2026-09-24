import numpy as np

def gauss_seidel(A,b, max_iter, TOL):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    x = np.zeros(n)
    for k in range (max_iter):
        x_new = x.copy()
        for i in range(n):
            sum_less = 0.0
            sum_greater = 0.0
            for j in range(n):
                if (i < j):
                    sum_less = sum_less + A[i,j] * x[j]
                if (i > j):
                    sum_greater = sum_greater + A[i,j] * x[j]
            x_new[i] = (b[i] - sum_greater - sum_less) / A[i,i]
        if np.linalg.norm(x_new - x, np.inf) < TOL:
            x = x_new
            break
        x = x_new
    return x

def main():
    A = np.array([[4.0, 1.0, -1.0],
                      [2.0, 7.0, 1.0],
                      [1.0, -3.0, 12.0]])
    b = np.array([3.0, 19.0, 31.0])
    real_solution = np.linalg.solve(A,b)
    print("The real solution is ", real_solution)

    for i in range(100):
        x = np.zeros(len(A))
        x =  gauss_seidel(A,b, i, 1e-10)
        print(f"Gauss-Sidel after {i} iterations is {x}".format(i,x))


if __name__ == "__main__":
    main()