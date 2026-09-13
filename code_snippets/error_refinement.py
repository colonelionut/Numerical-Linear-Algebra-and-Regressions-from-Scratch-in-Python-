import numpy as np
from functions_for_linear_systems import complete_gaussian_pivoting

def solve_using_error_refinement(A,b, max_iterations, tol):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    x = np.zeros(n)
    r = b

    e_norm_prev = np.inf
    for i in range(max_iterations):
        r = b - A @ x
        e = complete_gaussian_pivoting(A, r)
        x = x + e
        e_norm = np.linalg.norm(e, np.inf)
        if e_norm < tol:
            print(f"  CONVERGES in {i+1} iterations (||e||={e_norm:.2e})")
            break
        if e_norm >= e_norm_prev:
            print(f"  STALLS at iter {i+1} (||e||={e_norm:.2e}, tol={tol})")
            break
        e_norm_prev = e_norm
    else:
        print(f"  MAX_ITER hit without convergence")
    return x

def main():
    for order in range (1,10): 
        A = np.array([[1.0 / (i + j + 1) for j in range(order)] for i in range(order)])
        x_true = np.ones(order)          
        b = A @ x_true               

        x = solve_using_error_refinement(A, b, max_iterations=10000, tol=1e-25)

        print("True solution :", x_true)
        print("Solution found:", x)
        print("Max error:", np.linalg.norm(x - x_true, np.inf))
        print("Correct or not:", np.allclose(x, x_true))


if __name__ == "__main__":
    main()