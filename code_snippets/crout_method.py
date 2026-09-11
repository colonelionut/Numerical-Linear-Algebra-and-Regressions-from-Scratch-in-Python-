import numpy as np


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

def main():
    A = np.array([[1.0, 2.0, 3.0],
                  [2.0, 8.0, 11.0],
                  [3.0, 22.0, 42.0]])
    L, U = crout_method(A)
    print(L)
    print(U)
    if (np.allclose(L @ U, A)):
        print("Good job dumbass")
    else:
        print("Mai baga o fisa")

if __name__ == "__main__":
    main()

