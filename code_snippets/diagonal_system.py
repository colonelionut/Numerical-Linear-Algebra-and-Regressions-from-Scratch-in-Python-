import numpy as np

def diagonal_system(A,b):
    n = b.shape[0]
    x = np.zeros(n)
    for i in range (0, n):
        x[i] = b[i]/A[i,i]
    return x

def main():
    A = np.array([[1.0,0.0,0.0], [0.0,2.0,0.0], [0.0,0.0,3.0]])
    b = np.array([1.0,2.0,3.0])
    x = diagonal_system(A,b)
    print(x)

main()