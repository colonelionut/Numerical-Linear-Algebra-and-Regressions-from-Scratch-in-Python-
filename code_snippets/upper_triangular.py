import numpy as np

def back_substitution(A,b):
	n = b.shape[0]
	x = np.zeros(n)
	for i in range(n-1, -1, -1):
		s = 0.0
		for j in range(i+1, n):
			s = s + A[i,j]*x[j]
		x[i] = (b[i] - s)/A[i,i]
		pass
	return x




def main():
	A = np.array([
		[2.0, 1.0, -1.0],
		[0.0, 3.0,  2.0],
		[0.0, 0.0,  4.0],
	])
	b = np.array([3.0, 8.0, 8.0])
 
	x = back_substitution(A, b)
	print("x       =", x)
	print("A @ x   =", A @ x)   # trebuie sa dea inapoi b
	print("b       =", b)
	print("corect  :", np.allclose(A @ x, b))
 
 
if __name__ == "__main__":
	main()