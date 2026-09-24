import numpy as np
import matplotlib.pyplot as plt


def jacobi_history(A, b, x_true, max_iter):
	A = A.astype(float)
	b = b.astype(float)
	n = len(A)
	x = np.zeros(n)

    # We simply declare an array in which elements will be added
	errors = []
	for k in range(max_iter):
		x_new = np.zeros(n)
		for i in range(n):
			s = 0.0
			for j in range(n):
				if i != j:
					s = s + A[i, j] * x[j]
			x_new[i] = (b[i] - s) / A[i, i]
		x = x_new
		err = np.linalg.norm(x - x_true, np.inf)

        # We add err to the errors array
		errors.append(err)
	return errors


def make_diagonally_dominant(n, strength, rng):

	M = rng.standard_normal((n, n))
	for i in range(n):
		M[i, i] = strength * np.sum(np.abs(M[i])) + 1.0
	return M


def main():
	rng = np.random.default_rng(0)
	n = 20
	max_iter = 60

	plt.figure(figsize=(8, 5))
	for strength in [3.0, 1.5, 1.0, 0.7]:
		A = make_diagonally_dominant(n, strength, rng)
		x_true = np.ones(n)
		b = A @ x_true
		errors = jacobi_history(A, b, x_true, max_iter)
		# We want on the Ox axis to have the number of the iteration
		# We want on the Oy axis the norm of the error
		plt.plot(range(len(errors)), errors, label=f"strength={strength}")

	plt.yscale("log")          
	plt.xlabel("iteration k")
	plt.ylabel("error ||x_k - x_true||")
	plt.title("Jacobi convergence vs diagonal dominance")
	plt.legend()
	plt.grid(True, which="both", alpha=0.3)
	plt.savefig("jacobi_convergence.png", dpi=120, bbox_inches="tight")
	print("saved jacobi_convergence.png")


if __name__ == "__main__":
	main()