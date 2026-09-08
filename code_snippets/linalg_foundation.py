import numpy as np
import time


def matvec_loop(A, x):
	m, n = A.shape
	y = np.zeros(m)
	for i in range(m):
		for j in range(n):
			y[i] += A[i, j] * x[j]
	return y


def matvec_vec(A, x):
	return np.dot(A,x)



def dot_loop(u, v):
	s = 0.0
	for i in range(u.shape[0]):
		s += u[i] * v[i]
	return s


def dot_vec(u, v):
	return np.dot(u,v)


def norm_loop(u):
	s = 0.0
	for i in range(u.shape[0]):
		s += u[i] * u[i]
	return s ** 0.5


def norm_vec(u):
	return np.linalg.vector_norm(u);


def bench(fn, *args, reps=5):
	t0 = time.perf_counter()
	for _ in range(reps):
		out = fn(*args)
	dt = (time.perf_counter() - t0) / reps
	return out, dt


def main():
	rng = np.random.default_rng(0)
	m, n = 800, 800
	A = rng.standard_normal((m, n))
	x = rng.standard_normal(n)
	u = rng.standard_normal(n)
	v = rng.standard_normal(n)

	yl, tl = bench(matvec_loop, A, x)
	yv, tv = bench(matvec_vec, A, x)
	print("matvec  eq:", np.allclose(yl, yv), " speedup:", round(tl / tv, 1), "x")

	dl, _ = bench(dot_loop, u, v)
	dv, _ = bench(dot_vec, u, v)
	print("dot     eq:", np.allclose(dl, dv))

	nl, _ = bench(norm_loop, u)
	nv, _ = bench(norm_vec, u)
	print("norm    eq:", np.allclose(nl, nv))


if __name__ == "__main__":
	main()