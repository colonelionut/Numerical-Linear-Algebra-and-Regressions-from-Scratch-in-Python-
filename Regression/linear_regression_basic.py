import numpy as np



def build_design_matrix(x):

    # The function returns the matrix X, which has m values of 1 on the first column (the bias) and the array x
	# on the second column
    
	m = x.shape[0]
	X = np.ones((m, 2))
	X[:, 1] = x
	return X


def fit_normal_equation(X, y):
    A = np.transpose(X) @ X
    c = np.transpose(X) @ y
        
    theta = np.linalg.solve(A, c)
    return theta


def predict(X, theta):
	# TODO: predictiile = X @ theta
	return X @ theta


def mse(X, y, theta):
	e = predict(X, theta) - y
	return np.dot(e, e) / y.shape[0]


def main():
	rng = np.random.default_rng(0)
	m = 100
	x = np.linspace(-3, 3, m)
	w_true, b_true = 2.0, -1.0
	y = w_true * x + b_true + 0.5 * rng.standard_normal(m)

	X = build_design_matrix(x)
	theta = fit_normal_equation(X, y)

	print("theta gasit :", theta)                 # [b, w]
	print("adevarat    : b =", b_true, " w =", w_true)
	print("MSE         :", mse(X, y, theta))


if __name__ == "__main__":
	main()