import numpy as np


def predict(x, w, b):
	# TODO: modelul liniar, y_hat = w*x + b (vectorizat, x e un vector)
	return w * x + b



def cost(x, y, w, b):
	m = x.shape[0]
	e = predict(x, w, b) - y
	return np.dot(e, e) / m


def gradients(x, y, w, b):
	m = x.shape[0]
	e = predict(x, w, b) - y
	# derivata lui J in raport cu w (vectorizat)
	dw = (2/m) * np.dot(e, x)
	# derivata lui J in raport cu b
	db = (2/m) * np.sum(e)
	return dw, db


def train(x, y, lr=0.05, epochs=1000):
	w, b = 0.0, 0.0
	for t in range(epochs):
		dw, db = gradients(x, y, w, b)
		w = w - lr * dw
		b = b - lr * db
		if t % 100 == 0:
			print(f"epoch {t:4d}  cost {cost(x, y, w, b):.4f}  w {w:.3f}  b {b:.3f}")
	return w, b

def main():
	rng = np.random.default_rng(0)
	m = 100
	x = np.linspace(-3, 3, m)
	w_true, b_true = 2.0, -1.0
	y = w_true * x + b_true + 0.5 * rng.standard_normal(m)

	w, b = train(x, y, lr=0.05, epochs=1000)
	print(f"\nadevarat: w {w_true}  b {b_true}")
	print(f"invatat : w {w:.3f}  b {b:.3f}")


if __name__ == "__main__":
	main()