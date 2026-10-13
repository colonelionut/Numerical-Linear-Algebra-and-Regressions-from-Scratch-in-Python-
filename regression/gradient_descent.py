"""Simple linear regression y = w x + b trained by gradient descent on the MSE."""

import numpy as np


def predict(x, w, b):
	return w * x + b


def cost(x, y, w, b):
	e = predict(x, w, b) - y
	return np.dot(e, e) / x.shape[0]


def gradients(x, y, w, b):
	m = x.shape[0]
	e = predict(x, w, b) - y
	dw = (2 / m) * np.dot(e, x)
	db = (2 / m) * np.sum(e)
	return dw, db


def train(x, y, lr=0.05, epochs=1000):
	"""Returns w, b and the cost after every epoch."""
	w, b = 0.0, 0.0
	history = np.empty(epochs)
	for t in range(epochs):
		dw, db = gradients(x, y, w, b)
		w = w - lr * dw
		b = b - lr * db
		history[t] = cost(x, y, w, b)
	return w, b, history
