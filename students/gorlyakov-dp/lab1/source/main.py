import numpy as np
from sklearn.datasets import load_wine

wine = load_wine()

X = wine.data
y = wine.target

mask = (y == 0) | (y == 1)

X = X[mask]
y = y[mask]

y = np.where(y == 0, -1, 1)

w = np.empty(X.shape[1])

for j in range(X.shape[1]):
    w[j] = (y @ X[:, j]) / (X[:, j] @ X[:, j])

M = np.empty(X.shape[0])
# L = (1 - M) ^ 2
grad_L = np.empty((X.shape[0], X.shape[1]))

lambd = 0.1

sample_idx = np.random.choice(len(X), size=10, replace=False)
M_sample = y[sample_idx] * (X[sample_idx] @ w)
Q = np.mean((1 - M_sample) ** 2)

h = 0.001
max_iter = 10000

v = np.zeros_like(w)
gamma = 0.9
t = 0.001

for _ in range(max_iter):
    i = np.random.randint(X.shape[0])
    M[i] = y[i] * (X[i, :] @ w)
    loss_i = (1 - M[i]) ** 2
    grad_L[i] = -2 * (1 - M[i]) * y[i] * X[i]
    v = gamma * v + (1 - gamma) * grad_L[i]
    w = w * (1 - h * t) - h * v
    Q = lambd * loss_i + (1 - lambd) * Q
