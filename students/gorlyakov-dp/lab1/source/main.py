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

max_iter = 10000

for _ in range(max_iter):
    i = np.random.randint(X.shape[0])
    M[i] = y[i] * (X[i, :] @ w)
    loss_i = (1 - M[i]) ** 2
    grad_L[i] = -2 * (1 - M[i]) * y[i] * X[i]
    h_star = 1 / (2 * X[i] @ X[i])
    w = w - h_star * grad_L[i]
    Q = lambd * loss_i + (1 - lambd) * Q
