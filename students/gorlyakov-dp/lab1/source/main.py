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

for i in range(X.shape[0]):
    M[i] = y[i] * (X[i, :] @ w)

# L = (1 - M) ^ 2
grad_L = np.empty((X.shape[0], X.shape[1]))

for i in range(X.shape[0]):
    grad_L[i] = -2 * (1 - M) * y[i] * X[i]

lambd = 0.1
sample_idx = np.random.choice(len(M), size=10, replace=False)
Q = np.mean((1 - M[sample_idx]) ** 2)

for m in range(1, len(M)):
    Q = lambd * (1 - M[m]) ** 2 + (1 - lambd) * Q
