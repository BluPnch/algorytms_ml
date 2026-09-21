import matplotlib.pyplot as plt
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

plt.scatter(np.arange(len(M)), np.sort(M))
plt.axhline(0)
plt.show()

print("max: ", max(M), " min: ", min(M))
print("M > 0 count: ", np.sum(M > 0))
print("M < 0 count: ", np.sum(M < 0))
