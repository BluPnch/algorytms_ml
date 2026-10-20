import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()

X = wine.data
y = wine.target

mask = (y == 0) | (y == 1)

X = X[mask]
y = y[mask]

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30, stratify=y
)

w0 = np.empty(X_train.shape[1])

for j in range(X_train.shape[1]):
    w0[j] = np.dot(y_train, X_train[:, j]) / np.dot(X_train[:, j], X_train[:, j])


def train_classifier(X, y, w):
    w = w.copy()

    M = np.empty(X.shape[0])
    # L = (1 - M) ** 2
    grad_L = np.empty((X.shape[0], X.shape[1]))

    lambd = 0.1

    sample_idx = np.random.choice(len(X), size=10, replace=False)
    M_sample = y[sample_idx] * np.dot(X[sample_idx], w)
    Q = np.mean((1 - M_sample) ** 2)

    max_iter = 10000
    eps = 1e-8

    for _ in range(max_iter):
        M = y * np.dot(X, w)
        sample_weights = 1 / (np.abs(M) + eps)
        probabilities = sample_weights / np.sum(sample_weights)
        i = np.random.choice(X.shape[0], p=probabilities)
        loss_i = (1 - M[i]) ** 2
        grad_L[i] = -2 * (1 - M[i]) * y[i] * X[i]
        h_star = 1 / (2 * np.dot(X[i], X[i]))
        w = w - h_star * grad_L[i]
        Q = lambd * loss_i + (1 - lambd) * Q

    return w, Q


w, Q = train_classifier(X_train, y_train, w0)

y_pred_train = np.sign(np.dot(X_train, w))
y_pred_test = np.sign(np.dot(X_test, w))

print("Train accuracy:", np.mean(y_pred_train == y_train))
print("Test accuracy:", np.mean(y_pred_test == y_test))
print("Q:", Q)

w_multistart = None
Q_miltistart = None
old_loss = np.inf
for _ in range(10):
    w0 = np.random.uniform(
        -1 / (2 * X_train.shape[1]), 1 / (2 * X_train.shape[1]), size=X_train.shape[1]
    )
    w, Q = train_classifier(X_train, y_train, w0)
    M_train = y_train * np.dot(X_train, w)
    train_loss = np.mean((1 - M_train) ** 2)
    if train_loss < old_loss:
        old_loss = train_loss
        w_multistart = w
        Q_miltistart = Q

w = w_multistart
Q = Q_miltistart

y_pred_train = np.sign(np.dot(X_train, w))
y_pred_test = np.sign(np.dot(X_test, w))

print("Train accuracy:", np.mean(y_pred_train == y_train))
print("Test accuracy:", np.mean(y_pred_test == y_test))
print("Q:", Q)
