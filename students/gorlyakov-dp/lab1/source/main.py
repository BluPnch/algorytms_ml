import numpy as np
import pandas as pd
from metrics import accuracy_score, f1_score, precision_score, recall_score
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

w0_corr = np.empty(X_train.shape[1])

for j in range(X_train.shape[1]):
    w0_corr[j] = np.dot(y_train, X_train[:, j]) / np.dot(X_train[:, j], X_train[:, j])


def choose_random(M):
    return np.random.choice(len(M))


def choose_by_margin(M):
    eps = 1e-8
    sample_weights = 1 / (np.abs(M) + eps)
    probabilities = sample_weights / np.sum(sample_weights)
    return np.random.choice(len(M), p=probabilities)


def train_classifier(X, y, w, choose_object):
    w = w.copy()

    M = np.empty(X.shape[0])
    # L = (1 - M) ** 2
    grad_L = np.empty((X.shape[0], X.shape[1]))

    lambd = 0.1

    sample_idx = np.random.choice(len(X), size=10, replace=False)
    M_sample = y[sample_idx] * np.dot(X[sample_idx], w)
    Q = np.mean((1 - M_sample) ** 2)

    max_iter = 10000
    for _ in range(max_iter):
        M = y * np.dot(X, w)
        i = choose_object(M)
        loss_i = (1 - M[i]) ** 2
        grad_L[i] = -2 * (1 - M[i]) * y[i] * X[i]
        h_star = 1 / (2 * np.dot(X[i], X[i]))
        w = w - h_star * grad_L[i]
        Q = lambd * loss_i + (1 - lambd) * Q

    return w, Q


def evaluate_model(name, X, y, w):
    y_pred = np.sign(np.dot(X, w))
    return {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


results = []

w, Q = train_classifier(X_train, y_train, w0_corr, choose_random)
results.append(evaluate_model("correlation + random", X_test, y_test, w))

w_multistart = None
Q_miltistart = None
old_loss = np.inf
for _ in range(10):
    w0 = np.random.uniform(
        -1 / (2 * X_train.shape[1]), 1 / (2 * X_train.shape[1]), size=X_train.shape[1]
    )
    w, Q = train_classifier(X_train, y_train, w0, choose_random)
    M_train = y_train * np.dot(X_train, w)
    train_loss = np.mean((1 - M_train) ** 2)
    if train_loss < old_loss:
        old_loss = train_loss
        w_multistart = w
        Q_miltistart = Q

w = w_multistart
Q = Q_miltistart
results.append(evaluate_model("multistart + random", X_test, y_test, w))

w0 = np.random.uniform(
    -1 / (2 * X_train.shape[1]), 1 / (2 * X_train.shape[1]), size=X_train.shape[1]
)
w, Q = train_classifier(X_train, y_train, w0.copy(), choose_random)
results.append(evaluate_model("random init + random", X_test, y_test, w))

w, Q = train_classifier(X_train, y_train, w0.copy(), choose_by_margin)
results.append(evaluate_model("random init + margin", X_test, y_test, w))

print(pd.DataFrame(results).to_string(index=False))
