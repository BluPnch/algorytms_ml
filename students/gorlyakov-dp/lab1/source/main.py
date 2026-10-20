import numpy as np
import pandas as pd
from metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_wine
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

np.random.seed(30)

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

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


def calculate_margin(X, y, w):
    return y * np.dot(X, w)


def calculate_gradient(x, y, margin):
    return -2 * (1 - margin) * y * x


def plot_margins(M):
    import matplotlib.pyplot as plt

    plt.scatter(np.arange(len(M)), np.sort(M))
    plt.axhline(0, color="black")
    plt.xlabel("Object")
    plt.ylabel("Margin")
    plt.title("Sorted object margins")
    plt.show()


def choose_random(M):
    return np.random.choice(len(M))


def choose_by_margin(M):
    eps = 1e-8
    sample_weights = 1 / (np.abs(M) + eps)
    probabilities = sample_weights / np.sum(sample_weights)
    return np.random.choice(len(M), p=probabilities)


def train_classifier(X, y, w, choose_object):
    w = w.copy()
    velocity = np.zeros_like(w)

    lambd = 0.1
    h = 0.001
    gamma = 0.9
    l2 = 0.001
    max_iter = 10000

    sample_idx = np.random.choice(len(X), size=10, replace=False)
    M_sample = calculate_margin(X[sample_idx], y[sample_idx], w)
    Q = np.mean((1 - M_sample) ** 2)
    Q_history = [Q]

    for _ in range(max_iter):
        M = calculate_margin(X, y, w)
        i = choose_object(M)
        loss_i = (1 - M[i]) ** 2
        grad = calculate_gradient(X[i], y[i], M[i])

        velocity = gamma * velocity + (1 - gamma) * grad
        w = w * (1 - h * l2) - h * velocity
        Q = lambd * loss_i + (1 - lambd) * Q
        Q_history.append(Q)

    return w, Q, Q_history


def train_steepest(X, y, w, choose_object):
    w = w.copy()
    lambd = 0.1
    max_iter = 10000

    sample_idx = np.random.choice(len(X), size=10, replace=False)
    M_sample = calculate_margin(X[sample_idx], y[sample_idx], w)
    Q = np.mean((1 - M_sample) ** 2)
    Q_history = [Q]

    for _ in range(max_iter):
        M = calculate_margin(X, y, w)
        i = choose_object(M)
        loss_i = (1 - M[i]) ** 2
        grad = calculate_gradient(X[i], y[i], M[i])
        h_star = 1 / (2 * np.dot(X[i], X[i]))

        w = w - h_star * grad
        Q = lambd * loss_i + (1 - lambd) * Q
        Q_history.append(Q)

    return w, Q, Q_history


def initialize_by_correlation(X, y):
    w = np.empty(X.shape[1])

    for j in range(X.shape[1]):
        w[j] = np.dot(y, X[:, j]) / np.dot(X[:, j], X[:, j])

    return w


def initialize_random(X):
    return np.random.uniform(
        -1 / (2 * X.shape[1]),
        1 / (2 * X.shape[1]),
        size=X.shape[1],
    )


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

w0_corr = initialize_by_correlation(X_train, y_train)
w, Q, Q_history = train_classifier(X_train, y_train, w0_corr, choose_random)
results.append(evaluate_model("correlation + SGD", X_test, y_test, w))

w_multistart = None
old_loss = np.inf
for _ in range(10):
    w0 = initialize_random(X_train)
    w, Q, Q_history = train_classifier(X_train, y_train, w0, choose_random)
    M_train = calculate_margin(X_train, y_train, w)
    train_loss = np.mean((1 - M_train) ** 2)

    if train_loss < old_loss:
        old_loss = train_loss
        w_multistart = w

results.append(evaluate_model("multistart + SGD", X_test, y_test, w_multistart))

w0_random = initialize_random(X_train)

np.random.seed(30)
w_random, Q, Q_history = train_classifier(
    X_train, y_train, w0_random.copy(), choose_random
)
results.append(evaluate_model("random init + random", X_test, y_test, w_random))

np.random.seed(30)
w_margin, Q, Q_history = train_classifier(
    X_train, y_train, w0_random.copy(), choose_by_margin
)
results.append(evaluate_model("random init + margin", X_test, y_test, w_margin))

np.random.seed(31)
w_steepest, Q, Q_history = train_steepest(
    X_train, y_train, w0_corr, choose_random
)
results.append(evaluate_model("steepest descent", X_test, y_test, w_steepest))

ridge = RidgeClassifier()
with np.errstate(all="ignore"):
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
results.append(
    {
        "model": "RidgeClassifier",
        "accuracy": accuracy_score(y_test, y_pred_ridge),
        "precision": precision_score(y_test, y_pred_ridge),
        "recall": recall_score(y_test, y_pred_ridge),
        "f1": f1_score(y_test, y_pred_ridge),
    }
)

print(pd.DataFrame(results).to_string(index=False))
