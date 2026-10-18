"""Регрессия на главных компонентах (PCR): проверка эффективной размерности на задаче."""

import numpy as np
from sklearn.model_selection import KFold

from pca import PCA


def least_squares(F: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """МНК через SVD: α = V Σ⁻¹ Uᵀ (y − ȳ), свободный член — ȳ − x̄ᵀα."""
    x_mean, y_mean = F.mean(axis=0), y.mean()
    U, S, Vt = np.linalg.svd(F - x_mean, full_matrices=False)
    alpha = Vt.T @ ((U.T @ (y - y_mean)) / S)
    return alpha, y_mean - x_mean @ alpha


def pcr_cv_rmse(X: np.ndarray, y: np.ndarray, n_components: int,
                n_splits: int = 5, seed: int = 0) -> float:
    """RMSE на кросс-валидации; PCA обучается только на обучающей части фолда."""
    errors = []
    for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        pca = PCA(n_components).fit(X[train])
        alpha, b = least_squares(pca.transform(X[train]), y[train])
        pred = pca.transform(X[test]) @ alpha + b
        errors.append(np.mean((y[test] - pred) ** 2))
    return float(np.sqrt(np.mean(errors)))


def pcr_curve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([pcr_cv_rmse(X, y, m) for m in range(1, X.shape[1] + 1)])
