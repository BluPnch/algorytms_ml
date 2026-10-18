"""Собственная реализация PCA через сингулярное разложение."""

import numpy as np


class PCA:
    """Метод главных компонент.

    Центрированная матрица раскладывается как F = U Σ Vᵀ. Строки Vᵀ — главные
    направления, σ_j² / (ℓ − 1) — собственные значения ковариационной матрицы
    FᵀF / (ℓ − 1), а новые признаки G = F V_m = U_m Σ_m.
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCA":
        n_objects, n_features = X.shape
        m = n_features if self.n_components is None else self.n_components
        if not 1 <= m <= min(n_objects, n_features):
            raise ValueError(f"n_components должно быть от 1 до {min(n_objects, n_features)}, получено {m}")

        self.mean_ = X.mean(axis=0)
        F = X - self.mean_
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        U, Vt = self._fix_signs(U, Vt)

        eigenvalues = S ** 2 / (n_objects - 1)
        self.n_components_ = m
        self.singular_values_ = S[:m]
        self.components_ = Vt[:m]
        self.explained_variance_ = eigenvalues[:m]
        self.explained_variance_ratio_ = eigenvalues[:m] / eigenvalues.sum()
        # Полный спектр нужен для оценки эффективной размерности.
        self.all_eigenvalues_ = eigenvalues
        self._U = U[:, :m]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self._U * self.singular_values_

    def inverse_transform(self, G: np.ndarray) -> np.ndarray:
        return G @ self.components_ + self.mean_

    @staticmethod
    def _fix_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SVD определено с точностью до знака пары (u_j, v_j).

        Фиксируем знак так, чтобы наибольшая по модулю координата v_j была
        положительной — тогда результат детерминирован.
        """
        idx = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(Vt.shape[0]), idx])
        signs[signs == 0] = 1.0
        return U * signs, Vt * signs[:, None]
