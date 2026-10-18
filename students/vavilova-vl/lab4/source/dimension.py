"""Оценка эффективной размерности выборки по спектру ковариационной матрицы."""

import numpy as np


def residual_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """E_m = (λ_{m+1} + … + λ_n) / (λ_1 + … + λ_n) для m = 0, …, n.

    E_m — доля дисперсии, теряемая при проецировании на первые m компонент;
    она же равна относительной ошибке восстановления ‖F − G Uᵀ‖² / ‖F‖².
    """
    tail = np.concatenate([np.cumsum(eigenvalues[::-1])[::-1], [0.0]])
    return tail / eigenvalues.sum()


def effective_dimension(eigenvalues: np.ndarray, eps: float) -> int:
    """Наименьшее m, при котором E_m ≤ ε."""
    E = residual_variance(eigenvalues)
    return int(np.argmax(E <= eps + 1e-12))


def kaiser_dimension(eigenvalues: np.ndarray) -> int:
    """Правило Кайзера: число λ_j больше среднего собственного значения."""
    return int(np.sum(eigenvalues > eigenvalues.mean()))


def condition_number(eigenvalues: np.ndarray) -> float:
    """μ = λ_max / λ_min — мера мультиколлинеарности признаков."""
    return float(eigenvalues.max() / eigenvalues.min())
