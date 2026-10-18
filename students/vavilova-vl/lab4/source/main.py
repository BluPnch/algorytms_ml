"""Лабораторная работа №4: PCA через SVD.

Запуск из каталога lab4:  uv run python source/main.py
"""

from pathlib import Path

import numpy as np

import plots
from data import load_dataset
from dimension import condition_number, effective_dimension, kaiser_dimension, residual_variance
from pca import PCA
from reference import compare, reconstruction_errors
from regression import least_squares, pcr_curve

IMAGES = Path(__file__).resolve().parent.parent / "images"
EPS = 0.05
TOLERANCE = 1e-10


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    data = load_dataset()
    X, y = data.X, data.y
    n_objects, n_features = data.shape

    section("Датасет")
    print(f"Titanic: ℓ = {n_objects}, n = {n_features}")
    print("Признаки:", ", ".join(data.feature_names))

    section("PCA через SVD")
    pca = PCA().fit(X)
    lam = pca.all_eigenvalues_
    E = residual_variance(lam)
    print(f"{'j':>2} {'σ_j':>8} {'λ_j':>8} {'доля':>7} {'накоп.':>7} {'E_j':>7}")
    for j in range(n_features):
        print(f"{j + 1:>2} {pca.singular_values_[j]:8.3f} {lam[j]:8.4f} "
              f"{pca.explained_variance_ratio_[j]:7.2%} {1 - E[j + 1]:7.2%} {E[j + 1]:7.4f}")
    # Проверка: σ_j² / (ℓ − 1) совпадают с собственными значениями ковариационной матрицы.
    cov_eig = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    print(f"max |λ_SVD − λ_eig(cov)| = {np.max(np.abs(lam - cov_eig)):.2e}")

    section("Эффективная размерность")
    m_eff = effective_dimension(lam, EPS)
    print(f"Критерий E_m ≤ ε = {EPS}:  m = {m_eff}")
    for eps in (0.2, 0.1, 0.01):
        print(f"  при ε = {eps}: m = {effective_dimension(lam, eps)}")
    print(f"Правило Кайзера (λ_j > средн. λ): m = {kaiser_dimension(lam)}")
    print(f"Число обусловленности μ = λ_max / λ_min = {condition_number(lam):.1f}")

    section("Сравнение с sklearn.decomposition.PCA")
    print(f"{'m':>2} {'V':>9} {'λ':>9} {'доля':>9} {'σ':>9} {'G':>9} {'X̂':>9} {'знаки':>6}")
    worst = 0.0
    for m in range(1, n_features + 1):
        c = compare(X, m)
        worst = max(worst, c.max_error)
        print(f"{m:>2} {c.components:9.1e} {c.explained_variance:9.1e} {c.explained_variance_ratio:9.1e} "
              f"{c.singular_values:9.1e} {c.transformed:9.1e} {c.reconstructed:9.1e} {c.sign_flips:>6}")
    status = "OK" if worst < TOLERANCE else "РАСХОЖДЕНИЕ"
    print(f"Максимальное отклонение по всем m: {worst:.2e}  [{status}, допуск {TOLERANCE:g}]")
    rec_ours, rec_ref = reconstruction_errors(X)
    print(f"max |ошибка восстановления ours − sklearn| = {np.max(np.abs(rec_ours - rec_ref)):.2e}")
    print(f"max |ошибка восстановления − E_m|          = {np.max(np.abs(rec_ours - E[1:])):.2e}")

    section("Регрессия на главных компонентах (5-fold CV)")
    rmse = pcr_curve(X, y)
    for m, r in enumerate(rmse, start=1):
        mark = "  ← m_eff" if m == m_eff else ""
        print(f"m = {m:>2}: RMSE = {r:7.3f}{mark}")
    alpha_full, _ = least_squares(X, y)
    alpha_pcr, _ = least_squares(PCA(m_eff).fit(X).transform(X), y)
    print(f"‖α‖ МНК на всех признаках = {np.linalg.norm(alpha_full):.3f}, "
          f"‖α‖ в базисе {m_eff} компонент = {np.linalg.norm(alpha_pcr):.3f}")

    plots.spectrum(lam, E, EPS, m_eff, IMAGES / "spectrum.png")
    plots.projection(pca.transform(X), y, pca.explained_variance_ratio_, IMAGES / "projection.png")
    plots.loadings(pca.components_, data.feature_names, IMAGES / "loadings.png")
    plots.reconstruction(rec_ours, rec_ref, E, IMAGES / "reconstruction.png")
    plots.pcr(rmse, m_eff, IMAGES / "pcr.png")
    print(f"\nГрафики сохранены в {IMAGES}")


if __name__ == "__main__":
    main()
