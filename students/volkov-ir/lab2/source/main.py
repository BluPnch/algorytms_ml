from pathlib import Path

import report
from data_preprocessing import prepare_data
from model import calculate_loo, calculate_best_k, calculate_predict, calculate_margin, calculate_start_omega, greedy_addition
from report import (evaluate, find_errors, print_loo, print_metrics, print_matrices, print_sklearn_check, print_prototypes,
                    plot_loo_risk, plot_prototype_map, plot_ccv_history)
from sklearn_baseline import run_sklearn_baseline

# графики для README
report.SAVE_DIR = Path(__file__).parent.parent / "images"

X_train, X_test, y_train, y_test, feature_names, class_names = prepare_data()
n_classes = len(class_names)


# k_values – какие k перебираются в LOO, окну Парзена нужен (k+1)-й сосед среди l-1 объектов, поэтому k не больше l-2
# k_zoom – до какого k рисуется второй, увеличенный график LOO
# k_check – для каких k LOO своих функций сверяется со sklearn
# n_prototypes – сколько эталонов оставить, None – все, что отобрал жадный алгоритм
k_values = list(range(1, len(X_train) - 1))
k_zoom = 50
k_check = list(range(1, 51))
n_prototypes = None
methods = ["knn", "parzen", "parzen_lecture"]

SHOW_PLOTS = True
RUN_SKLEARN = True


# 3, 4 – подбор k по LOO на train
print("Подбор k по LOO ...")
all_loo = {}
best_k = {}
for method in methods:
    all_loo[method] = calculate_loo(X_train, y_train, k_values, method)
    best_k[method] = calculate_best_k(k_values, all_loo[method])
print_loo(all_loo, best_k, k_values, len(X_train))

if SHOW_PLOTS:
    plot_loo_risk(all_loo, best_k, k_values, k_values[-1], "все k", "loo_risk_all.png")
    plot_loo_risk(all_loo, best_k, k_values, k_zoom, f"k от 1 до {k_zoom}", "loo_risk_zoom.png")


# 5 – качество на test с лучшим k
test_rows = []
for method in methods:
    pred_labels = calculate_predict(X_train, y_train, X_test, best_k[method], method)
    test_rows.append(("своя", report.SHORT_NAMES[method], best_k[method], evaluate(y_test, pred_labels, n_classes)))

# 5 – эталон sklearn
if RUN_SKLEARN:
    print("Сравнение со sklearn ...")
    sklearn_results = run_sklearn_baseline(X_train, y_train, X_test, k_check)
    print_sklearn_check(all_loo, sklearn_results, k_check)
    for name in sklearn_results:
        result = sklearn_results[name]
        test_rows.append(("sklearn", report.SHORT_NAMES[name], result["k"], evaluate(y_test, result["pred"], n_classes)))

print_metrics("Качество на test", test_rows, len(X_test))
print_matrices(test_rows, class_names)


# 6 – отбор эталонов, старт по отступу окна Парзена с его лучшим k
print("Отбор эталонов ...")
margins = calculate_margin(X_train, y_train, best_k["parzen"])
start_idx = calculate_start_omega(margins, y_train)
omega_idx, history = greedy_addition(X_train, y_train, start_idx)
if n_prototypes is not None:
    omega_idx = omega_idx[:n_prototypes]
    history = history[:n_prototypes - len(start_idx) + 1]
print_prototypes(start_idx, omega_idx, history, y_train, class_names)

# 7 – визуализация, графики с n_prototypes сохраняются отдельно и в README не идут
if SHOW_PLOTS:
    suffix = "" if n_prototypes is None else f"_n{n_prototypes}"
    errors_idx = find_errors(X_train, y_train, omega_idx)
    plot_prototype_map(X_train, y_train, omega_idx, len(start_idx), errors_idx, class_names, f"prototypes_map{suffix}.png")
    plot_ccv_history(history, len(start_idx), len(X_train), f"prototypes_ccv{suffix}.png")


# 8 – 1NN с отбором эталонов и без
pred_full = calculate_predict(X_train, y_train, X_test, 1, "knn")
pred_omega = calculate_predict(X_train[omega_idx], y_train[omega_idx], X_test, 1, "knn")
compare_rows = [
    ("своя", f"1NN, вся выборка ({len(X_train)})", 1, evaluate(y_test, pred_full, n_classes)),
    ("своя", f"1NN, эталоны ({len(omega_idx)})", 1, evaluate(y_test, pred_omega, n_classes)),
]
print_metrics("1NN с отбором эталонов и без", compare_rows, len(X_test))
