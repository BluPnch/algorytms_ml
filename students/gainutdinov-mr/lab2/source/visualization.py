"""Графики и GIF для лабораторной работы № 2."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image, ImageSequence

GRID_SIZE = 120
ANIMATION_GRID_SIZE = 75
PREDICTION_BATCH_SIZE = 512


class Visualizer:
    """Показывает результаты экспериментов в заданных парах признаков.

    По умолчанию: u + redshift, alpha + redshift и delta + redshift.
    Все входные X уже стандартизованы. Отбор выполнен по всем признакам;
    для раскраски строятся отдельные модели на двух координатах.
    Обычные точки показывают test с истинными классами, звёзды показывают
    выбранные строки train. Повторный отбор для каждой проекции не выполняется.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        output_dir: str | Path = "figures",
        projections: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.output_dir = Path(output_dir)
        self.projections = (
            list(projections)
            if projections is not None
            else [
                ("u", "redshift"),
                ("alpha", "redshift"),
                ("delta", "redshift"),
            ]
        )

    def plot_risk(
        self,
        results: pd.DataFrame,
        *,
        output_name: str = "loo_risk.png",
    ) -> Path:
        """Показать ошибки LOO: столбцы k, parzen_error, sklearn_error.

        Ошибки задаются долями. Сохранить график и вернуть путь к нему.
        """
        results = results.sort_values("k")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(results["k"], results["parzen_error"], "o-", label="Парзен")
        ax.plot(results["k"], results["sklearn_error"], "s--", label="sklearn KNN")
        ax.set(
            xlabel="Число соседей k",
            ylabel="Доля ошибок LOO",
            title="Риск скользящего контроля по всем признакам",
            ylim=(0, None),
        )
        ax.grid(alpha=0.25)
        ax.legend()
        return self._save_figure(fig, output_name)

    def plot_comparison(
        self,
        results: pd.DataFrame,
        *,
        output_name: str = "parzen_comparison.png",
    ) -> Path:
        """Показать accuracy, macro_f1 и predict_seconds для каждого name.

        Это метрики моделей на всех признаках, рассчитанные в main.py.
        """
        metrics = [
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro-F1"),
            ("predict_seconds", "Время predict, с"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (column, label) in zip(axes, metrics):
            values = results[column].to_numpy(dtype=float)
            bars = ax.bar(results["name"], values, color="tab:blue", alpha=0.8)
            ax.bar_label(bars, fmt="%.3f", fontsize=9)
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", labelrotation=20)
            ax.grid(axis="y", alpha=0.25)
            if column != "predict_seconds":
                ax.set_ylim(0, 1.08)
            for position in np.flatnonzero(~np.isfinite(values)):
                ax.text(position, 0, "нет данных", ha="center", va="bottom", fontsize=8)
        fig.suptitle("Сравнение на test по всем признакам")
        return self._save_figure(fig, output_name)

    def plot_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        predictors: Mapping[
            str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        ],
        output_prefix: str = "classification",
    ) -> list[Path]:
        """Сравнить переданные классификаторы в каждой выбранной проекции.

        predictors сопоставляет имя модели с функцией (X_fit, y_fit, X_query).
        В эту функцию передаются две координаты. Она обучает модель на X_fit
        и возвращает ответы для X_query. Точки взяты из test.
        Значение accuracy относится к классификации в 2D.
        """
        if not predictors:
            raise ValueError("Нужен хотя бы один классификатор")
        classes = np.unique(np.concatenate((y_train, y_test)))
        paths = []
        for projection in self.projections:
            train_2d = self._project(X_train, projection)
            test_2d = self._project(X_test, projection)
            grid = self._make_grid(train_2d, test_2d, GRID_SIZE)
            fig, axes = plt.subplots(
                1, len(predictors), figsize=(7 * len(predictors), 6), squeeze=False
            )
            for ax, (name, predictor) in zip(axes[0], predictors.items()):
                accuracy = self._draw_classification(
                    ax, train_2d, y_train, test_2d, y_test, grid, classes, predictor
                )
                ax.set_title(f"{name}\nAccuracy на test в 2D: {accuracy:.1%}")
                self._label_axes(ax, projection)
            fig.suptitle(" + ".join(projection))
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(classes))
            fig.tight_layout(rect=(0, 0.09, 1, 0.94))
            name = f"{output_prefix}_{projection[0]}_{projection[1]}.png"
            paths.append(self._save_figure(fig, name, tight_layout=False))
        return paths

    def plot_prototypes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        indices: np.ndarray,
        *,
        title: str,
        output_prefix: str = "prototypes",
    ) -> list[Path]:
        """Сравнить 1NN по всему train и выбранным эталонам в каждой проекции.

        Эталоны из train показаны звёздами. Обычные точки представляют
        объекты test с истинными классами.
        Фон и accuracy относятся к отдельным двумерным моделям 1NN.
        """
        indices = np.sort(np.asarray(indices))
        classes = np.unique(np.concatenate((y_train, y_test)))
        support_sets = [
            ("1NN по всему train", np.arange(len(X_train))),
            ("1NN по эталонам", indices),
        ]
        paths = []
        for projection in self.projections:
            train_2d = self._project(X_train, projection)
            test_2d = self._project(X_test, projection)
            grid = self._make_grid(train_2d, test_2d, GRID_SIZE)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
            for ax, (name, support_indices) in zip(axes, support_sets):
                accuracy = self._draw_classification(
                    ax,
                    train_2d[support_indices],
                    y_train[support_indices],
                    test_2d,
                    y_test,
                    grid,
                    classes,
                    self._predict_1nn,
                )
                self._draw_prototypes(ax, train_2d, y_train, indices, classes)
                ax.set_title(
                    f"{name}: {len(support_indices)} объектов\n"
                    f"Accuracy на test в 2D: {accuracy:.1%}"
                )
                self._label_axes(ax, projection)
            fig.suptitle(f"{title}: {' + '.join(projection)}")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(classes) + 1)
            fig.text(
                0.5,
                0.07,
                f"Отбор по всем {X_train.shape[1]} признакам. Фон и accuracy: 1NN в 2D.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.12, 1, 0.94))
            name = f"{output_prefix}_{projection[0]}_{projection[1]}.png"
            paths.append(self._save_figure(fig, name, tight_layout=False))
        return paths

    def plot_selection_history(
        self,
        history: list[dict[str, object]],
        *,
        title: str,
        test_errors: Sequence[float] | None = None,
        mark_ccv_minimum: bool = False,
        output_name: str = "selection_history.png",
    ) -> Path:
        """Показать CCV и, при наличии, ошибку 1NN на test по всем признакам.

        test_errors соответствует всем записям history, включая начальную.
        mark_ccv_minimum отмечает минимум CCV вертикальной пунктирной линией.
        При равных минимумах выбрать меньшее число эталонов.
        """
        if not history:
            raise ValueError("История отбора пуста")
        if test_errors is not None and len(test_errors) != len(history):
            raise ValueError("test_errors должен соответствовать полной history")
        counts = [entry["n_prototypes"] for entry in history]
        ccv_values = [entry["ccv"] for entry in history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(counts, ccv_values, "o-", markersize=3, label="CCV отбора (1NN)")
        if test_errors is not None:
            ax.plot(counts, test_errors, "--", label="Ошибка 1NN на test")
        ax.scatter(
            counts[-1],
            ccv_values[-1],
            marker="s",
            color="black",
            zorder=4,
            label=f"Итог: {counts[-1]} эталонов",
        )
        if mark_ccv_minimum:
            minimum_position = min(
                range(len(history)), key=lambda i: (ccv_values[i], counts[i])
            )
            ax.axvline(
                counts[minimum_position],
                color="0.4",
                linestyle="--",
                label=f"Минимум CCV: {counts[minimum_position]} эталонов",
            )
        ax.set(
            xlabel="Количество эталонов",
            ylabel="Доля ошибок",
            title=f"{title}: метрики 1NN по всем признакам",
            ylim=(0, None),
        )
        ax.grid(alpha=0.25)
        ax.legend()
        return self._save_figure(fig, output_name)

    def animate_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        history: list[dict[str, object]],
        *,
        operation: Literal["addition", "removal"],
        accuracy_all_features: Sequence[float],
        output_prefix: str,
        frame_duration_ms: int = 2000,
        final_hold_ms: int = 6000,
        max_steps: int | None = None,
    ) -> list[Path]:
        """Сохранить GIF по выбранным проекциям и вернуть пути.

        Показывать начальное множество и принятые изменения из history.
        Один график на кадр, выбранный объект выделяется окружностью.
        max_steps ограничивает число показываемых изменений с указанием
        исходных номеров шагов. На кадре: CCV, accuracy 2D и accuracy 1NN
        по всем признакам из accuracy_all_features. Массив метрик соответствует
        полной history; ошибки красными кольцами не отмечаются.
        """
        if operation not in ("addition", "removal"):
            raise ValueError("operation должен быть 'addition' или 'removal'")
        if not history or len(accuracy_all_features) != len(history):
            raise ValueError("Нужны непустая history и accuracy для каждого её шага")
        if frame_duration_ms < 10 or final_hold_ms < 10:
            raise ValueError("Длительности кадров должны быть не меньше 10 мс")
        if max_steps is not None and (
            not isinstance(max_steps, (int, np.integer)) or max_steps < 1
        ):
            raise ValueError("max_steps должен быть положительным целым числом")

        positions = np.arange(len(history))
        if max_steps is not None and len(history) - 1 > max_steps:
            if max_steps == 1:
                positions = np.array([0, len(history) - 1])
            else:
                shown_steps = np.linspace(1, len(history) - 1, max_steps, dtype=int)
                positions = np.concatenate(([0], shown_steps))

        classes = np.unique(np.concatenate((y_train, y_test)))
        operation_name = (
            "Жадное добавление" if operation == "addition" else "Жадное удаление"
        )
        paths = []
        for projection in self.projections:
            train_2d = self._project(X_train, projection)
            test_2d = self._project(X_test, projection)
            grid = self._make_grid(train_2d, test_2d, ANIMATION_GRID_SIZE)
            fig, ax = plt.subplots(figsize=(9, 7), dpi=90)
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.22, top=0.78)
            fig.text(
                0.5,
                0.02,
                f"Отбор: все {X_train.shape[1]} признаков. Фон: 1NN в 2D.",
                ha="center",
                fontsize=9,
            )
            frames = []
            for position in positions:
                entry = history[position]
                # Меньший исходный индекс первым участвует в разрешении равных расстояний.
                indices = np.sort(np.asarray(entry["indices"]))
                ax.clear()
                accuracy_2d = self._draw_classification(
                    ax,
                    train_2d[indices],
                    y_train[indices],
                    test_2d,
                    y_test,
                    grid,
                    classes,
                    self._predict_1nn,
                )
                self._draw_prototypes(ax, train_2d, y_train, indices, classes)
                changed_index = entry["changed_index"]
                action = "Начальное множество"
                if changed_index is not None:
                    point = train_2d[changed_index]
                    action = (
                        f"Добавлен эталон {changed_index}"
                        if operation == "addition"
                        else f"Удалён эталон {changed_index}"
                    )
                    ax.scatter(
                        point[0],
                        point[1],
                        s=240,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.8,
                        zorder=6,
                        label="Выбранный объект",
                    )
                ax.set_title(
                    f"Эталонов: {len(indices)}; CCV (все признаки): {entry['ccv']:.5f}\n"
                    f"Accuracy 1NN на test: 2D = {accuracy_2d:.1%}; "
                    f"все признаки = {accuracy_all_features[position]:.1%}",
                    fontsize=10,
                )
                self._label_axes(ax, projection)
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.17),
                    ncol=3,
                    fontsize=8,
                )
                sampling_note = ""
                if len(positions) < len(history):
                    sampling_note = "\nПоказаны выбранные шаги; номера исходные"
                fig.suptitle(
                    f"{operation_name}: {' + '.join(projection)}\n"
                    f"Шаг {entry['step']}/{history[-1]['step']}. {action}"
                    f"{sampling_note}",
                    fontsize=11,
                )
                fig.canvas.draw()
                pixels = np.asarray(fig.canvas.buffer_rgba())
                frames.append(Image.fromarray(pixels.copy()).convert("RGB"))
            plt.close(fig)
            durations = [frame_duration_ms] * len(frames)
            durations[-1] = max(frame_duration_ms, final_hold_ms)
            path = self._output_path(
                f"{output_prefix}_{projection[0]}_{projection[1]}.gif"
            )
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                disposal=2,
            )
            paths.append(path)
        return paths

    def save_fast_gif(
        self,
        source_path: str | Path,
        *,
        output_path: str | Path | None = None,
        edge_frames: int = 4,
        edge_duration_ms: int = 1200,
        middle_duration_ms: int = 120,
        final_hold_ms: int = 8000,
    ) -> Path:
        """Сохранить GIF с замедленными краями и ускоренной серединой.

        Меняются длительности кадров; изображения сохраняются.
        Без output_path добавить суффикс _fast к имени исходного файла.
        """
        if not isinstance(edge_frames, (int, np.integer)) or edge_frames < 0:
            raise ValueError("edge_frames должен быть неотрицательным целым числом")
        if min(edge_duration_ms, middle_duration_ms, final_hold_ms) < 10:
            raise ValueError("Длительности кадров должны быть не меньше 10 мс")
        source_path = Path(source_path)
        if output_path is None:
            output_path = source_path.with_name(f"{source_path.stem}_fast.gif")
        output_path = Path(output_path)
        with Image.open(source_path) as source:
            if source.format != "GIF":
                raise ValueError("Ожидается исходный файл GIF")
            frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(source)]
        durations = [middle_duration_ms] * len(frames)
        for position in range(len(frames)):
            if position < edge_frames or position >= len(frames) - edge_frames - 1:
                durations[position] = edge_duration_ms
        durations[-1] = final_hold_ms
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
        )
        return output_path

    def _project(self, X: np.ndarray, projection: tuple[str, str]) -> np.ndarray:
        """Взять две заданные колонки без повторного масштабирования."""
        columns = [self.feature_names.index(name) for name in projection]
        return X[:, columns]

    @staticmethod
    def _make_grid(
        X_train: np.ndarray, X_test: np.ndarray, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Построить общую сетку для train и test с небольшим отступом."""
        points = np.vstack((X_train, X_test))
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        padding = np.maximum((upper - lower) * 0.05, 0.05)
        return np.meshgrid(
            np.linspace(lower[0] - padding[0], upper[0] + padding[0], size),
            np.linspace(lower[1] - padding[1], upper[1] + padding[1], size),
        )

    def _draw_classification(
        self, ax, X_fit, y_fit, X_test, y_test, grid, classes, predictor
    ) -> float:
        """Нарисовать фон и test, вернуть accuracy в этих двух координатах."""
        if len(X_test) == 0:
            raise ValueError("Для accuracy нужен непустой test")
        xx, yy = grid
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        queries = np.vstack((grid_points, X_test))
        predictions = self._predict_batches(predictor, X_fit, y_fit, queries)
        if not np.isin(predictions, classes).all():
            raise ValueError("Классификатор вернул неизвестные метки классов")
        background = np.searchsorted(classes, predictions[: len(grid_points)])
        colors = plt.get_cmap("tab10")(np.arange(len(classes)))
        ax.pcolormesh(
            xx,
            yy,
            background.reshape(xx.shape),
            cmap=ListedColormap(colors),
            vmin=-0.5,
            vmax=len(classes) - 0.5,
            alpha=0.25,
            shading="auto",
        )
        for label, color in zip(classes, colors):
            points = X_test[y_test == label]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color=color,
                s=12,
                alpha=0.6,
                edgecolors="white",
                linewidths=0.3,
                label=f"{label}: test",
            )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        return float(np.mean(predictions[len(grid_points) :] == y_test))

    @staticmethod
    def _draw_prototypes(ax, X_train, y_train, indices, classes) -> None:
        """Показать выбранные строки train звёздами в цветах их классов."""
        colors = plt.get_cmap("tab10")(np.arange(len(classes)))
        marker_size = 120 if len(indices) < 300 else 35
        for label, color in zip(classes, colors):
            selected = indices[y_train[indices] == label]
            points = X_train[selected]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color=color,
                marker="*",
                s=marker_size,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )
        ax.scatter([], [], marker="*", s=100, color="black", label="Эталоны train")

    @staticmethod
    def _predict_1nn(X_fit, y_fit, X_query) -> np.ndarray:
        """Ответить по ближайшему объекту; X_fit подаётся по порядку индексов."""
        distances = np.linalg.norm(X_query[:, None, :] - X_fit[None, :, :], axis=2)
        return y_fit[np.argmin(distances, axis=1)]

    @staticmethod
    def _predict_batches(predictor, X_fit, y_fit, X_query) -> np.ndarray:
        """Ограничить размер матрицы расстояний при раскраске большой сетки."""
        predictions = []
        for start in range(0, len(X_query), PREDICTION_BATCH_SIZE):
            queries = X_query[start : start + PREDICTION_BATCH_SIZE]
            batch = np.asarray(predictor(X_fit, y_fit, queries))
            if batch.shape != (len(queries),):
                raise ValueError(
                    "Классификатор должен вернуть одну метку на каждый запрос"
                )
            predictions.append(batch)
        return np.concatenate(predictions)

    @staticmethod
    def _label_axes(ax, projection: tuple[str, str]) -> None:
        ax.set_xlabel(f"{projection[0]} (стандартизован)")
        ax.set_ylabel(f"{projection[1]} (стандартизован)")

    def _output_path(self, name: str) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save_figure(self, fig, name: str, *, tight_layout: bool = True) -> Path:
        path = self._output_path(name)
        if tight_layout:
            fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path
