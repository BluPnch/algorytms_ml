"""Загрузка и подготовка датасета Titanic для линейной регрессии."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "Titanic-Dataset.csv"


@dataclass
class Dataset:
    X: np.ndarray            # матрица «объекты–признаки», ℓ × n
    y: np.ndarray            # целевая переменная, ℓ
    feature_names: list[str]

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


def standardize(X: np.ndarray) -> np.ndarray:
    """Приводит каждый признак к нулевому среднему и единичной дисперсии.

    Без стандартизации главные компоненты определялись бы масштабом признаков,
    а не их связью друг с другом.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Числовые признаки пассажира.

    Age пропущен у 177 пассажиров — заполняем медианой; Embarked пропущен у двух —
    самым частым портом S. Порт кодируется двумя индикаторами (S — базовый),
    иначе сумма трёх индикаторов давала бы точную линейную зависимость.
    """
    embarked = df["Embarked"].fillna("S")
    return pd.DataFrame({
        "Pclass": df["Pclass"],
        "Sex": (df["Sex"] == "female").astype(float),
        "Age": df["Age"].fillna(df["Age"].median()),
        "SibSp": df["SibSp"],
        "Parch": df["Parch"],
        "HasCabin": df["Cabin"].notna().astype(float),
        "Embarked_C": (embarked == "C").astype(float),
        "Embarked_Q": (embarked == "Q").astype(float),
        "Survived": df["Survived"],
    })


def load_dataset(path: Path = DATA_PATH) -> Dataset:
    """Titanic: 891 пассажир; цель — log(1 + Fare), стоимость билета в лог-шкале.

    Fare сильно скошен (медиана 14, максимум 512), логарифм делает зависимость
    ближе к линейной.
    """
    df = pd.read_csv(path)
    features = build_features(df)
    X = standardize(features.to_numpy(dtype=float))
    y = np.log1p(df["Fare"].to_numpy(dtype=float))
    return Dataset(X=X, y=y, feature_names=list(features.columns))
