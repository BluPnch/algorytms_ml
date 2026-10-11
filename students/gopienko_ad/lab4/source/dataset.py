import shutil
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_NOTEBOOK_URL = "https://www.kaggle.com/code/meet3010/linear-regression-on-the-diabetes-dataset"
DATASET_PATH = Path(__file__).resolve().parent / "data" / "diabetes_regression.csv"
FEATURE_COLUMNS = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
TARGET_COLUMN = "target"

def load_dataset(
    output_path: str | Path | None = None,
    source_path: str | Path = DATASET_PATH,
) -> pd.DataFrame:

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Diabetes Regression CSV not found: {source}")

    df = pd.read_csv(source)
    expected = (*FEATURE_COLUMNS, TARGET_COLUMN)

    if tuple(df.columns) != expected:
        raise ValueError(f"Unexpected CSV columns: {df.columns.tolist()}, expected: {list(expected)}")

    if len(df) != 442:
        raise ValueError(f"Expected 442 samples, got {len(df)}")

    for column in expected:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dataset contains missing or non-finite values")

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.resolve() != source.resolve():
            shutil.copyfile(source, destination)

    return df

def analyze_dataset(df: pd.DataFrame, output_path: str | Path | None = None) -> None:

    print("\nDataset preview:\n", df.head().to_string(index=False))
    print("\nDataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum().to_string())
    print("\nDescriptive statistics:\n", df.describe().to_string())
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.describe().to_csv(path)

def train_val_test_split(
    df: pd.DataFrame,
    target_column: str = "target",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(df))
    n_test = round(len(df) * test_size)
    n_val = round(len(df) * val_size)

    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    X = df.drop(columns=[target_column]).to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)

    return (
        X[train_indices], X[val_indices], X[test_indices],
        y[train_indices], y[val_indices], y[test_indices],
    )

class StandardScaler:

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.means = X.mean(axis=0)
        stds = X.std(axis=0)
        self.stds = np.where(stds < self.eps, 1.0, stds)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
