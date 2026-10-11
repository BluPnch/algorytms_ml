import numpy as np

class PCAfromSVD:
	def __init__(self, n_components: int):
		self.n_components = n_components
		self.mean_: np.ndarray
		self.components_: np.ndarray
		self.singular_values_: np.ndarray
		self.explained_variance_: np.ndarray
		self.explained_variance_ratio_: np.ndarray
		self.all_explained_variance_: np.ndarray
		self.all_explained_variance_ratio_: np.ndarray
		self.rank_: int

	def fit(self, X: np.ndarray) -> "PCAfromSVD":
		X = np.asarray(X, dtype=float)
		self.mean_ = X.mean(axis=0)
		X_centered = X - self.mean_
		_, s, vt = np.linalg.svd(X_centered, full_matrices=False)

		eigenvalues = s ** 2 / (len(X) - 1)
		total_variance = float(eigenvalues.sum())

		explained_ratios = eigenvalues / total_variance

		self.n_components_ = self.n_components
		self.rank_ = int(np.linalg.matrix_rank(X_centered))
		self.components_ = vt[:self.n_components]
		self.singular_values_ = s[:self.n_components]
		self.explained_variance_ = eigenvalues[:self.n_components]
		self.explained_variance_ratio_ = explained_ratios[:self.n_components]
		self.all_singular_values_ = s
		self.all_explained_variance_ = eigenvalues
		self.all_explained_variance_ratio_ = explained_ratios
		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		X = np.asarray(X, dtype=float)
		return (X - self.mean_) @ self.components_.T

	def inverse_tranform(self, Z: np.ndarray) -> np.ndarray:
		Z = np.asarray(Z, dtype=float)
		return Z @ self.components_ + self.mean_

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
			return self.fit(X).transform(X)

	def effective_dimension(self, epsilon: float = 0.05) -> int:
		remaining = np.maximum(0.0, 1.0 - np.cumsum(self.all_explained_variance_ratio_))
		indices = np.flatnonzero(remaining <= epsilon + 1e-12)
		return int(indices[0] + 1) if indices.size else len(remaining)

	def reconstruction_error_ratio(self, n_components: int) -> float:
		return float(max(0.0, 1.0 - self.all_explained_variance_ratio_[:n_components].sum()))

	