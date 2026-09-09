from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp


def get_rng(rng: int | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def validate_expression(X):
    if not (isinstance(X, np.ndarray) or sp.issparse(X)):
        raise TypeError("X must be a NumPy array or SciPy sparse matrix.")
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional cells-by-genes matrix.")
    return X


def validate_coords(coords: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be an n_cells x >=2 coordinate matrix.")
    if n_rows is not None and coords.shape[0] != n_rows:
        raise ValueError("coords and expression must contain the same number of rows.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contain non-finite values.")
    return coords


def standardize_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    centered = X - X.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=0, keepdims=True)
    out = np.zeros_like(centered)
    np.divide(centered, scale, out=out, where=scale > eps)
    return out


def as_column_matrix(values: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("Expected a vector or two-dimensional matrix.")
    if n_rows is not None and values.shape[0] != n_rows:
        raise ValueError("Input has the wrong number of rows.")
    return values


def row_subset_dense(X, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if sp.issparse(X):
        return X[rows][:, cols].toarray().astype(float, copy=False)
    return np.asarray(X[np.ix_(rows, cols)], dtype=float)


def add_dense_block(X, rows: np.ndarray, cols: np.ndarray, delta: np.ndarray):
    """Add a dense row/column block while preserving sparse input when possible."""
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (rows.size, cols.size):
        raise ValueError("delta has incompatible shape.")

    if sp.issparse(X):
        rr = np.repeat(rows, cols.size)
        cc = np.tile(cols, rows.size)
        values = delta.ravel()
        keep = values != 0
        update = sp.coo_matrix(
            (values[keep], (rr[keep], cc[keep])), shape=X.shape, dtype=float
        ).tocsr()
        out = X.astype(float, copy=True).tocsr() + update
        out.eliminate_zeros()
        return out

    out = np.array(X, dtype=float, copy=True)
    out[np.ix_(rows, cols)] += delta
    return out


def labels_to_strata(*labels: Iterable) -> list[np.ndarray]:
    arrays = [np.asarray(x) for x in labels]
    if not arrays:
        raise ValueError("At least one stratum label is required.")
    n = arrays[0].shape[0]
    if any(x.ndim != 1 or x.shape[0] != n for x in arrays):
        raise ValueError("All stratum labels must be one-dimensional and equally sized.")
    return arrays


def centered_frobenius_energy(X, rows: np.ndarray | None = None) -> float:
    """Squared Frobenius norm after column centering, without densifying sparse X."""
    X = validate_expression(X)
    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        X = X[rows]
    n = X.shape[0]
    if n == 0:
        return 0.0
    if sp.issparse(X):
        sumsq = float(X.multiply(X).sum())
        means = np.asarray(X.mean(axis=0)).ravel()
    else:
        arr = np.asarray(X, dtype=float)
        sumsq = float(np.square(arr).sum())
        means = arr.mean(axis=0)
    value = sumsq - n * float(np.square(means).sum())
    return max(value, 0.0)
