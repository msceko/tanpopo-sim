from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


def orthonormal_basis(loadings: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    V = np.asarray(loadings, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    if V.ndim != 2:
        raise ValueError("loadings must be a matrix.")
    U, singular, _ = np.linalg.svd(V, full_matrices=False)
    if not singular.size or singular[0] == 0:
        return U[:, :0]
    rank = int(np.sum(singular > tol * singular[0]))
    return U[:, :rank]


def principal_angles(true_loadings: np.ndarray, estimated_loadings: np.ndarray) -> np.ndarray:
    Q_true = orthonormal_basis(true_loadings)
    Q_est = orthonormal_basis(estimated_loadings)
    if Q_true.shape[0] != Q_est.shape[0]:
        raise ValueError("True and estimated loadings must contain the same genes.")
    singular = np.linalg.svd(Q_true.T @ Q_est, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return np.arccos(singular)


def subspace_overlap(true_loadings: np.ndarray, estimated_loadings: np.ndarray) -> float:
    """Fraction of the true subspace captured by the estimated subspace.

    This is the mean squared cosine of the true principal directions. If the
    estimated rank is smaller than the true rank, the maximum possible value is
    correspondingly smaller. It is invariant to rotations within either subspace.
    """
    Q_true = orthonormal_basis(true_loadings)
    Q_est = orthonormal_basis(estimated_loadings)
    if Q_true.shape[0] != Q_est.shape[0]:
        raise ValueError("True and estimated loadings must contain the same genes.")
    if Q_true.shape[1] == 0 or Q_est.shape[1] == 0:
        return 0.0
    numerator = np.linalg.norm(Q_true.T @ Q_est, ord="fro") ** 2
    value = float(numerator / Q_true.shape[1])
    return float(np.clip(value, 0.0, 1.0))


def projector_distance(true_loadings: np.ndarray, estimated_loadings: np.ndarray) -> float:
    Q_true = orthonormal_basis(true_loadings)
    Q_est = orthonormal_basis(estimated_loadings)
    P_true = Q_true @ Q_true.T
    P_est = Q_est @ Q_est.T
    return float(np.linalg.norm(P_true - P_est, ord="fro"))


def matched_loading_correlations(
    true_loadings: np.ndarray,
    estimated_loadings: np.ndarray,
) -> np.ndarray:
    V = np.asarray(true_loadings, dtype=float)
    W = np.asarray(estimated_loadings, dtype=float)
    if V.shape[0] != W.shape[0]:
        raise ValueError("True and estimated loadings must contain the same genes.")
    V = V - V.mean(axis=0, keepdims=True)
    W = W - W.mean(axis=0, keepdims=True)
    V /= np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-12)
    W /= np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    corr = V.T @ W
    true_idx, est_idx = linear_sum_assignment(-np.abs(corr))
    return corr[true_idx, est_idx]


def recovery_curve(
    true_loadings: np.ndarray,
    estimated_loadings: np.ndarray,
    *,
    max_rank: int | None = None,
) -> np.ndarray:
    """Cumulative true-subspace recovery using the first 1..M estimated modes.

    The result has two columns: estimated rank and subspace overlap. This avoids
    assuming that k injected programs must correspond to the first k estimated
    modes.
    """
    estimated = np.asarray(estimated_loadings, dtype=float)
    if estimated.ndim == 1:
        estimated = estimated[:, None]
    if estimated.ndim != 2:
        raise ValueError("estimated_loadings must be a matrix.")
    if np.asarray(true_loadings).shape[0] != estimated.shape[0]:
        raise ValueError("True and estimated loadings must contain the same genes.")
    M = estimated.shape[1] if max_rank is None else min(int(max_rank), estimated.shape[1])
    if M <= 0:
        return np.empty((0, 2), dtype=float)
    values = np.empty((M, 2), dtype=float)
    for m in range(1, M + 1):
        values[m - 1] = (m, subspace_overlap(true_loadings, estimated[:, :m]))
    return values


def recovery_rank(
    true_loadings: np.ndarray,
    estimated_loadings: np.ndarray,
    *,
    threshold: float = 0.9,
    max_rank: int | None = None,
) -> int | None:
    """Smallest estimated rank reaching a chosen cumulative recovery threshold."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie in (0, 1].")
    curve = recovery_curve(true_loadings, estimated_loadings, max_rank=max_rank)
    hits = curve[curve[:, 1] >= threshold]
    return None if hits.size == 0 else int(hits[0, 0])


def area_under_recovery_curve(
    true_loadings: np.ndarray,
    estimated_loadings: np.ndarray,
    *,
    max_rank: int | None = None,
) -> float:
    """Mean cumulative subspace recovery over ranks 1..M, in [0, 1]."""
    curve = recovery_curve(true_loadings, estimated_loadings, max_rank=max_rank)
    return 0.0 if curve.size == 0 else float(np.mean(curve[:, 1]))


def best_mode_matches(
    true_loadings: np.ndarray,
    estimated_loadings: np.ndarray,
) -> dict[str, np.ndarray]:
    """Best individual estimated mode for each true direction.

    This is secondary to subspace recovery because rotations among tied modes can
    make individual matches arbitrary.
    """
    true = orthonormal_basis(true_loadings)
    estimated = np.asarray(estimated_loadings, dtype=float)
    if estimated.ndim == 1:
        estimated = estimated[:, None]
    estimated = estimated / np.maximum(
        np.linalg.norm(estimated, axis=0, keepdims=True), 1e-12
    )
    similarity = np.abs(true.T @ estimated)
    best = np.argmax(similarity, axis=1)
    return {
        "estimated_rank": best.astype(int) + 1,
        "absolute_cosine": similarity[np.arange(true.shape[1]), best],
    }


def source_leakage_matrix(
    true_sources: Mapping[str, np.ndarray],
    estimated_sources: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Projection overlap of every true source with every estimated workflow."""
    true_names = list(true_sources)
    estimated_names = list(estimated_sources)
    output = np.zeros((len(true_names), len(estimated_names)), dtype=float)
    for i, true_name in enumerate(true_names):
        for j, estimated_name in enumerate(estimated_names):
            output[i, j] = subspace_overlap(
                true_sources[true_name], estimated_sources[estimated_name]
            )
    return output, true_names, estimated_names
