from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds

from .metrics import orthonormal_basis
from .spatial_stats import spatial_score_summary
from .truth import ComponentTruth
from .utils import centered_frobenius_energy, get_rng, row_subset_dense, validate_coords, validate_expression


def _column_means(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(float)
    return np.asarray(X, dtype=float).mean(axis=0)


def _centered_matmul(X, vectors: np.ndarray, means: np.ndarray | None = None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if means is None:
        means = _column_means(X)
    scores = np.asarray(X @ vectors)
    return scores - (means @ vectors)[None, :]


def _centered_linear_operator(X) -> LinearOperator:
    n, g = X.shape
    means = _column_means(X)

    def matvec(v):
        return np.asarray(X @ v).ravel() - float(means @ v)

    def rmatvec(u):
        raw = np.asarray(X.T @ u).ravel()
        return raw - means * float(np.sum(u))

    def matmat(V):
        return np.asarray(X @ V) - np.ones((n, 1)) @ (means @ V)[None, :]

    def rmatmat(U):
        raw = np.asarray(X.T @ U)
        return raw - means[:, None] * np.sum(U, axis=0, keepdims=True)

    return LinearOperator(
        shape=(n, g),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=float,
    )


def pca_reference_scores(X, *, rank: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Top ordinary PCA scores and their variances without using spatial information."""
    X = validate_expression(X)
    n, g = X.shape
    max_rank = min(n, g)
    if max_rank <= 1:
        raise ValueError("PCA reference requires at least two rows and two genes.")
    rank = min(max(int(rank), 1), max_rank - 1)

    if not sp.issparse(X) and n * g <= 5_000_000:
        arr = np.asarray(X, dtype=float)
        arr = arr - arr.mean(axis=0, keepdims=True)
        U, singular, _ = np.linalg.svd(arr, full_matrices=False)
        U = U[:, :rank]
        singular = singular[:rank]
    else:
        U, singular, _ = svds(
            _centered_linear_operator(X),
            k=rank,
            which="LM",
            return_singular_vectors=True,
        )
        order = np.argsort(singular)[::-1]
        U = U[:, order]
        singular = singular[order]
    scores = U * singular[None, :]
    variance = np.var(scores, axis=0, ddof=0)
    return scores, variance


def random_projection_reference_scores(
    X,
    *,
    n_projections: int = 30,
    rng: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary non-spatial random orthogonal gene projections."""
    X = validate_expression(X)
    generator = get_rng(rng)
    q = min(max(int(n_projections), 1), X.shape[1])
    Q, _ = np.linalg.qr(generator.normal(size=(X.shape[1], q)))
    scores = _centered_matmul(X, Q)
    return scores, np.var(scores, axis=0, ddof=0)


def expression_subspace_fraction(X, loadings, *, rows=None) -> float:
    """Fraction of centered expression energy contained in a loading subspace."""
    X = validate_expression(X)
    V = orthonormal_basis(loadings)
    if V.shape[0] != X.shape[1]:
        raise ValueError("loadings must contain one row per gene.")
    if rows is None:
        rows = np.arange(X.shape[0])
    rows = np.asarray(rows, dtype=int)
    support = np.flatnonzero(np.any(np.abs(V) > 1e-14, axis=1))
    scores = row_subset_dense(X, rows, support) @ V[support]
    centered = scores - scores.mean(axis=0, keepdims=True)
    subspace_energy = float(np.square(centered).sum())
    total_energy = centered_frobenius_energy(X, rows)
    return subspace_energy / total_energy if total_energy > 0 else 0.0


def _percentiles(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        return np.full(values.shape, np.nan)
    return np.asarray([np.mean(reference <= value) for value in values], dtype=float)


def _to_lists(mapping: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value, dtype=float).tolist() for key, value in mapping.items()}


def characterize_injection(
    X_baseline,
    X_simulated,
    coords,
    truth: ComponentTruth,
    *,
    k: int = 15,
    reference_rank: int = 30,
    random_projections: int = 30,
    max_reference_cells: int | None = 20_000,
    rng: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    """Quantify an injection without using Tanpopo's spatial operator.

    The true program is characterized by expression variance and generic kNN
    Moran's I, Geary's C and neighbour-predictive R². Its strength is additionally
    expressed as a percentile of ordinary PCA components and random gene
    projections in the pre-injection baseline.
    """
    X_baseline = validate_expression(X_baseline)
    X_simulated = validate_expression(X_simulated)
    if X_baseline.shape != X_simulated.shape:
        raise ValueError("Baseline and simulated matrices must have the same shape.")
    coords = validate_coords(coords, n_rows=X_baseline.shape[0])
    if truth.loadings.shape[0] != X_baseline.shape[1]:
        raise ValueError("Truth loadings and expression matrix have incompatible genes.")

    active_rows = np.flatnonzero(np.asarray(truth.active_rows, dtype=bool))
    if active_rows.size < 3:
        raise ValueError("At least three active rows are required for characterization.")

    generator = get_rng(rng)
    if max_reference_cells is not None and active_rows.size > max_reference_cells:
        reference_rows = np.sort(
            generator.choice(active_rows, size=int(max_reference_cells), replace=False)
        )
    else:
        reference_rows = active_rows

    V = orthonormal_basis(truth.loadings)
    support = np.flatnonzero(np.any(np.abs(V) > 1e-14, axis=1))
    baseline_scores = row_subset_dense(X_baseline, reference_rows, support) @ V[support]
    simulated_scores = row_subset_dense(X_simulated, reference_rows, support) @ V[support]
    driver_scores = np.asarray(truth.spatial_scores)[reference_rows]
    if driver_scores.ndim == 1:
        driver_scores = driver_scores[:, None]

    ref_coords = coords[reference_rows]
    baseline_spatial = spatial_score_summary(baseline_scores, ref_coords, k=k)
    simulated_spatial = spatial_score_summary(simulated_scores, ref_coords, k=k)
    driver_spatial = spatial_score_summary(driver_scores, ref_coords, k=k)

    X_ref = X_baseline[reference_rows]
    pca_scores, pca_variance = pca_reference_scores(X_ref, rank=reference_rank)
    pca_spatial = spatial_score_summary(pca_scores, ref_coords, k=k)
    random_scores, random_variance = random_projection_reference_scores(
        X_ref, n_projections=random_projections, rng=generator
    )
    random_spatial = spatial_score_summary(random_scores, ref_coords, k=k)

    baseline_variance = np.var(baseline_scores, axis=0, ddof=0)
    simulated_variance = np.var(simulated_scores, axis=0, ddof=0)
    delta = simulated_scores - baseline_scores

    pca_percentiles = {
        "expression_variance_before": _percentiles(baseline_variance, pca_variance),
        "expression_variance_after": _percentiles(simulated_variance, pca_variance),
    }
    random_percentiles = {
        "expression_variance_before": _percentiles(baseline_variance, random_variance),
        "expression_variance_after": _percentiles(simulated_variance, random_variance),
    }
    for metric in ("morans_i", "knn_predictive_r2"):
        pca_percentiles[f"{metric}_before"] = _percentiles(
            baseline_spatial[metric], pca_spatial[metric]
        )
        pca_percentiles[f"{metric}_after"] = _percentiles(
            simulated_spatial[metric], pca_spatial[metric]
        )
        random_percentiles[f"{metric}_before"] = _percentiles(
            baseline_spatial[metric], random_spatial[metric]
        )
        random_percentiles[f"{metric}_after"] = _percentiles(
            simulated_spatial[metric], random_spatial[metric]
        )

    baseline_fraction = expression_subspace_fraction(
        X_baseline, V, rows=active_rows
    )
    simulated_fraction = expression_subspace_fraction(
        X_simulated, V, rows=active_rows
    )

    result = {
        "component": truth.name,
        "kind": truth.kind,
        "n_active_cells": int(active_rows.size),
        "n_reference_cells": int(reference_rows.size),
        "n_programs": int(V.shape[1]),
        "expression": {
            "subspace_variance_fraction_before": float(baseline_fraction),
            "subspace_variance_fraction_after": float(simulated_fraction),
            "program_variance_before": baseline_variance.tolist(),
            "program_variance_after": simulated_variance.tolist(),
            "score_change_variance": np.var(delta, axis=0, ddof=0).tolist(),
        },
        "spatial_before": _to_lists(baseline_spatial),
        "spatial_after": _to_lists(simulated_spatial),
        "spatial_driver": _to_lists(driver_spatial),
        "reference_pca": {
            "rank": int(pca_scores.shape[1]),
            "expression_variance": pca_variance.tolist(),
            **_to_lists(pca_spatial),
        },
        "reference_random_projections": {
            "count": int(random_scores.shape[1]),
            "expression_variance": random_variance.tolist(),
            **_to_lists(random_spatial),
        },
        "percentile_vs_pca": _to_lists(pca_percentiles),
        "percentile_vs_random_projections": _to_lists(random_percentiles),
        "truth_metadata": dict(truth.metadata),
    }
    return result
