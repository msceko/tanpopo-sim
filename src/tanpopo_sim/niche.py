from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .injection import inject_program_subspace
from .truth import ComponentTruth
from .utils import as_column_matrix, get_rng, standardize_columns, validate_coords, validate_expression


def knn_weight_matrix(
    target_coords,
    neighbour_coords,
    *,
    k: int = 15,
    sigma: float | None = None,
) -> sp.csr_matrix:
    """Row-normalized Gaussian kNN weights from targets to neighbour cells."""
    target_coords = validate_coords(target_coords)
    neighbour_coords = validate_coords(neighbour_coords)
    if k <= 0:
        raise ValueError("k must be positive.")
    if neighbour_coords.shape[0] == 0:
        raise ValueError("No neighbour cells were supplied.")
    k_eff = min(k, neighbour_coords.shape[0])
    distances, indices = cKDTree(neighbour_coords).query(target_coords, k=k_eff)
    if k_eff == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    positive = distances[distances > 0]
    if sigma is None:
        sigma = float(np.median(positive)) if positive.size else 1.0
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    weights = np.exp(-0.5 * np.square(distances / sigma))
    rows = np.repeat(np.arange(target_coords.shape[0]), k_eff)
    cols = indices.ravel()
    data = weights.ravel()
    matrix = sp.coo_matrix(
        (data, (rows, cols)), shape=(target_coords.shape[0], neighbour_coords.shape[0])
    ).tocsr()
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    return sp.diags(inverse, format="csr") @ matrix


def neighbour_exposure(weights: sp.spmatrix, neighbour_scores) -> np.ndarray:
    scores = as_column_matrix(neighbour_scores)
    if weights.shape[1] != scores.shape[0]:
        raise ValueError("weights and neighbour_scores have incompatible dimensions.")
    return np.asarray(weights @ scores)


def simulate_niche_coupling(
    X,
    coords,
    cell_types,
    *,
    target_type,
    neighbour_type,
    neighbour_loadings: np.ndarray,
    target_loadings: np.ndarray,
    neighbour_spatial_scores: np.ndarray,
    neighbour_spatial_fraction=None,
    neighbour_spatial_variance_fraction=None,
    coupling=None,
    coupling_variance_fraction=None,
    neighbour_expression_variance_fraction: float | None = None,
    target_expression_variance_fraction: float | None = None,
    coupling_matrix: np.ndarray | None = None,
    k: int = 15,
    sigma: float | None = None,
    name: str = "niche",
    rng: int | np.random.Generator | None = None,
) -> tuple[object, dict[str, ComponentTruth], np.ndarray]:
    """Inject a neighbour program and target response driven by local exposure.

    Preferred strength parameters are the variance fractions:
    ``neighbour_spatial_variance_fraction`` and ``coupling_variance_fraction``.
    The legacy ``neighbour_spatial_fraction`` and ``coupling`` parameters are rho
    coefficients and remain supported for backwards compatibility.
    """
    X = validate_expression(X)
    coords = validate_coords(coords, n_rows=X.shape[0])
    cell_types = np.asarray(cell_types)
    if cell_types.ndim != 1 or cell_types.shape[0] != X.shape[0]:
        raise ValueError("cell_types must contain one value per cell.")
    generator = get_rng(rng)
    neighbour_rows = np.flatnonzero(cell_types == neighbour_type)
    target_rows = np.flatnonzero(cell_types == target_type)
    if neighbour_rows.size == 0 or target_rows.size == 0:
        raise ValueError("Both target_type and neighbour_type must be present.")

    neighbour_Z = as_column_matrix(neighbour_spatial_scores)
    if neighbour_Z.shape[0] == X.shape[0]:
        neighbour_Z = neighbour_Z[neighbour_rows]
    if neighbour_Z.shape[0] != neighbour_rows.size:
        raise ValueError("neighbour_spatial_scores must match all cells or neighbour cells.")

    neighbour_kwargs = {}
    if neighbour_spatial_variance_fraction is not None:
        neighbour_kwargs["spatial_variance_fraction"] = neighbour_spatial_variance_fraction
    else:
        neighbour_kwargs["spatial_fraction"] = (
            0.7 if neighbour_spatial_fraction is None else neighbour_spatial_fraction
        )

    result, neighbour_truth = inject_program_subspace(
        X,
        neighbour_loadings,
        neighbour_Z,
        rows=neighbour_rows,
        expression_variance_fraction=neighbour_expression_variance_fraction,
        name=f"{name}__neighbour",
        kind="niche_neighbour_program",
        rng=generator,
        **neighbour_kwargs,
    )

    W = knn_weight_matrix(coords[target_rows], coords[neighbour_rows], k=k, sigma=sigma)
    exposure = neighbour_exposure(W, neighbour_truth.scores[neighbour_rows])

    q_neighbour = exposure.shape[1]
    q_target = np.asarray(target_loadings).reshape(X.shape[1], -1).shape[1]
    if coupling_matrix is None:
        if q_neighbour != q_target:
            raise ValueError(
                "coupling_matrix is required when neighbour and target program counts differ."
            )
        coupling_matrix = np.eye(q_neighbour)
    coupling_matrix = np.asarray(coupling_matrix, dtype=float)
    if coupling_matrix.shape != (q_neighbour, q_target):
        raise ValueError("coupling_matrix must be neighbour_program x target_program.")

    target_driver = standardize_columns(exposure @ coupling_matrix)
    target_kwargs = {}
    if coupling_variance_fraction is not None:
        target_kwargs["spatial_variance_fraction"] = coupling_variance_fraction
    else:
        target_kwargs["spatial_fraction"] = 0.7 if coupling is None else coupling

    result, target_truth = inject_program_subspace(
        result,
        target_loadings,
        target_driver,
        rows=target_rows,
        expression_variance_fraction=target_expression_variance_fraction,
        name=f"{name}__target",
        kind="niche_target_response",
        rng=generator,
        **target_kwargs,
    )
    target_truth.metadata.update(
        {
            "target_type": str(target_type),
            "neighbour_type": str(neighbour_type),
            "coupling_matrix": coupling_matrix.tolist(),
            "k": int(k),
            "sigma": None if sigma is None else float(sigma),
        }
    )
    neighbour_truth.metadata.update(
        {
            "target_type": str(target_type),
            "neighbour_type": str(neighbour_type),
        }
    )
    full_exposure = np.zeros((X.shape[0], q_neighbour), dtype=float)
    full_exposure[target_rows] = exposure
    return result, {
        neighbour_truth.name: neighbour_truth,
        target_truth.name: target_truth,
    }, full_exposure
