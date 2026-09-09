from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .loadings import orthonormalize_loadings
from .truth import ComponentTruth
from .utils import (
    add_dense_block,
    as_column_matrix,
    centered_frobenius_energy,
    get_rng,
    row_subset_dense,
    standardize_columns,
    validate_expression,
)


def _prepare_rows(n_rows: int, rows=None) -> np.ndarray:
    if rows is None:
        return np.arange(n_rows, dtype=int)
    rows = np.asarray(rows)
    if rows.dtype == bool:
        if rows.shape != (n_rows,):
            raise ValueError("Boolean row mask has the wrong shape.")
        rows = np.flatnonzero(rows)
    rows = rows.astype(int, copy=False)
    if rows.ndim != 1 or np.any((rows < 0) | (rows >= n_rows)):
        raise ValueError("rows contains invalid indices.")
    return rows


def _vector_parameter(value, n_programs: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.repeat(float(array), n_programs)
    if array.shape != (n_programs,):
        raise ValueError(f"{name} must be scalar or contain one value per program.")
    return array


def _resolve_rho(
    n_programs: int,
    *,
    spatial_fraction=None,
    spatial_variance_fraction=None,
) -> tuple[np.ndarray, str]:
    if spatial_fraction is not None and spatial_variance_fraction is not None:
        raise ValueError(
            "Use only one of spatial_fraction (legacy rho) or spatial_variance_fraction."
        )
    if spatial_variance_fraction is not None:
        fraction = _vector_parameter(
            spatial_variance_fraction, n_programs, "spatial_variance_fraction"
        )
        if np.any((fraction < 0) | (fraction > 1)):
            raise ValueError("spatial_variance_fraction must lie in [0, 1].")
        return np.sqrt(fraction), "spatial_variance_fraction"
    if spatial_fraction is None:
        spatial_fraction = 0.5
    rho = _vector_parameter(spatial_fraction, n_programs, "spatial_fraction")
    if np.any((rho < 0) | (rho > 1)):
        raise ValueError("spatial_fraction must lie in [0, 1].")
    return rho, "spatial_fraction_legacy_rho"


def _orthogonalize_spatial_against_baseline(T: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if T.shape[0] <= T.shape[1]:
        return standardize_columns(Z)
    coefficients, *_ = np.linalg.lstsq(T, Z, rcond=None)
    residual = Z - T @ coefficients
    bad = residual.std(axis=0) < 1e-10
    if np.any(bad):
        residual[:, bad] = Z[:, bad]
    return standardize_columns(residual)


def _target_program_sd(
    *,
    n_rows: int,
    baseline_score_energy: np.ndarray,
    perpendicular_energy: float,
    program_sd,
    expression_variance_fraction,
    program_variance_multiplier,
    program_weights,
) -> tuple[np.ndarray, str]:
    requested = [
        program_sd is not None,
        expression_variance_fraction is not None,
        program_variance_multiplier is not None,
    ]
    if sum(requested) > 1:
        raise ValueError(
            "Use at most one of program_sd, expression_variance_fraction, or "
            "program_variance_multiplier."
        )
    n_programs = baseline_score_energy.size

    if expression_variance_fraction is not None:
        fraction = float(expression_variance_fraction)
        if not 0 < fraction < 1:
            raise ValueError("expression_variance_fraction must lie strictly in (0, 1).")
        target_total = fraction / (1.0 - fraction) * perpendicular_energy
        if program_weights is None:
            weights = baseline_score_energy.copy()
            if float(weights.sum()) <= 1e-12:
                weights = np.ones(n_programs, dtype=float)
        else:
            weights = _vector_parameter(program_weights, n_programs, "program_weights")
            if np.any(weights < 0) or float(weights.sum()) <= 0:
                raise ValueError("program_weights must be non-negative with positive sum.")
        weights = weights / weights.sum()
        target_energy = target_total * weights
        return np.sqrt(target_energy / max(n_rows, 1)), "expression_variance_fraction"

    if program_variance_multiplier is not None:
        multiplier = _vector_parameter(
            program_variance_multiplier, n_programs, "program_variance_multiplier"
        )
        if np.any(multiplier <= 0):
            raise ValueError("program_variance_multiplier must be positive.")
        fallback = np.where(baseline_score_energy > 1e-12, baseline_score_energy, n_rows)
        target_energy = fallback * multiplier
        return np.sqrt(target_energy / max(n_rows, 1)), "program_variance_multiplier"

    if program_sd is not None:
        sd = _vector_parameter(program_sd, n_programs, "program_sd")
        if np.any(sd <= 0):
            raise ValueError("program_sd must be positive.")
        return sd, "program_sd"

    fallback = np.sqrt(np.maximum(baseline_score_energy / max(n_rows, 1), 0.0))
    fallback[fallback < 1e-12] = 1.0
    return fallback, "preserve_baseline_subspace_variance"


def inject_program_subspace(
    X,
    loadings: np.ndarray,
    spatial_scores: np.ndarray,
    spatial_fraction=None,
    *,
    spatial_variance_fraction=None,
    rows=None,
    program_sd=None,
    expression_variance_fraction: float | None = None,
    program_variance_multiplier=None,
    program_weights=None,
    orthogonalize_spatial: bool = True,
    name: str = "program",
    kind: str = "spatial_program",
    rng: int | np.random.Generator | None = None,
) -> tuple[object, ComponentTruth]:
    """Replace variation inside a known gene subspace with controlled spatial signal.

    Let V be the orthonormal loading matrix and T the standardized baseline score.
    The replacement score is

        sd_k [sqrt(1-rho_k^2) T_k + rho_k Z_k].

    The preferred spatial parameter is ``spatial_variance_fraction=rho^2``.
    ``spatial_fraction`` remains as a backwards-compatible alias for rho.

    Expression amplitude can be set independently of spatial coherence. In
    particular, ``expression_variance_fraction=f`` chooses the program SD so the
    full injected loading subspace contributes fraction f of centered expression
    energy among active rows. This calibration contains no Tanpopo spatial operator.
    """
    X = validate_expression(X)
    generator = get_rng(rng)
    rows = _prepare_rows(X.shape[0], rows)
    V = orthonormalize_loadings(loadings)
    if V.shape[0] != X.shape[1]:
        raise ValueError("loadings must have one row per gene in X.")
    n_programs = V.shape[1]
    rho, spatial_parameterization = _resolve_rho(
        n_programs,
        spatial_fraction=spatial_fraction,
        spatial_variance_fraction=spatial_variance_fraction,
    )

    Z = as_column_matrix(spatial_scores)
    if Z.shape[0] == X.shape[0]:
        Z = Z[rows]
    elif Z.shape[0] != rows.size:
        raise ValueError("spatial_scores must have one row per cell or active row.")
    if Z.shape[1] != n_programs:
        raise ValueError("spatial_scores and loadings must contain the same number of programs.")

    support = np.flatnonzero(np.any(np.abs(V) > 1e-14, axis=1))
    V_support = V[support]
    X_support = row_subset_dense(X, rows, support)
    original_scores = X_support @ V_support
    baseline_mean = original_scores.mean(axis=0)
    centered_original_scores = original_scores - baseline_mean[None, :]
    baseline_score_energy = np.sum(np.square(centered_original_scores), axis=0)

    total_energy_before = centered_frobenius_energy(X, rows)
    subspace_energy_before = float(baseline_score_energy.sum())
    perpendicular_energy = max(total_energy_before - subspace_energy_before, 0.0)

    T = standardize_columns(original_scores)
    flat = np.sqrt(baseline_score_energy / max(rows.size, 1)) < 1e-12
    if np.any(flat):
        T[:, flat] = standardize_columns(
            generator.normal(size=(rows.size, int(flat.sum())))
        )

    Z = standardize_columns(Z)
    if orthogonalize_spatial:
        Z = _orthogonalize_spatial_against_baseline(T, Z)

    sd, amplitude_parameterization = _target_program_sd(
        n_rows=rows.size,
        baseline_score_energy=baseline_score_energy,
        perpendicular_energy=perpendicular_energy,
        program_sd=program_sd,
        expression_variance_fraction=expression_variance_fraction,
        program_variance_multiplier=program_variance_multiplier,
        program_weights=program_weights,
    )

    new_scores = baseline_mean[None, :] + sd * (
        np.sqrt(1.0 - rho**2)[None, :] * T + rho[None, :] * Z
    )
    delta = (new_scores - original_scores) @ V_support.T
    X_out = add_dense_block(X, rows, support, delta)

    centered_new = new_scores - new_scores.mean(axis=0, keepdims=True)
    subspace_energy_after = float(np.square(centered_new).sum())
    total_energy_after = perpendicular_energy + subspace_energy_after
    fraction_before = (
        subspace_energy_before / total_energy_before if total_energy_before > 0 else 0.0
    )
    fraction_after = (
        subspace_energy_after / total_energy_after if total_energy_after > 0 else 0.0
    )
    delta_energy = float(np.square(delta).sum())

    full_scores = np.zeros((X.shape[0], n_programs), dtype=float)
    full_spatial = np.zeros_like(full_scores)
    full_scores[rows] = new_scores
    full_spatial[rows] = Z
    active = np.zeros(X.shape[0], dtype=bool)
    active[rows] = True

    truth = ComponentTruth(
        name=name,
        kind=kind,
        loadings=V,
        scores=full_scores,
        spatial_scores=full_spatial,
        active_rows=active,
        metadata={
            "rho": rho.tolist(),
            "spatial_variance_fraction": np.square(rho).tolist(),
            "spatial_parameterization": spatial_parameterization,
            "program_sd": sd.tolist(),
            "amplitude_parameterization": amplitude_parameterization,
            "requested_expression_variance_fraction": expression_variance_fraction,
            "expression_variance_fraction_before": float(fraction_before),
            "expression_variance_fraction_after": float(fraction_after),
            "total_centered_expression_energy_before": float(total_energy_before),
            "subspace_centered_energy_before": float(subspace_energy_before),
            "subspace_centered_energy_after": float(subspace_energy_after),
            "perpendicular_centered_energy": float(perpendicular_energy),
            "injected_delta_energy": delta_energy,
            "injected_delta_fraction_of_baseline_energy": (
                delta_energy / total_energy_before if total_energy_before > 0 else 0.0
            ),
            "orthogonalize_spatial": bool(orthogonalize_spatial),
        },
    )
    return X_out, truth


def inject_grouped_program_subspace(
    X,
    loadings: np.ndarray,
    spatial_scores: np.ndarray,
    group_labels,
    spatial_fraction_by_group: Mapping | None = None,
    *,
    spatial_variance_fraction_by_group: Mapping | None = None,
    rows=None,
    program_sd=None,
    expression_variance_fraction: float | None = None,
    program_variance_multiplier=None,
    orthogonalize_spatial: bool = True,
    name: str = "differential_program",
    rng: int | np.random.Generator | None = None,
) -> tuple[object, ComponentTruth]:
    """Inject the same gene programs with group-specific spatial strengths."""
    if spatial_fraction_by_group is not None and spatial_variance_fraction_by_group is not None:
        raise ValueError("Specify only one grouped spatial-strength parameterization.")
    grouped = (
        spatial_variance_fraction_by_group
        if spatial_variance_fraction_by_group is not None
        else spatial_fraction_by_group
    )
    if grouped is None:
        raise ValueError("A group-specific spatial strength mapping is required.")

    X = validate_expression(X)
    generator = get_rng(rng)
    rows_all = _prepare_rows(X.shape[0], rows)
    groups = np.asarray(group_labels)
    if groups.ndim != 1 or groups.shape[0] != X.shape[0]:
        raise ValueError("group_labels must contain one value per row in X.")
    Z = as_column_matrix(spatial_scores, n_rows=X.shape[0])
    V = orthonormalize_loadings(loadings)
    result = X
    combined_scores = np.zeros((X.shape[0], V.shape[1]))
    combined_spatial = np.zeros_like(combined_scores)
    active = np.zeros(X.shape[0], dtype=bool)
    per_group_metadata = {}

    for group, strength in grouped.items():
        group_rows = rows_all[groups[rows_all] == group]
        if group_rows.size == 0:
            continue
        kwargs = (
            {"spatial_variance_fraction": strength}
            if spatial_variance_fraction_by_group is not None
            else {"spatial_fraction": strength}
        )
        result, component = inject_program_subspace(
            result,
            V,
            Z,
            rows=group_rows,
            program_sd=program_sd,
            expression_variance_fraction=expression_variance_fraction,
            program_variance_multiplier=program_variance_multiplier,
            orthogonalize_spatial=orthogonalize_spatial,
            name=name,
            kind="differential_spatial_program",
            rng=generator,
            **kwargs,
        )
        combined_scores[group_rows] = component.scores[group_rows]
        combined_spatial[group_rows] = component.spatial_scores[group_rows]
        active[group_rows] = True
        per_group_metadata[str(group)] = component.metadata

    truth = ComponentTruth(
        name=name,
        kind="differential_spatial_program",
        loadings=V,
        scores=combined_scores,
        spatial_scores=combined_spatial,
        active_rows=active,
        metadata={
            "spatial_fraction_by_group": (
                None
                if spatial_fraction_by_group is None
                else {str(k): np.asarray(v, dtype=float).tolist() for k, v in grouped.items()}
            ),
            "spatial_variance_fraction_by_group": (
                None
                if spatial_variance_fraction_by_group is None
                else {str(k): np.asarray(v, dtype=float).tolist() for k, v in grouped.items()}
            ),
            "group_diagnostics": per_group_metadata,
        },
    )
    return result, truth
