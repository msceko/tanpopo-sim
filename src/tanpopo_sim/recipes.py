from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from .fields import generate_fields, generate_samplewise_fields
from .injection import inject_grouped_program_subspace, inject_program_subspace
from .loadings import loadings_from_supports, sparse_loadings
from .niche import simulate_niche_coupling
from .permutation import stratified_permute_expression
from .truth import SimulationResult
from .utils import get_rng, validate_coords, validate_expression


def _indices_from_gene_names(gene_names, names) -> np.ndarray:
    lookup = {str(name): i for i, name in enumerate(gene_names)}
    missing = [str(name) for name in names if str(name) not in lookup]
    if missing:
        raise KeyError(f"Genes not present in gene_names: {missing[:10]}")
    return np.asarray([lookup[str(name)] for name in names], dtype=int)


def _make_loadings(spec, n_genes, gene_names, n_programs, rng):
    spec = {} if spec is None else dict(spec)
    supports = spec.get("supports")
    if supports is not None:
        parsed = []
        for support in supports:
            if support and isinstance(support[0], str):
                if gene_names is None:
                    raise ValueError("gene_names are required for named supports.")
                parsed.append(_indices_from_gene_names(gene_names, support))
            else:
                parsed.append(np.asarray(support, dtype=int))
        if len(parsed) != n_programs:
            raise ValueError("Number of supports must equal n_programs.")
        return loadings_from_supports(
            n_genes,
            parsed,
            positive=bool(spec.get("positive", False)),
            rng=rng,
        )

    gene_pool = spec.get("gene_pool")
    if gene_pool is not None and gene_pool and isinstance(gene_pool[0], str):
        if gene_names is None:
            raise ValueError("gene_names are required for a named gene_pool.")
        gene_pool = _indices_from_gene_names(gene_names, gene_pool)
    return sparse_loadings(
        n_genes,
        n_programs=n_programs,
        genes_per_program=int(spec.get("genes_per_program", 50)),
        gene_pool=gene_pool,
        disjoint=bool(spec.get("disjoint", True)),
        positive=bool(spec.get("positive", False)),
        rng=rng,
    )


def _field_matrix(coords, rows, spec, n_programs, obs, rng):
    spec = deepcopy(spec or {"kind": "rff"})
    kind = spec.pop("kind", "rff")
    sample_key = spec.pop("sample_key", None)
    if sample_key is not None:
        labels = np.asarray(obs[sample_key])[rows]
        return generate_samplewise_fields(
            coords[rows], labels, n_programs, kind, rng=rng, **spec
        )
    return generate_fields(coords[rows], n_programs, kind, rng=rng, **spec)


def _amplitude_kwargs(spec: dict) -> dict:
    output = {}
    for key in (
        "program_sd",
        "expression_variance_fraction",
        "program_variance_multiplier",
        "program_weights",
        "orthogonalize_spatial",
    ):
        if key in spec:
            output[key] = spec.pop(key)
    return output


def _spatial_kwargs(spec: dict) -> dict:
    if "spatial_variance_fraction" in spec:
        return {"spatial_variance_fraction": spec.pop("spatial_variance_fraction")}
    if "spatial_fraction" in spec:
        return {"spatial_fraction": spec.pop("spatial_fraction")}
    return {"spatial_variance_fraction": 0.25}


def run_recipe(
    X,
    coords,
    obs: Mapping[str, np.ndarray],
    recipe: Mapping,
    *,
    gene_names=None,
) -> SimulationResult:
    """Run a reproducible simulation recipe on arrays, without requiring AnnData."""
    X = validate_expression(X)
    coords = validate_coords(coords, n_rows=X.shape[0])
    obs = {key: np.asarray(value) for key, value in obs.items()}
    for key, values in obs.items():
        if values.ndim != 1 or values.shape[0] != X.shape[0]:
            raise ValueError(f"obs[{key!r}] must contain one value per cell.")
    if gene_names is not None and len(gene_names) != X.shape[1]:
        raise ValueError("gene_names must contain one name per gene.")

    seed = recipe.get("seed", None)
    rng = get_rng(seed)
    result = SimulationResult(X=X.copy(), metadata={"seed": seed, "recipe": dict(recipe)})

    baseline = recipe.get("baseline")
    if baseline:
        within = list(baseline.get("permute_within", []))
        if within:
            missing = [key for key in within if key not in obs]
            if missing:
                raise KeyError(f"Missing permutation strata columns: {missing}")
            result.X, result.permutation_indices = stratified_permute_expression(
                result.X,
                *(obs[key] for key in within),
                rng=rng,
                return_indices=True,
            )
    result.baseline_X = result.X.copy()

    for component_spec in recipe.get("components", []):
        spec = deepcopy(component_spec)
        name = str(spec.pop("name"))
        kind = str(spec.pop("kind", "global")).lower().replace("-", "_")
        n_programs = int(spec.pop("n_programs", 1))

        if kind in {"global", "within_type", "differential"}:
            rows = np.arange(X.shape[0])
            if kind == "within_type":
                cell_type_key = spec.pop("cell_type_key")
                target = spec.pop("target")
                rows = np.flatnonzero(obs[cell_type_key] == target)
                if rows.size == 0:
                    raise ValueError(f"No rows found for {cell_type_key}={target!r}.")

            V = _make_loadings(
                spec.pop("loadings", None), X.shape[1], gene_names, n_programs, rng
            )
            Z_rows = _field_matrix(coords, rows, spec.pop("field", None), n_programs, obs, rng)
            Z = np.zeros((X.shape[0], n_programs), dtype=float)
            Z[rows] = Z_rows
            amplitude_kwargs = _amplitude_kwargs(spec)

            if kind == "differential":
                condition_key = spec.pop("condition_key")
                if "spatial_variance_fraction_by_condition" in spec:
                    grouped_kwargs = {
                        "spatial_variance_fraction_by_group": spec.pop(
                            "spatial_variance_fraction_by_condition"
                        )
                    }
                elif "spatial_fraction_by_condition" in spec:
                    grouped_kwargs = {
                        "spatial_fraction_by_group": spec.pop("spatial_fraction_by_condition")
                    }
                else:
                    raise ValueError(
                        "Differential components require spatial_variance_fraction_by_condition."
                    )
                result.X, truth = inject_grouped_program_subspace(
                    result.X,
                    V,
                    Z,
                    obs[condition_key],
                    rows=rows,
                    name=name,
                    rng=rng,
                    **grouped_kwargs,
                    **amplitude_kwargs,
                )
            else:
                result.X, truth = inject_program_subspace(
                    result.X,
                    V,
                    Z,
                    rows=rows,
                    name=name,
                    kind=(
                        "within_type_program" if kind == "within_type" else "spatial_program"
                    ),
                    rng=rng,
                    **_spatial_kwargs(spec),
                    **amplitude_kwargs,
                )
            truth.metadata.update(spec)
            result.add_component(truth)
            continue

        if kind == "niche":
            cell_type_key = spec.pop("cell_type_key")
            target = spec.pop("target")
            neighbour = spec.pop("neighbour")
            neighbour_rows = np.flatnonzero(obs[cell_type_key] == neighbour)
            if neighbour_rows.size == 0:
                raise ValueError(f"No neighbour cells found for {neighbour!r}.")
            neighbour_programs = int(spec.pop("neighbour_programs", n_programs))
            target_programs = int(spec.pop("target_programs", n_programs))
            neighbour_V = _make_loadings(
                spec.pop("neighbour_loadings", None),
                X.shape[1],
                gene_names,
                neighbour_programs,
                rng,
            )
            target_V = _make_loadings(
                spec.pop("target_loadings", None),
                X.shape[1],
                gene_names,
                target_programs,
                rng,
            )
            neighbour_Z = _field_matrix(
                coords,
                neighbour_rows,
                spec.pop("neighbour_field", None),
                neighbour_programs,
                obs,
                rng,
            )

            niche_kwargs = {
                "neighbour_expression_variance_fraction": spec.pop(
                    "neighbour_expression_variance_fraction", None
                ),
                "target_expression_variance_fraction": spec.pop(
                    "target_expression_variance_fraction", None
                ),
                "coupling_matrix": spec.pop("coupling_matrix", None),
                "k": int(spec.pop("k", 15)),
                "sigma": spec.pop("sigma", None),
            }
            if "neighbour_spatial_variance_fraction" in spec:
                niche_kwargs["neighbour_spatial_variance_fraction"] = spec.pop(
                    "neighbour_spatial_variance_fraction"
                )
            else:
                niche_kwargs["neighbour_spatial_fraction"] = spec.pop(
                    "neighbour_spatial_fraction", 0.7
                )
            if "coupling_variance_fraction" in spec:
                niche_kwargs["coupling_variance_fraction"] = spec.pop(
                    "coupling_variance_fraction"
                )
            else:
                niche_kwargs["coupling"] = spec.pop("coupling", 0.7)

            result.X, truths, _ = simulate_niche_coupling(
                result.X,
                coords,
                obs[cell_type_key],
                target_type=target,
                neighbour_type=neighbour,
                neighbour_loadings=neighbour_V,
                target_loadings=target_V,
                neighbour_spatial_scores=neighbour_Z,
                name=name,
                rng=rng,
                **niche_kwargs,
            )
            for truth in truths.values():
                truth.metadata.update(spec)
                result.add_component(truth)
            continue

        raise ValueError(f"Unknown recipe component kind: {kind!r}")

    return result
