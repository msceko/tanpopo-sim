from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .anndata_io import component_truth_from_adata, read_h5ad, simulate_adata
from .characterization import characterize_injection
from .metrics import (
    area_under_recovery_curve,
    best_mode_matches,
    orthonormal_basis,
    principal_angles,
    projector_distance,
    recovery_curve,
    recovery_rank,
    subspace_overlap,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Semi-synthetic spatial transcriptomics simulation using real AnnData geometry.",
)


def _load_recipe(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _optional_layer(value: str | None) -> str | None:
    if value is None:
        return None
    return None if value.lower() in {"none", "null", "off", ""} else value


def _matrix_from_adata(adata, layer: str | None):
    if layer is None or layer == "X":
        return adata.X
    if layer not in adata.layers:
        raise typer.BadParameter(f"Layer {layer!r} was not found in adata.layers.")
    return adata.layers[layer]


def _simulate_and_write(
    input_path: Path,
    output_path: Path,
    recipe: dict,
    layer: str | None,
    output_layer: str | None,
    baseline_output_layer: str | None,
    spatial_key: str,
    prefix: str,
):
    adata = read_h5ad(input_path)
    output, result = simulate_adata(
        adata,
        recipe,
        layer=layer,
        output_layer=output_layer,
        baseline_output_layer=baseline_output_layer,
        spatial_key=spatial_key,
        prefix=prefix,
    )
    output.write_h5ad(output_path)
    typer.echo(
        f"Wrote {output_path} with {len(result.components)} truth component(s) "
        f"in {'X' if output_layer in (None, 'X') else f'layers[{output_layer!r}]'}."
    )


@app.command("simulate")
def simulate_command(
    input_path: Annotated[Path, typer.Argument(help="Input .h5ad")],
    output_path: Annotated[Path, typer.Argument(help="Output .h5ad")],
    recipe_path: Annotated[Path, typer.Option("--recipe", "-r", help="JSON recipe")],
    layer: Annotated[str | None, typer.Option(help="Input expression layer; default X")] = None,
    output_layer: Annotated[
        str, typer.Option(help="Layer receiving simulated expression; use X to replace adata.X")
    ] = "tanpopo_sim",
    baseline_output_layer: Annotated[
        str,
        typer.Option(
            help="Layer receiving the actual pre-injection baseline; use 'none' to disable"
        ),
    ] = "tanpopo_sim_baseline",
    spatial_key: Annotated[str, typer.Option(help="adata.obsm coordinate key")] = "spatial",
    prefix: Annotated[str, typer.Option(help="Prefix for truth keys")] = "tanpopo_sim",
    seed: Annotated[int | None, typer.Option(help="RNG seed, overrides recipe")] = None,
):
    """Run a multi-component JSON recipe on an AnnData object."""
    recipe = _load_recipe(recipe_path)
    if seed is not None:
        recipe["seed"] = seed
    _simulate_and_write(
        input_path,
        output_path,
        recipe,
        layer,
        output_layer,
        _optional_layer(baseline_output_layer),
        spatial_key,
        prefix,
    )


@app.command("permute")
def permute_command(
    input_path: Annotated[Path, typer.Argument(help="Input .h5ad")],
    output_path: Annotated[Path, typer.Argument(help="Output .h5ad")],
    within: Annotated[
        list[str], typer.Option("--within", help="obs column defining a permutation stratum")
    ],
    seed: Annotated[int, typer.Option()] = 1,
    layer: Annotated[str | None, typer.Option()] = None,
    output_layer: Annotated[str, typer.Option()] = "tanpopo_sim",
    spatial_key: Annotated[str, typer.Option()] = "spatial",
):
    """Permute complete expression vectors within one or more obs strata."""
    recipe = {"seed": seed, "baseline": {"permute_within": within}, "components": []}
    _simulate_and_write(
        input_path,
        output_path,
        recipe,
        layer,
        output_layer,
        "tanpopo_sim_baseline",
        spatial_key,
        "tanpopo_sim",
    )


@app.command("inject-program")
def inject_program_command(
    input_path: Annotated[Path, typer.Argument(help="Input .h5ad")],
    output_path: Annotated[Path, typer.Argument(help="Output .h5ad")],
    name: Annotated[str, typer.Option()] = "program",
    field: Annotated[str, typer.Option(help="gradient, hotspot, stripe, or rff")] = "rff",
    spatial_variance_fraction: Annotated[
        float, typer.Option(help="Fraction of program variance driven by the spatial field")
    ] = 0.36,
    expression_variance_fraction: Annotated[
        float | None,
        typer.Option(help="Target fraction of centered expression energy in the true subspace"),
    ] = None,
    n_programs: Annotated[int, typer.Option()] = 1,
    genes_per_program: Annotated[int, typer.Option()] = 50,
    cell_type_key: Annotated[str | None, typer.Option()] = None,
    target: Annotated[str | None, typer.Option()] = None,
    sample_key: Annotated[str | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 1,
    layer: Annotated[str | None, typer.Option()] = None,
    output_layer: Annotated[str, typer.Option()] = "tanpopo_sim",
    spatial_key: Annotated[str, typer.Option()] = "spatial",
):
    """Inject one global or within-cell-type spatial expression subspace."""
    component = {
        "name": name,
        "kind": "global" if cell_type_key is None else "within_type",
        "n_programs": n_programs,
        "spatial_variance_fraction": spatial_variance_fraction,
        "loadings": {"genes_per_program": genes_per_program},
        "field": {"kind": field},
    }
    if expression_variance_fraction is not None:
        component["expression_variance_fraction"] = expression_variance_fraction
    if sample_key:
        component["field"]["sample_key"] = sample_key
    if cell_type_key is not None:
        if target is None:
            raise typer.BadParameter("--target is required with --cell-type-key")
        component["cell_type_key"] = cell_type_key
        component["target"] = target
    recipe = {"seed": seed, "components": [component]}
    _simulate_and_write(
        input_path,
        output_path,
        recipe,
        layer,
        output_layer,
        "tanpopo_sim_baseline",
        spatial_key,
        "tanpopo_sim",
    )


@app.command("inject-niche")
def inject_niche_command(
    input_path: Annotated[Path, typer.Argument(help="Input .h5ad")],
    output_path: Annotated[Path, typer.Argument(help="Output .h5ad")],
    cell_type_key: Annotated[str, typer.Option()] = "cell_type",
    target: Annotated[str, typer.Option()] = ...,
    neighbour: Annotated[str, typer.Option()] = ...,
    name: Annotated[str, typer.Option()] = "niche",
    field: Annotated[str, typer.Option()] = "rff",
    coupling_variance_fraction: Annotated[float, typer.Option()] = 0.49,
    neighbour_spatial_variance_fraction: Annotated[float, typer.Option()] = 0.49,
    neighbour_expression_variance_fraction: Annotated[float | None, typer.Option()] = None,
    target_expression_variance_fraction: Annotated[float | None, typer.Option()] = None,
    genes_per_program: Annotated[int, typer.Option()] = 50,
    k: Annotated[int, typer.Option()] = 15,
    sample_key: Annotated[str | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 1,
    layer: Annotated[str | None, typer.Option()] = None,
    output_layer: Annotated[str, typer.Option()] = "tanpopo_sim",
    spatial_key: Annotated[str, typer.Option()] = "spatial",
):
    """Inject a known neighbour-context -> target-response program pair."""
    field_spec = {"kind": field}
    if sample_key:
        field_spec["sample_key"] = sample_key
    component = {
        "name": name,
        "kind": "niche",
        "cell_type_key": cell_type_key,
        "target": target,
        "neighbour": neighbour,
        "n_programs": 1,
        "neighbour_field": field_spec,
        "neighbour_loadings": {"genes_per_program": genes_per_program},
        "target_loadings": {"genes_per_program": genes_per_program},
        "neighbour_spatial_variance_fraction": neighbour_spatial_variance_fraction,
        "coupling_variance_fraction": coupling_variance_fraction,
        "k": k,
    }
    if neighbour_expression_variance_fraction is not None:
        component["neighbour_expression_variance_fraction"] = (
            neighbour_expression_variance_fraction
        )
    if target_expression_variance_fraction is not None:
        component["target_expression_variance_fraction"] = target_expression_variance_fraction
    recipe = {"seed": seed, "components": [component]}
    _simulate_and_write(
        input_path,
        output_path,
        recipe,
        layer,
        output_layer,
        "tanpopo_sim_baseline",
        spatial_key,
        "tanpopo_sim",
    )


@app.command("evaluate-subspace")
def evaluate_subspace_command(
    input_path: Annotated[
        Path, typer.Argument(help="AnnData containing true and estimated loadings")
    ],
    true_key: Annotated[str, typer.Option(help="True adata.varm key")],
    estimated_key: Annotated[str, typer.Option(help="Estimated adata.varm key")],
    max_rank: Annotated[
        int | None, typer.Option(help="Maximum estimated rank to evaluate")
    ] = None,
    threshold: Annotated[float, typer.Option(help="Additional recovery threshold")] = 0.9,
    curve_csv: Annotated[Path | None, typer.Option(help="Optional recovery-curve CSV")] = None,
    output_json: Annotated[Path | None, typer.Option(help="Optional JSON output path")] = None,
):
    """Evaluate truth anywhere in the retained estimated spectrum, not only the first k modes."""
    adata = read_h5ad(input_path)
    if true_key not in adata.varm or estimated_key not in adata.varm:
        raise typer.BadParameter("Both --true-key and --estimated-key must exist in adata.varm")
    true = np.asarray(adata.varm[true_key])
    estimated = np.asarray(adata.varm[estimated_key])
    M = estimated.shape[1] if max_rank is None else min(max_rank, estimated.shape[1])
    retained = estimated[:, :M]
    curve = recovery_curve(true, retained)
    same_rank = min(orthonormal_basis(true).shape[1], M)
    best = best_mode_matches(true, retained)
    result = {
        "true_rank": int(orthonormal_basis(true).shape[1]),
        "estimated_modes_evaluated": int(M),
        "subspace_overlap_first_true_rank": (
            subspace_overlap(true, retained[:, :same_rank]) if same_rank > 0 else 0.0
        ),
        "subspace_overlap_all_evaluated_modes": subspace_overlap(true, retained),
        "projector_distance_all_evaluated_modes": projector_distance(true, retained),
        "principal_angles_radians_all_evaluated_modes": principal_angles(true, retained).tolist(),
        "recovery_rank_50": recovery_rank(true, retained, threshold=0.5),
        "recovery_rank_80": recovery_rank(true, retained, threshold=0.8),
        "recovery_rank_90": recovery_rank(true, retained, threshold=0.9),
        f"recovery_rank_{int(round(threshold * 100))}": recovery_rank(
            true, retained, threshold=threshold
        ),
        "area_under_recovery_curve": area_under_recovery_curve(true, retained),
        "best_individual_mode_rank": best["estimated_rank"].tolist(),
        "best_individual_mode_absolute_cosine": best["absolute_cosine"].tolist(),
        "recovery_curve": [
            {"rank": int(rank), "subspace_overlap": float(overlap)} for rank, overlap in curve
        ],
    }
    if curve_csv is not None:
        with curve_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "subspace_overlap"])
            writer.writerows((int(rank), float(overlap)) for rank, overlap in curve)
    text = json.dumps(result, indent=2)
    if output_json is not None:
        output_json.write_text(text + "\n")
    typer.echo(text)


@app.command("characterize-injection")
def characterize_injection_command(
    input_path: Annotated[Path, typer.Argument(help="Simulated AnnData")],
    component: Annotated[str, typer.Option(help="Stored simulation component name")],
    baseline_layer: Annotated[
        str, typer.Option(help="Pre-injection baseline layer or X")
    ] = "tanpopo_sim_baseline",
    simulated_layer: Annotated[str, typer.Option(help="Simulated layer or X")] = "tanpopo_sim",
    spatial_key: Annotated[str, typer.Option()] = "spatial",
    prefix: Annotated[str, typer.Option()] = "tanpopo_sim",
    k: Annotated[int, typer.Option(help="k for method-agnostic spatial diagnostics")] = 15,
    reference_rank: Annotated[int, typer.Option(help="Number of ordinary baseline PCs")] = 30,
    random_projections: Annotated[
        int, typer.Option(help="Number of random non-spatial gene projections")
    ] = 30,
    max_reference_cells: Annotated[
        int, typer.Option(help="Maximum active cells used for reference calibration")
    ] = 20_000,
    seed: Annotated[int, typer.Option()] = 1,
    output_json: Annotated[Path | None, typer.Option()] = None,
):
    """Measure injection strength relative to naturally occurring baseline structure."""
    adata = read_h5ad(input_path)
    if spatial_key not in adata.obsm:
        raise typer.BadParameter(f"adata.obsm[{spatial_key!r}] was not found.")
    truth = component_truth_from_adata(adata, component, prefix=prefix)
    result = characterize_injection(
        _matrix_from_adata(adata, baseline_layer),
        _matrix_from_adata(adata, simulated_layer),
        np.asarray(adata.obsm[spatial_key]),
        truth,
        k=k,
        reference_rank=reference_rank,
        random_projections=random_projections,
        max_reference_cells=max_reference_cells,
        rng=seed,
    )
    text = json.dumps(result, indent=2)
    if output_json is not None:
        output_json.write_text(text + "\n")
    typer.echo(text)


if __name__ == "__main__":
    app()
