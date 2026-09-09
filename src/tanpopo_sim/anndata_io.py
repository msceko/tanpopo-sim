from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .recipes import run_recipe
from .truth import ComponentTruth


def _anndata_module():
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "AnnData support requires the 'anndata' package. Install tanpopo-sim normally "
            "or `pip install anndata`."
        ) from exc
    return ad


def read_h5ad(path: str | Path):
    return _anndata_module().read_h5ad(path)


def _slug(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return value or "component"


def store_simulation_truth(adata, result, *, prefix: str = "tanpopo_sim") -> None:
    components = {}
    for name, truth in result.components.items():
        slug = _slug(name)
        loading_key = f"{prefix}_{slug}_loadings"
        score_key = f"{prefix}_{slug}_scores"
        spatial_key = f"{prefix}_{slug}_spatial_scores"
        active_key = f"{prefix}_{slug}_active"
        adata.varm[loading_key] = truth.loadings
        adata.obsm[score_key] = truth.scores
        adata.obsm[spatial_key] = truth.spatial_scores
        adata.obs[active_key] = truth.active_rows
        components[name] = {
            "kind": truth.kind,
            "loadings_key": loading_key,
            "scores_key": score_key,
            "spatial_scores_key": spatial_key,
            "active_key": active_key,
            "metadata_json": json.dumps(truth.metadata, default=str),
        }

    if result.permutation_indices is not None:
        adata.obs[f"{prefix}_source_index"] = result.permutation_indices.astype(np.int64)

    adata.uns[prefix] = {
        "components": components,
        "metadata_json": json.dumps(result.metadata, default=str),
    }


def component_truth_from_adata(
    adata,
    component: str,
    *,
    prefix: str = "tanpopo_sim",
) -> ComponentTruth:
    """Reconstruct a :class:`ComponentTruth` from keys stored in AnnData."""
    if prefix not in adata.uns:
        raise KeyError(f"adata.uns[{prefix!r}] was not found.")
    components = adata.uns[prefix].get("components", {})
    if component not in components:
        raise KeyError(f"Simulation component {component!r} was not found.")
    info = components[component]
    metadata = json.loads(str(info.get("metadata_json", "{}")))
    return ComponentTruth(
        name=component,
        kind=str(info["kind"]),
        loadings=np.asarray(adata.varm[str(info["loadings_key"])]),
        scores=np.asarray(adata.obsm[str(info["scores_key"])]),
        spatial_scores=np.asarray(adata.obsm[str(info["spatial_scores_key"])]),
        active_rows=np.asarray(adata.obs[str(info["active_key"])]).astype(bool),
        metadata=metadata,
    )


def simulate_adata(
    adata,
    recipe,
    *,
    layer: str | None = None,
    output_layer: str | None = "tanpopo_sim",
    baseline_output_layer: str | None = "tanpopo_sim_baseline",
    spatial_key: str = "spatial",
    prefix: str = "tanpopo_sim",
    copy: bool = True,
):
    """High-level AnnData adapter around the array-based recipe engine."""
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{spatial_key!r}] was not found.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"adata.layers[{layer!r}] was not found.")
    output = adata.copy() if copy else adata
    X = output.X if layer is None else output.layers[layer]
    obs = {column: np.asarray(output.obs[column]) for column in output.obs.columns}
    result = run_recipe(
        X,
        np.asarray(output.obsm[spatial_key]),
        obs,
        recipe,
        gene_names=np.asarray(output.var_names).astype(str),
    )
    if baseline_output_layer is not None:
        if baseline_output_layer == "X":
            raise ValueError("baseline_output_layer='X' would overwrite the original expression.")
        output.layers[baseline_output_layer] = result.baseline_X
    if output_layer is None or output_layer == "X":
        output.X = result.X
    else:
        output.layers[output_layer] = result.X
    store_simulation_truth(output, result, prefix=prefix)
    return output, result


def simulate_h5ad(
    input_path: str | Path,
    output_path: str | Path,
    recipe,
    **kwargs,
):
    adata = read_h5ad(input_path)
    output, result = simulate_adata(adata, recipe, **kwargs)
    output.write_h5ad(output_path)
    return result
