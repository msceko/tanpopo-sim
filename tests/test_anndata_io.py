import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("anndata") is None,
    reason="anndata is not installed in the test environment",
)


def test_simulate_adata_stores_truth():
    import anndata as ad
    from tanpopo_sim.anndata_io import simulate_adata

    rng = np.random.default_rng(13)
    adata = ad.AnnData(rng.normal(size=(40, 20)))
    adata.obsm["spatial"] = rng.uniform(size=(40, 2))
    adata.obs["cell_type"] = np.repeat(["a", "b"], 20)
    recipe = {
        "seed": 1,
        "components": [
            {
                "name": "p",
                "kind": "within_type",
                "cell_type_key": "cell_type",
                "target": "a",
                "spatial_fraction": 0.5,
                "loadings": {"genes_per_program": 5},
                "field": {"kind": "gradient"},
            }
        ],
    }
    out, _ = simulate_adata(adata, recipe)
    assert "tanpopo_sim" in out.layers
    assert "tanpopo_sim_p_loadings" in out.varm
    assert "tanpopo_sim_p_scores" in out.obsm
