import numpy as np

from tanpopo_sim import generate_fields, simulate_niche_coupling, sparse_loadings


def test_niche_coupling_produces_target_scores_correlated_with_exposure():
    rng = np.random.default_rng(7)
    n = 160
    coords = rng.uniform(0, 10, size=(n, 2))
    cell_types = np.where(np.arange(n) % 2 == 0, "target", "neighbour")
    X = rng.normal(size=(n, 30))
    neighbour_rows = cell_types == "neighbour"
    Z = generate_fields(coords[neighbour_rows], 1, "gradient", rng=8)
    Vn = sparse_loadings(30, 1, 5, gene_pool=np.arange(0, 10), rng=9)
    Vt = sparse_loadings(30, 1, 5, gene_pool=np.arange(10, 20), rng=10)

    _, truths, exposure = simulate_niche_coupling(
        X,
        coords,
        cell_types,
        target_type="target",
        neighbour_type="neighbour",
        neighbour_loadings=Vn,
        target_loadings=Vt,
        neighbour_spatial_scores=Z,
        neighbour_spatial_fraction=0.8,
        coupling=0.9,
        k=8,
        rng=11,
    )

    target = cell_types == "target"
    target_scores = truths["niche__target"].scores[target, 0]
    corr = np.corrcoef(target_scores, exposure[target, 0])[0, 1]
    assert corr > 0.65
