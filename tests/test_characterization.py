import numpy as np
import scipy.sparse as sp

from tanpopo_sim import (
    characterize_injection,
    inject_program_subspace,
    pca_reference_scores,
    sparse_loadings,
)


def test_characterization_reports_relative_strength_percentiles():
    rng = np.random.default_rng(31)
    n = 180
    g = 50
    coords = np.column_stack([np.linspace(0, 10, n), rng.normal(scale=0.1, size=n)])
    X = rng.normal(size=(n, g))
    V = sparse_loadings(g, 1, 8, rng=32)
    Z = np.sin(coords[:, 0])
    X_sim, truth = inject_program_subspace(
        X,
        V,
        Z,
        spatial_variance_fraction=0.81,
        expression_variance_fraction=0.08,
        rng=33,
    )

    result = characterize_injection(
        X,
        X_sim,
        coords,
        truth,
        k=8,
        reference_rank=10,
        random_projections=10,
        max_reference_cells=180,
        rng=34,
    )

    assert abs(result["expression"]["subspace_variance_fraction_after"] - 0.08) < 1e-8
    assert result["spatial_after"]["morans_i"][0] > result["spatial_before"]["morans_i"][0]
    p = result["percentile_vs_pca"]["morans_i_after"][0]
    assert 0 <= p <= 1


def test_sparse_pca_reference_scores():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(90, 35))
    X[np.abs(X) < 0.7] = 0
    scores, variance = pca_reference_scores(sp.csr_matrix(X), rank=5)
    assert scores.shape == (90, 5)
    assert variance.shape == (5,)
    assert np.all(np.diff(variance) <= 1e-8)
