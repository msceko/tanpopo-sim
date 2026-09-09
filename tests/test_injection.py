import numpy as np
import scipy.sparse as sp

from tanpopo_sim import inject_program_subspace, sparse_loadings


def test_injection_preserves_orthogonal_complement():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 40))
    V = sparse_loadings(40, n_programs=2, genes_per_program=6, rng=3)
    Z = rng.normal(size=(120, 2))

    X_sim, truth = inject_program_subspace(
        X, V, Z, spatial_fraction=[0.4, 0.8], rng=4
    )

    P = truth.loadings @ truth.loadings.T
    residual_change = (X_sim - X) @ (np.eye(X.shape[1]) - P)
    assert np.linalg.norm(residual_change) < 1e-9


def test_within_type_injection_changes_only_selected_rows():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 25))
    V = sparse_loadings(25, 1, 5, rng=1)
    rows = np.arange(0, 80, 2)
    Z = np.linspace(-2, 2, rows.size)

    X_sim, truth = inject_program_subspace(
        X, V, Z, spatial_fraction=0.7, rows=rows
    )

    inactive = np.setdiff1d(np.arange(80), rows)
    np.testing.assert_allclose(X_sim[inactive], X[inactive])
    assert truth.active_rows[rows].all()
    assert not truth.active_rows[inactive].any()


def test_sparse_injection_preserves_sparse_type():
    rng = np.random.default_rng(5)
    dense = rng.normal(size=(50, 20))
    dense[np.abs(dense) < 0.8] = 0
    X = sp.csr_matrix(dense)
    V = sparse_loadings(20, 1, 4, rng=6)
    Z = rng.normal(size=50)
    X_sim, _ = inject_program_subspace(X, V, Z, spatial_fraction=0.5)
    assert sp.issparse(X_sim)
    assert X_sim.shape == X.shape


def test_zero_spatial_fraction_leaves_scores_unchanged():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(60, 24))
    V = sparse_loadings(24, 1, 5, rng=16)
    Z = rng.normal(size=60)
    X_sim, _ = inject_program_subspace(X, V, Z, spatial_fraction=0.0)
    np.testing.assert_allclose(X_sim, X, atol=1e-10)


def test_expression_variance_fraction_controls_program_strength():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(300, 80))
    V = sparse_loadings(80, 2, 10, rng=22)
    Z = rng.normal(size=(300, 2))

    X_sim, truth = inject_program_subspace(
        X,
        V,
        Z,
        spatial_variance_fraction=0.49,
        expression_variance_fraction=0.10,
        rng=23,
    )

    assert abs(truth.metadata["expression_variance_fraction_after"] - 0.10) < 1e-8
    np.testing.assert_allclose(truth.metadata["spatial_variance_fraction"], [0.49, 0.49])
    assert X_sim.shape == X.shape
