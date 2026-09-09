import numpy as np
import scipy.sparse as sp

from tanpopo_sim import stratified_permute_expression


def test_stratified_permutation_preserves_rows_within_groups():
    X = np.arange(48).reshape(12, 4)
    sample = np.repeat(["s1", "s2"], 6)
    cell_type = np.tile(np.repeat(["a", "b"], 3), 2)

    out, permutation = stratified_permute_expression(
        X, sample, cell_type, rng=4, return_indices=True
    )

    np.testing.assert_array_equal(out, X[permutation])
    np.testing.assert_array_equal(sample, sample[permutation])
    np.testing.assert_array_equal(cell_type, cell_type[permutation])
    assert sorted(map(tuple, out)) == sorted(map(tuple, X))


def test_stratified_permutation_supports_sparse_input():
    X = sp.csr_matrix(np.arange(30).reshape(10, 3))
    groups = np.repeat([0, 1], 5)
    out = stratified_permute_expression(X, groups, rng=1)
    assert sp.issparse(out)
    np.testing.assert_array_equal(np.sort(out.toarray(), axis=0), np.sort(X.toarray(), axis=0))
