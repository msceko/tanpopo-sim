import numpy as np

from tanpopo_sim import projector_distance, source_leakage_matrix, subspace_overlap


def test_subspace_overlap_is_rotation_invariant():
    rng = np.random.default_rng(1)
    V, _ = np.linalg.qr(rng.normal(size=(60, 3)))
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    W = V @ R

    assert subspace_overlap(V, W) > 1 - 1e-12
    assert projector_distance(V, W) < 1e-10


def test_source_leakage_recovers_diagonal_for_orthogonal_sources():
    eye = np.eye(12)
    true = {"between": eye[:, :2], "within": eye[:, 2:4], "niche": eye[:, 4:6]}
    estimated = {"between": eye[:, :2], "within": eye[:, 2:4], "niche": eye[:, 4:6]}
    matrix, _, _ = source_leakage_matrix(true, estimated)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)


def test_recovery_curve_does_not_assume_truth_is_in_first_modes():
    from tanpopo_sim import area_under_recovery_curve, recovery_curve, recovery_rank

    eye = np.eye(20)
    true = eye[:, :2]
    estimated = np.column_stack([eye[:, 2:7], eye[:, :2], eye[:, 7:10]])
    curve = recovery_curve(true, estimated)

    assert curve[1, 1] == 0.0
    assert curve[5, 1] == 0.5
    assert curve[6, 1] == 1.0
    assert recovery_rank(true, estimated, threshold=0.9) == 7
    assert 0 < area_under_recovery_curve(true, estimated) < 1
