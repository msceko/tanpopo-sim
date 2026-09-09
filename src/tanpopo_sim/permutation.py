from __future__ import annotations

import numpy as np

from .utils import get_rng, labels_to_strata, validate_expression


def stratified_permutation_indices(
    *strata,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Return a row permutation that never moves a row outside its stratum.

    Passing sample and cell-type labels therefore preserves the real spatial
    distribution of cell types and all within-cell expression vectors while
    destroying within-stratum expression-position association.
    """
    arrays = labels_to_strata(*strata)
    generator = get_rng(rng)
    n = arrays[0].shape[0]
    permutation = np.arange(n)

    # np.unique over a structured string representation avoids requiring pandas.
    keys = np.empty(n, dtype=object)
    for i in range(n):
        keys[i] = tuple(array[i] for array in arrays)
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)

    for indices in groups.values():
        idx = np.asarray(indices, dtype=int)
        if idx.size > 1:
            permutation[idx] = generator.permutation(idx)
    return permutation


def stratified_permute_expression(
    X,
    *strata,
    rng: int | np.random.Generator | None = None,
    return_indices: bool = False,
):
    """Permute complete expression vectors within supplied strata."""
    X = validate_expression(X)
    permutation = stratified_permutation_indices(*strata, rng=rng)
    out = X[permutation].copy()
    if return_indices:
        return out, permutation
    return out
