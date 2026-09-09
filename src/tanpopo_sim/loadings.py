from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .utils import get_rng


def orthonormalize_loadings(loadings: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    loadings = np.asarray(loadings, dtype=float)
    if loadings.ndim == 1:
        loadings = loadings[:, None]
    if loadings.ndim != 2:
        raise ValueError("loadings must be gene x program.")
    support = np.flatnonzero(np.any(np.abs(loadings) > tol, axis=1))
    if support.size < loadings.shape[1]:
        raise ValueError("Too few supported genes for the requested number of programs.")
    Q, R = np.linalg.qr(loadings[support], mode="reduced")
    if np.any(np.abs(np.diag(R)) <= tol):
        raise ValueError("loadings are rank deficient.")
    # Stabilize arbitrary QR signs.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q *= signs
    out = np.zeros_like(loadings, dtype=float)
    out[support] = Q
    return out


def loadings_from_supports(
    n_genes: int,
    supports: Sequence[Sequence[int]],
    *,
    positive: bool = False,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    generator = get_rng(rng)
    n_programs = len(supports)
    loadings = np.zeros((n_genes, n_programs), dtype=float)
    used: set[int] = set()
    for k, support in enumerate(supports):
        support = np.asarray(support, dtype=int)
        if support.ndim != 1 or support.size == 0:
            raise ValueError("Every support must contain at least one gene index.")
        if np.any((support < 0) | (support >= n_genes)):
            raise ValueError("A support contains an out-of-range gene index.")
        overlap = used.intersection(support.tolist())
        if overlap:
            raise ValueError(
                "Supports must be disjoint for sparse orthonormal truth; "
                f"overlap contains {sorted(overlap)[:5]}."
            )
        used.update(support.tolist())
        weights = np.abs(generator.normal(size=support.size)) if positive else generator.normal(
            size=support.size
        )
        norm = np.linalg.norm(weights)
        if norm == 0:
            weights[0] = 1.0
            norm = 1.0
        loadings[support, k] = weights / norm
    return loadings


def sparse_loadings(
    n_genes: int,
    n_programs: int = 1,
    genes_per_program: int = 50,
    *,
    gene_pool: Sequence[int] | None = None,
    disjoint: bool = True,
    positive: bool = False,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    if n_genes <= 0 or n_programs <= 0 or genes_per_program <= 0:
        raise ValueError("n_genes, n_programs, and genes_per_program must be positive.")
    generator = get_rng(rng)
    pool = np.arange(n_genes) if gene_pool is None else np.asarray(gene_pool, dtype=int)
    if np.any((pool < 0) | (pool >= n_genes)):
        raise ValueError("gene_pool contains out-of-range indices.")
    if disjoint and n_programs * genes_per_program > pool.size:
        raise ValueError("Not enough genes for disjoint program supports.")

    supports = []
    if disjoint:
        selected = generator.choice(
            pool, size=n_programs * genes_per_program, replace=False
        ).reshape(n_programs, genes_per_program)
        supports = [row for row in selected]
    else:
        for _ in range(n_programs):
            supports.append(generator.choice(pool, size=genes_per_program, replace=False))
        # Overlapping supports are allowed by request, but we cannot guarantee sparse
        # orthogonality. QR would spread loadings over the union, which is intentional.
        raw = np.zeros((n_genes, n_programs), dtype=float)
        for k, support in enumerate(supports):
            raw[support, k] = (
                np.abs(generator.normal(size=genes_per_program))
                if positive
                else generator.normal(size=genes_per_program)
            )
        return orthonormalize_loadings(raw)

    return loadings_from_supports(
        n_genes, supports, positive=positive, rng=generator
    )
