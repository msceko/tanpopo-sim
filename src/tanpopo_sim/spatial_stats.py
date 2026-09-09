from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .utils import as_column_matrix, validate_coords


def knn_adjacency(coords, *, k: int = 15, weighted: bool = False) -> sp.csr_matrix:
    """Build a symmetric zero-diagonal kNN graph independent of Tanpopo kernels.

    By default this is an unweighted graph. With ``weighted=True`` Gaussian
    weights are used with bandwidth equal to the median positive neighbour
    distance. The graph is symmetrized by taking the maximum of both directions.
    """
    coords = validate_coords(coords)
    n = coords.shape[0]
    if n < 2:
        raise ValueError("At least two coordinates are required.")
    if k <= 0:
        raise ValueError("k must be positive.")
    k_eff = min(int(k), n - 1)
    distances, indices = cKDTree(coords).query(coords, k=k_eff + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    if weighted:
        positive = distances[distances > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
        data = np.exp(-0.5 * np.square(distances / max(sigma, 1e-12))).ravel()
    else:
        data = np.ones(n * k_eff, dtype=float)

    rows = np.repeat(np.arange(n), k_eff)
    cols = indices.ravel()
    graph = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def _validate_graph_scores(graph: sp.spmatrix, scores) -> tuple[sp.csr_matrix, np.ndarray]:
    graph = graph.tocsr().astype(float)
    scores = as_column_matrix(scores, n_rows=graph.shape[0])
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square.")
    return graph, scores


def morans_i(scores, graph: sp.spmatrix) -> np.ndarray:
    """Classical Moran's I for each score column."""
    graph, scores = _validate_graph_scores(graph, scores)
    z = scores - scores.mean(axis=0, keepdims=True)
    denominator = np.sum(np.square(z), axis=0)
    s0 = float(graph.sum())
    if s0 <= 0:
        return np.full(scores.shape[1], np.nan)
    numerator = np.sum(z * (graph @ z), axis=0)
    return np.divide(
        graph.shape[0] * numerator,
        s0 * denominator,
        out=np.full(scores.shape[1], np.nan),
        where=denominator > 0,
    )


def gearys_c(scores, graph: sp.spmatrix) -> np.ndarray:
    """Classical Geary's C for each score column."""
    graph, scores = _validate_graph_scores(graph, scores)
    z = scores - scores.mean(axis=0, keepdims=True)
    denominator = np.sum(np.square(z), axis=0)
    s0 = float(graph.sum())
    if s0 <= 0:
        return np.full(scores.shape[1], np.nan)

    coo = graph.tocoo()
    diff = scores[coo.row] - scores[coo.col]
    numerator = np.sum(coo.data[:, None] * np.square(diff), axis=0)
    scale = (graph.shape[0] - 1.0) / (2.0 * s0)
    return np.divide(
        scale * numerator,
        denominator,
        out=np.full(scores.shape[1], np.nan),
        where=denominator > 0,
    )


def neighbour_predictive_r2(scores, graph: sp.spmatrix) -> np.ndarray:
    """Leave-one-cell-out R² from the row-normalized kNN neighbour average."""
    graph, scores = _validate_graph_scores(graph, scores)
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(row_sum, dtype=float),
        where=row_sum > 0,
    )
    weights = sp.diags(inverse, format="csr") @ graph
    prediction = weights @ scores
    centered = scores - scores.mean(axis=0, keepdims=True)
    sst = np.sum(np.square(centered), axis=0)
    sse = np.sum(np.square(scores - prediction), axis=0)
    return np.divide(
        sst - sse,
        sst,
        out=np.full(scores.shape[1], np.nan),
        where=sst > 0,
    )


def spatial_score_summary(scores, coords, *, k: int = 15) -> dict[str, np.ndarray]:
    """Method-agnostic spatial diagnostics for one or more cell score vectors."""
    graph = knn_adjacency(coords, k=k, weighted=False)
    return {
        "morans_i": morans_i(scores, graph),
        "gearys_c": gearys_c(scores, graph),
        "knn_predictive_r2": neighbour_predictive_r2(scores, graph),
    }
