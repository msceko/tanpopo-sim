from __future__ import annotations

import numpy as np

from .utils import get_rng, standardize_columns, validate_coords


def _diameter(coords: np.ndarray) -> float:
    span = np.ptp(coords[:, :2], axis=0)
    value = float(np.linalg.norm(span))
    return value if value > 0 else 1.0


def gradient_field(coords, angle: float = 0.0) -> np.ndarray:
    coords = validate_coords(coords)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    return coords[:, :2] @ direction


def hotspot_field(coords, centre=None, sigma: float | None = None) -> np.ndarray:
    coords = validate_coords(coords)
    if centre is None:
        centre = np.median(coords[:, :2], axis=0)
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (2,):
        raise ValueError("centre must contain x and y coordinates.")
    if sigma is None:
        sigma = 0.15 * _diameter(coords)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    distance2 = np.square(coords[:, :2] - centre).sum(axis=1)
    return np.exp(-distance2 / (2.0 * sigma**2))


def stripe_field(
    coords,
    angle: float = 0.0,
    wavelength: float | None = None,
    phase: float = 0.0,
) -> np.ndarray:
    coords = validate_coords(coords)
    if wavelength is None:
        wavelength = 0.35 * _diameter(coords)
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    projected = gradient_field(coords, angle=angle)
    return np.sin(2.0 * np.pi * projected / wavelength + phase)


def random_fourier_field(
    coords,
    length_scale: float | None = None,
    n_features: int = 128,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Approximate an RBF Gaussian-process field without using Tanpopo's kernel."""
    coords = validate_coords(coords)
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if length_scale is None:
        length_scale = 0.2 * _diameter(coords)
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")
    generator = get_rng(rng)
    W = generator.normal(
        scale=1.0 / length_scale,
        size=(coords.shape[1], n_features),
    )
    phase = generator.uniform(0.0, 2.0 * np.pi, size=n_features)
    weights = generator.normal(size=n_features)
    features = np.sqrt(2.0 / n_features) * np.cos(coords @ W + phase)
    return features @ weights


def distance_field(
    distances,
    transform: str = "linear",
    scale: float | None = None,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1:
        raise ValueError("distances must be one-dimensional.")
    if scale is None:
        positive = np.abs(distances[np.isfinite(distances)])
        scale = float(np.median(positive)) if positive.size else 1.0
        if scale <= 0:
            scale = 1.0
    x = distances / scale
    if transform == "linear":
        return x
    if transform == "exp":
        return np.exp(-np.maximum(x, 0.0))
    if transform == "logistic":
        return 1.0 / (1.0 + np.exp(x))
    raise ValueError("transform must be 'linear', 'exp', or 'logistic'.")


def make_field(
    coords,
    kind: str,
    *,
    rng: int | np.random.Generator | None = None,
    **params,
) -> np.ndarray:
    generator = get_rng(rng)
    kind = kind.lower().replace("-", "_")
    if kind == "gradient":
        params = dict(params)
        params.setdefault("angle", float(generator.uniform(0, 2 * np.pi)))
        values = gradient_field(coords, **params)
    elif kind == "hotspot":
        params = dict(params)
        if params.get("centre") is None:
            coords_array = validate_coords(coords)
            params["centre"] = coords_array[generator.integers(coords_array.shape[0]), :2]
        values = hotspot_field(coords, **params)
    elif kind in {"stripe", "streak"}:
        params = dict(params)
        params.setdefault("angle", float(generator.uniform(0, 2 * np.pi)))
        params.setdefault("phase", float(generator.uniform(0, 2 * np.pi)))
        values = stripe_field(coords, **params)
    elif kind in {"rff", "random_fourier", "smooth"}:
        values = random_fourier_field(coords, rng=generator, **params)
    else:
        raise ValueError(f"Unknown field kind: {kind!r}")
    return standardize_columns(values).ravel()


def generate_fields(
    coords,
    n_fields: int,
    kind: str,
    *,
    rng: int | np.random.Generator | None = None,
    **params,
) -> np.ndarray:
    if n_fields <= 0:
        raise ValueError("n_fields must be positive.")
    generator = get_rng(rng)
    values = [make_field(coords, kind, rng=generator, **params) for _ in range(n_fields)]
    return np.column_stack(values)


def generate_samplewise_fields(
    coords,
    sample_labels,
    n_fields: int,
    kind: str,
    *,
    rng: int | np.random.Generator | None = None,
    **params,
) -> np.ndarray:
    """Generate independent realizations of the same field family per sample."""
    coords = validate_coords(coords)
    sample_labels = np.asarray(sample_labels)
    if sample_labels.ndim != 1 or sample_labels.shape[0] != coords.shape[0]:
        raise ValueError("sample_labels must contain one label per coordinate row.")
    generator = get_rng(rng)
    out = np.zeros((coords.shape[0], n_fields), dtype=float)
    for sample in np.unique(sample_labels):
        rows = np.flatnonzero(sample_labels == sample)
        out[rows] = generate_fields(
            coords[rows], n_fields=n_fields, kind=kind, rng=generator, **params
        )
    return out
