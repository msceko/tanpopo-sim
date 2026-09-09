from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ComponentTruth:
    """Ground truth for one injected expression subspace."""

    name: str
    kind: str
    loadings: np.ndarray
    scores: np.ndarray
    spatial_scores: np.ndarray
    active_rows: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_programs(self) -> int:
        return int(self.loadings.shape[1])

    @property
    def support_indices(self) -> np.ndarray:
        return np.flatnonzero(np.any(np.abs(self.loadings) > 0, axis=1))


@dataclass
class SimulationResult:
    """A simulated matrix together with exact truth needed for benchmarking.

    ``baseline_X`` is the matrix immediately after any baseline permutation and
    before the first injected component. Keeping it makes it possible to quantify
    the injection against the actual background used by the simulator.
    """

    X: Any
    baseline_X: Any | None = None
    components: dict[str, ComponentTruth] = field(default_factory=dict)
    permutation_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_component(self, component: ComponentTruth) -> None:
        if component.name in self.components:
            raise ValueError(f"Duplicate simulation component name: {component.name!r}")
        self.components[component.name] = component
