"""Semi-synthetic simulations for conditional spatial transcriptomics benchmarks."""

from .characterization import (
    characterize_injection,
    expression_subspace_fraction,
    pca_reference_scores,
    random_projection_reference_scores,
)
from .fields import (
    distance_field,
    generate_fields,
    generate_samplewise_fields,
    gradient_field,
    hotspot_field,
    random_fourier_field,
    stripe_field,
)
from .injection import inject_grouped_program_subspace, inject_program_subspace
from .loadings import loadings_from_supports, orthonormalize_loadings, sparse_loadings
from .metrics import (
    area_under_recovery_curve,
    best_mode_matches,
    matched_loading_correlations,
    principal_angles,
    projector_distance,
    recovery_curve,
    recovery_rank,
    source_leakage_matrix,
    subspace_overlap,
)
from .niche import knn_weight_matrix, neighbour_exposure, simulate_niche_coupling
from .permutation import stratified_permutation_indices, stratified_permute_expression
from .recipes import run_recipe
from .spatial_stats import (
    gearys_c,
    knn_adjacency,
    morans_i,
    neighbour_predictive_r2,
    spatial_score_summary,
)
from .truth import ComponentTruth, SimulationResult

__all__ = [
    "ComponentTruth",
    "SimulationResult",
    "area_under_recovery_curve",
    "best_mode_matches",
    "characterize_injection",
    "distance_field",
    "expression_subspace_fraction",
    "generate_fields",
    "generate_samplewise_fields",
    "gearys_c",
    "gradient_field",
    "hotspot_field",
    "inject_grouped_program_subspace",
    "inject_program_subspace",
    "knn_adjacency",
    "knn_weight_matrix",
    "loadings_from_supports",
    "matched_loading_correlations",
    "morans_i",
    "neighbour_exposure",
    "neighbour_predictive_r2",
    "orthonormalize_loadings",
    "pca_reference_scores",
    "principal_angles",
    "projector_distance",
    "random_fourier_field",
    "random_projection_reference_scores",
    "recovery_curve",
    "recovery_rank",
    "run_recipe",
    "simulate_niche_coupling",
    "source_leakage_matrix",
    "sparse_loadings",
    "spatial_score_summary",
    "stratified_permutation_indices",
    "stratified_permute_expression",
    "stripe_field",
    "subspace_overlap",
]

__version__ = "0.2.0"
