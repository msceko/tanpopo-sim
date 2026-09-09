import numpy as np

from tanpopo_sim import run_recipe


def test_recipe_combines_permutation_within_and_differential_components():
    rng = np.random.default_rng(12)
    n = 120
    g = 50
    X = rng.normal(size=(n, g))
    coords = rng.uniform(size=(n, 2))
    obs = {
        "sample": np.repeat(["s1", "s2", "s3", "s4"], 30),
        "cell_type": np.tile(np.repeat(["fib", "mac"], 15), 4),
        "condition": np.repeat(["A", "A", "B", "B"], 30),
    }
    recipe = {
        "seed": 2,
        "baseline": {"permute_within": ["sample", "cell_type"]},
        "components": [
            {
                "name": "fib",
                "kind": "within_type",
                "cell_type_key": "cell_type",
                "target": "fib",
                "n_programs": 1,
                "spatial_fraction": 0.6,
                "loadings": {"genes_per_program": 6},
                "field": {"kind": "gradient", "sample_key": "sample"},
            },
            {
                "name": "diff",
                "kind": "differential",
                "condition_key": "condition",
                "n_programs": 1,
                "spatial_fraction_by_condition": {"A": 0.1, "B": 0.8},
                "loadings": {"genes_per_program": 6},
                "field": {"kind": "rff", "sample_key": "sample"},
            },
        ],
    }
    result = run_recipe(X, coords, obs, recipe)
    assert result.X.shape == X.shape
    assert result.permutation_indices is not None
    assert set(result.components) == {"fib", "diff"}
    fib_truth = result.components["fib"]
    assert fib_truth.active_rows[obs["cell_type"] == "fib"].all()
    assert not fib_truth.active_rows[obs["cell_type"] == "mac"].any()
