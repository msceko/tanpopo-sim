# tanpopo-sim

`tanpopo-sim` creates semi-synthetic spatial transcriptomics benchmarks with exact
multigene ground truth while retaining realistic expression backgrounds and real tissue
geometry.

The numerical API operates on NumPy arrays or SciPy sparse matrices. AnnData is only
used in the high-level I/O and CLI layer.

## Why version 0.2 changes the strength model

A spatial benchmark should not force injected programs to become the first modes of
Tanpopo, SPACO, NSF, or any other method. Version 0.2 therefore separates three
questions:

1. **How transcriptionally strong is the injected subspace?**
2. **How spatial is variation inside that subspace?**
3. **How far down an estimated spectrum must one search before the truth is recovered?**

None of these quantities uses Tanpopo's Wendland operator.

## Mathematical injection model

For an orthonormal gene loading matrix `V`, baseline scores are

\[
T_0 = XV.
\]

After standardization, an independently generated spatial field `Z` is injected as

\[
T^*_k = \mu_k + \sigma_k\left[
\sqrt{1-\rho_k^2}\,T_k + \rho_k Z_k
\right].
\]

The final expression matrix is

\[
X^* = X + (T^*-T_0)V^\top.
\]

Consequently the expression complement orthogonal to `V` is unchanged exactly.

The preferred spatial-strength parameter is

\[
f_{\mathrm{spatial},k}=\rho_k^2.
\]

Use `spatial_variance_fraction` to specify this directly. The old
`spatial_fraction` parameter remains supported and is interpreted as `rho`.

### Expression amplitude

`expression_variance_fraction=f` chooses the program standard deviations so the true
loading subspace accounts for fraction `f` of centered expression energy among the
active cells after injection:

\[
f = \frac{E_V}{E_\perp + E_V}.
\]

This is independent of any spatial-analysis method. Alternatives are:

- `program_variance_multiplier`: multiply the pre-existing variance along each true
  direction;
- `program_sd`: set absolute score standard deviations;
- omit all three to preserve the original subspace variance.

Expression amplitude and spatial coherence can therefore be swept independently.

## Core API

```python
from tanpopo_sim import (
    inject_program_subspace,
    characterize_injection,
    recovery_curve,
    recovery_rank,
)

X_sim, truth = inject_program_subspace(
    X,
    true_loadings,
    spatial_fields,
    expression_variance_fraction=0.05,
    spatial_variance_fraction=0.64,
)
```

Here the true gene subspace explains 5% of centered expression energy and 64% of its
score variance is attributable to the supplied spatial field.

## Real-background calibration

`characterize_injection()` compares the injected program against naturally occurring
structure in the actual pre-injection matrix.

It reports:

- expression variance fraction before and after injection;
- program-score Moran's I;
- Geary's C;
- kNN neighbour-predictive R2;
- the same quantities for the supplied spatial driver;
- reference distributions from ordinary non-spatial PCA components;
- reference distributions from random orthogonal gene projections;
- percentiles of the injected program relative to both reference distributions.

```python
summary = characterize_injection(
    X_baseline,
    X_sim,
    coords,
    truth,
    reference_rank=30,
    random_projections=30,
)

print(summary["percentile_vs_pca"]["morans_i_after"])
```

This permits statements such as "the injected program has expression variance near the
60th percentile of baseline PCs but spatial autocorrelation near the 90th percentile."

The spatial diagnostics use a generic zero-diagonal kNN graph, not Tanpopo's kernel.

## Recovery without assuming the first k modes are truth

If `V_true` contains two injected programs and an analysis method returns 30 ordered
modes, do not compare only against the first two. Use cumulative recovery:

```python
curve = recovery_curve(V_true, estimated_modes, max_rank=30)
r90 = recovery_rank(V_true, estimated_modes, threshold=0.90)
```

For estimated rank `m`, the recovery curve is

\[
R(m)=\frac{1}{q}\|Q_{\mathrm{true}}^\top Q_{1:m}\|_F^2,
\]

where `q` is the true subspace rank. This is rotation invariant and asks how much of
the true subspace is present anywhere among the first `m` estimated modes.

Useful summaries are:

- `R(5)`, `R(10)`, `R(20)`;
- `r50`, `r80`, `r90`;
- area under the recovery curve;
- best individual mode rank as a secondary diagnostic.

## Baseline regimes

Three benchmark regimes are recommended.

### 1. Stratified-permutation background

Permute whole expression vectors within sample × cell type. This preserves realistic
non-spatial covariance and cell-type composition while removing most within-type
spatial association.

```python
from tanpopo_sim import stratified_permute_expression

X0 = stratified_permute_expression(X, sample, cell_type, rng=1)
```

### 2. Real unpermuted background

Inject directly into the observed matrix. This leaves unknown real spatial programs in
place and is the main stress test for ranking/recovery.

### 3. External simulated background

Use an external simulator such as scDesign3 for count realism, then apply the same
array-level injections. This should remain external to this package.

## Spatial fields

Spatial fields deliberately do not use Tanpopo's Wendland kernel. Included generators
are:

- rotated gradients;
- hotspots;
- stripes;
- random Fourier fields approximating smooth RBF-GP structure;
- supplied distance-to-boundary vectors.

## Niche coupling

`simulate_niche_coupling()` creates a known directed pair:

\[
\text{neighbour gene program}
\rightarrow
\text{local exposure}
\rightarrow
\text{target response program}.
\]

Neighbour exposure is generated with a Gaussian kNN graph, again independently of
Tanpopo's kernel.

Preferred parameters are:

- `neighbour_expression_variance_fraction`;
- `neighbour_spatial_variance_fraction`;
- `target_expression_variance_fraction`;
- `coupling_variance_fraction`.

## Differential simulations

`inject_grouped_program_subspace()` can hold the true gene loading matrix fixed while
changing only spatial variance fraction between groups. This creates a differential
spatial-covariance alternative without differential mean expression.

## Recipes

The same controls are available in JSON recipes. See
`examples/crohn_like_recipe.json`.

A within-cell-type component can be specified as:

```json
{
  "name": "fibroblast_within",
  "kind": "within_type",
  "cell_type_key": "cell_type",
  "target": "Fibroblasts",
  "n_programs": 2,
  "expression_variance_fraction": 0.05,
  "spatial_variance_fraction": [0.25, 0.64],
  "loadings": {"genes_per_program": 40},
  "field": {"kind": "rff", "sample_key": "sample"}
}
```

## AnnData CLI

Install:

```bash
pip install tanpopo_sim-0.2.0-py3-none-any.whl
```

Simulate:

```bash
tanpopo-sim simulate input.h5ad output.h5ad \
  --recipe examples/crohn_like_recipe.json \
  --layer log1p \
  --output-layer tanpopo_sim
```

By default two layers are stored:

- `layers["tanpopo_sim_baseline"]`: actual matrix after baseline permutation and before
  injection;
- `layers["tanpopo_sim"]`: final simulated matrix.

This makes post-hoc injection characterization unambiguous.

### Characterize an injection

```bash
tanpopo-sim characterize-injection output.h5ad \
  --component fibroblast_within \
  --baseline-layer tanpopo_sim_baseline \
  --simulated-layer tanpopo_sim \
  --output-json fibroblast_strength.json
```

### Evaluate an estimated subspace

```bash
tanpopo-sim evaluate-subspace result.h5ad \
  --true-key tanpopo_sim_fibroblast_within_loadings \
  --estimated-key tanpopo_spatial_gene_loadings \
  --max-rank 30 \
  --curve-csv recovery.csv \
  --output-json recovery.json
```

The command reports recovery across the entire retained spectrum rather than assuming
the injected programs should be the first modes.

## Principal benchmark design

For each real reference dataset, sweep a grid such as:

```text
expression_variance_fraction = 0.01, 0.025, 0.05, 0.10, 0.20
spatial_variance_fraction    = 0.05, 0.25, 0.50, 0.75, 0.95
```

For every setting:

1. inject several independently sampled gene subspaces;
2. characterize their strength relative to baseline PCs/random projections;
3. run every competing method with enough retained components;
4. evaluate cumulative subspace recovery;
5. report recovery rank and AURC;
6. repeat on permuted, real, and externally simulated backgrounds.

This does not engineer the signal to be a leading Tanpopo mode. It explicitly tests
whether a known signal can be recovered in the presence of stronger naturally
occurring programs.
