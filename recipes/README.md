# Figure 1 simulation recipes

These recipes target the narrowed Tanpopo Figure 1 claim:
**recovery should depend on spatial organization, not merely expression variance.**

Observed nearest-neighbour spacing is approximately 7 coordinate units.

## Core grid

`core/` contains the primary factorial benchmark:

- expression variance fraction: 0.005, 0.01, 0.025, 0.05
- spatial variance fraction: 0.00, 0.25, 0.50, 0.75
- two injected fibroblast programs
- 40 disjoint genes per program
- random Fourier spatial fields
- length scale: 70 units (~10 nearest-neighbour spacings)
- 128 random Fourier features
- orthogonalized spatial driver
- seed placeholder: 1

Each spatial fraction is applied equally to both programs in the core grid. This keeps
the truth rank fixed and makes recovery curves interpretable as a function of a single
spatial-strength variable.

Two background regimes are provided:

- `core/permuted/`: whole expression vectors permuted within `CellAnnotation.Level0`
  before injection. This removes most within-cell-type endogenous spatial structure
  while preserving cell-type composition and realistic expression covariance.
- `core/real/`: injection directly into the real expression background. This measures
  recovery in competition with endogenous spatial programs.

The core grid contains 32 recipes.

## Heterogeneous pair

`heterogeneous/` uses spatial variance fractions `[0.25, 0.64]`, matching the
diagnostic setting already used in development. This is useful for showing that
Tanpopo need not recover the two injected programs at identical ranks.

## Spatial-scale sensitivity

`scale/` holds expression variance at 0.025 and spatial variance at 0.50 while varying
the RFF length scale:

- 21 units (~3 neighbour spacings)
- 42 units (~6 neighbour spacings)
- 70 units (~10 neighbour spacings)
- 140 units (~20 neighbour spacings)

This is best treated as a supplementary robustness experiment rather than part of the
main Figure 1 factorial grid.

## Field-family sensitivity

`field_family/` holds expression variance at 0.025 and spatial variance at 0.50 and
uses gradient, hotspot, or stripe drivers. Together with the RFF recipes, this checks
that recovery is not specific to the same smooth-field family used in the main grid.

## Seed handling

Every file currently uses:

```json
"seed": 1
```

For replicate runs, change only this value. Keep every other field fixed within a
configuration so that between-seed variation represents simulation stochasticity
rather than a changed experimental condition.

## Example

```bash
tanpopo-sim simulate RUN1_SLIDE1_S2.h5ad RUN1_SLIDE1_S2_sim.h5ad \
  --recipe figure1_recipes/core/permuted/expr0.025_spatial0.50.json
```

Then run global and fibroblast-conditioned Tanpopo exactly as in the manuscript plan.
