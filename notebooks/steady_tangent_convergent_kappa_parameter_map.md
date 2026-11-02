# Steady convergent-basin bed--surface coupling sweep

## Parameterization

The calculation fixes total length at 6 km and domain-mean vertical thickness at 200 m, so L/H=30 and beta H/eta=0.08.  For the piecewise-sine flank shape f with exact mean 1/2,

\[b=\Delta b f,\qquad s=H+(1-\kappa)\Delta b/2+\kappa\Delta b f.\]

Thus kappa=Delta s/Delta b=1 is the previous constant-thickness geometry.  Lower kappa flattens the surface while preserving ice area: the confluence thickens and the divides thin.  Positive thickness requires

\[h_{\min}/H=1-|1-\kappa|(\Delta b/H)/2>0.\]

The kappa=0 endpoint is exactly hydrostatic and stagnant.  It has no depositing source family, so steady J is undefined; the solved maps begin at kappa=0.10.

Every positive-thickness case uses the central 5--95% of depositing surface flux and a local two-vertical-cell ablation buffer.  The validated time step scales as 1/kappa.  Low-kappa safety caps were enlarged as 240 kyr/kappa; the uncensored reused kappa=1 baseline retains its validated 120 kyr cap.

## Result

The grid has 880 cells: 715 resolved positive-thickness cases and 165 nonpositive-thickness geometries.  No trajectory reaches the safety cap.

| kappa | valid | overturned | broad-source overturned | unresolved | ordered | minimum J |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 88 | 16 | 15 | 7 | 65 | -1.59416 |
| 0.25 | 110 | 29 | 28 | 7 | 74 | -2.38359 |
| 0.50 | 165 | 71 | 69 | 12 | 82 | -6.09414 |
| 0.75 | 176 | 76 | 68 | 5 | 95 | -9.37032 |
| 1.00 | 176 | 62 | 50 | 1 | 113 | -7.30852 |

The common positive-thickness rectangle Delta b/H<=2 contains the same 88 cells in every slice:

| kappa | overturned | unresolved | ordered |
|---:|---:|---:|---:|
| 0.10 | 16 | 7 | 65 |
| 0.25 | 16 | 3 | 69 |
| 0.50 | 15 | 4 | 69 |
| 0.75 | 13 | 1 | 74 |
| 1.00 | 10 | 0 | 78 |

On this paired domain, reducing kappa makes J smaller in nearly every cell relative to kappa=1:

| kappa | cells with J below kappa=1 | paired cells | median Delta J |
|---:|---:|---:|---:|
| 0.10 | 85 | 88 | -0.503111 |
| 0.25 | 88 | 88 | -0.461075 |
| 0.50 | 88 | 88 | -0.36056 |
| 0.75 | 88 | 88 | -0.189564 |

This is not a strict monotonic law: 72/88 paired cells are monotone across all five sampled kappa values.  The comparison to kappa=1 is nevertheless systematic.

For 0.10<=kappa<=1, flattening the surface therefore does not remove the folding mechanism in this controlled family.  Dimensional speed and required mass balance weaken approximately with kappa, while residence ages grow approximately as 1/kappa.  Those two scalings largely cancel in accumulated steady strain; the J shift instead reflects the changing normalized strain and streamline geometry, the flatter initial surface tangent and the divide-to-confluence thickness redistribution.  This experiment does not separate those effects, and the one-parameter kappa coupling still forces the bed and surface extrema to coincide on both flanks.

The full-map counts are not directly comparable because the low-kappa slices lose high-relief cells to nonpositive thickness. The paired Delta b/H<=2 comparison is the controlled result.  Cells nearest the positivity boundary have only 12.5--20 m minimum divide thickness; their negative signs validate, but they are a thin-divide limit.  The plotted isolated minimum asks whether any retained source overturns; the broad-source counts show that exact onset remains sensitive to source width.

## Targeted validation

| scenario | minimum J across variants | maximum J across variants | variants |
|---|---:|---:|---:|
| high-relief sampler retry | -1.79971 | -1.75378 | 3 |
| kappa 0.75 extreme | -9.37032 | -9.17691 | 2 |
| low-kappa onset | -0.0638013 | -0.0632611 | 4 |
| low-kappa strong | -1.02581 | -0.91946 | 3 |
| thin positive-thickness edge | -1.59416 | -1.55694 | 3 |

All 15 validation variants pass the mechanical diagnostics and preserve the sign, including the low-kappa onset, the thin positive-thickness edge and the cells requiring a more interior velocity sampler.

Point-location retries were required in 38 of 715 resolved cells; 32 use interior sampler bounds.  Most offsets are slight; the widest bounds are sigma=0.02 to 0.95.  5 cells have an upper sampler bound below the sigma=0.995 source level and therefore extrapolate there; their negative signs are more robust than their most saturated minimum-J magnitudes.  Mesh and sampler variants preserve the signs in the targeted retry scenarios.

## Outputs

- `figures/steady_tangent_convergent_kappa_parameter_map.png`
- `figures/steady_tangent_convergent_kappa_parameter_map.pdf`
- `data/steady_tangent_convergent_kappa_parameter_map.csv`
- `data/steady_tangent_convergent_kappa_validation.csv`

No manuscript source is modified by this analysis.
