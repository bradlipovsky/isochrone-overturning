# Sinusoidal steady-characteristic validation

This audit is deliberately smaller than the production parameter sweep.  It
uses exact-surface characteristic sources and changes one numerical control at
a time around the lowest-`J` successful production geometry. Every row records
the diagnostic-method version so results produced with different source or
surface-buffer conventions cannot be mixed. A separate
regime survey samples the analytic flat-bed limit, a small-amplitude bed,
short waves down to `lambda/H = 1/12`, a long-wave extension at 24, and the
approach to the singular thin-neck limit at `A/H = 0.85`, 0.90, and 0.95.
The `A/H = 0.95` family is evaluated at `lambda/H = 1/2`, `1/4`, `1/8`, and
`1/12`; the shortest case is repeated on a 128 x 80 mesh. Short-wave meshes
and samplers cluster points at both vertical boundaries.

The primary minimum includes each depositing source at `J(0) = 1` and excludes
only the upper 5% of local thickness beneath the ablating surface. The raw
minimum to surface exit is retained as a secondary check. Every numerical case
uses the central 80% of accumulation flux unless the retained-flux fraction is
the quantity being varied explicitly.

The validation driver compares:

- the production Stokes mesh and 0.75x/1.5x horizontal, 0.75x/1.5x
  vertical-resolution variants (48 x 18, 64 x 24, and 96 x 36 when the
  focus geometry uses 64 horizontal cells);
- 50, 25, and 12.5 yr characteristic steps;
- coarser and finer velocity-sampling grids;
- 121, 241, and 481 candidate surface sources;
- central 70%, 80%, and 90% accumulation-flux retention; and
- a quarter-wavelength translation of the bed as a periodic gauge test.

The focus reference and in-grid regime cases reuse the exact production rows.
All perturbed settings and out-of-grid asymptotic probes are independent audit
solves; the CSV identifies the origin of every row.

Run from the repository root with the Firedrake environment:

```bash
/Users/bradlipovsky/firedrake/bin/python \
  notebooks/steady_characteristic_sinusoidal_validation.py
```

Results are written to
`data/steady_characteristic_sinusoidal_validation.csv`.  The interpretation
and numerical differences are given below.

## Results

The completed production grid contains 750 successful numerical cases, 80
analytic flat-bed cases, and 210 cells whose surface signal is below the
numerical-leakage criterion. The lowest resolved production value is
`J = 0.642810` at
`H = 50 m`, `A/H = 0.85`, and `lambda/H = 3`; its raw minimum to surface exit
is 0.442762. The long safety age used for this slow case is a numerical ceiling,
not a modeled duration: every retained path exits by 363.9 kyr. All primary
and raw production minima remain positive. The minimum horizontal path speed
over the production grid is `1.11e-4 m/yr`, and no path reverses horizontal
direction.

At that minimum, the exact steady factorization gives
`u/u_source = 1.1159` and `C = -0.4240`, so the speed change alone would
stretch the tangent. Accumulated adverse vertical shear supplies the
compression. The maximum relative residual of the independent streamfunction
identity is `9.5e-4` for this case.

### Numerical sensitivity at the production minimum

The Stokes mesh is the largest numerical uncertainty in the magnitude of the
minimum. All tested resolutions remain far from `J = 0`.

| Control | Tested minimum-`J` range | Full span | Span/reference |
|---|---:|---:|---:|
| Stokes mesh, 48 x 18 to 96 x 36 | 0.620275-0.654840 | 0.034564 | 5.38% |
| Characteristic step, 12.5-50 yr | 0.642788-0.642810 | 0.000022 | 0.0034% |
| Velocity sampler, 241 x 81 to 481 x 181 | 0.640520-0.643747 | 0.003228 | 0.50% |
| Candidate sources, 121-481 | 0.640667-0.645997 | 0.005330 | 0.83% |
| Retained accumulation flux, 70%-90% | 0.642810-0.642810 | 0 | 0% |

The raw minimum changes from 0.448873 at 70% flux retention to 0.442762 at
80%-90%, but the primary minimum is identical. Thus the ablation-surface
buffer prevents the slow separatrix tail from controlling the stated result.
A quarter-wave translation changes the minimum by 0.001664 (0.26%) and moves
the bed-relative minimizing-source phase by 0.00104 wavelength. This residual
is small finite-grid sampling error rather than a physical phase dependence.

The largest sampled surface-mass-balance closure error among audit cases is
`6.0e-6`; the largest discrete weak-form residual is `7.7e-12`. These checks
are much smaller than the mesh-resolution uncertainty. The calculation does
not establish a mesh-converged value of the positive minimum, but it does
establish its sign with a margin of about 0.62 in the tested production range.

### Asymptotic probes

Short waves recover the ordered slab limit. At `H = 160 m` and `A/H = 0.65`,
decreasing `lambda/H` from 1 to `1/2`, `1/4`, and `1/8` raises the minimum
from 0.843329 to 0.973107, 0.999634, and 0.9999999. At `1/12` no depositing
surface source remains above the absolute sampling tolerance. The `1/4`
surface signal is already only 9.96 times the RMS basal leakage, and the
shorter values are numerical probes rather than production-quality
measurements. The long-wave probes are also nearly
ordered: the raw minima are 0.989827 at `lambda/H = 12` and 0.983909 at
`lambda/H = 24`. The latter point is not monotonically closer to one, so it
supports a near-flat long-wave regime but not a precise asymptotic rate. Both
extension cases are successful, retain positive horizontal velocity, and have
zero path reversals.

| Probe at `H = 160 m` | `A/H` | `lambda/H` | Minimum `J` | Raw minimum `J` | Minimum path `u` (m/yr) |
|---|---:|---:|---:|---:|---:|
| Formal flat bed | 0 | 4 | 1 | 1 | -- |
| Small amplitude | 0.05 | 4 | 0.948637 | 0.948637 | 0.0686 |
| Short-wave extension | 0.65 | 0.5 | 0.973107 | 0.973105 | 0.00348 |
| Short-wave extension | 0.65 | 0.25 | 0.999634 | 0.999634 | 0.00314 |
| Short-wave extension | 0.65 | 0.125 | 1.000000 | 1.000000 | 0.00284 |
| Long-wave extension | 0.65 | 24 | 0.984037 | 0.983909 | 0.117 |
| Thin-neck production edge | 0.85 | 3 | 0.643929 | 0.444095 | 0.00290 |
| Thin-neck extension | 0.90 | 3 | 0.609519 | 0.384637 | 0.00234 |
| Thin-neck extension | 0.95 | 3 | 0.580291 | 0.331584 | 0.00202 |
| Thin-neck extension, 96 x 36 mesh | 0.95 | 3 | 0.553517 | 0.299773 | 0.00209 |
| Thin-neck short wave | 0.95 | 0.5 | 0.835920 | 0.411054 | $1.48\times10^{-4}$ |
| Thin-neck short wave | 0.95 | 0.25 | 0.800755 | 0.667148 | $9.29\times10^{-5}$ |
| Thin-neck short wave | 0.95 | 0.125 | 0.860199 | 0.860171 | $7.44\times10^{-5}$ |
| Thin-neck short wave | 0.95 | 0.0833 | 0.939373 | 0.939352 | $6.46\times10^{-5}$ |
| Thin-neck short wave, 128 x 80 mesh | 0.95 | 0.0833 | 0.946596 | 0.946584 | $6.82\times10^{-5}$ |
| Thin-neck short wave, refined | 0.95 | 0.5 | 0.758426 | 0.179392 | $4.91\times10^{-5}$ |
| Thin-neck short wave, refined | 0.95 | 0.25 | 0.812725 | 0.729462 | $1.04\times10^{-4}$ |

The thin-neck sequence continues downward as `A/H` approaches one, but does
not approach overturning over the tested interval. Refining the `A/H = 0.95`
case changes the primary minimum by 0.0268, or 4.6% of the production-mesh
value. Its magnitude is therefore mesh-sensitive. Both resolutions remain
positive, have positive horizontal velocity, and show no path reversal. This
probe cannot justify extrapolating through the singular `A/H -> 1` limit.
The thin-neck short-wave family has a separate positive trough near
`lambda/H = 1/4`, then returns toward one. Refining the `1/12` case changes
the primary minimum from 0.9394 to 0.9466; the sign is insensitive even though
the magnitude is not fully converged.  At `lambda/H = 1/2`, the more demanding
mesh/sampler/source audit changes the primary value from 0.8359 to 0.7584 and
the raw value from 0.4111 to 0.1794.  At `lambda/H = 1/4`, the corresponding
primary values are 0.8008 and 0.8127.  Thus the positive sign is stable while
the near-exit raw magnitude at `1/2` remains resolution-sensitive.

Thirty numerical audit rows end in successful surface exits, one
`A/H = 0.65`, `lambda/H = 1/12` row has no resolvable depositing source, and
one row is the analytic flat-bed limit. No audit row contains retained-path
counterflow or a horizontal reversal.
