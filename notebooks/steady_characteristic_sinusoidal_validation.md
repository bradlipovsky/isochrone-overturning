# Sinusoidal steady-characteristic validation

This audit is deliberately smaller than the production parameter sweep.  It
uses exact-surface characteristic sources and changes one numerical control at
a time around the lowest-`J` successful production geometry. Every row records
the diagnostic-method version so results produced with different source or
surface-buffer conventions cannot be mixed. A separate
regime survey samples the analytic flat-bed limit, a small-amplitude bed, map
edges at `lambda/H = 1` and 12, asymptotic extensions at `lambda/H = 0.5` and
24, and the approach to the singular thin-neck limit at `A/H = 0.85`, 0.90,
and 0.95 for `lambda/H = 3`. The `A/H = 0.95` probe is repeated on a 96 x 36
mesh so censoring or mesh sensitivity is not mistaken for physical evidence.

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

The completed production grid contains 720 successful numerical cases and 60
analytic flat-bed cases. The lowest production value is `J = 0.642810` at
`H = 50 m`, `A/H = 0.85`, and `lambda/H = 3`; its raw minimum to surface exit
is 0.442762. The long safety age used for this slow case is a numerical ceiling,
not a modeled duration: every retained path exits by 363.9 kyr. All primary
and raw production minima remain positive. The minimum horizontal path speed
over the production grid is `1.11e-4 m/yr`, and no path reverses horizontal
direction.

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
`1.84e-6`; the largest discrete weak-form residual is `2.81e-13`. These checks
are much smaller than the mesh-resolution uncertainty. The calculation does
not establish a mesh-converged value of the positive minimum, but it does
establish its sign with a margin of about 0.62 in the tested production range.

### Asymptotic probes

Short waves recover the ordered slab limit. At `H = 160 m` and `A/H = 0.65`,
decreasing `lambda/H` from the map edge 1 to the out-of-grid value 0.5 raises
the minimum from 0.843329 to 0.972268. The long-wave probes are also nearly
ordered: the raw minima are 0.989827 at `lambda/H = 12` and 0.983909 at
`lambda/H = 24`. The latter point is not monotonically closer to one, so it
supports a near-flat long-wave regime but not a precise asymptotic rate. Both
extension cases are successful, retain positive horizontal velocity, and have
zero path reversals.

| Probe at `H = 160 m` | `A/H` | `lambda/H` | Minimum `J` | Raw minimum `J` | Minimum path `u` (m/yr) |
|---|---:|---:|---:|---:|---:|
| Formal flat bed | 0 | 4 | 1 | 1 | -- |
| Small amplitude | 0.05 | 4 | 0.948637 | 0.948637 | 0.0686 |
| Short-wave extension | 0.65 | 0.5 | 0.972268 | 0.972268 | 0.00345 |
| Long-wave extension | 0.65 | 24 | 0.984037 | 0.983909 | 0.117 |
| Thin-neck production edge | 0.85 | 3 | 0.643929 | 0.444095 | 0.00290 |
| Thin-neck extension | 0.90 | 3 | 0.609519 | 0.384637 | 0.00234 |
| Thin-neck extension | 0.95 | 3 | 0.580291 | 0.331584 | 0.00202 |
| Thin-neck extension, 96 x 36 mesh | 0.95 | 3 | 0.553517 | 0.299773 | 0.00209 |

The thin-neck sequence continues downward as `A/H` approaches one, but does
not approach overturning over the tested interval. Refining the `A/H = 0.95`
case changes the primary minimum by 0.0268, or 4.6% of the production-mesh
value. Its magnitude is therefore mesh-sensitive. Both resolutions remain
positive, have positive horizontal velocity, and show no path reversal. This
probe cannot justify extrapolating through the singular `A/H -> 1` limit.

All 21 numerical audit rows end in successful surface exits; the remaining
row is the analytic flat-bed limit. No audit row is censored or fails, and none
contains counterflow.
