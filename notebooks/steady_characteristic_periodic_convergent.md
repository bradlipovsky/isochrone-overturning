# Smooth periodic convergent basin

## Question and geometry

This calculation asks whether the asymmetric convergent basin can overturn
isochrones after removing its finite side boundaries and its piecewise-smooth
joins. One periodic cell has total length 6000 m, constant thickness 200 m,
and nominal right/left flank ratio `q`. The original piecewise-sin-squared
relief is convolved with a wrapped Gaussian and renormalized to retain the
specified relief `Delta b/H`. Bed and surface are parallel.

The horizontal boundaries are identified exactly. Gravity is vertical in the
zero-mean study. A continuation adds a mean downslope component without
changing the geometry, thereby moving continuously from two-sided convergence
toward one-directional throughflow. All cases retain the viscosity, passive
linear sliding law, and basal drag used elsewhere on PR40.

## Steady diagnostic

For each fixed Stokes solution, surface-origin characteristics carry the
equal-age tangent directly to exit. The order diagnostic is `J=dx/dx0`; no
deposited cohort or time-dependent glacier simulation is used.

The sampled horizontal velocity is integrated vertically to a periodic
streamfunction. Differentiating that one representation gives both velocity
components and their gradient. It therefore enforces incompressibility and
zero bed-normal velocity exactly, eliminating the spurious bed exits caused
by accumulating tiny Nitsche leakage along very old separatrix paths.

Accumulation zones are detected cyclically. Each is split wherever sampled
surface `u` changes sign, so source families follow the actual flow divide
rather than the geometric maximum or arbitrary coordinate seam. Positive
islands carrying less than 0.1% of total sampled accumulation are rejected as
interpolation noise. Each remaining family retains its central 90% of
accumulation flux.

## Result: smooth periodic convergence can overturn

The symmetric control (`q=1`, `Delta b/H=4`) is ordered at every smoothing
width and mesh (`J_min=1`). The `q=8`, `Delta b/H=2.75`, zero-mean cases are
also ordered in the smoothing study after flow-divide-aware source splitting.

The stronger-asymmetry `q=12`, `Delta b/H=2`,
`sigma/L_short=0.10` geometry contains a narrow negative-J source band. Its
isolated minimum remains negative on 180x24, 240x32, 320x40, 360x48, and
480x64 meshes: -2.33, -1.15, -4.33, -7.84, and -64.2. The sign is persistent,
but the magnitude is emphatically not converged.

At fixed 360x48 resolution, increasing the candidate-source count gives:

| candidates | minimum J | negative sources | negative retained flux | phase width |
|---:|---:|---:|---:|---:|
| 481 | +1.00 | 0 | 0 | 0 |
| 721 | -7.84 | 1 | 0.523% | 0 |
| 961 | -34.3 | 1 | 0.394% | 0 |
| 1441 | -7.82 | 2 | 0.523% | 0.000694 periods |

Thus this is a real but very narrow, separatrix-associated overturning band,
not a converged estimate of fold amplitude. It is not literally a zero-flux
path: the sampled negative sources carry about 0.4-0.5% of retained
accumulation flux. Nevertheless, the negative set is only one or two sampled
sources wide, and its measured width remains at most one source-grid interval.
The per-source first percentile stays `J=1`, because the band occupies less
than 1% of the retained source family.

A quarter-period coordinate translation preserves the refined sign while
moving it away from the seam: at 360x48 and 721 candidates, `J_min` changes
from -7.84 at phase 0.9986 to -2.81 at phase 0.2483. This establishes that the
sign is not a periodic-seam artifact, while the magnitude's gauge sensitivity
reinforces that it should not be interpreted quantitatively.

## Mean-slope continuation

Mean forcing exposes the flow-topology control. For the `q=8`,
`Delta b/H=2.75`, `sigma/L_short=0.10` geometry, a narrow negative band occurs
inside the counterflow regime near `alpha=0.10`. Fine, full, and ultra meshes
give `J_min=-0.0998`, -0.0980, and -0.1209, with no horizontal reversal before
the minimum and mechanism `C` near -3.6. A quarter-period translation gives
-0.314, confirming the sign away from the original seam.

This band is also sub-percent and source-grid sensitive. At fixed 360x48,
481/721/961/1441 candidate sources give `J_min=+0.274/-0.098/+0.207/-0.165`;
when detected, one source carries 0.245-0.489% of retained accumulation flux.
The first percentile remains positive (about 0.80).

Counterflow disappears between `alpha=0.13` and 0.14. Once the sampled
horizontal velocity is positive everywhere, the flow is ordered: production
values at `alpha=0.14`, 0.15, 0.20, and 0.30 are 0.747, 0.791, 0.936, and
0.924. This supports the interpretation that the fold is associated with a
narrow slow/counterflow separatrix, not bed roughness alone.

## Numerical interpretation

All 71 final v9 rows complete successfully. The reconstructed bed-normal
speed is exactly zero. On the key q=12 source-density cases, the identity
residual on the minimum-J source is 0.00020-0.00041; on the refined
`alpha=0.10` cases it is below 4e-5. These checks make the negative sign much
more credible than the discarded pre-v9 results, which mixed source branches
at the seam and accumulated basal leakage.

The defensible conclusion is therefore:

> A wrapped-Gaussian-smoothed, finite-sliding periodic basin can produce
> overturning under sufficiently asymmetric convergence. Open boundaries and
> sharp bed joins are not necessary. In the sampled periodic family the fold
> occupies a narrow, sub-percent accumulation-flux band adjacent to a flow separatrix;
> its existence is mesh- and gauge-robust at refined resolution, but its
> amplitude and physical width are unresolved.

## Outputs

Reproduce the table and figure from the repository root with:

```bash
OMP_NUM_THREADS=1 python notebooks/steady_characteristic_periodic_convergent.py --jobs 2
MPLBACKEND=Agg OMP_NUM_THREADS=1 python notebooks/plot_steady_characteristic_periodic_convergent.py
```

- `data/steady_characteristic_periodic_convergent.csv`
- `data/steady_characteristic_periodic_convergent_metadata.json`
- `figures/steady_characteristic_periodic_convergent.png`
- `figures/steady_characteristic_periodic_convergent.pdf`
- `notebooks/steady_characteristic_periodic_convergent.py`
- `notebooks/plot_steady_characteristic_periodic_convergent.py`

No manuscript or LaTeX source is modified by this analysis.
