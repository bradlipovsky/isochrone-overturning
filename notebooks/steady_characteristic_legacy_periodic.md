# Lifetime audit of the legacy periodic beds

## Question

The original 15-case periodic screen stopped every cohort at 80 kyr while
23--71 markers were still active.  This audit asks whether the three cases
closest to loss of order, or any of the other 12 cases, overturn before each
surface-connected path actually exits.

## Method

The bed definitions are imported unchanged from
`realistic_geometry_parameter_search.py`.  Each finite-thickness profile is
placed in its original periodic cell beneath a flat surface.  Gravity drives
the same linear-viscous Stokes model and the bed has the same passive linear
sliding law, with

\[
\beta=10^5\ {\rm Pa\,yr\,m^{-1}}.
\]

The calculation otherwise uses the current steady diagnostic developed for
the Fourier beds.  Sources begin exactly on every accumulating part of the
surface; the central 80% of accumulation flux in each connected interval is
retained.  Along each steady characteristic it integrates

\[
\dot{\boldsymbol X}=\boldsymbol v(\boldsymbol X),\qquad
\dot{\boldsymbol T}=\nabla\boldsymbol v(\boldsymbol X)\boldsymbol T,
\qquad \boldsymbol T(0)=(1,0),
\]

and diagnoses equal-age order as \(J=T_x\).  Each path stops independently
at its first physical surface or bed exit.  The uppermost 5% of local ice
thickness is excluded only on the final ablating side for the primary
diagnostic; `raw minimum J to exit` retains that thin exit layer.

The production calculation uses a 72 by 18 Stokes mesh, a 481 by 161
terrain-following velocity sampler, 321 surface candidates, and a 10 yr
Runge--Kutta step.  The nominal 1 Myr ceiling is only a safety cap: all paths
exit by 322 kyr.  Every case passes the surface-signal/leakage screen by a
factor of at least 853.

## Results

All 15 cases remain ordered for their complete retained lifetimes.  All
retained paths also keep \(u>0\), so none experiences horizontal reversal or
counterflow.  The closest primary result is \(J=0.467\) for the basin between
narrow riegels.  The steep asymmetric trough reaches nearly the same primary
minimum, \(J=0.472\), and the lowest raw value, \(J=0.241\), in the shallow
ablating-side exit layer.

For a positive-speed path the exact steady identity is

\[
J={u\over u_0}(1+C).
\]

Overturn therefore requires \(C<-1\) in the absence of reversal.  The most
negative sampled values are only \(-0.704\) in production and \(-0.714\) in
the refined audit.  These geometries create substantial distortion but not
enough accumulated adverse shear to fold an isochrone.

| # | Periodic geometry | Primary min \(J\) | Raw min \(J\) | Min \(C\) | Min path \(u\) (m/yr) | Last exit (kyr) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | asymmetric trough 15m 250x75 | 0.622 | 0.489 | -0.406 | 0.0011 | 318.4 |
| 1 | asymmetric trough 25m 250x75 | 0.650 | 0.524 | -0.374 | 0.0021 | 193.7 |
| 2 | asymmetric trough 50m 250x75 | 0.730 | 0.628 | -0.284 | 0.0064 | 87.7 |
| 3 | reverse trough 25m 75x250 | 0.856 | 0.790 | -0.332 | 0.0037 | 91.5 |
| 4 | steep asymmetric trough 10m | 0.472 | 0.241 | -0.704 | 0.0013 | 160.9 |
| 5 | steep asymmetric trough 25m | 0.574 | 0.362 | -0.580 | 0.0024 | 110.0 |
| 6 | steep reverse trough 25m | 0.695 | 0.560 | -0.519 | 0.0015 | 208.7 |
| 7 | narrow overdeepening 25m | 0.811 | 0.652 | -0.349 | 0.0025 | 97.4 |
| 8 | deep valley 25m | 0.640 | 0.436 | -0.473 | 0.0017 | 250.8 |
| 9 | overdeepening and exit sill | 0.627 | 0.492 | -0.341 | 0.0041 | 134.4 |
| 10 | overdeepening and entry sill | 0.964 | 0.924 | -0.292 | 0.0057 | 97.2 |
| 11 | basin between narrow riegels | 0.467 | 0.318 | -0.682 | 0.0020 | 111.4 |
| 12 | basin between broad riegels | 0.637 | 0.497 | -0.501 | 0.0036 | 77.2 |
| 13 | trough sill doublet | 0.604 | 0.474 | -0.300 | 0.0041 | 120.7 |
| 14 | two asymmetric overdeepenings | 0.672 | 0.544 | -0.332 | 0.0023 | 321.7 |

## Convergence of the three legacy closest cases

The 48 by 12, 72 by 18, and 96 by 24 rows change only the Stokes mesh.  The
last row at 96 by 24 also increases the sampler from 481 by 161 to 641 by
241, increases surface candidates from 321 to 481, and halves the trajectory
step from 10 to 5 yr.  The final trace refinement changes primary \(J\) by
0.0002, 0.0076, and 0.0002 in the three cases.  Every sign and classification
is stable.

| Geometry | Resolution | Primary min \(J\) | Raw min \(J\) | Min \(C\) | Max identity residual |
|---|---|---:|---:|---:|---:|
| steep asymmetric trough 10m | 48x12 | 0.4319 | 0.1955 | -0.7465 | 0.0118 |
|  | 72x18 | 0.4719 | 0.2409 | -0.7042 | 0.0045 |
|  | 96x24 | 0.4668 | 0.2338 | -0.7138 | 0.0019 |
|  | 96x24 + trace refinement | 0.4666 | 0.2334 | -0.7143 | 0.0029 |
| steep reverse trough 25m | 48x12 | 0.7327 | 0.5753 | -0.5221 | 0.0049 |
|  | 72x18 | 0.6947 | 0.5599 | -0.5186 | 0.0041 |
|  | 96x24 | 0.7053 | 0.5475 | -0.5250 | 0.0023 |
|  | 96x24 + trace refinement | 0.6978 | 0.5473 | -0.5254 | 0.0047 |
| basin between narrow riegels | 48x12 | 0.4548 | 0.3332 | -0.6651 | 0.0050 |
|  | 72x18 | 0.4667 | 0.3179 | -0.6818 | 0.0018 |
|  | 96x24 | 0.4643 | 0.3423 | -0.6575 | 0.0014 |
|  | 96x24 + trace refinement | 0.4642 | 0.3421 | -0.6577 | 0.0015 |

Among these three cases, the largest horizontal RK4 step is 0.012% of one
period in production and 0.006% after trace refinement.  The refined
streamfunction-identity residual is below 0.5% in all three cases.  No
validation path exits through the bed, hits the safety cap, or reverses
horizontal velocity.

## Interpretation and limits

This replaces the old time-bounded statement with a lifetime statement for
the retained surface-connected flux: the legacy periodic troughs, riegels,
and sills do not overturn under this model.  Their failure to fold is
consistent with the one- and two-mode Fourier results.  Even sharp and
strongly asymmetric periodic roughness is insufficient here when the flow
remains one-directional.

The numerical values should not be compared one-for-one with the old 80 kyr
centered-marker results.  This audit starts on the exact surface, evolves the
variational tangent rather than a finite source difference, stops paths at
individual exits, and uses the current Stokes boundary-penalty convention.
The qualitative no-fold conclusion is unchanged, but is now based on full
retained path lifetimes.

The conclusion is limited to two-dimensional, linear-viscous, steady flow
with diagnosed surface mass balance, a flat surface in flow-following
coordinates, one periodic cell, the specified linear sliding coefficient,
and the central 80% accumulation-flux convention.  It does not exclude folds
arbitrarily close to accumulation separatrices or under a different sliding,
rheological, or surface-forcing regime.

## Reproduction

From the repository root in the Firedrake environment:

```bash
python notebooks/steady_characteristic_legacy_periodic_sweep.py --mode production --jobs 2
python notebooks/steady_characteristic_legacy_periodic_sweep.py --mode validation --jobs 2
```

Outputs are
`data/steady_characteristic_legacy_periodic_sweep.csv`, its JSON metadata,
and the analogous `steady_characteristic_legacy_periodic_validation` pair.
