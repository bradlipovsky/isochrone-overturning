# Additional two-dimensional steady-geometry survey

This note collects the four follow-up tests performed with the direct steady
characteristic diagnostic.  Every calculation uses the existing
linear-viscous Stokes model and passive linear basal sliding law.  No
time-dependent isochrone simulation, three-dimensional model, or manuscript
source is involved.

## 1. Smooth periodic convergent basin

Replacing the finite asymmetric basin by a smooth, exactly periodic
wrapped-Gaussian geometry does **not** remove overturning altogether.  The
strongly asymmetric case (`q=12`, `Delta b/H=2`) contains a very narrow
negative-`J` band next to a flow separatrix.  Its sign persists from 180x24
through 480x64 meshes and after a quarter-period coordinate translation.
At fixed 360x48 resolution, 721--1441 candidate sources place 1--2 sources in
the band, carrying about 0.4--0.5% of retained accumulation flux.  The
minimum value ranges from order one to order 100 in magnitude and is not
converged; only the existence of the narrow band is defensible.

A mean-slope continuation shows why this occurs.  Sub-percent folds can
appear while opposing flow branches and a separatrix remain.  Once the mean
slope is large enough to make horizontal flow one-directional everywhere
(near `alpha=0.14` in the sampled `q=8` family), every case is ordered.

Details: `steady_characteristic_periodic_convergent.md`.

## 2. Lifetime audit of the 15 legacy periodic beds

All retained surface-connected paths were followed to their individual
exits rather than stopped at 80 kyr.  Every one of the 15 trough, sill,
riegel, and overdeepening geometries remains ordered; all paths exit by
322 kyr.  The smallest primary value is `J=0.467`, and refinement preserves
the sign.  None of these cases develops horizontal counterflow.

Details: `steady_characteristic_legacy_periodic.md`.

## 3. Smooth localized two-scale obstacle

An analytic broad trough plus a narrow rounded riegel does overturn, but only
in an extreme near-pinching regime.  The robust adverse example has
`A/H=0.90`, `L/H=4`, and imposed-kernel FWHM `ell/H=1/12`; its minimum gap is
0.10H and maximum bed slope is 17.2.  On the quasi-uniform refined mesh,
`J_min=-0.271`; 14 of 138 retained sources overturn and carry 0.924% of
retained accumulation flux.  Every negative path undergoes horizontal
velocity reversal.  Wider or lower-amplitude obstacles remain ordered in the
resolved tests, apart from one borderline width whose threshold is not yet
resolved.

Details: `steady_characteristic_localized_packet_report.md`.

## 4. Smooth icefall-to-tongue transition

The full 80-cell sweep over transition length, steep/gentle slope ratio, and
thick/thin ratio remains ordered.  The minimum production value is
`J=0.7102`, with a refined value of `0.7074`; an additional sharper stress
test gives `J=0.8370`.  The layer compression is caused mainly by longitudinal
deceleration (`u/u0=0.679` at the minimum), while the adverse-shear factor is
small and no path reverses.

Details: `steady_characteristic_icefall_transition.md`.

## Synthesis

The combined result is more specific than “a sharp boundary is required.”
An analytic bed can fold layers, so geometric nondifferentiability is not
necessary.  In every robust or sign-persistent folding case found here,
however, the dynamics contain a stagnation/counterflow separatrix or local
velocity reversal.  Smooth, one-directional periodic roughness and severe
longitudinal compression distort layers but preserve their order.  The
physically useful discriminator in this survey is therefore flow topology,
not smoothness by itself.
