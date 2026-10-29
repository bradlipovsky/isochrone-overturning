# Steady variational-tangent validation

## Calculation

The full-Stokes velocity is solved once for each fixed geometry.  Individual surface-origin characteristics then carry the material tangent by `dT/dt = grad(v) T`.  This is a steady-flow sensitivity calculation: particle age advances, but the glacier does not evolve.  Each characteristic ends at its first bed, surface, or side exit; 120 kyr is only a numerical safety cap.

The deposition line is horizontal at `z=-0.05 m`, so `T(0)=(1,0)`.  Velocity and its analytical bilinear derivative use the same 601 by 181 bed-following sampling grid as the existing material map.  Comparisons use only the fixed central 80% source band.

## Result

The direct tangent and existing centered-source Jacobian differ by at most 7.73e-05 at the four comparison points.  Changing the adaptive maximum step from 8 to 1 yr changes T_x by at most 2.33e-06.  At the smallest tested source half-spacing, centered differences agree with T_x to at most 8.08e-06.  Changing the counterflow velocity sampling grid from 301 by 91 to 901 by 271 changes its comparison T_x over a range of 6.26e-05.

- selected characteristic, ordered slow flow: minimum T_x=0.6174; no T_x=0 crossing before exit; exit at 9197.3 yr (surface); minimum path u=0.003581 m/yr; horizontal velocity reversals=0.
- selected characteristic, overturned unidirectional flow: minimum T_x=-0.5966; first T_x=0 at 1417.5 yr; exit at 1988.1 yr (surface); minimum path u=0.009663 m/yr; horizontal velocity reversals=0.
- selected characteristic, ordered early exit: minimum T_x=0.1085; no T_x=0 crossing before exit; exit at 1676.0 yr (surface); minimum path u=0.01695 m/yr; horizontal velocity reversals=0.
- selected characteristic, overturned counterflow: minimum T_x=-0.9704; first T_x=0 at 1458.4 yr; exit at 3382.8 yr (surface); minimum path u=-0.0436 m/yr; horizontal velocity reversals=2, at 277.7 and 3104.8 yr.

The calculation took 121.8 s on this machine, including four Stokes solves and all convergence integrations.

## Interpretation and limitations

A negative `T_x` is the same loss of horizontal source order detected by neighboring trajectories, but it does not require selecting a reporting age or differencing the last surviving markers.  The method still integrates cumulative deformation along steady characteristics; it removes transient glacier simulation, not material history.

The exit surface is the conservative inset mask used by the existing elliptical calculation (0.994 of each side and sigma=0.006--0.99994), because the sampled velocity grid is inset from the curved finite-element boundary.  The validation checks transport and source-spacing convergence on the production 10 m Stokes mesh; it does not replace the existing Stokes mesh-convergence tests.  The companion stationary Eulerian solve treats open surface-fed streamtubes; closed recirculation and exact stagnation still require separate treatment.

## Machine-readable outputs

- `data/steady_tangent_elliptical_validation.csv`
- `data/steady_tangent_elliptical_convergence.csv`
- `data/steady_tangent_elliptical_validation_metadata.json`
