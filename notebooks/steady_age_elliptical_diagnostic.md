# Stationary overturn diagnostic

## Question

Can a steady isochrone be shown to overturn without evolving many deposited
cohorts to selected reporting ages?

Yes.  The calculation is still nonlocal because deformation accumulates along
a flow path, but it can be posed as a stationary transport problem.  It does
not require a transient glacier simulation or an arbitrary 2 or 8 kyr cutoff.

## Calculation

Let \(\mathbf{X}(x_0,a)\) be the position at age \(a\) of ice deposited at
surface coordinate \(x_0\), and define its equal-age material tangent by

\[
\mathbf{T}
=\left.\frac{\partial\mathbf{X}}{\partial x_0}\right|_a
=(J,K).
\]

For a fixed velocity field \(\mathbf{v}=(u,w)\), age and tangent obey

\[
\mathbf{v}\mathbin{\cdot}\nabla\tau=1,
\qquad
\mathbf{v}\mathbin{\cdot}\nabla\mathbf{T}
=(\nabla\mathbf{v})\mathbf{T}.
\]

The boundary data are \(\tau=0\) and
\(\mathbf{T}=(1,s_x)\) on accumulating surface inflow.  The elliptical tests
have a horizontal surface, so \(\mathbf{T}=(1,0)\).  If
\(\dot a_0=u_s s_x-w_s>0\) is the accumulation rate at deposition,
incompressibility gives the independent identity

\[
\boxed{J=T_x=-\dot a_0\tau_z}.
\]

Thus \(J<0\), or equivalently \(\tau_z>0\), diagnoses an overturned limb.
The first-overturn age is the smallest stationary age on the \(T_x=0\)
contour.  No assumption that \(u>0\) is needed.

The implementation uses upwind discontinuous Galerkin transport for age,
tangent, source position, source accumulation and source provenance.  The
provenance field excludes ice entering through a lateral boundary.  Diagnostics
also omit two cells next to an outflow boundary, where discontinuous transport
fields are least accurate.  The reported stationary zero crossing is estimated
after projecting discontinuous \(T_x\) to continuous linear vertices, so its
age is a validation estimate rather than an exact contour of the DG field.

## Elliptical-trough result

The 7.5 m stationary solve gives the following results for the central 80% of
the accumulating source interval:

| Geometry | Retained stationary cells at provenance \(\geq0.95\) | Cohort check |
|---|---:|---:|
| \(H_b=70\) m, \(H_c=150\) m | \(\min T_x=-0.855\); projected first-\(T_x=0\) estimate 1421.8 yr | first \(J=0\) at 1416.8 yr |
| \(H_b=70\) m, \(H_c=270\) m | \(\min T_x=+0.460\); no negative retained cells; oldest retained age 33.9 kyr | \(\min J=+0.705\) through 8 kyr |

The stationary full-streamtube minimum follows every retained source to its own
exit.  It is therefore more complete than, and not numerically identical to,
the fixed-cohort minimum that stops when the first retained source exits.
The extrema and oldest represented age depend on the numerical provenance
mask.  Across thresholds 0.90--0.9999, the overturned retained minimum ranges
from -0.869 to -0.768, whereas its 1421.8 yr zero crossing is unchanged.  The
ordered retained minimum remains +0.460, while its oldest retained age ranges
from 20.4 to 37.8 kyr.  The sign classification is therefore robust; the
retained extrema and lifetime are not interpreted as converged physical
quantities.

The tangent and age-gradient fields agree without using the cohort calculation.
At 7.5 m their full-streamtube correlations are 0.9996 in the overturned case
and 0.9975 in the ordered case.  Their root-mean-square differences are 0.0186
and 0.0218, respectively, with no sign disagreements where both are resolved.
The 10 and 7.5 m meshes give the same ordered/overturned classification; their
projected stationary tangent crossing estimates are 1437.4 and 1421.8 yr.

The [diagnostic figure](../figures/steady_age_elliptical_diagnostic.png) shows
the stationary tangent Jacobian with stationary-age contours and compares it
with \(-\dot a_0\tau_z\).  The underlying values are in
[`steady_age_elliptical_diagnostic.csv`](../data/steady_age_elliptical_diagnostic.csv).

## Independent characteristic validation

An adaptive characteristic calculation integrates
\(d\mathbf{T}/da=(\nabla\mathbf{v})\mathbf{T}\) to the first physical exit.
It does not finite-difference neighboring trajectories.  Direct \(T_x\) agrees
with the existing centered-source Jacobian within \(7.73\times10^{-5}\) in four
tests, and changing the maximum adaptive step from 8 to 1 yr changes \(T_x\)
by at most \(2.33\times10^{-6}\).  These characteristic checks begin 0.05 m
below the surface to match the existing sampler, whereas the stationary
boundary data are applied at the true surface; the comparison is therefore
not identical at deposition.  Changing the counterflow velocity sampling grid
from 301 by 91 to 901 by 271 changes its comparison \(T_x\) over a range of
\(6.26\times10^{-5}\).

- The selected slow-case characteristic remains ordered to its 9.20 kyr
  surface exit, with
  \(\min T_x=0.617\).
- The selected unidirectional characteristic crosses \(T_x=0\) at 1417.5 yr
  and reaches \(\min T_x=-0.597\).
- The selected early-exit characteristic remains ordered to its 1.68 kyr exit,
  with
  \(\min T_x=0.109\).
- The selected counterflow characteristic crosses \(T_x=0\) at 1458.4 yr and
  reaches \(\min T_x=-0.970\).  Its selected path has
  \(\min u=-0.0436\) m yr\(^{-1}\)
  and reverses horizontal direction twice, so the result does not rely on a
  downstream-\(x\) coordinate.

Details and machine-readable outputs are in
[`steady_tangent_elliptical_validation.md`](steady_tangent_elliptical_validation.md)
and the corresponding CSV files in `data/`.

## Interpretation and limits

The existence question is therefore independent of a selected plot time:
solve the stationary fields once over each open, surface-fed streamtube and
test whether \(T_x\) becomes negative before physical exit.  A 2 or 8 kyr map
still answers the different question of whether overturning has occurred by
that age.

There is no purely local bed-geometry or strain threshold.  The stationary
transport equation encodes the accumulated upstream deformation that controls
the sign of \(J\).

A single-valued steady age requires a unique surface origin.  Closed
streamlines are incompatible with
\(\mathbf{v}\mathbin{\cdot}\nabla\tau=1\), exact stagnation points are
degenerate, and source-basin separatrices can be discontinuous.  Such regions
must be diagnosed separately rather than filled by numerical diffusion.
Horizontal counterflow alone is not a mathematical limitation.  However, the
Eulerian DG comparison above uses the two single-basin cases; the adaptive
characteristic calculation is the reliable counterflow check until numerical
mixing across the counterflow source-basin boundary is resolved.

The model predicts that an observed steady age field is overturned wherever
age increases upward at fixed horizontal position, \(\tau_z>0\).  In a
reconstructed velocity field, the same prediction is \(T_x<0\) before the
surface-fed characteristic exits the glacier.
