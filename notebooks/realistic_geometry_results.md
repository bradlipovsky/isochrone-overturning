# Steady overturning in additional glacier geometries

## Question and diagnostic

A single admissible counterexample disproves a universal no-overturning claim.
The calculations here ask a different question: does the passive mechanism
operate in more than one special geometry?

All models solve two-dimensional, Newtonian full-Stokes flow with density
$917\,\mathrm{kg\,m^{-3}}$, viscosity
$2.5\times10^8\,\mathrm{Pa\,yr}$, and mean surface slope $0.05$ rad. Every
bed is impermeable and obeys the same passive linear law

$$
u_t=C\tau_b, \qquad
C=10^{-5}\,\mathrm{m\,yr^{-1}\,Pa^{-1}}.
$$

No boundary prescribes velocity. The predicted observable is the
surface-deposition-map Jacobian
$J=\left.\partial x/\partial x_0\right|_\tau$. A negative value means that an
initially ordered isochrone has passed through vertical. Centered source
derivatives omit two points at every valid interval end.

## Finite elliptical troughs

The clean additional counterexample is a narrow elliptical trough. Its
vertical radius is 250 m, horizontal radius is 175 m, maximum thickness is
150 m, and both open cuts have 70 m of ice. The surface is traction free, the
cuts carry hydrostatic traction, and the bed retains the same sliding law.

| Target spacing (m) | Cells | $J_{\min}$ | First $J<-0.2$ (yr) | Valid sources at threshold | Source span (m) | Minimum path $u$ (m yr$^{-1}$) |
|---:|---:|---:|---:|---:|---:|---:|
| 15.0 | 382 | -0.95951 | 1582 | 383 | 112.74 | 0.00949 |
| 10.0 | 786 | -0.95970 | 1581 | 383 | 112.74 | 0.00946 |
| 7.5 | 1471 | -0.95969 | 1581 | 383 | 112.74 | 0.00945 |

The late minimum and first resolved age also converge with the material map.
Using 201, 401, and 801 surface points gives $J_{\min}=-0.9565$, $-0.9597$,
and $-0.9612$, and first resolved ages of 1583, 1581, and 1580 yr. Time steps
of 2.0, 1.0, and 0.5 yr give $J_{\min}=-0.95950$, $-0.95970$, and
$-0.95970$.

The result is not tied to the exact position of the open cuts. Moving the
dimensionless cut fraction from 0.70 to 0.90 changes boundary thickness from
91.8 to 41.3 m. All five cases overturn, with
$-0.991\leq J_{\min}\leq-0.826$ and first resolved ages of 1.317--1.956 kyr.

Hydrostatic side power must be interpreted as throughflow energy flux, not as
a negligible boundary effect. In the 10 m, 70 m-boundary case, the left and
right powers are $+1.343\times10^6$ and
$-1.343\times10^6\,\mathrm{Pa\,m^2\,yr^{-1}}$, while gravity supplies
$1.077\times10^6\,\mathrm{Pa\,m^2\,yr^{-1}}$. The side contributions nearly
cancel in the domain-integrated energy balance, but each is individually
large. The result is therefore robust to cut location within this
hydrostatic-open-boundary family, not independent of its boundary condition.

Aspect ratio separates ordered and overturned regimes. At 10 m target
spacing, otherwise identical troughs with horizontal radii 175, 350, and
500 m give $J_{\min}=-0.960$, 0.229, and 0.624. Several asymmetric ellipses
also overturn. The steep-entry/gentle-exit case is mesh-converged at
$J_{\min}\simeq-0.343$, while the reverse, deeper, and shallower asymmetric
cases cross $J=-0.2$ in the 10 m survey. These comparisons show that
overturning is selective rather than generic.

## Periodic finite-thickness beds

The periodic screen contains 15 finite-thickness overdeepenings, sills,
riegels, double obstacles, and asymmetric variants. None reaches $J=0$ in the
80 kyr calculation. This is a time-bounded negative result: 23--71 surface
markers remain active at 80 kyr, so it is not a full-lifetime preservation
proof.

Three cases closest to loss of order were remeshed through 80 kyr. Their
minima remain positive on 48x12, 72x18, and 96x24 meshes:

| Geometry | $J_{\min}$, 48x12 | $J_{\min}$, 72x18 | $J_{\min}$, 96x24 |
|---|---:|---:|---:|
| Steep asymmetric trough | 0.394 | 0.423 | 0.424 |
| Steep reverse trough | 0.413 | 0.460 | 0.430 |
| Basin between narrow riegels | 0.420 | 0.437 | 0.441 |

## Figure note

`realistic_geometry_overturned_maps` shows the narrow symmetric and
steep-entry/gentle-exit maps when each first reaches $J<-0.2$. Full geometry,
tight material-map views, and $x(x_0)$ plots make the negative source-order
slope explicit. The narrow case retains 383 valid sources spanning 112.7 m at
that threshold.

## Reproduction

- `realistic_geometry_parameter_search.py`: 15-case periodic screen.
- `realistic_geometry_elliptical_family.py`: finite elliptical family and
  material-map figure.
- `realistic_geometry_validation.py`: mesh, source, time-step, cut-location,
  and periodic remeshing checks.
- CSV diagnostics are in `../data/realistic_geometry_*.csv`.
- PNG/PDF summaries are in `../figures/realistic_geometry_*`.
