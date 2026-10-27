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

Those lifetime minima occur after most of the cohort has left the domain. A
second diagnostic removes that moving-exit ambiguity: it uses the fixed
central 80% of the original deposition origins and stops before the first
member of that set exits. Its minima on the 15, 10, and 7.5 m meshes are
$-0.5922$, $-0.5942$, and $-0.5948$. Thus the additional counterexample does
not depend on the final surviving markers at a late exit edge.

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

### Two-dimensional thickness slice

Minimum and maximum ice thickness do not fully specify a finite symmetric
ellipse. After horizontal and vertical translations are removed, the geometry
has four parameters. One convenient set is center thickness, boundary
thickness, half-domain length, and a horizontal curvature scale. With the
surface fixed, the first two are equivalent to specifying the deepest and
shallowest sampled bed elevations, but two horizontal scales are still
required.

The requested map is therefore a two-dimensional slice of a four-dimensional
geometric space. It varies boundary and center thickness while fixing the
horizontal radius at 175 m and half-domain length at 128.312 m. It also fixes
mean slope, rheology, basal mobility, hydrostatic open-side tractions, 10 m
target mesh spacing, 1 yr trajectory step, 401 deposition origins, and a
4 kyr age ceiling. The 8 by 8 grid has 55 valid troughs; the nine cells with
center thickness less than or equal to boundary thickness are shown as
invalid rather than interpolated.

The color field reports the minimum fixed-interior $J$ through 4 kyr or the
first exit from that fixed source subset, whichever occurs first. Evaluation
ends range from 73 yr to the 4 kyr ceiling across the grid. The solid zero
contour is the fixed-interior result; the dashed contour shows the raw
lifetime diagnostic. For example, the 20/80 m and 140/180 m boundary/center
cases have raw minima $-0.194$ and $-0.631$ but fixed-interior minima $+0.922$
and $+0.800$. Those raw signs are not retained by the fixed-interior
diagnostic and are sensitive to the moving-exit convention.

The fixed-interior map nevertheless contains a resolved negative band. At
150 m center thickness, boundary thicknesses 40--80 m are negative. At 180 m
center thickness, boundary thicknesses 70--120 m are negative; at 210 m, the
sampled 120 and 140 m boundary cases are negative. The existing 70/150 m case
lies inside this band at $J_{\mathrm{int}}=-0.594$.

The closest sampled crossing is stable under numerical changes. At 70/180 m,
the fixed-interior minimum is $-0.04624$, $-0.04668$, and $-0.04740$ on 15,
10, and 7.5 m meshes; $-0.04479$, $-0.04668$, and $-0.04759$ with 201, 401,
and 801 sources; and $-0.04668$ at trajectory steps 2, 1, and 0.5 yr. The
neighboring 60/180 m case remains positive on all three meshes
($0.2671$, $0.2662$, and $0.2658$). The near-zero 70/180 m case contains local
reverse horizontal flow, whereas the 150 m-center negative band retains
positive downstream velocity.

The map is age-conditioned, not a lifetime geometry boundary. Several cells
that are positive at the 4 kyr ceiling become negative in 8 kyr tests:
20/150 m changes from $+0.312$ to $-0.719$, 60/180 m from $+0.266$ to
$-0.743$, 100/210 m from $+0.283$ to $-0.659$, and 140/240 m from $+0.531$
to $-0.415$. The 70/270 m case remains positive, changing from $+0.908$ to
$+0.703$.

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

`realistic_geometry_overturned_maps` shows ages 0, 0.7, 1.4, 2.1, and 2.8 kyr
for the narrow symmetric and steep-entry/gentle-exit troughs. Curve color is
deposition coordinate. Open red rings mark sampled locations with centered
$J<0$, and markers are removed permanently at their first surface or side
exit. Every panel retains the full bed geometry.

`realistic_geometry_elliptical_parameter_map` shows the two-dimensional
thickness slice. The diverging color scale is centered exactly at a white
$J=0$; sampled cells, the existing narrow case, invalid cells, and the
fixed-interior and raw zero contours are identified explicitly.

## Reproduction

- `realistic_geometry_parameter_search.py`: 15-case periodic screen.
- `realistic_geometry_elliptical_family.py`: finite elliptical family and
  time-evolution figure.
- `realistic_geometry_elliptical_parameter_map.py`: fixed-horizontal 8 by 8
  center/boundary-thickness map.
- `realistic_geometry_elliptical_parameter_validation.py`: mesh, source,
  time-step, and 8 kyr checks for the map.
- `realistic_geometry_validation.py`: mesh, source, time-step, cut-location,
  and periodic remeshing checks.
- CSV diagnostics are in `../data/realistic_geometry_*.csv`.
- PNG/PDF summaries are in `../figures/realistic_geometry_*`.
