# Finite nonperiodic bed-spike test

## Calculation

This calculation asks whether one upward bedrock spike overturns steady,
surface-sourced isochrones in a finite, nonperiodic glacier reach. It is not a
whole divide-to-terminus glacier model.

The model solves two-dimensional incompressible Stokes flow with constant
viscosity `2.5e8 Pa yr`, density `917 kg m^-3`, surface slope `0.05 rad`, and
ice thickness `160 m`. The bed obeys no penetration and the passive linear
sliding law

`tangential traction + beta * tangential velocity = 0`,

with `beta = 1e5 Pa yr m^-1` (mobility `1e-5 m yr^-1 Pa^-1`). The upper
surface is flat in slope-following coordinates and traction free. A compact
raised-cosine spike is the only bed feature. The two vertical ends apply the
exact hydrostatic and shear tractions of the corresponding uniform flat slab;
no velocity is prescribed. The primary reach is `1.2 km` long.

The fixed surface's accumulation is diagnosed from its computed normal
velocity. Surface trajectories begin `0.05 m` below the surface on the
accumulating side of the numerically located ablation--accumulation transition.
The material tangent obeys

`D F / D t = grad(u) F`, with `F(0) = (1, 0)`.

The order diagnostic is `J = F_x = partial x / partial x_0` at fixed age.
An overturn requires `J < 0`. Finite differences between neighboring
trajectories provide an independent sign check but are not used as the primary
quantitative diagnostic.

## Result

No tested spike overturns an isochrone. The six-case `160 x 24` survey spans
spike heights of `80--145 m`, analytic maximum bed angles of `38.1--76.6
degrees`, and minimum ice thicknesses of `15--80 m`. The minimum `J` values
are `0.841`, `0.587`, `0.478`, `0.411`, `0.408`, and `0.221`.

The closest case is a `145 m`-high spike with `60 m` half width, a `75.24
degree` maximum bed angle, and only `15 m` of ice over its crest. For a source
`0.25 m` downstream of the diagnosed accumulation onset:

| Mesh | Minimum `J` | Neighbor difference | Age of minimum | Last valid age |
|---|---:|---:|---:|---:|
| `160 x 24` | 0.220816 | 0.224515 | 6.252 kyr | 29.486 kyr |
| `240 x 36` | 0.247411 | 0.250957 | 6.674 kyr | 30.172 kyr |
| `320 x 48` | 0.258331 | 0.262361 | 6.894 kyr | 30.528 kyr |

The sign therefore moves away from overturning with mesh refinement. On the
finest mesh, time steps of `4`, `2`, `1`, and `0.5 yr` all give
`J_min = 0.258331`; the age varies by at most `2 yr`.

The onset-limit test places individual sources `0.01--10 m` downstream of the
numerically located accumulation transition. The closest source has
`J_min = 0.232813` at `7.038 kyr`, and all values are positive. All ten
particles leave the modeled reach by `30.546 kyr`, well before the `100 kyr`
ceiling. The fine `0.5--4 yr` integration comparison covers the minimum event;
the full-residence check uses `2 yr` steps.

Moving the open boundaries changes quantitative residence times but not the
sign. At approximately `7.5 m` horizontal spacing, lengths of `0.8`, `1.2`,
and `1.8 km` give `J_min = 0.281401`, `0.220816`, and `0.219763`; their minimum
ages are `11.474`, `6.252`, and `4.370 kyr`. Thus the `1.2` and `1.8 km`
order minima nearly agree, although flow speed and event age remain
length-dependent. Every source in these three reaches completes its modeled
residence by `41.372 kyr`.

The flat-slab control reproduces the analytical velocity with relative `L2`
error `7.1e-15`. For the finest closest case, the relative mechanical-energy
residual is `9.16e-4`, RMS basal normal speed is `2.16e-7 m yr^-1`, and all
contributing trajectories retain positive horizontal speed.

## Interpretation and scope

In this fixed-surface steady state, the spike produces surface ablation on its
upstream side and accumulation on its downstream side. Surface-sourced layers
therefore enter on the lee side and flow away from the obstacle. The spike
strongly bends and compresses those layers but does not reverse their
horizontal order.

This is a negative result for an open, finite glacier reach with transparent
through-flow boundaries and diagnosed surface mass balance. It does not rule
out overturning in a whole glacier with a divide, terminus, evolving surface,
or independently prescribed accumulation. The domain-length comparison also
shows that quantitative ages should not be interpreted independently of the
chosen reach length.

## Reproducible files

- `notebooks/finite_spike_open_boundary.py`
- `data/finite_spike_survey.csv`
- `data/finite_spike_mesh_convergence.csv`
- `data/finite_spike_map_convergence.csv`
- `data/finite_spike_domain_convergence.csv`
- `data/finite_spike_accumulation_boundary.csv`
- `figures/finite_spike_open_boundary.png`
- `figures/finite_spike_open_boundary.pdf`
