# Idealized Allan Hills steady-flow experiment

## Question and setup

This calculation tests whether an idealized thin-left/thick-right geometry, together with leftward influx from the thick side and alternative treatments of the thin-left boundary, can overturn material lines. It is a two-dimensional, fixed-geometry, incompressible, linear-viscous full-Stokes calculation, not a free-surface evolution.

The domain is 2.0 km long. The variable-thickness geometry grows smoothly from 70 to 360 m; the surface slope transitions from -0.035 to 0.012. The bed has no penetration and linear Robin drag $\beta=1.00e+05$ Pa yr m$^{-1}$, and the surface is traction free. At the right boundary, only $u$ is prescribed using an $n=1$ SIA-shaped profile scaled to the target flux; $w$ is not prescribed and retains its natural zero-tangential-traction condition. The reference right flux is -146.618 m$^2$ yr$^{-1}$, and the reduced cases use 1% of it.

The thickness construction reaches 300 m through a smooth 0--1.1 km transition, then adds a second transition beginning at 0.8 km. That second ramp formally ends at 2.2 km outside the domain and is normalized to give exactly 360 m at the right boundary. Consequently, the bed is still descending at the right inlet rather than turning upward there.

The principal case uses an open left boundary with hydrostatic-plus-shear traction corresponding to a reference parallel slab. This condition does not prescribe the left flux: its magnitude and sign are outputs. A second case prescribes an SIA-shaped horizontal left influx exactly matched in magnitude to the right influx. Closed-left cases impose $u=0$ at the left boundary. The uniform-thickness run is only a sensitivity case because changing its left thickness also changes the open-boundary response.

The surface mass balance needed to keep the surface fixed is diagnosed as $a_s=u s_x-w$ and checked against the exact finite-element surface-normal flux. Boundary fluxes below use the outward normal: negative lateral flux is inflow, while positive surface-outward flux requires negative steady surface mass balance (net ablation).

## Steady material-line diagnostic

Along 193 right-inlet depth sources, the code integrates $\dot F=\nabla u F$ from $F=I$. For each incoming tangent $(1,m_0)$ it evaluates $J(m_0)=F_{00}+m_0F_{01}$. Negative $J$ is overturning. The reported decomposition integrates $u_xT_x$ and $u_zT_z$ separately, and `det(F)` checks incompressibility along paths.

| title | minimum J, m0=0 | minimum J, m0=m_right | left outward flux (m2/yr) | computed right outward flux (m2/yr) | surface outward flux (m2/yr) | age-capped paths | maximum abs detF-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Variable thickness, open-left convergence | -0.2929 | -0.2591 | -1.376 | -1.466 | 2.842 | 0 | 6.12e-04 |
| Variable thickness, flux-matched convergence | -0.3135 | -0.2782 | -1.466 | -1.466 | 2.932 | 0 | 6.06e-04 |
| Variable thickness, closed-left control | 0.0776 | 0.0820 | -0.000 | -1.466 | 1.466 | 0 | 6.35e-04 |
| Uniform thickness, open-left sensitivity | -0.7722 | -0.5401 | -27.731 | -1.466 | 29.197 | 0 | 1.34e-04 |
| Variable thickness, full reference influx | 0.0487 | 0.0451 | 0.000 | -146.618 | 146.618 | 0 | 2.75e-04 |

For initially horizontal material lines, overturning occurs in: **Variable thickness, open-left convergence, Variable thickness, flux-matched convergence, Uniform thickness, open-left sensitivity**. For material lines initially parallel to the right surface, overturning occurs in: **Variable thickness, open-left convergence, Variable thickness, flux-matched convergence, Uniform thickness, open-left sensitivity**.

## What the experiment shows

The open-left variable-thickness case overturns: $J_{min}(m_0=0)=-0.2929$ at 118.71 kyr. 16 of 193 sampled inlet depths overturn for horizontal incoming layers. The reported minimum comes from the explicit basal-source cutoff $\sigma_0=0.01000$ and occurs at $x=0.401$ km, local $\sigma=0.504$; it is therefore a cutoff-conditioned screening value, not a claimed continuum-global minimum. The tangent budget identifies the mechanism: vertical shear contributes -2.422, while longitudinal deformation contributes +1.129 and therefore opposes the fold.

The flux-matched two-sided case also overturns ($J_{min}=-0.3135$), so the positive result does not depend on using the open-traction boundary. In contrast, closing the thin-left boundary leaves the same variable-thickness geometry ordered ($J_{min}=0.0776$), and even the full reference right influx with a closed left remains ordered ($J_{min}=0.0487$). Thus thickness contrast plus right-side influx is not sufficient in these tests; thin-left counterflow and the resulting two-sided convergence are the decisive ingredients.

The principal case imports 1.376 m$^2$ yr$^{-1}$ from the left and 1.466 m$^2$ yr$^{-1}$ from the right, balanced by 2.842 m$^2$ yr$^{-1}$ of surface outflux. Its left quarter is not uniformly downwelling: 41% of sampled points have $w<0$, while the mean is 1.972e-03 m yr$^{-1}$. The idealized vertical-velocity field therefore only partly reproduces the attached 3-D section.

The largest tangent-budget closure error is 1.34e-13.

## Resolution and source-depth validation

| variant | source count | bed buffer sigma | minimum J, m0=0 | source sigma at minimum J, m0=0 | negative horizontal sources | bed-buffer terminations | maximum abs detF-1 | relative flux closure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coarse | 193 | 0.0005 | -0.29078 | 0.01000 | 16 | 0 | 2.30e-03 | 1.48e-07 |
| production | 193 | 0.0005 | -0.29291 | 0.01000 | 16 | 0 | 6.12e-04 | 1.94e-08 |
| production-dense-depth | 385 | 0.0005 | -0.29291 | 0.01000 | 32 | 0 | 6.12e-04 | 1.94e-08 |
| refined | 193 | 0.0005 | -0.29350 | 0.01000 | 16 | 0 | 4.20e-04 | 5.99e-09 |
| buffer-0.001 | 193 | 0.001 | -0.29293 | 0.01000 | 16 | 0 | 6.14e-04 | 1.94e-08 |
| buffer-0.00025 | 193 | 0.00025 | -0.29284 | 0.01000 | 16 | 0 | 6.53e-04 | 1.94e-08 |

At the fixed source cutoff $\sigma_0=0.01$, the production, doubled source-density, and refined-mesh values are -0.29291, -0.29291, and -0.29350. Moving the bed buffer from $\sigma=0.001$ to $\sigma=0.00025$ changes the minimum from -0.29293 to -0.29284.

## Interpretation limits

This is a sufficient-mechanism test, not a complete representation of Allan Hills. The specialized calculation satisfies $u_x+v_y+w_z=0$, whereas this section sets the omitted lateral strain $v_y$ to zero. A positive 2-D result shows that the represented cross-sectional convergence can suffice; a negative result would not rule out folding driven by 3-D lateral convergence. The incoming tangent $(1,m_0)$ is assumed boundary stratigraphy, and the depth sources are trajectories carrying that local deformation diagnostic, not a claim that a simultaneous isochrone lies along the vertical inlet boundary. The open-left traction is a reference mechanical boundary condition, not an observational constraint. The fixed geometry also requires the diagnosed surface mass balance.

## Numerical settings

Mesh: 160 by 32; terrain sampler: 501 by 161 over $\sigma=[0.0005,0.99]$; trajectory sources: 193 uniformly spaced over $\sigma_0=[0.01,0.95]$; requested RK4 step: 5 yr (automatically reduced by a terrain-grid CFL bound); maximum age: 500000 yr; incoming-slope range: $[-0.1,0.1]$.
