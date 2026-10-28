# Convergent-flow tests

## Question and models

These calculations test whether two steady flows converging into an ablation
zone can overturn a surface-deposited isochrone. The diagnostic is the
horizontal material-map tangent

\[
J_x=\left.\frac{\partial x}{\partial x_0}\right|_\tau .
\]

An initially ordered curve has passed through vertical when (J_x=0), and an
overturned segment has (J_x<0). Two mechanically distinct models are useful:

- an exact analytical convergent flow, which provides a clean positive
  kinematic benchmark but has free-slip straight boundaries; and
- a gravity-driven finite glacier with the manuscript's linear sliding law,
  (u_t=C\tau_b), (C=10^{-5}\ \mathrm{m\,yr^{-1}\,Pa^{-1}}).

## Analytical convergent-flow benchmark

The analytical field is

\[
\psi=-r^\lambda\sin(\lambda\theta),\qquad
\lambda=2\pi/\alpha,\qquad \alpha=3\pi/4.
\]

The circular outer boundary contains two disconnected accumulation arcs,
separated by a central ablation/outflow arc. The right, shallow-arm source
first reaches (J_x=0) at the dimensionless age
(tU_R/R=0.34417); its source-arc midpoint reaches zero at 0.35715. The
left, steep-arm source remains ordered through (tU_R/R=0.6), with
(J_{x,\min}=0.99273). The right-branch crossing converges to 0.344197,
0.344179, and 0.344169 for 601, 1201, and 2401 source markers with
corresponding time-step refinement.

The result is not a corner-only numerical sign change. In the original
(R=2, A=-1) scaling, active right-branch markers deposited at least
(15^\circ) from the arm have (-0.2415\le J_x\le-0.1836) at (t=0.10).
The trajectories are open and the diagnosed crossing remains inside the
outer boundary.

The redesigned benchmark figure makes each visual encoding explicit:

- panel (a) uses blue and orange only to identify the two accumulation
  sources; the black middle arc is ablation/outflow and grey curves are
  streamlines;
- panel (b) uses color for age and line style for source branch; curves are
  shown through (tU_R/R=3); and
- panel (c) uses an open marker for the first (J_x=0) event and shading for
  (J_x<0).

There is no deposition source on the middle outer arc, so no forward-time
equal-age curve starts there. Older curves from the two side sources fill most
of the apparent central gap. The small region that remains close to the exact
stagnation point requires ages beyond the plotted range; travel time to that
point diverges.

The mechanical caveat is exact: tangential traction is zero while tangential
velocity is nonzero on both straight arms. This is free slip
((B=0, C=\infty)), not the finite linear sliding law. It is therefore a
kinematic benchmark rather than the requested finite-sliding counterexample.

## Finite-sliding convergent glacier

The baseline glacier has a 1 km left flank and a 5 km right flank. Both rise
150 m from the confluence, the ice is 200 m thick, and the bed and surface are
parallel. The upper surface is traction free, the outer boundaries are
symmetry divides, and the fixed-surface mass balance is diagnosed as
(\dot a=u_s s_x-w_s). This produces accumulation on the two outer flanks and
ablation over the central upwelling zone.

The primary material tangent is now integrated along every trajectory from

\[
\dot{\mathbf F}=\nabla\mathbf u\,\mathbf F,
\]

with the initial tangent taken along the source curve. This avoids estimating
(J_x) from the spacing between the last two markers near an exit. Neighbor
finite differences are retained only as an independent comparison.

The baseline remains ordered. On 120 by 16, 180 by 24, and 240 by 32 meshes,
the variational minima are 0.85428, 0.85409, and 0.85406. The corresponding
neighbor-difference minima are about 0.9236, agreeing on the sign but not
serving as the primary diagnostic. The finest solve has relative mechanical
energy residual (7.0\times10^{-8}) and rms basal-normal speed
(8.0\times10^{-8}\ \mathrm{m\,yr^{-1}}).

The summary figure now uses 45 normalized arrows on a 9 by 5 grid in panel
(a), with speed contours carrying the magnitude information. Panel (b) shows
the diagnosed accumulation and ablation, and panel (c) shows equal-age curves.

## Two-dimensional geometry study

The useful first parameterization separates asymmetry from roughness:

\[
q=L_R/L_L,\qquad R_b=\Delta b/H.
\]

The screen varies (q=1,\ldots,9) and (R_b=0.25,0.50,\ldots,2.50), while
holding total length at 6 km, thickness at 200 m, viscosity, sliding mobility,
boundary treatment, mesh, marker count, and marker time step fixed. All 90
cells pass the mechanical diagnostics. Eight cells overturn, all in the
high-relief, strongly asymmetric part of the slice. The plotted (J=0)
contour is therefore evidence for a transition within this specified slice,
not a universal boundary for convergent glaciers.

The phase-map figure pairs the heat map with a schematic of the exact
piecewise-sinusoidal geometry. Dimension arrows define the vertical bed
relief $\Delta b$, vertical ice thickness $H$, and the two horizontal flank
lengths $L_{\rm left}$ and $L_{\rm right}$; no particular sampled geometry is
singled out.

A mesh-stable bracket occurs at (q=7):

| (\Delta b/H) | (J_{\min}), coarse | medium | fine | refined map |
|---:|---:|---:|---:|---:|
| 2.25 | 0.0821 | 0.0603 | 0.0790 | 0.0766 |
| 2.50 | -0.1773 | -0.1744 | -0.1606 | -0.1548 |

The extreme sampled case, (q=9\), (\Delta b/H=2.5), gives
(J_{\min}=-1.849,-1.877,-1.871) over the three meshes. At fixed ages its
minimum is about -0.151 at 3 kyr and -0.647 at 4 kyr. At 3 kyr, negative
material tangents extend about 49.1 m below the surface. The 3 kyr value is
stable to gradient-sampling refinement
(-0.15132, -0.15095, and -0.15112), and the sign also persists under source
count and time-step refinement. The first negative tangent at least 10 m
below the surface appears at about 2.66--2.70 kyr.

The next useful parameter studies would add one axis at a time: sliding
mobility (C), total aspect ratio (L/H), smoothing radius at the confluence,
or the accumulation/ablation distribution. Those tests would distinguish a
geometric threshold from a boundary-condition or residence-time effect
without immediately creating an uninterpretable high-dimensional sweep.

## Reproduction

Run from the repository root in the Firedrake environment:

```bash
OMP_NUM_THREADS=1 /Users/bradlipovsky/firedrake/bin/python \
  notebooks/stagnation_flow_linear_sliding.py

OMP_NUM_THREADS=1 /Users/bradlipovsky/firedrake/bin/python \
  notebooks/stagnation_flow_parameter_study.py --workers 3

/Users/bradlipovsky/firedrake/bin/python \
  notebooks/analytical_convergent_flow_benchmark.py
```

Flattened diagnostics and PNG/PDF figures are written to
`notebooks/stagnation_flow_results/`.
