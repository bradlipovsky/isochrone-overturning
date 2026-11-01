# Steady characteristics through a smooth icefall-to-tongue transition

## Question

An icefall creates strong along-flow acceleration and a transition into a
thicker, slower tongue.  This calculation tests whether that longitudinal
compression is sufficient to overturn an equal-age surface in steady,
two-dimensional flow.

## Geometry and boundary conditions

The domain contains one exactly periodic transition pair.  The mean surface
slope is 0.05 and the local downhill slope is

\[
m(x)=m_g\{1+(r_s-1)W(x)\},
\qquad
W(x)=\left[{1+\cos(2\pi x/L)\over2}\right]^4.
\]

The gentle slope \(m_g\) is chosen so that the spatial mean of \(m\) remains
0.05.  Thus \(r_s\) is a tangent-slope ratio, not an exact angle ratio.  The
same window makes the steep reach thin and the gentle reach thick while
preserving a mean thickness of 160 m.  The parameter \(r_H\) is the ratio of
the thick-reach to thin-reach thickness.  The sweep spans

- \(L/H=(2,4,8,12)\),
- \(r_s=(1,2,4,8,12)\), and
- \(r_H=(1,1.5,2,3)\).

The geometry is periodic because this removes an open-boundary explanation
for any sign change in \(J\).  Periodicity also means that the cell contains
both the icefall-to-tongue transition and the return transition into the next
icefall.  The result therefore applies to a periodic transition pair, not to
an isolated finite glacier terminus.

The flow is linear-viscous Stokes flow with viscosity
\(2.5\times10^8\) Pa yr.  The bed is impermeable and obeys the same passive
linear sliding law as the other calculations,
\(\tau_t+\beta u_t=0\), with
\(\beta=10^5\) Pa yr m\(^{-1}\).  The surface is traction free.  The steady
surface mass balance is diagnosed from the solved velocity.

## Direct steady diagnostic

No time-dependent isochrone simulation is used.  Characteristics begin on
every depositing part of the surface and satisfy

\[
\dot{\boldsymbol X}=\boldsymbol v(\boldsymbol X),\qquad
\dot{\boldsymbol T}=\nabla\boldsymbol v(\boldsymbol X)\boldsymbol T,
\qquad
\boldsymbol T(0)=(1,s_x).
\]

The horizontal order diagnostic is \(J=T_x\).  Each path stops at its first
surface or bed exit.  For a path that does not reverse horizontal direction,

\[
J={u\over u_0}(1+C).
\]

Consequently, adverse shear must accumulate to \(C<-1\) to produce a fold
unless the path enters counterflow.

The implementation traces physical coordinates rather than a
surface-flattened vertical velocity.  This avoids artificial velocity jumps
at columns of the piecewise-curved coordinate mesh.  The initial tangent and
the streamfunction identity are evaluated on the represented physical
surface.

## Result

No sampled geometry overturns.  Seventy-five numerical cells and four
analytic uniform-slab limits are ordered.  One weak-signal, nearly
uniform-thickness cell has one failed production characteristic; its refined
retry exits cleanly and remains ordered at \(J=0.995\).

The smallest production value is

\[
J_{\min}=0.7102
\]

at \(L/H=2\), \(r_s=12\), and \(r_H=3\).  The first percentile across
sources is 0.7134, and the raw value carried through surface exit is also
0.7102.  A combined mesh, sampler, source, and trajectory-step refinement
gives 0.7074.  The sign and margin from zero are therefore stable.

This minimum occurs in a severe geometry: the thickness ranges from 65.2 to
195.7 m and the maximum absolute bed slope is 2.31.  An additional stress
test with \(r_s=32\), \(r_H=5\), a sharper eighth-power transition, and a
37.96 m minimum thickness remains ordered at \(J=0.8370\).

The parameter trend is controlled primarily by wavelength.  The minimum at
each wavelength is 0.710, 0.724, 0.842, and 0.889 for
\(L/H=2,4,8,12\), respectively.  Strong compression is therefore confined
to short transitions, but it does not approach the fold threshold.

## Mechanics

At the global minimum, the speed factor is \(u/u_0=0.679\), whereas the
integrated shear factor is \(C=+0.046\).  The reduction in \(J\) is therefore
caused by along-path deceleration into the tongue, not by accumulated adverse
vertical shear.  Across all successful paths, the most negative pre-reversal
value is only \(C=-0.247\), and no path reverses horizontal velocity.

This separates compression from overturning.  A steep-to-gentle transition
can bring neighboring source labels closer together, but longitudinal
deceleration alone preserves their order.  The calculation does not produce
the adverse-shear integral or counterflow needed to change the sign of
\(J\).

## Numerical checks

- The global-minimum refinement changes \(J\) from 0.7102 to 0.7074.
- Its maximum streamfunction-identity residual decreases from 0.041% to
  0.017%, and every retained characteristic exits through the surface.
- The largest horizontal step in the refined minimum case is
  \(1.59\times10^{-4}\) wavelengths.
- The median identity residual over successful production cells is 0.050%.
  The largest relative values occur in near-uniform, weak-surface-signal
  cases and are far from the minimum-\(J\) regime.
- No successful production or validation path reverses, reaches the safety
  cap, or exits through the bed.

## Interpretation

This model does not support an icefall transition as an independent folding
mechanism.  Its observational prediction is distortion without overturning:
isochrones should steepen or bunch through the decelerating reach but retain
their horizontal order.  The conclusion is limited to steady,
two-dimensional, linear-viscous flow, the specified linear sliding law, and
the periodic transition-pair geometry.

## Reproduction

From the repository root in the Firedrake environment:

```bash
python notebooks/steady_characteristic_icefall_transition.py --sweep
python notebooks/steady_characteristic_icefall_transition.py \
  --validation \
  --output data/steady_characteristic_icefall_transition_validation.csv
python notebooks/plot_steady_characteristic_icefall_transition.py
```

The calculations write the production and validation CSV files under
`data/`.  The figure is
`figures/steady_characteristic_icefall_transition.png`.
