# Two-sided stagnation-flow test

## Question and model lineage

The public `bradlipovsky/wedge` notebook prescribes the analytical field

\[
\psi=-r^\lambda\sin(\lambda\theta),\qquad
\lambda=2\pi/\alpha,\qquad \alpha=3\pi/4.
\]

The two wedge arms are impermeable, flow enters through two parts of an outer
circular boundary, and it leaves through the middle of that boundary.  This is
the stagnation-flow construction used in the public JM/Allan Hills temperature
repository.  I tested its material map directly, then constructed a separate
gravity-driven glacier that retains a finite linear basal sliding law.

## Analytical wedge: overturning, but free slip

The outer circular boundary was treated as a fixed surface.  Its two inward
flow arcs are separate accumulation sources, and markers were removed at their
first outward crossing.  Equal-age curves were evaluated with
\(J_x=\partial x/\partial x_0\); layer relief or dip was not used as the
criterion.

The shallow/right source develops a vertical tangent at
\(tU_R/R=0.34417\).  In the original notebook scaling \(R=2\), \(A=-1\), this
is \(t=0.081305\).  The source-arc midpoint passes through vertical at
\(tU_R/R=0.35715\).  The sign change is not confined to the surface--bed
corner: at original time \(t=0.10\), the active markers deposited at least
\(15^\circ\) from that corner have \(-0.2415\leq J_x\leq-0.1836\).  The
shallow/right curve is therefore genuinely overturned.  The steep/left source
does not overturn through \(tU_R/R=0.6\); its minimum is \(J_x=0.99273\).

The threshold is converged in source count and time step: 601, 1201, and 2401
source markers give \(tU_R/R=0.344197\), 0.344179, and 0.344169.

This is an open-trajectory result, not recirculation.  The overturned markers
remain inside the outer boundary, and the diagnosed crossing lies away from
the zero-speed wedge corner.  The two source arcs nevertheless define two
separate material families rather than one global deposition coordinate.

The important mechanical caveat is exact.  On both wedge arms the analytical
solution has nonzero tangential velocity but zero tangential traction.  It is
therefore a free-slip solution (basal drag \(B=0\), mobility \(C=\infty\)), not
a solution of the requested finite law \(u_t=C\tau_b\).  It is a useful
positive kinematic benchmark, but it cannot by itself answer the finite-sliding
question.

## Finite-mobility V basin: no overturning found

The finite model has a 1 km left flank and a 5 km right flank.  Both rise 150 m
from the confluence, giving maximum absolute slopes 0.236 and 0.047.  Bed and
surface are parallel, ice thickness is 200 m, and gravity drives both flows
toward the central low point.  Ice is linear viscous with
\(\eta=2.5\times10^8\ \mathrm{Pa\,yr}\).  The bed is impermeable and obeys
\(u_t=C\tau_b\) with \(C=10^{-5}\ \mathrm{m\,yr^{-1}\,Pa^{-1}\).  The upper
surface is traction free, the outer vertical boundaries are symmetry divides,
and the fixed-surface mass balance is diagnosed as
\(\dot a=u_s s_x-w_s\).

The solved balance has two accumulation zones and one broad central ablation
zone.  On the finest mesh it ranges from -0.08383 to 0.16563 m/yr.  Integrated
positive and negative balances are 147.870 and -147.981 m²/yr; their 0.08%
mismatch measures boundary sampling error.  Surface markers were drawn from
the central 5--95% of accumulation flux on each source and removed
individually at first exit.

No surface-source curve reaches a vertical tangent.  Derivatives at the two
ends of every active marker interval are excluded because they become
one-sided when neighboring markers exit.  On 120x12, 180x18, and 240x24
meshes, the minimum \(J_x\) is 0.923588, 0.923549, and 0.923529.  It occurs on
the short, steep source at about 3.0 kyr.  The gentle/right family remains at
or above one in the baseline material map.  Horizontal flow reverses across
the confluence and turns into central upwelling as intended, but speed along
all evaluated open trajectories stays above 0.0507 m/yr.  There are no closed
trajectories in the evaluated families.

Source-count and time-step effects were tested separately on the finest Stokes
mesh.  At 161 source markers, steps of 100, 50, and 25 yr give lifetime minima
0.88860, 0.88858, and 0.88740.  At the same interior marker
\((x_0=-800\ \mathrm{m},t=3.0\ \mathrm{kyr})\), they give 1.23391, 1.23404,
and 1.23387.  With the step fixed at 25 yr, 41, 81, and 161 markers give
lifetime minima 0.99465, 0.92269, and 0.88740.  The lifetime minimum shifts as
a denser map retains points closer to a newly truncated active interval; at the
same interior marker and age the corresponding values are 1.21070, 1.20928,
and 1.23387.  Thus the lifetime minimum is exit-edge and source-density
sensitive, whereas the sign and fixed-interior value are stable at the
few-percent level.  Every test remains positive and contains zero overturned
samples.

The finest Stokes solve has a
relative mechanical-energy residual of \(7.1\times10^{-8}\) and an rms basal
normal speed of \(8.0\times10^{-8}\) m/yr.

The confluence separatrix is a material-origin boundary between the two source
families.  A single downstream \(x\)-coordinate is invalid across the flow
reversal, but time integration of each open family remains admissible.  The
separatrix itself is not assigned a finite surface-source age, and no claim is
made about inherited ice placed exactly on it.

## Interpretation and prediction

The JM analytical field does overturn a connected equal-age curve, but its
basal condition is free slip.  Replacing it with a smooth, gravity-driven
two-slope glacier with the manuscript's finite linear sliding mobility produces
strong bending but preserves order by a clear margin.
Convergence and upwelling are therefore not sufficient by themselves; this
finite-C calculation is a negative example, not a no-overturning theorem.

The finite-C model predicts steep but order-preserving reflectors from the
short flank into the ablation zone.  The free-slip benchmark instead predicts
a connected shallow-flank reflector that passes through vertical and has an
overturned segment away from the basal corner.

## Reproduction

Run from the repository root with the Firedrake environment:

```bash
OMP_NUM_THREADS=1 /Users/bradlipovsky/firedrake/bin/python notebooks/stagnation_flow_linear_sliding.py
/Users/bradlipovsky/firedrake/bin/python notebooks/stagnation_flow_wedge_benchmark.py
```

The scripts write flattened CSV diagnostics and PNG figures to
`notebooks/stagnation_flow_results/`.
