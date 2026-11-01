# Smooth localized periodic bed: pilot result

## Question and model

This calculation asks whether a smooth but spectrally localized bed feature can
produce steady isochrone overturning when a single Fourier mode does not.  The
domain is one exactly periodic broad wavelength, the surface is flat, and the
same constant downslope forcing and linear sliding law are used as in the
sinusoidal-bed calculation.  Characteristics are integrated directly to their
individual exits; there is no time-dependent ice simulation.

The bed is

\[
b(x)=-H+A f(\theta),\qquad
f=\frac{\cos\theta+p r[V_\kappa(\theta-\phi)-\overline V_\kappa]}{M},
\]

where \(V_\kappa(\delta)=\exp[\kappa(\cos\delta-1)]\), \(M\) is the
positive maximum of the numerator, and \(p=+1\) gives a rounded riegel while
\(p=-1\) gives a notch.  This normalization fixes the spatial mean thickness
at \(H\) and the minimum thickness at \(H-A\).  The parameter \(A\) is
therefore a minimum-gap/high-bed normalization parameter, not half of the
actual relief.  The symbol \(\ell\) denotes the FWHM of the imposed von Mises
kernel, not the FWHM of the composite bed.  The phase \(\phi\) is measured
relative to the crest of the broad cosine.

The pilot used \(H=160\) m and equal raw amplitudes \(r=1\).  A 40-cell shape
scan varied \(\ell/H=(1/12,1/8,1/4,1/2,1)\), four phases, and both polarities
at \(A/H=0.90\) and \(L/H=4\).  A second 24-cell scan varied
\(A/H=(0.65,0.80,0.90,0.95)\) and \(L/H=(0.5,1,2,3,4,6)\) for the most adverse
shape.

## Observations

The smooth periodic bed does produce negative \(J\), but only in an extreme
obstacle regime.  The clearest case has

\[
 A/H=0.90,\quad L/H=4,\quad \ell/H=1/12,\quad
 \phi/(2\pi)=1/4,\quad p=+1.
\]

Its minimum gap is 16 m, the imposed-kernel FWHM is 13.3 m, the actual relief
is 1.839\(H\), and the maximum bed slope is 17.2.  Thus the bed is analytic,
but it is effectively a near-vertical, near-pinching riegel rather than an
ordinary smooth undulation.

A quasi-uniform vertical coordinate gives \(J_{\min}=-0.257\) on the
production mesh and \(-0.271\) on the refined mesh.  A quadratically
endpoint-graded coordinate gives \(-0.539\) and \(-0.405\), respectively.
The sign is robust to both mesh refinement and the vertical coordinate, while
the magnitude remains coordinate-sensitive.

In the quasi-uniform refined calculation, 14 of 138 retained surface sources
have negative \(J\).  They span source phase 0.0478--0.0748 and carry 0.924%
of the retained positive accumulation flux.  The most negative source is the
first source at the 80% retained-flux cutoff, so the exact minimum is sensitive
to that cutoff.  The overturning set nevertheless has finite phase width and
finite accumulation flux; it is not a single sampled separatrix.

At the same \(A/H=0.90\) and \(L/H=4\), the \(\ell/H=1/8\) case changes from
\(J_{\min}=0.112\) to \(-0.068\) under refinement and should be regarded as a
borderline, unresolved threshold case.  All wider features remain ordered in
the pilot.  All sampled notch polarities remain ordered.  At fixed
\(\ell/H=1/12\), \(A/H=0.80\), and \(L/H=4\), refinement changes
\(J_{\min}=0.290\) to 0.255 and no counterflow occurs.  In the production
minimum-gap/period map, \(A/H=0.90\) first becomes negative near \(L/H=3\);
the sampled \(A/H\leq0.80\) cases remain ordered.

The \(A/H=0.95\), long-period cells generate very large negative numerical
values together with large characteristic-identity residuals.  Those values
are marked as unreliable in the figure and are not interpreted physically.

## Mechanism

Every negative-\(J\) path in the refined adverse case undergoes two horizontal
velocity reversals.  Before reversal, the most adverse accumulated mechanism
factor is \(C=-0.731>-1\); forward-flow slow-path deformation alone does not
reach the folding threshold.  The earliest retained path crosses \(J=0\) at
94.2 kyr and 0.183 local-thickness units below the surface, and negative
\(J\) eventually reaches a surface clearance of 0.343.  The result is
therefore a counterflow fold produced when the near-pinching obstacle creates
a local reversal/stagnation topology.

This links the localized-riegel result to the asymmetric convergent basin more
closely than to ordinary Fourier roughness.  Mean flow is unidirectional, but
the obstacle locally creates the counterflow that the single-mode and sampled
two-harmonic beds lack.

## Numerical checks and interpretation

The refined quasi-uniform run uses 576 by 144 Stokes elements, a 2353 by 481
velocity sampler, 481 candidate surface sources, and a 6.25 yr characteristic
step.  The production/refined pair keeps the sign and changes the minimum by
0.014.  The exact weak-form residual is of order \(10^{-13}\).  The maximum
streamfunction-identity residual is 2.36% in the quasi-uniform pair.  The
endpoint-graded refinement reduces that residual to 1.06%, but gives a more
negative \(J\).  These differences and the severe terrain-following cell
skewness justify treating the magnitude, and especially the \(A/H=0.95\)
limit, cautiously.

The result answers a narrow mathematical question: a discontinuous or
nondifferentiable bed boundary is not strictly necessary.  It does not show
that realistic smooth roughness commonly folds layers.  Within this pilot,
overturning requires a smooth bed that is dynamically almost a sharp wall and
that generates counterflow.  A useful next sweep would use minimum-gap to
feature-width ratio and maximum bed slope as axes, with a mesh designed for
near-vertical boundaries, to locate the counterflow threshold without entering
the nearly pinched limit.

No manuscript or LaTeX source is modified by this analysis.

## Reproduction

From the repository root in the Firedrake environment, the two pilot grids
and the primary endpoint-graded refinement are the defaults for their modes:

```bash
python notebooks/steady_characteristic_localized_packet_parameter_sweep.py \
  --mode shape --jobs 2 \
  --output data/steady_characteristic_localized_packet_shape_pilot.csv
python notebooks/steady_characteristic_localized_packet_parameter_sweep.py \
  --mode selected --jobs 2
python notebooks/steady_characteristic_localized_packet_parameter_sweep.py \
  --mode focused --jobs 2
```

The three named sensitivity pairs use explicit, non-overlapping outputs:

```bash
python notebooks/steady_characteristic_localized_packet_parameter_sweep.py \
  --mode focused --jobs 2 --focused-grading-power 1 \
  --output data/steady_characteristic_localized_packet_focused_uniform.csv
python notebooks/steady_characteristic_localized_packet_parameter_sweep.py \
  --mode focused --jobs 2 --focused-width-ratio 0.125 \
  --output data/steady_characteristic_localized_packet_focused_width_1_8.csv
python notebooks/steady_characteristic_localized_packet_parameter_sweep.py \
  --mode focused --jobs 2 --focused-amplitude-ratio 0.8 \
  --output data/steady_characteristic_localized_packet_focused_amplitude_0_8.csv
python notebooks/steady_characteristic_localized_packet_path_audit.py --grading-power 2
python notebooks/steady_characteristic_localized_packet_path_audit.py --grading-power 1
python notebooks/steady_characteristic_localized_packet_plot.py
```
