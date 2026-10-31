# Smooth two-harmonic beds and steady isochrone ordering

## Question

The one-mode sinusoidal bed stays ordered over the original parameter sweep.
This analysis asks whether the missing ingredient is a sharp bed boundary, or
whether a second smooth Fourier mode can supply enough localized, asymmetric
strain to overturn a steady isochrone.

The answer must distinguish two statements.  Failure of a finite smooth
Fourier family to fold cannot establish that a sharp boundary is necessary.
The elliptical calculations contain robust negative-$J$ paths away from the
finite-domain joins, but no rounded-join continuation has yet been computed.
The two-harmonic calculation therefore tests whether one particular minimal
smooth family is sufficient; it is not a proof about all smooth beds.

## Geometry

One exactly periodic wavelength has a flat surface and bed

\[
b(x)=-H+A f(\theta;q,\phi),\qquad
f=\frac{\cos\theta+q\cos(2\theta+\phi)}
        {\max_\theta[\cos\theta+q\cos(2\theta+\phi)]},
\qquad \theta=2\pi x/\lambda.
\]

This normalization fixes the mean thickness $H$ and minimum thickness $H-A$
for every harmonic ratio $q$ and relative phase $\phi$.  It recovers the
one-mode sinusoid exactly at $q=0$.  Relief is recorded rather than fixed,
because an asymmetric zero-mean profile cannot simultaneously preserve its
mean, upper excursion, and total range with one scale factor.

The quick shape scan uses $H=160$ m, $A/H=0.85$, 13 fundamental wavelengths
from $\lambda/H=1/12$ to 12, six $q$ values from 0 to 1, and eight phases for
each nonzero $q$.  A focused thin-neck scan at $A/H=0.95$ and
$\lambda/H=3$ samples eleven $q$ values and twelve phases.  A third sweep
fixes the most compressive $A/H=0.85$ quick-scan shape and varies $H$, $A/H$,
and $\lambda/H$; it includes $A/H=0.90$ and 0.95 to test the thin-neck direction.
The second harmonic has wavelength $\lambda/2$, so the shortest modeled
component has wavelength $H/24$.

## Exact steady mechanism

Let $(J,K)=\partial(x,z)/\partial x_0$ be the equal-age tangent carried from a
surface source $x_0$, and let $a_0=-w(x_0,0)>0$ be its accumulation.  The
streamfunction identity is

\[
-wJ+uK=a_0.
\]

For a path with no horizontal reversal,

\[
\frac{d}{dt}\left(\frac{J}{u}\right)
  =\frac{a_0 u_z}{u^2},\qquad
J=\frac{u}{u_0}(1+C),\qquad
C=a_0u_0\int_0^t\frac{u_z}{u^2}\,dt.
\]

Thus a positive-speed path overturns only if $C<-1$.  At the first crossing
$J=0$, $dJ/dt=a_0u_z/u$, so a positive-to-negative crossing requires adverse
vertical shear, $u_z/u<0$.  This separates two effects in the figures: the
speed ratio $u/u_0$ and the accumulated orientation factor $1+C$.  Near a
horizontal reversal the factorization is singular, so $J$ and the
streamfunction identity remain the authoritative diagnostics.

## Short-wave limit and numerical reach

A Fourier disturbance is screened above the bed over its inverse wavenumber.
The useful local shielding number is

\[
\chi=\frac{2\pi(H-A)}{\lambda},
\]

not $2\pi H/\lambda$, because the thinnest gap controls the strongest possible
bed-to-surface coupling.  At fixed $A/H<1$, every finite smooth harmonic family
has $\chi\to\infty$ as $\lambda/H\to0$.  Surface-fed paths then approach the
ordered slab limit even while bed slope grows.

For $\lambda/H\le1/4$, the finite-element mesh and velocity sampler therefore
cluster points at both the bed and surface.  Longer waves use a uniform
terrain-following coordinate; extending endpoint clustering into that regime
creates increasingly anisotropic thin-neck cells.  A case is called
*screened below numerical signal* when
the maximum sampled surface mass balance is less than ten times the RMS basal
normal-velocity leakage, or when relative surface mass-balance closure exceeds
$10^{-3}$.  Such a cell is not assigned a numerical $J$: the physical
$J\to1$ statement is an asymptotic inference, not a measurement extracted
from roundoff-level surface velocities.  The calculation also records the
largest Runge--Kutta displacement in wavelengths and an independent residual
of $-wJ+uK=a_0$.

## Results

The three sweeps contain 1,629 sweep cells, including 14 cross-sweep
overlaps.  In the selected-shape sweep
($q=0.125$, $\phi=3\pi/2$), 695 of 975 cells are resolved numerical cases,
65 are analytic flat-bed limits, and 215 are screened below the numerical
surface signal.  Every resolved value is positive.  The lowest central-flux
value is

\[
J_{\min}=0.44431
\]

at $H=50$ m, $A/H=0.95$, and $\lambda/H=3$.  Extending every retained path
all the way to its exit gives a lower raw minimum, $J=0.17393$, at the same
$H$ and $A/H$ but $\lambda/H=2$.  This case also gives the most adverse
accumulated shear, $C=-0.82576$, which remains above the folding threshold
$C=-1$.

The broad shape scan resolves 465 of 533 cells at $H=160$ m and $A/H=0.85$;
68 very short-wave cells are screened.  Its lowest central-flux value is
$J=0.54290$ at $\lambda/H=3$, $q=0.125$, and $\phi=3\pi/2$.  Refinement moves
that value to 0.51680, without changing its sign.  The focused thin-neck
shape scan resolves all 121 cells at $A/H=0.95$ and $\lambda/H=3$.  Its
minimum is $J=0.44790$ at the same $q=0.125$, $\phi=3\pi/2$ shape.  The raw
minimum over that scan is 0.20751 and its most adverse value is
$C=-0.78993$, both at $q=0.125$, $\phi=\pi/2$.

The explicitly requested short wavelengths do not reveal a hidden folding
branch.  In the broad $A/H=0.85$ shape scan, the lowest resolved values at
$\lambda/H=1/2$, $1/4$, and $1/8$ are 0.81712, 0.94512, and 0.99829.  All 41
shapes resolve at $1/2$ and $1/4$; 14 resolve and 27 are screened at $1/8$;
all 41 are screened at $1/12$.  In the more demanding $H=160$ m,
$A/H=0.95$ selected shape, the production value at $\lambda/H=1/12$ is
0.72060, but refinement raises it to 0.94250 and reduces the most adverse
$C$ from $-0.13043$ to $-0.00366$.  Refinement therefore moves this
finite-wavelength result much closer to one and provides no evidence of a
hidden folding branch.  The strict slab limit $J\to1$ refers to
$\lambda/H\to0$ at fixed $A/H<1$.

The characteristic sampler resolves small Eulerian reverse-flow pockets in
155 of the 695 selected-shape cells; the finite-element nodal diagnostic
flags 159.  They are confined away from every retained surface-fed
characteristic: no resolved path in any of the three sweeps has negative $u$
or a horizontal reversal.  The low values of $J$ are therefore caused by
accumulated adverse shear along positive-speed paths, rather than by a
characteristic entering counterflow.

## Numerical checks

The 25-run audit includes higher-resolution repeats of the important
intermediate, raw-minimum, short-wave, and long-wave cells, together with
grading, source-count, and coordinate-gauge checks.  The refined
selected-shape minimum remains positive at
$J=0.42729$ (raw value 0.20987, $C_{\min}=-0.79592$).  The refined
$\lambda/H=2$ raw-minimum case remains positive at $J_{\rm raw}=0.14110$ with
$C_{\min}=-0.85924$.  Source-count and quarter-wavelength coordinate-gauge
audits of the slow $q=0.24$, $\phi=\pi$ case change the reported central-flux
minimum only from 0.73811 to 0.73896 and 0.74137, respectively.

An initially tested quadratic endpoint grading produced a false sign change
for the thin-neck long wave by creating increasingly anisotropic cells.  That
result was discarded.  With the adopted quasi-uniform coordinate, the
$\lambda/H=12$ case gives $J=1$ on $128\times96$, $160\times120$, and
$192\times144$ meshes; $C_{\min}$ converges from $-0.06104$ through
$-0.06169$ to $-0.06238$.  Quadratic grading is consequently restricted to
$\lambda/H\le1/4$, where it resolves the vertically screened boundary layers.
Audits on both sides of this switch preserve positive $J$.

For the 695 selected-shape numerical cells, the median, 99th-percentile, and
maximum relative residual of $-wJ+uK=a_0$ are $9.6\times10^{-4}$,
$7.8\times10^{-3}$, and $1.66\times10^{-2}$.  The corresponding maximum
Runge--Kutta displacement is 0.0167 wavelengths.  Across the 25 focused
audits the largest identity residual is $3.12\times10^{-3}$, no path reverses,
and neither $J$ nor the raw-to-exit $J$ becomes negative.

## Interpretation

Short wavelengths create large bed slopes but poor communication with the
surface.  Long wavelengths communicate with surface-fed characteristics but
produce weak spatial gradients.  The strongest distortion should therefore
remain at an intermediate wavelength.  A localized feature can evade this
tradeoff by containing a spectrum: longer modes couple to the surface while
shorter modes concentrate adverse shear.  Two modes are the minimal smooth
test of that proposed mechanism; failure of this family to fold would not be
a proof that a nonsmooth boundary is required.

In fact, the modeled elliptical bed itself is analytic over the modeled bed
arc; its finite domain has hydrostatic open cuts and corners where those cuts
meet the bed and surface.  The ellipse calculation therefore does not isolate
a local bed kink as the cause: the open side joins remain sharp and can
influence the whole Stokes field.  The comparison instead highlights
nonperiodic open-boundary throughflow and its boundary-energy flux as
plausible distinctions that remain to be separated from corner regularity.
The periodic one- and two-mode calculations isolate bed spectral content but
do not reproduce those boundary conditions.  No fold occurs in the sampled slices: the
full $(q,\phi,\lambda/H)$ scan at $A/H=0.85$, the dense $(q,\phi)$ scan at
$A/H=0.95$ and $\lambda/H=3$, or the $(H,A/H,\lambda/H)$ maps at the selected
$q=0.125$, $\phi=3\pi/2$ shape.  These slices do not exclude a fold at an
unsampled joint combination of thin neck, wavelength, and harmonic shape,
and they cannot identify mathematical boundary nonsmoothness as necessary.
Because Stokes flow communicates boundary conditions globally, the fact that
an overturned path lies away from a corner is not by itself a numerical proof
that the corner is irrelevant.  Explicitly rounding the finite-domain side
junctions while retaining the same open-boundary tractions is the clean next
test of that narrower question.

## Reproducible outputs

- `data/steady_characteristic_two_harmonic_shape_scan.csv`
- `data/steady_characteristic_two_harmonic_thin_neck_shape_scan.csv`
- `data/steady_characteristic_two_harmonic_parameter_map.csv`
- `data/steady_characteristic_two_harmonic_validation.csv`
- `figures/steady_characteristic_two_harmonic_parameter_map.png` and `.pdf`
- `notebooks/steady_characteristic_two_harmonic_parameter_sweep.py`
- `notebooks/steady_characteristic_two_harmonic_validation.py`
- `notebooks/plot_steady_characteristic_two_harmonic_parameter_map.py`

No manuscript or LaTeX source is modified.
