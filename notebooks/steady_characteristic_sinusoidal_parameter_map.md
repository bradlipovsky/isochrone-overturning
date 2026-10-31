# Steady-characteristic analysis over a sinusoidal bed

## Main result

No isochrone overturning occurs anywhere that the surface-fed diagnostic is
numerically resolved.  The extended grid contains 750 successful numerical
cases and 80 formal flat-bed cells.  Another 210 short-wave cells are labeled
as screened below numerical signal rather than assigned a potentially
roundoff-controlled $J$.  The smallest resolved production diagnostic is

\[
J_{\min}=0.642810,
\]

at mean thickness $H=50$ m, amplitude $A/H=0.85$, and wavelength
$\lambda/H=3$.  Even the raw minimum carried through the final near-surface
ablation return remains positive, $J_{\min}^{\rm raw}=0.442762$.  The
corresponding first-percentile values across sources are 0.645418 and
0.445389, so this is not a single-source outlier.

This is a direct steady diagnostic, not a time-dependent isochrone
simulation.  Characteristics are integrated through one fixed velocity field
and their equal-age tangent is evolved variationally.

## Geometry and nondimensional controls

The model contains exactly one periodic wavelength, $0\le x<\lambda$, with
a flat surface at $z=0$ and bed

\[
b(x)=-H+A\cos(2\pi x/\lambda).
\]

Here $H$ is mean thickness, $A$ is bed-elevation amplitude, peak-to-trough
relief is $2A$, minimum thickness is $H-A$, and maximum bed slope is
$2\pi A/\lambda$.  The domain length is rebuilt for every wavelength, so it
always contains exactly one period.

With the viscosity and linear sliding law fixed, the flow depends on three
nondimensional controls,

\[
\epsilon=A/H,\qquad \Lambda=\lambda/H,\qquad
\delta=\beta H/\eta=H/(2500\ {\rm m}).
\]

The production grid contains

- $H=(50,100,160,240,320)$ m, or $\delta=0.020$--0.128;
- 13 amplitudes, $A/H=0$--0.85; and
- 16 wavelengths, $\lambda/H=1/12$--12, including $1/12$, $1/8$, $1/4$,
  and $1/2$.

Thus the dimensional sweep spans wavelengths from 4.2 to 3840 m and bed
amplitudes from 0 to 272 m.  The flow uses linear viscosity
$2.5\times10^8$ Pa yr, basal drag $\beta=10^5$ Pa yr m$^{-1}$, surface slope
0.05 rad, and the same linear sliding and Nitsche bed treatment used by the
other PR40 calculations.

## Diagnostic and source convention

The full periodic surface is searched for every connected accumulation
interval using $a_s=-w$.  The central 80% of positive accumulation flux in
each interval supplies the characteristics; this removes slow separatrix
endpoints without assuming which part of the surface is accumulating.

Sources lie exactly at the physical surface.  This detail is essential in the
flat-bed limit: a fixed dimensional source depth eventually lies beneath the
vanishingly shallow surface-connected streamtube as $A\to0$.  Exact-surface
initialization recovers $J\to1$ smoothly down to $A=0.001$ m in direct tests.

For every source,

\[
\dot{\mathbf X}=\mathbf v(\mathbf X),\qquad
\dot{\mathbf T}=\nabla\mathbf v(\mathbf X)\mathbf T,
\qquad \mathbf T(0)=(1,0),
\]

and $J=T_x$.  Horizontal positions remain unwrapped while velocity queries
are wrapped into the periodic Stokes cell.  Each path ends at its own first
sampled surface or bed-boundary exit (the numerical mask is inset by 0.5% of
local thickness).  The primary minimum excludes only the upper
5% of local thickness beneath an ablating surface; the accumulating side is
eligible from its source.  A separate raw minimum retains every point through
surface exit.  Both diagnostics remain positive throughout the sweep.

## What controls the pattern?

The map contains one broad distortion trough centered at intermediate
wavelength, rather than the several re-entrant $J=0$ branches found for the
elliptical trough.

### Short-wave screening

Bed-induced Fourier modes decay upward over the length $k^{-1}$.  The useful
local shielding number is $\chi=2\pi(H-A)/\lambda$, because the minimum gap
sets the strongest possible connection to the surface.  As $\lambda/H\to0$
at fixed $A/H<1$, the bed perturbation is screened from surface-connected
flow and $J\to1$.

The new production cells resolve this return directly where the surface
signal remains above numerical leakage.  At $A/H=0.85$, $J$ changes from
0.8209 at $\lambda/H=1/2$ to 0.9512 at $1/4$ and 0.99852 at $1/8$; the values
are nearly independent of the five sampled thicknesses.  At $1/12$, however,
the maximum surface signal is only about 0.8 times the RMS basal normal
leakage, so those cells are screened rather than interpreted.  The plotted
dots mark every cell for which the surface signal is less than ten times that
leakage.  This is a numerical-reach boundary, not a physical phase boundary.

The thin-neck audit remains surface-coupled farther into the short-wave
regime.  At $A/H=0.95$, the primary minima for $\lambda/H=(1/2,1/4,1/8,1/12)$
are $(0.8359,0.8008,0.8602,0.9394)$.  Refining the $1/12$ mesh changes the last
value to 0.9466.  Even this extreme family is ordered and turns back toward
one at the shortest wavelengths; its positive short-wave trough is near
$\lambda/H=1/4$.

### Intermediate coupling

At $\lambda/H\simeq2$--4, the bed signal reaches much of the ice column while
retaining strong spatial velocity gradients.  Surface pumping and accumulated
tangent compression are both large.  For $H=160$ m and $A/H=0.85$, normalized
gross surface exchange peaks at 0.635 near $\lambda/H=2.5$, and $J$ reaches
its minimum, 0.6439, at $\lambda/H=3$.

Surface exchange alone is not a complete scaling: over the grid its
correlation with $1-J_{\min}$ is 0.756.  Multiplying it by the empirical
gradient factor $\sqrt{kH}$ raises the correlation to 0.844, but that is a
descriptive collapse rather than a proposed law.

### Ordered long-wave limit

The long-wave branch approaches lubrication flow and is provably ordered.
With $y$ measured above the bed and
$G=\rho g\sin\alpha$,

\[
u=G\left[\frac{h}{\beta}+\frac{hy-y^2/2}{\eta}\right],\qquad
q=G\left[\frac{h^2}{\beta}+\frac{h^3}{3\eta}\right].
\]

For a deposited streamline with streamfunction $\psi_0=q(h_0)$,
differentiating its travel-time integral gives

\[
J=u(x)\left[\frac{1}{u_s(x_0)}-q_x(x_0)
\int_{x_0}^{x}\partial_\psi(1/u)\,d\xi\right].
\]

Deposition implies $q_x>0$, while
$\partial_\psi(1/u)=-u_y/u^3\le0$, hence $J>0$.  In the additional
$\lambda/H=24$, $A/H=0.65$ audit, the primary/raw values are
0.9840/0.9839.  In the sliding-dominated plug sublimit, the minimum tends
toward one.

### Thin-neck direction

The limit $A/H\to1$ is singular because the minimum thickness vanishes.
Nevertheless, at $H=160$ m and $\lambda/H=3$, primary/raw $J$ remains
positive at $A/H=0.90$ (0.6095/0.3846) and 0.95 (0.5803/0.3316).  Refining
the $A/H=0.95$ mesh changes these to 0.5535/0.2998.  The mesh dependence is
visible, but it is far too small to change the ordering conclusion at the
sampled point; results should not be extrapolated all the way to the pinched
domain at $A/H=1$.

### Mean thickness and linear sliding

At fixed $(A/H,\lambda/H)$ the five maps nearly collapse because all sampled
thicknesses have $\beta H/\eta\ll1$.  The median same-geometry spread in $J$
is $2.1\times10^{-4}$, the 95th percentile is 0.00477, and the maximum is
0.0105.  At the global-minimum geometry, $J$ changes only from 0.6428 at
$H=50$ m to 0.6453 at 320 m.

For small amplitude, the form-drag coefficient scales as $\eta A^2k^3$.
The corresponding form-to-basal drag number is

\[
R_f=\frac{(A/H)^2(kH)^3}{\beta H/\eta}.
\]

For $A/H\le0.12$, this collapses the computed drag partition well
($r=0.997$ between the logarithms of the predicted ratio and measured
non-basal/basal dissipation ratio).

## Counterflow is not the mechanism

Small Eulerian reverse-flow pockets occur in 160 of 750 resolved numerical
cases, mostly for short, steep beds, but the largest reverse speed is only
about 1.8% of the case maximum.  No retained deposition characteristic has
$u<0$, and none reverses horizontal direction.  At $A/H=0.85$, the Eulerian
reverse pockets are confined to $\lambda/H\le2$, whereas the strongest
tangent compression occurs at $\lambda/H=3$.  The intermediate $J$ trough is
therefore a surface-coupling and finite-strain effect, not counterflow
folding.

## Numerical checks

- Every resolved production cell completes.  The longest actual exit age is
  0.412 Myr, well below the 1.5 Myr safety ceiling.
- Sampled periodic surface-mass-balance closure error is at most
  $1.1\times10^{-6}$, and the discrete weak-form residual is at most
  $8.2\times10^{-13}$.
- The 210 cells whose surface signal is not at least ten times the RMS basal
  normal leakage are explicitly screened.  They do not enter the minimum-$J$
  count.  All resolved characteristic steps move less than 0.0151 wavelength;
  the 99th percentile is 0.0122 wavelength.
- The validation table changes mesh, characteristic step, sampler, source
  count, flux retention, and phase independently around the lowest-$J$
  geometry.  Its detailed values are reported in
  `notebooks/steady_characteristic_sinusoidal_validation.md`.
- The raw-to-exit diagnostic is lower than the primary metric in the roughest
  beds, but its global minimum remains 0.4428.  The no-overturn result
  therefore does not depend on removing the final ablation-surface layer.
- A method-version field is present in every row.  Restart logic reuses only
  successful rows from the exact-surface, ablation-only-buffer convention;
  failed or censored rows are always retried.

The separately reported physical strong-boundary energy imbalance omits the
Nitsche consistency and penalty work and is not a solver residual.  The
discrete weak-form residual is the appropriate algebraic check.

## Output files

- `data/steady_characteristic_sinusoidal_parameter_map.csv`
- `data/steady_characteristic_sinusoidal_parameter_map_metadata.json`
- `data/steady_characteristic_sinusoidal_validation.csv`
- `data/steady_characteristic_sinusoidal_smoke.csv`
- `figures/steady_characteristic_sinusoidal_parameter_map.png` and `.pdf`
- `figures/steady_characteristic_sinusoidal_limits.png` and `.pdf`
- `notebooks/steady_characteristic_sinusoidal_parameter_map.py`
- `notebooks/steady_characteristic_sinusoidal_parameter_sweep.py`
- `notebooks/steady_characteristic_sinusoidal_validation.py`
- `notebooks/plot_steady_characteristic_sinusoidal_parameter_map.py`

No manuscript or LaTeX source is modified.
