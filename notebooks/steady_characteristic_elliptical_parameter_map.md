# Direct steady-characteristic map for symmetric elliptical troughs

## Question

The earlier parameter map stopped a common-age cohort when the first retained marker left the domain and reported results at 2 and 8 kyr.  The present calculation instead asks whether each surface-origin path overturns before its own exit.  It therefore removes both the reporting-age choice and common-first-exit censoring.

## Calculation

For each fixed geometry, the full-Stokes velocity is solved with the same linear basal sliding law as the earlier study.  The equal-age tangent is then carried along steady characteristics,

\[
\frac{d\boldsymbol X}{dt}=\boldsymbol v(\boldsymbol X),\qquad
\frac{d\boldsymbol T}{dt}=\nabla\boldsymbol v(\boldsymbol X)\boldsymbol T,
\qquad \boldsymbol T(0)=(1,0).
\]

The horizontal order Jacobian is \(J=T_x\).  A negative value means that neighboring deposition sources have reversed horizontal order.  Particle age parameterizes a path through a steady velocity field; neither the glacier nor a cohort evolves in time.

The production calculation uses a 10 m Stokes mesh, a 601 by 181 bed-following velocity sampler, a 10 yr fourth-order Runge--Kutta step, and 161 candidate source positions.  Only positions with positive surface deposition are retained.  The central 90% are traced once, and nested central 70%, 80%, and 90% summaries are calculated from those paths.  The figure shows the conventional central-80% minimum.  Each path ends independently at its first sampled bed, surface, or side boundary.  Six slow cells required increasing the numerical safety ceiling from 120 to 500 kyr; every path then exited, so the final map contains no age-censored or failed cells.

## Parameter-map result

The grid contains 702 cells: 231 are geometrically invalid, 50 have no depositing surface source, and 421 have complete steady paths.  Applying a numerical resolution band of \(|J|\leq0.05\), 277 cells are resolved as overturned, 136 are resolved as ordered, five remain near zero, and three change sign when the retained source fraction is changed between 70%, 80%, and 90%.  These three cells are classified as trim-sensitive rather than overturned or ordered.

The absolute minimum lies at an edge of the central-80% source interval in 230 of 421 cells.  This makes the stated source convention important.  However, the first percentile of the per-source minima differs from the absolute minimum by at most 0.0092 and by 0.00066 at the median, so the resolved negative regions are not generally produced by a single isolated path.

The diagonal bands have a direct geometric origin.  With fixed horizontal radius \(a=175\) m and half-domain length \(L=128.3\) m, the maximum modeled bed slope is

\[
\max |b_x|=0.019255\,(H_c-H_b).
\]

Consequently, lines of constant \(H_c-H_b\) are also lines of constant ellipse vertical radius and bed slope.  The first resolved-overturn cells occur at thickness contrasts of 30--60 m, corresponding to maximum slopes of 0.58--1.16.  There is no special geometry at \(H_c\simeq2H_b\).  Nevertheless, slope is not a global phase criterion: the best single slope threshold misclassifies 95 of the 421 complete cells.

Horizontal counterflow identifies one overturning regime but is not required.  All 80 cells containing a retained path with a horizontal velocity reversal overturn.  An additional 197 cells overturn without a reversal, while 144 non-overturning cells also have no reversal.  The map therefore contains at least two kinematic regimes rather than one universal roughness threshold.

## Difference from finite-age cohort maps

At the same \(J<-0.05\) criterion, the direct-to-exit map finds 277 overturned cells.  The earlier common-cohort maps find 49 by 2 kyr and 115 by 8 kyr.  Every old negative cell is also negative in the direct map.  Of the 228 additional direct overturns missed at 2 kyr, 170 first cross after 2 kyr and 58 cross earlier but were hidden by common-first-exit censoring.  Of the 162 additional overturns missed at 8 kyr, 96 first cross after 8 kyr and 66 cross earlier but were censored.

The \(H_b=40\) m, \(H_c=130\) m case illustrates the censoring mechanism.  The legacy cohort stops at 1.676 kyr when its first interior marker exits and therefore remains ordered.  A representative retained source crosses \(J=0\) at about 2.20 kyr and exits at about 3.43 kyr.  The production minimum-source path gives the same interpretation, crossing at 2.21 kyr and exiting at 3.54 kyr.

## Numerical checks

The direct audit contains 95 rows: 86 completed calculations and nine consistent no-deposition cases.  It varies the integration step from 20 to 2.5--5 yr, the candidate source grid from 81 to 321 points, the velocity sampler from 301 by 91 to 901 by 271, and the Stokes mesh from 10 to 5 m.  No numerical check changes the sign of the central-80% minimum.  Only the \(H_b=10\) m, \(H_c=70\) m case moves across the \(|J|=0.05\) resolution threshold.  The largest changes in minimum \(J\) are 0.00028 for time step, 0.0066 for source count, 0.0028 for sampler size, and 0.0109 for mesh spacing.

Adaptive integration of the selected minimum-source characteristic agrees with the vectorized fixed-step result to at most \(1.01\times10^{-4}\).  The counterflow audit retains two horizontal reversals and gives \(J_{\min}=-0.9701\).  The slow ordered case remains positive, with \(J_{\min}=0.1059\).

The buffered Eulerian tangent field is retained as a numerical audit, not as the production estimator.  Its value depends on the outflow and provenance masks.  In particular, the \(H_b=60\) m, \(H_c=240\) m Eulerian estimate is mask-sensitive and positive at the conventional 20 m buffer, whereas the independent direct path calculation gives \(J_{\min}=-0.5013\).  Direct characteristics remove this arbitrary spatial buffer and preserve each source identity through counterflow.

## Outputs

- `figures/steady_characteristic_elliptical_parameter_map.png`
- `data/steady_characteristic_elliptical_parameter_map.csv`
- `data/steady_characteristic_elliptical_parameter_map_audit.csv`
- `data/steady_tangent_elliptical_parameter_map.csv` (buffered Eulerian audit)
- `data/steady_tangent_elliptical_parameter_map_audit.csv`
- `notebooks/steady_characteristic_elliptical_parameter_map.py`
- `notebooks/steady_characteristic_elliptical_parameter_audit.py`

No LaTeX manuscript file is modified by this analysis.
