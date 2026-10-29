# Steady convergent-flow parameter map

## Diagnostic

For each fixed geometry, the full-Stokes velocity is solved once and held fixed. The equal-age tangent is then carried along surface-origin characteristics by

\[\frac{d\mathbf X}{da}=\mathbf v(\mathbf X),\qquad \frac{d\mathbf T}{da}=(\nabla\mathbf v)\mathbf T,\qquad J=T_x.\]

Every characteristic is followed to its first physical bed, surface, or divide exit. The 120 kyr ceiling is only a reported safety cap, not an analysis horizon. Particle age parameterizes a characteristic, but neither the glacier nor deposited cohorts are advanced in time. The plotted resolved map asks whether a retained path overturns before entering the stated ablating-surface buffer; the raw to-exit minimum remains in the CSV. No 2 or 8 kyr reporting choice is involved.

The source interval is fixed a priori to the central 5--95% of positive surface accumulation flux in each basin. The primary minimum also excludes the final two vertical mesh cells (16.7 m) below an ablating surface, where gradient interpolation and a one-path extremum are least reliable. The raw full-residence minimum and its surface clearance remain in the CSV.

This source trim is a physical definition, not merely a convergence setting. As accumulation approaches zero at a source-basin endpoint, residence age becomes singular or extremely long, so an untrimmed global minimum is not stable. The plotted phase map is explicitly conditional on the central 5--95% accumulation-flux convention; trim sensitivity is reported separately below.

The effect is first-order near onset. At q=7 and Delta b/H=2.50, the resolved minimum is -12.68, -0.150, and +0.147 under the 1--99%, 5--95%, and 10--90% trims; q=7 and Delta b/H=2.25 also changes sign. By contrast, the strongly overturned q=1/8, Delta b/H=3.5 case remains negative (-44.3, -3.70, -1.30). Source trim therefore dominates transition-boundary uncertainty, while the strong-overturn region is robust.

## Result

The screen contains 336 valid cells. Using an a-priori numerical classification, J_min<-0.05 is resolved overturning, J_min>0.05 is resolved ordered, and |J_min|<=0.05 is unresolved. The resolved diagnostic overturns in 124 cells and is unresolved in 2; the unbuffered diagnostic overturns in 124. The most negative resolved value is -7.309, while the most negative across-case 1st percentile of per-source minima is -5.265. In 20 mapped cells the single-source minimum crosses -0.05 while the per-source 1st percentile remains above +0.05, flagging onset controlled by a very narrow retained-source interval. Across the grid, 0 trajectories reached the safety cap.

## Flow-mechanics comparison

Peak required ablation ranges from 0.024 to 0.440 m/yr and generally strengthens as relief increases. Its gridwise Pearson correlation with J_min is -0.373. The significant-counterflow panel measures the fraction of sampled area moving opposite the local descent toward the basin center, after requiring speed greater than 1% of the domain maximum; its correlation with J_min is 0.575.

Counter-slope pockets are not themselves the overturn criterion: 196 cells contain significant counterflow but are not resolved as overturned, while 0 overturned cells contain no significant counterflow at this threshold. The panels therefore support cumulative strain along the retained characteristic—not ablation strength or counterflow alone—as the direct mechanism controlling J.

Across this sampled family, significant counterflow appears on the ordered side of the phase boundary and is present in every overturned cell: it precedes overturning as roughness/asymmetry increases, but is far from sufficient for overturning.

A compact geometric summary nevertheless emerges. Let S_max be the maximum absolute bed slope on the shorter flank, S_max=(pi/2) Delta b/L_short. The empirical threshold S_max approximately 0.917 (42.5 degrees) reproduces the isolated-minimum classification in 328/336 sampled cells; S_max approximately 1.10 (47.7 degrees) reproduces the broader-source criterion p01(J_min)<-0.05 in 328/336 cells. These are phase-map summaries, not universal local overturn criteria: J still accumulates along the full upstream path.

The first sampled relief ratio with resolved overturning at each affected q is:

| q | first sampled Delta b/H with J<-0.05 |
|---:|---:|
| 0.0833333 | 1.25 |
| 0.1 | 1.5 |
| 0.111111 | 1.75 |
| 0.125 | 2 |
| 0.142857 | 2.5 |
| 0.166667 | 2.75 |
| 0.2 | 3 |
| 0.25 | 3.75 |
| 4 | 3.75 |
| 5 | 3 |
| 6 | 2.75 |
| 7 | 2.5 |
| 8 | 2 |
| 9 | 1.75 |
| 10 | 1.5 |
| 12 | 1.25 |

## Numerical checks

Transition and reciprocal-counterflow cases use 120x16, 180x24, and 240x32 Stokes meshes; the extreme case uses 180x24, 240x32, and 300x40. All mesh comparisons hold physical clearance fixed at 16.7 m. Additional checks span 10--25 m clearances; 41--161 sources per basin; 12.5--50 yr RK4 steps; 1--99%, 5--95%, and 10--90% source-flux trims; and 1201x101 versus 1801x151 velocity samplers in transition and extreme cases.

Maximum within-case ranges are kept separate for numerical convergence and for deliberate changes to the diagnostic definition:

| category | check | range of isolated minimum J | range of per-source p01 J |
|---|---|---:|---:|
| numerical convergence | mesh at fixed clearance | 0.296 | 0.0923 |
| definition sensitivity | outflow clearance | 1.48 | 2.17 |
| numerical convergence | source count | 0 | 0.27 |
| definition sensitivity | source flux trim | 44.2 | 7.65 |
| numerical convergence | time step | 1.06 | 0.506 |
| numerical convergence | velocity sampler | 2.64 | 1.93 |

The isolated minimum is intentionally retained for the existence question, but the 1st percentile distinguishes a broad overturning source band from a one-source onset. Individual variants and all exit/counterflow diagnostics are in the validation CSV.

Applying the 16.7 m clearance only beneath ablating surface, rather than globally, changes J_min in 38/336 cells (maximum absolute change 0.0801) but changes no ordered/unresolved/overturned classifications.

At the transition, the isolated-minimum ranges are 0.00486 across the fixed-clearance mesh test, 0.00431 across velocity samplers, and 0.0218 across time steps; its sign classification is stable under those numerical changes.

The saturated extreme magnitude is not converged as a precise value: q=12, Delta b/H=4 spans 1.06 in J_min across time steps and 2.64 across velocity samplers (p01 ranges 1.93). Its negative sign is stable and the figure clips J below -1, so the extreme colors support robust occurrence and order of magnitude, not a converged amplitude.

## Why the global stationary-DG field is not mapped

A stationary Eulerian upwind-DG tangent solve was also tested with the correct curved-surface boundary value `T=(1,s_x)`, transported source labels, separate source-basin masks, and the same two-cell outflow buffer. It is reliable in the single-basin elliptical tests, but not here: the two inflow basins meet at a strongly compressed internal separatrix, where numerical branch mixing creates large spurious negative extrema.

The audit gives characteristic ordered: min -31.01, p01 -3.66, transition: min -30.32, p01 -2.10, strong counterflow: min -344.10, p01 -9.86. In particular, the independently ordered q=7, Delta b/H=2.25 case is falsely classified as strongly overturned. The Eulerian extrema are therefore rejected rather than plotted. Direct steady characteristics preserve basin identity and remain well posed through horizontal counterflow.

## Outputs

- `figures/steady_tangent_convergent_parameter_map.png`
- `data/steady_tangent_convergent_parameter_map.csv`
- `data/steady_tangent_convergent_parameter_map_validation.csv`
- `data/steady_tangent_convergent_eulerian_audit.csv`
- `data/steady_tangent_convergent_parameter_map_metadata.json`

No manuscript source is modified by this analysis.
