# Steady convergent-basin aspect-ratio sweep

## Parameterization

The figure uses the same 2x3 visual grammar as the sinusoidal-bed
parameter map: one geometry schematic and five heat maps with one
shared diverging scale. The auxiliary ablation and counterflow panels
have been removed.

Only the reflection-unique half of the family is shown. Interchanging
the left and right flank lengths is an exact horizontal reflection and
does not change the minimum J, so q=L_long/L_short>=1 is sufficient.

The independent third geometric coordinate is total basin aspect L/H.
The sweep holds H=200 m, beta H/eta=0.080, viscosity, basal drag, and
the linear sliding law fixed while using L/H=(15,20,30,45,60), or
L=(3,4,6,9,12) km. Each panel spans q=1--12 and Delta b/H=0.25--4.

The characteristic diagnostic retains the central 5--95% of positive
depositing flux and excludes the final H/12 beneath an ablating surface.
All cases use the validated 25 yr characteristic step. New aspect
cases use a 480 kyr safety ceiling; the reused, uncensored L/H=30
baseline retains its original 120 kyr ceiling. These are censoring
guards rather than analysis horizons.

## Result

The table contains 880/880 expected cells, of which
880 are numerically resolved. The global minimum is
J=-12.1555 at L/H=15,
q=12, and
Delta b/H=0.25.

Resolved classification by aspect (J<-0.05 overturned, |J|<=0.05
unresolved, J>0.05 ordered):

| L/H | resolved | overturned | unresolved | ordered | minimum J |
|---:|---:|---:|---:|---:|---:|
| 15 | 176 | 85 | 2 | 89 | -12.1555 |
| 20 | 176 | 96 | 4 | 76 | -8.73638 |
| 30 | 176 | 62 | 1 | 113 | -7.30852 |
| 45 | 176 | 0 | 0 | 176 | 0.194001 |
| 60 | 176 | 0 | 0 | 176 | 0.650153 |

The classification above is explicitly the buffered diagnostic.
At L/H=15 and 20, respectively, 11 and 6 cells classified as
ordered outside the H/12 surface buffer have negative raw minima
during the final near-surface return; no such disagreement occurs
at L/H=30, 45, or 60.

The isolated-minimum existence criterion is intentionally shown.
At L/H=15, 85 cells have J_min<-0.05 but only 51 have a negative
first percentile across retained source paths, so part of the
high-relief island is a narrow-source feature.

## Targeted validation

The re-entrant short-aspect branch and the long-residence
ordered limit were rerun with independent 240x32 Stokes-mesh,
12.5 yr characteristic-step, and 1801x151 velocity-sampler
refinements, all using the same physical H/12 outflow
clearance.

| scenario | minimum J across variants | maximum J across variants |
|---|---:|---:|
| intermediate-aspect closest-to-zero ordered | 0.070256 | 0.19673 |
| long-aspect long-residence ordered | 1.00013 | 1.00026 |
| short-aspect high-relief ordered | 0.477142 | 0.509839 |
| short-aspect low-relief overturn | -12.1557 | -9.45055 |

Every sentinel preserves its sign. The saturated negative
magnitude is sampler-sensitive, so it supports robust occurrence
of overturning rather than a precision estimate of the extremum.

## Outputs

- `figures/steady_tangent_convergent_parameter_map.png`
- `figures/steady_tangent_convergent_parameter_map.pdf`
- `data/steady_tangent_convergent_aspect_parameter_map.csv`
- `data/steady_tangent_convergent_aspect_parameter_map_metadata.json`
- `data/steady_tangent_convergent_aspect_validation.csv`

No manuscript source is modified by this analysis.
