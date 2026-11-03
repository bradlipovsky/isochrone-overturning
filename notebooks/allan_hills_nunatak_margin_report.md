# Allan Hills nunatak-margin experiment

## Setup

The ice domain ends at a true grounded point margin: bed and free surface meet at $x=0$, and exposed rock lies to the left. There is no left boundary facet, no left boundary condition, and no prescribed left influx. The bed has no penetration plus the same linear sliding law as the other experiments. The primary surface is fully traction free; a control instead imposes the normal velocity corresponding to uniform steady ablation while retaining zero tangential traction. At the right inlet, the horizontal velocity is the reduced SIA-shaped profile and $w=0$.

The thickness is $H(x)=H_R[1-\exp(-x/\ell_H)]/[1-\exp(-L/\ell_H)]$, with $H_R=360$ m, $L=2.0$ km, and $\ell_H=600$ m. Thus $H(0)=0$ with a finite opening angle. The immediate contact-line region is excluded from the material-line minimum because terrain coordinates are singular at the apex: the primary uses $H>25$ m and the uniform-ablation control uses the more permissive $H>10$ m.

For the fixed geometry, the effective surface mass balance $a_{\mathrm{eff}}=u_s s_x-w_s$ is diagnosed. In strict two dimensions it absorbs the right-side influx; in the real Allan Hills section it may also represent omitted lateral flux divergence.

## Results

| case | left thickness (m) | mesh cells | resolution (m) | surface mode | minimum J, m0=0 | left outward flux (m2/yr) | surface outward flux (m2/yr) | age-capped paths | maximum abs detF-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point_margin_coarse | 0 | 3283 | 30 | traction_free | -0.19464 | 0.000e+00 | 1.46618 | 0 | 1.98e-02 |
| point_margin | 0 | 7283 | 20 | traction_free | -0.19360 | 0.000e+00 | 1.46618 | 0 | 1.19e-02 |
| point_margin_refined | 0 | 17069 | 13 | traction_free | -0.19343 | 0.000e+00 | 1.46618 | 0 | 7.20e-03 |
| truncated_20m | 20 | 7562 | 20 | traction_free | -0.12785 | -7.221e-02 | 1.53839 | 0 | 1.25e-02 |
| truncated_10m | 10 | 7381 | 20 | traction_free | -0.18383 | -1.222e-01 | 1.58835 | 0 | 1.22e-02 |
| truncated_5m | 5 | 7384 | 20 | traction_free | -0.19165 | -7.278e-02 | 1.53896 | 0 | 1.20e-02 |
| truncated_2p5m | 2.5 | 7275 | 20 | traction_free | -0.19323 | -3.790e-02 | 1.50408 | 0 | 1.28e-02 |
| point_margin_uniform_ablation | 0 | 7283 | 20 | uniform_ablation | 0.30750 | 0.000e+00 | 1.46456 | 0 | 1.23e-02 |

With the fully traction-free surface, the production point-margin value is $J_{min}=-0.19360$ for initially horizontal layers; the refined value is -0.19343. 14 of 193 sampled right-inlet source heights overturn. The sign does not depend solely on the lowest source: after excluding all sources below $\sigma_0=0.05$, the minimum is still -0.05739. The global horizontal-layer minimum occurs at $x=486$ m, where $H=207$ m, well outside the contact-line exclusion. The exact point margin has zero left flux by construction. However, that solution diagnoses a spatially alternating surface balance that supplies a small rightward flow near the nunatak. When a uniform steady ablation rate is imposed instead, the minimum is 0.30750, so the layers remain ordered. All control paths exit and the minimum stays positive. The margin geometry alone is therefore not sufficient: the surface/lateral mass-flux closure controls whether the convergent counterflow exists.

The production right-boundary RMS vertical velocity is 0.00e+00 m yr$^{-1}$. The uniform-ablation control prescribes $a_s=-7.331e-04$ m yr$^{-1}$ and has RMS surface normal-speed error 2.34e-05 m yr$^{-1}$.

## Diagnostic validation

| variant | sample nx | sample nz | source count | source plane x (m) | actual time step (yr) | minimum J, m0=0 | minimum J for source sigma >= 0.05 | negative horizontal sources | maximum abs detF-1 | age-capped paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 501 | 161 | 193 | 1999.0 | 3.261 | -0.19360 | -0.05739 | 14 | 1.19e-02 | 0 |
| small-time-step | 501 | 161 | 193 | 1999.0 | 1.500 | -0.19360 | -0.05739 | 14 | 1.19e-02 | 0 |
| dense-sampler-and-sources | 751 | 241 | 385 | 1999.0 | 2.174 | -0.19330 | -0.06463 | 27 | 1.74e-02 | 0 |
| source-plane-20m-inside | 501 | 161 | 193 | 1980.0 | 3.966 | -0.17685 | -0.06935 | 15 | 6.13e-03 | 0 |
| larger-terrain-buffer | 501 | 161 | 193 | 1999.0 | 3.255 | -0.19355 | -0.05740 | 14 | 1.14e-02 | 0 |

The overturning sign is unchanged by halving the trajectory step, increasing terrain and source resolution, enlarging the terrain buffer, or moving the source plane from 1 m to 20 m inside the inlet. The displaced source plane gives $J_{min}=-0.17685$, ruling out the lower-right boundary corner as the sole cause.

## Interpretation limits

The margin corner can carry a local stress singularity, so only minima outside the stated thickness cutoff are interpreted. Positive-thickness truncations and the exact-margin mesh sequence test this sensitivity; their shrinking artificial faces are traction free and can carry small fluxes, so they are numerical checks rather than physical boundary models. This remains a two-dimensional fixed-geometry calculation: any implausible diagnosed surface balance is evidence that lateral convergence around the nunatak matters dynamically.
