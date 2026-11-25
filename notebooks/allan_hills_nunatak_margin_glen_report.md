# Allan Hills nunatak-margin experiment (glen)

## Setup

The ice domain ends at a true grounded point margin: bed and free surface meet at $x=0$, and exposed rock lies to the left. There is no left boundary facet, no left boundary condition, and no prescribed left influx. Ice obeys isotropic Glen rheology ($n=3$, $A_{\mathrm{G}}=6.80e-26$ Pa$^{-3}$ s$^{-1}$ $=2.146e-18$ Pa$^{-3}$ yr$^{-1}$ at $-25^\circ$C, regularization $D_0=1.0e-10$ yr$^{-1}$). The bed has no penetration plus the same linear sliding law as the other experiments. The primary surface is fully traction free; a control instead imposes the normal velocity corresponding to uniform steady ablation while retaining zero tangential traction. At the right inlet, the horizontal velocity is the reduced SIA-shaped profile and $w=0$.

The thickness is $H(x)=H_R[1-\exp(-x/\ell_H)]/[1-\exp(-L/\ell_H)]$, with $H_R=360$ m, $L=2.0$ km, and $\ell_H=600$ m. Thus $H(0)=0$ with a finite opening angle. The immediate contact-line region is excluded from the material-line minimum because terrain coordinates are singular at the apex: the primary uses $H>25$ m and the uniform-ablation control uses the more permissive $H>10$ m.

For the fixed geometry, the effective surface mass balance $a_{\mathrm{eff}}=u_s s_x-w_s$ is diagnosed. In strict two dimensions it absorbs the right-side influx; in the real Allan Hills section it may also represent omitted lateral flux divergence.

## Results

| case | left thickness (m) | mesh cells | resolution (m) | surface mode | minimum J, m0=0 | left outward flux (m2/yr) | surface outward flux (m2/yr) | age-capped paths | maximum abs detF-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point_margin_coarse | 0 | 3283 | 30 | traction_free | -0.29278 | 0.000e+00 | 1.46431 | 0 | 2.61e-02 |
| point_margin | 0 | 7283 | 20 | traction_free | -0.29156 | 0.000e+00 | 1.46431 | 0 | 1.41e-02 |
| point_margin_refined | 0 | 17069 | 13 | traction_free | -0.29128 | 0.000e+00 | 1.46431 | 0 | 9.19e-03 |
| truncated_20m | 20 | 7562 | 20 | traction_free | -0.08968 | 3.257e-03 | 1.46105 | 0 | 1.70e-02 |
| truncated_10m | 10 | 7381 | 20 | traction_free | -0.25379 | -1.042e-01 | 1.56847 | 0 | 2.18e-02 |
| truncated_5m | 5 | 7384 | 20 | traction_free | -0.28323 | -6.243e-02 | 1.52674 | 0 | 1.70e-02 |
| truncated_2p5m | 2.5 | 7275 | 20 | traction_free | -0.28987 | -3.235e-02 | 1.49666 | 0 | 2.03e-02 |
| point_margin_uniform_ablation | 0 | 7283 | 20 | uniform_ablation | 0.30714 | 0.000e+00 | 1.46270 | 0 | 1.68e-02 |

With the fully traction-free surface, the production point-margin value is $J_{min}=-0.29156$ for initially horizontal layers; the refined value is -0.29128. 25 of 193 sampled right-inlet source heights overturn. The sign does not depend solely on the lowest source: after excluding all sources below $\sigma_0=0.05$, the minimum is still -0.17279. The global horizontal-layer minimum occurs at $x=512$ m, where $H=214$ m, well outside the contact-line exclusion. The exact point margin has zero left flux by construction. However, that solution diagnoses a spatially alternating surface balance that supplies a small rightward flow near the nunatak. When a uniform steady ablation rate is imposed instead, the minimum is 0.30714, so the layers remain ordered. All control paths exit and the minimum stays positive. The margin geometry alone is therefore not sufficient: the surface/lateral mass-flux closure controls whether the convergent counterflow exists.

For the plotted 30 m horizontal segment at $\sigma_0=0.049$, the endpoint reaches $J=0$ at 146.3 kyr and $J=-0.186$ at 154.5 kyr.

The production right-boundary RMS vertical velocity is 0.00e+00 m yr$^{-1}$. The uniform-ablation control prescribes $a_s=-7.322e-04$ m yr$^{-1}$ and has RMS surface normal-speed error 2.34e-05 m yr$^{-1}$.

## Diagnostic validation

| variant | sample nx | sample nz | source count | source plane x (m) | actual time step (yr) | minimum J, m0=0 | minimum J for source sigma >= 0.05 | negative horizontal sources | maximum abs detF-1 | age-capped paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 501 | 161 | 193 | 1999.0 | 3.358 | -0.29156 | -0.17279 | 25 | 1.41e-02 | 0 |
| small-time-step | 501 | 161 | 193 | 1999.0 | 1.500 | -0.29156 | -0.17279 | 25 | 1.41e-02 | 0 |
| dense-sampler-and-sources | 751 | 241 | 385 | 1999.0 | 2.238 | -0.29188 | -0.17935 | 49 | 1.55e-02 | 0 |
| source-plane-20m-inside | 501 | 161 | 193 | 1980.0 | 4.083 | -0.25572 | -0.17213 | 25 | 7.19e-03 | 0 |
| larger-terrain-buffer | 501 | 161 | 193 | 1999.0 | 3.349 | -0.29158 | -0.17279 | 25 | 1.50e-02 | 0 |

The overturning sign is unchanged by halving the trajectory step, increasing terrain and source resolution, enlarging the terrain buffer, or moving the source plane from 1 m to 20 m inside the inlet. The displaced source plane gives $J_{min}=-0.25572$, ruling out the lower-right boundary corner as the sole cause.

## Interpretation limits

The margin corner can carry a local stress singularity, so only minima outside the stated thickness cutoff are interpreted. Positive-thickness truncations and the exact-margin mesh sequence test this sensitivity; their shrinking artificial faces are traction free and can carry small fluxes, so they are numerical checks rather than physical boundary models. This remains a two-dimensional fixed-geometry calculation: any implausible diagnosed surface balance is evidence that lateral convergence around the nunatak matters dynamically.
