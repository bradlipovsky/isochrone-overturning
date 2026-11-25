# Steady-characteristic analysis over a sinusoidal bed (nonlinear Glen rheology)

## Main result

No isochrone overturning occurs anywhere that the surface-fed diagnostic is numerically resolved under isotropic nonlinear Glen rheology ($n=3$). Across the 1,040-cell grid, 745 numerical cases resolve cleanly and 80 formal flat-bed cells satisfy $J=1$ analytically. Another 215 short-wave cells have surface vertical-velocity signals smaller than ten times the RMS basal normal leakage and are explicitly left unclassified rather than assigned roundoff-controlled values; zero cells hit the characteristic safety ceiling.

The smallest resolved production diagnostic is

$$
J_{\min} = 0.368583,
$$

at mean thickness $H = 50$ m, amplitude $A/H = 0.85$, and wavelength $\lambda/H = 4$. The raw minimum carried through the final near-surface ablation return also remains positive, $J_{\min}^{\mathrm{raw}} = 0.256183$ (first-percentile values across retained sources: $0.369432$ and $0.256294$).

## Geometry, rheology, and nondimensional controls

One periodic cell occupies $0 \le x < \lambda$ with flat surface $z = 0$, bed $b(x) = -H + A\cos(2\pi x/\lambda)$, surface inclination $0.05$ rad, ice density $\rho = 917$ kg m$^{-3}$, and passive linear sliding ($\beta = 10^5$ Pa yr m$^{-1}$). Isotropic Glen rheology uses $n = 3$, cold-ice fluidity $A_{\mathrm{G}} = 6.8\times 10^{-26}$ Pa$^{-3}$ s$^{-1}$ $= 2.146\times 10^{-18}$ Pa$^{-3}$ yr$^{-1}$ ($-25^{\circ}$C; Cuffey and Paterson, 2010), and strain-rate regularization $D_0 = 10^{-10}$ yr$^{-1}$.

Measuring internal shear with the characteristic viscosity at mean driving stress $\tau_{\mathrm{d}} = \rho g H\sin(0.05\text{ rad})$,

$$
\eta_{\mathrm{G}} = \frac{1}{2 A_{\mathrm{G}} \tau_{\mathrm{d}}^2},
$$

gives sliding-thickness ratios $\beta H/\eta_{\mathrm{G}} = (0.011, 0.087, 0.355, 1.20, 2.84)$ for $H = (50, 100, 160, 240, 320)$ m. Unlike the constant-viscosity sweep ($\beta H/\eta = 0.020$--$0.128 \ll 1$), the Glen sweep spans the transition from sliding-dominated flow at $H = 50$ m to internal-shear-dominated flow at $H = 320$ m. Stronger surface-fast internal shear at larger $H$ slightly increases the minimum $J$ at $(A/H = 0.85, \lambda/H = 4)$ from $0.3686$ at $50$ m to $0.3982$ at $320$ m (maximum same-geometry spread across $H$: $0.0499$).

## Regime behavior and counterflow check

- **Short-wave screening ($\lambda/H \to 0$):** Bed-induced modes decay upward over $k^{-1}$, returning $J \to 1$ before falling below the basal-leakage floor. At $H = 160$ m and $A/H = 0.85$, $J_{\min}$ rises from $0.8841$ at $\lambda/H = 1/2$ to $0.9788$ at $1/4$ and $0.9988$ at $1/8$.
- **Intermediate coupling ($\lambda/H \simeq 3$--$5$):** Strain-rate softening over the bed crests deepens the intermediate-wavelength distortion trough relative to linear-viscous ice ($0.3686$ vs. $0.6428$), while keeping $J > 0$ throughout.
- **Long-wave and thin-neck probes:** At $H = 160$ m and $A/H = 0.65$, extending to $\lambda/H = 24$ gives $J_{\min} = 0.9656$ (raw $0.9537$). Thin-neck audits at $H = 160$ m, $\lambda/H = 3$ remain ordered at $A/H = 0.90$ ($0.3603$ primary / $0.1893$ raw) and $A/H = 0.95$ ($0.3067$ / $0.1140$; $0.2884$ / $0.0912$ on the $96\times 36$ mesh), and the $A/H = 0.95$ short-wave sequence returns monotonically from $0.7766$ ($\lambda/H = 1/2$, finer mesh) to $0.8537$ ($1/4$), $0.9149$ ($1/8$), and $0.9735$ ($1/12$, finer mesh).
- **Counterflow:** Small Eulerian reverse-flow pockets occur in deep short-wavelength troughs in 156 of 745 resolved cases (peak reverse speed $0.81\%$ of $\max|u|$), but zero retained surface-deposition trajectories encounter $u < 0$ or reverse horizontal direction.

## Numerical checks and validation

Each characteristic integrates with Courant-scaled RK4 step $\Delta t_{\mathrm{eff}} = 0.002(\Delta t / 25\text{ yr})(\lambda / \max|u|)$ and safety ceiling $\max(T_{\max}, 25\lambda/\max|u|)$, so every resolved production case advances by at most $0.0020$ wavelengths per step across seven orders of magnitude in Glen flow speed. Sampled periodic surface-mass-balance closure error is at most $1.6\times 10^{-6}$, and the relative discrete weak-form residual is at most $3.3\times 10^{-5}$.

Single-factor perturbations around the global-minimum cell ($H = 50$ m, $A/H = 0.85$, $\lambda/H = 4$, reference $J_{\min} = 0.368583$):

| Control | Tested minimum-$J$ range | Full span |
| --- | --- | --- |
| Stokes mesh ($48\times 18$ to $96\times 36$) | $0.351144$--$0.381221$ | $0.030077$ |
| Characteristic step ($0.5\times$ to $2\times$) | $0.367917$--$0.368614$ | $0.000697$ |
| Velocity sampler ($241\times 81$ to $481\times 181$) | $0.367396$--$0.368632$ | $0.001236$ |
| Candidate sources ($121$ to $481$) | $0.368583$--$0.369000$ | $0.000417$ |
| Retained accumulation flux ($70\%$ to $90\%$) | $0.368583$--$0.368583$ | $0.000000$ |
| Quarter-wave phase translation | $0.368274$ | $0.000309$ |

## Output files

- `data/steady_characteristic_sinusoidal_glen_parameter_map.csv`
- `data/steady_characteristic_sinusoidal_glen_parameter_map_metadata.json`
- `data/steady_characteristic_sinusoidal_glen_validation.csv`
- `data/steady_characteristic_sinusoidal_glen_smoke.csv`
- `figures/steady_characteristic_sinusoidal_glen_parameter_map.png` and `.pdf`
- `figures/steady_characteristic_sinusoidal_glen_limits.png` and `.pdf`
