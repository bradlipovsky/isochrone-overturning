# Revision checklist: *Isochrone overturning in steady glacier flow* (JOG-2026-0127)

Revision audit for the *Journal of Glaciology* major-revision submission (due **23 November 2026**). All edits live in `latex/manuscript.tex` (mirrored to `latex/main_v1.tex`), `latex/references.bib`, `figures/Figure1.tex`, and `latex/response_to_reviewers.tex`, with tracked-changes diffs generated against the original 1 September 2026 submission snapshot (`latex/main_v0.tex`).

Status key: 🟢 done; 🟡 partly done / author check; 🔴 not started / open.

## Audit authority

| Artifact | Pages | SHA-256 (prefix) | Status / note |
|---|---:|---|---|
| `latex/manuscript.tex` / `latex/main_v1.tex` | — | `d7fb16e2ac3e77d4` | Identical revised LaTeX source |
| `latex/manuscript.pdf` | 39 | `e1146bcc0a5486c9` | Clean build; zero errors, zero undefined refs/citations |
| `latex/main_v1.pdf` | 39 | `1ad9ba3429cf3ff0` | Clean build matching `manuscript.tex` |
| `latex/response_to_reviewers.tex` | — | `2c96e420e11d9879` | Consistent first-person (`I`/`my`) replies to all 40 R1 + R2 items |
| `latex/response_to_reviewers.pdf` | 10 | `e8e2270b1c7ce681` | Clean build; zero errors |
| `latex/main_diff.pdf` | 48 | `ce8a33a6a60f657f` | `latexdiff` (underline/strikeout) vs. `main_v0.tex` |
| `latex/manuscript-diff.pdf` | 46 | `49079f8a931aca87` | `latexdiff -t CFONT` vs. `main_v0.tex` |
| `figures/Figure1.pdf` | 1 | `609b717e89cbf31b` | New 3-panel incompressible schematic (Figure 2 in PDF) |
| `latex/references.bib` | — | `bbffdc1160a26dc9` | Added `DahlJensen2013` and `SchoofClarke2008` |

## Status at a glance

| Group | Total | 🟢 Done | 🟡 Partly done / author check | 🔴 Open |
|---|---:|---:|---:|---:|
| Reviewer 1 — General & Remarks (`R1G`, `R1C1`–`R1C5`) | 6 | 6 | 0 | 0 |
| Reviewer 1 — Specific comments (`R1S1`–`R1S10`) | 10 | 10 | 0 | 0 |
| Reviewer 2 — General & Major comments (`R2G`, `R2C1`–`R2C7`) | 8 | 6 | 2 | 0 |
| Reviewer 2 — Minor comments (`R2M8`–`R2M23`) | 16 | 16 | 0 | 0 |
| Editorial hygiene & submission (`S1`–`S4`) | 4 | 2 | 0 | 2 |
| **Total** | **44** | **40** | **2** | **2** |

---

## Reviewer 1 (`latex/LipovskyReview.pdf`)

### General & Remarks

| Status | ID | Reviewer comment | Action taken (manuscript & response) | Remaining / author check |
|---|---|---|---|---|
| 🟢 | R1G | Venue fit for *Journal of Glaciology* (superficial glaciological engagement; theorems with proofs) | Deepened glaciological grounding (Figure 1 Allan Hills map and dated core reversals; NEEM and basal spiral-flow citations; nunatak-motivated point-margin calculation in Figure 4); moved formal proofs to Appendix C and Parrenin–Hindmarsh comparison to Appendix E; replied in `response_to_reviewers.tex`. | — |
| 🟢 | R1C1 | Remark 1: Deepen Allan Hills context (schematic, return to real glacier after Stokes results, radargrams?) | Added Figure 1 (`figures/allan_hills_context.pdf`: 10 m Sentinel-2 map, Antarctica inset, ALHIC1901 and ALHIC1902 $^{40}\mathrm{Ar}_{\mathrm{atm}}$ reversals), added idealized nunatak point-margin experiment (Figure 4), and expanded Discussion (*Implications for old-ice sites*). Updated `R1C1` reply to explain that steeply dipping isochrones are hard to image with existing radar data (Holschuh and others, 2014) and that imaging them is a major goal of COLDEX Phase II (started 1 October 2026). | — |
| 🟢 | R1C2 | Remark 2: Method-of-characteristics interpretation (orig. Eqs. 57–58) | Added characteristic interpretation right after Equation (5), explicitly framed Equations (47)–(49) and the deformation-gradient form Equations (50)–(51) as a method-of-characteristics implementation, and explained glacier-scale application in Discussion. | — |
| 🟢 | R1C3 | Remark 3: Role of accumulation vs. melt; streamlines hitting surface again | Added explicit text right after Equation (4) and in Discussion (*Limits of the steady coordinate construction*) clarifying that accumulation need operate only on the originating portion of the surface, while downstream ablation or basal melt truncates trajectories at exit without invalidating the interior order relation. | — |
| 🟢 | R1C4 | Remark 4: Estimate percentage of work completed by an LLM | Expanded *Acknowledgements* with a task-by-task breakdown of author vs. AI-tool roles updated through November 2026 (OpenAI Codex and Google Antigravity/Gemini); replied explaining why a single percentage is not meaningful across heterogeneous tasks. Approved by author. | — |
| 🟢 | R1C5 | Remark 5: Appendix section letters; connection between $\mathcal{J}$ and continuum-mechanics Jacobian $\det|\partial\mathbf{x}/\partial\mathbf{X}|$ | Enabled appendix section lettering (`Appendix A`–`Appendix H`) with per-appendix equation numbering (`C1`–`C9`, `D1`–`D6`, `E1`–`E3`); added explanation in *Exact order theorem* and full derivation in Appendix D (Equations D4–D6) linking $\mathcal{J} = (\mathbf{F}\mathbf{T}_0)_x$ to $\det\mathbf{F} = 1$ and $\det[\partial(x,z)/\partial(\tau,x_0)] = \dot a_0 > 0$. | — |

### Specific comments

| Status | ID | Reviewer comment | Action taken (manuscript & response) | Remaining / author check |
|---|---|---|---|---|
| 🟢 | R1S1 | Line 5: Include time periods for epochs (Miocene, Pliocene) | Added `Miocene (23–5.3 Ma)` in Abstract and `Pliocene (5.3–2.6 Ma) and Miocene (23–5.3 Ma)` in Introduction. | — |
| 🟢 | R1S2 | Line 7: Word `"initially"` is jarring | Rephrased Abstract sentence to `"Whether steady glacier flow can overturn layers deposited in chronological order remains uncertain."` | — |
| 🟢 | R1S3 | Line 26: Swap `"useful only"` to `"most useful"` | Updated opening sentence of Introduction to `"Recovering old ice is most useful where its age and stratigraphic context can be established."` | — |
| 🟢 | R1S4 | Around line 50: Mention Dahl-Jensen et al. (2013) NEEM folded ice core | Added sentence in Introduction on NEEM Eemian reconstruction from folded basal ice and added `DahlJensen2013` to `latex/references.bib`. | — |
| 🟢 | R1S5 | Around line 66: Cite Schoof and Clarke (2008) | Cited Schoof and Clarke (2008) in Introduction and Discussion for non-Newtonian normal-stress secondary spiral flows and added `SchoofClarke2008` to `latex/references.bib`. | — |
| 🟢 | R1S6 | Line 187: Move Parrenin–Hindmarsh subsection to Appendix | Moved entire subsection (and the $x_0$-vs.-$\pi_0$ remark) to Appendix E, leaving a one-sentence pointer in the main text. | — |
| 🟢 | R1S7 | Line 201: Use `"Stokes flow"` rather than `"Full Stokes"` | Replaced `"full-Stokes"` / `"Full-Stokes"` with `"Stokes"` / `"Stokes flow"` throughout headings, keywords, prose, and tables; updated residual symbol $R_{\mathrm{FS}} \to R_{\mathrm{S}}$. | — |
| 🟢 | R1S8 | Line 274: Move proofs to Appendix with full derivation steps | Moved formal proofs of both Theorem 1 and Proposition 1 to Appendix C (Equations C1–C9) with every intermediate step written out explicitly. | — |
| 🟢 | R1S9 | Orig. Eq. (61): Write $-\mathbf{t}\cdot\boldsymbol{\sigma}\cdot\mathbf{n}$ | Updated Equation (53) to $\tau_b = -\mathbf{t}\cdot\boldsymbol{\sigma}\cdot\mathbf{n}$. | — |
| 🟢 | R1S10 | Line 333: Explain what role dissipating mechanical energy plays | Added explicit explanation after Equation (53): $C > 0$ implies $\tau_b u_t = C\tau_b^2 \ge 0$, so the bed passively resists motion rather than injecting mechanical work to force overturning. | — |

---

## Reviewer 2 (`latex/Review_for_Lipovsky.pdf`)

### General & Major comments

| Status | ID | Reviewer comment | Action taken (manuscript & response) | Remaining / author check |
|---|---|---|---|---|
| 🟡 | R2G | Summary: Derivations hard to follow; weak link between theory and numerics; Figure 3 unclear; single idealized run insufficient | Reorganized kinematic section around physical question, expanded schematic to 3 panels (Figure 2), replaced circular-trough Figure 3 with point-margin (Figure 4 + Figure S1) and 750-case periodic sinusoidal-bed sweep (Figure 5), and added explicit theoretical links in text and response letter. | Pending author confirmation on `R2C2` inlet-line vs. surface-deposition distinction. |
| 🟢 | R2C1 | Major 1: Relation to Allan Hills (either mimic Allan Hills partially or frame strictly as theoretical) | Added Figure 1 Allan Hills context, replaced circular trough with nunatak-motivated point-margin experiment (Figure 4) + periodic control (Figure 5), and explicitly bounded all Allan Hills claims across Abstract, *Stokes numerical experiments*, Discussion, and Conclusions; added dedicated `R2C1` reply. | — |
| 🟡 | R2C2 | Major 2: Clarify Figure 3 panels; show velocity/streamlines to distinguish from recirculation; plot $\mathcal{P}, \mathcal{R}, 1+\mathcal{P}-\mathcal{R}$; run more simulations varying geometry, viscosity, temperature, sliding law | Removed open-cut circular trough; added steady Stokes point-margin experiment (Figure 4: geometry, BCs, $u, w$, open streamlines, $(\sigma_0, m_0)$ map, fold evolution, close-up; Figure S1: mesh/truncation convergence) and 750-case periodic sinusoidal-bed sweep (Figure 5). Explained why $\mathcal{J}_{\mathrm{T}} = T_x$ is used instead of $1 + \mathcal{P} - \mathcal{R}$ (regular through counterflow). Fixed viscosity, isothermal state, and linear sliding approved by author. | Pending author confirmation on the framing distinction in Figure 4 (steady Stokes flow folding an incoming horizontal material line from the right inflow boundary vs. surface deposition inside the 2 km domain). |
| 🟢 | R2C3 | Major 3: Reorganize derivations around physical purpose; compress orig. lines 83–133; show missing steps to orig. Eq. (16); long nomenclature | Rewrote opening of *Steady age transport and the order Jacobian* around the two-parcel physical question and introduced $\mathcal{J}$ immediately in Equation (1); cut verbose streamfunction prose; moved Parrenin–Hindmarsh to Appendix E; wrote full step-by-step derivation from Equation (10) to $\tau_\psi$ and $\mathcal{J}$ in Appendix C (Equations C1–C8); audited and completed Appendix A (Table 1: added $b(x)$ and $\tau_{zz}$, clarified $z_0$, removed unused Poisson bracket). | — |
| 🟢 | R2C4 | Major 4: Clarify $\tau$ vs. $t$; define `"passive"` and `"non-diffusive"` | Explicitly defined `"passive"` and `"non-diffusive"` before Equation (5) (noting neither implies age is conserved along a trajectory) and clarified that $\tau(x,z)$ is the Eulerian age field while $t$ is elapsed time since deposition ($\tau = t$ along trajectories started at $\tau = 0$). | — |
| 🟢 | R2C5 | Major 5: Why are streamfunctions, age coordinates, and Jacobian derivations before orig. Eq. (22) needed? | Added dedicated motivation in paragraph 2 of *Steady age transport and the order Jacobian* (contrasting numerical pair tracking against isolating dynamical controls $u_z/u^3$, proving analytical preservation bounds, and linking to $-\dot a_0\tau_z = \lambda_z^{-1}$) and replied in `R2C5`. | — |
| 🟢 | R2C6 | Major 6: SIA sign step too compressed ($\tau_{xz} \simeq \eta u_z$ needed); distinguish uphill slopes, stagnation, flow reversals from $s_x < 0$ | Rewrote *Shallow-ice flow* (Equations 34–35) to state both the hydrostatic shear-stress balance and the leading-order SIA strain-rate approximation ($w_x \ll u_z \Rightarrow \tau_{xz} \simeq \eta u_z$), qualified the result explicitly to strictly downhill one-way SIA flow ($s_x < 0, u > 0$), excluded divides/stagnation/reversals/$s_x > 0$, and updated Abstract and Conclusions. | — |
| 🟢 | R2C7 | Major 7: Demonstrate thin-film scaling $R_{\mathrm{FS}} = O(\varepsilon^2 U/H)$; remind local $R_{\mathrm{FS}}$ vs. integrated $\mathcal{F}$; show substitution into $\mathcal{F}$ | Rewrote *Regular thin-film Stokes flow* (Equations 36–39) deriving $R_{\mathrm{S}} = O(\varepsilon^2 U/H)$ explicitly from $H, L, U, W = O(\varepsilon U), \eta_0, \tau_{xz}, \tau_{xx}, p_{\mathrm{nh}}, \tau_{xz}^s$, reminded local $R_{\mathrm{S}}$ ($\mathrm{yr}^{-1}$) vs. integrated dimensionless $\mathcal{F}$ (Equation 31), and wrote out the substitution in Equation (38). | — |

### Minor comments

| Status | ID | Reviewer comment | Action taken (manuscript & response) | Remaining / author check |
|---|---|---|---|---|
| 🟢 | R2M8 | Orig. Figure 1: Add illustrative diagrams for ordered, vertical tangent, and overturned limb | Rebuilt `figures/Figure1.tex` (now Figure 2 in `manuscript.pdf`) as a 3-panel exact incompressible schematic: *(a) Ordered stratigraphy ($\mathcal{J}>0$)*, *(b) Vertical tangent ($\mathcal{J}=0$)*, *(c) Overturned limb ($\mathcal{J}<0$)*. Visually inspected. | — |
| 🟢 | R2M9 | Orig. lines 94–96: Unclear last sentence of paragraph | Rewrote forward pointer clearly to explain that the Discussion subsection on limits of the steady coordinate construction examines cases where Equations (2)–(5) break down (basal sources, multiple accumulation zones, stagnation, recirculation). | — |
| 🟢 | R2M10 | Orig. Figure 1: $\tau$ annotated as `"isochrone τ"` instead of stratigraphic age | Updated all three panels in `figures/Figure1.tex` to read `"isochrone (age τ)"`. | — |
| 🟢 | R2M11 | Orig. lines 97–99: Remove or move elementary streamfunction definition | Removed the elementary fluid-mechanics sentence explaining constant-$\psi$ curves and flux per unit width. | — |
| 🟢 | R2M12 | Orig. line 103: `"wherever the assumptions hold"` — which assumptions? | Replaced with `"wherever Equations (2)–(5) hold."` | — |
| 🟢 | R2M13 | Orig. line 105: Define `"stratigraphic order"` vs. `"stratigraphic age"` | Explicitly defined *stratigraphic order* in paragraph 1 of *Steady age transport and the order Jacobian* as the horizontal spatial sequence of parcels along an equal-age surface, distinct from stratigraphic age $\tau$. | — |
| 🟢 | R2M14 | Orig. lines 117–120: Unclear purpose of $x_0$-vs.-$\pi_0$ paragraph | Removed standalone paragraph from main text and integrated it as the opening sentence of Appendix E. | — |
| 🟢 | R2M15 | Orig. lines 187–200: Move Parrenin–Hindmarsh subsection to Appendix | Moved to Appendix E (same as `R1S6`). | — |
| 🟢 | R2M16 | Orig. line 207: Why specify $\eta > 0$ and `"may depend on"`? | Clarified after Equation (24) that $\eta$ is the positive scalar effective viscosity (constant for linear-viscous ice, or dependent on temperature and effective strain rate for Glen-law ice) and that positivity is required for the stress–shear sign relations. | — |
| 🟢 | R2M17 | Orig. lines 225–248: Confusing reverse-shear vs. trough-recirculation section | Rewrote *Stokes mechanics permits reverse shear* into two focused paragraphs cleanly separating (1) why Stokes terms in $R_{\mathrm{S}}$ allow local $u_z < 0$ and (2) how closed recirculating eddies differ from open-trajectory overturning. | — |
| 🟢 | R2M18 | Orig. line 232: Define `"one-way-flow assumption"` | Explicitly defined as $u(x,z) > 0$ along trajectories (Equation 3), excluding flow reversal and closed recirculation, both at Equation (3) and in *Stokes mechanics permits reverse shear*. | — |
| 🟢 | R2M19 | Orig. lines 234–236: Trim Moffatt (1964) corner-flow details | Removed the similarity-exponent and $146^\circ$ included-angle details. | — |
| 🟢 | R2M20 | Orig. line 234: `"two stationary rigid planes"` unclear | Removed along with the overly detailed Moffatt (1964) description. | — |
| 🟢 | R2M21 | Orig. line 247: Missing citation for `"published rough-bed solutions"` | Merged into paragraph 1 of *Stokes mechanics permits reverse shear* with explicit citations to Gudmundsson (1997a) and Waddington (2010). | — |
| 🟢 | R2M22 | Orig. line 214: Factor of 2 in $\tau_{xz} = \eta(u_z + w_x)$ vs. $2\eta(u_z + w_x)$ | Verified standard glaciological convention ($\boldsymbol{\tau} = 2\eta\mathbf{D}$ with $D_{xz} = \tfrac{1}{2}(u_z + w_x) \Rightarrow \tau_{xz} = 2\eta D_{xz} = \eta(u_z + w_x)$); made the intermediate $2\eta D_{xz}$ step explicit before Equation (28) and explained in `R2M22`. | — |
| 🟢 | R2M23 | Boxed equations throughout manuscript | Removed all `\boxed{}` frames from equations across `latex/manuscript.tex`. | — |

---

## Editorial hygiene & submission

| Status | ID | Item | Action taken | Remaining |
|---|---|---|---|---|
| 🟢 | S1 | Compilation, cross-references & numerical reconciliation | Compiled `manuscript.pdf`, `main_v1.pdf`, `response_to_reviewers.pdf`, `main_diff.pdf`, `manuscript-diff.pdf`, and `Figure1.pdf`/`.eps` with zero LaTeX errors and zero undefined main-text refs/citations; verified all numerical values against `notebooks/allan_hills_nunatak_margin_report.md` and `notebooks/steady_characteristic_sinusoidal_parameter_map.md`. | — |
| 🟢 | S2 | Copyedit polish in `latex/response_to_reviewers.tex` | Harmonized all author prose to singular first person (`I`/`my`) and replaced all three `"Section 2"` references with the unnumbered section title *Steady age transport and the order Jacobian*. Verified zero stray `we`/`our`/`us` outside quoted comments. | — |
| 🔴 | S3 | Author read-through of final PDFs | — | Final read-through of `latex/manuscript.pdf`, `latex/response_to_reviewers.pdf`, and `latex/main_diff.pdf`. |
| 🔴 | S4 | Upload revision to ScholarOne (`JOG-2026-0127`) | — | Submit clean revised manuscript, tracked-changes diff, figures, and response to reviewers at `mc.manuscriptcentral.com/jog` by **Monday, 23 November 2026**. |
