# Synthesis: Physics & Algorithms for the Kerr + Accretion-Disk Offline Renderer

**Stage 3 (Opus) cross-model synthesis.** Distilled from six source papers, each analyzed independently by **Gemini 3.5 Flash** (thinking: high) and **Claude Sonnet**. Only information **cross-validated by both models** — or explicitly flagged where they diverge — is retained here. This is a research digest for orientation; the **authoritative formulas to implement in code remain `skills/kerr-physics/SKILL.md`** (project policy: no re-derivation; flag suspicious formulas for human review). The papers below are the *provenance* of those formulas, not a substitute for the skill file.

---

## 1. Source Corpus & Role

| Paper | Role in pipeline | Coordinate basis |
|---|---|---|
| **Bardeen, Press & Teukolsky 1972** (ApJ 178) | LNRF/ZAMO tetrad; all critical radii; circular-orbit E, L, Ω; geodesic RHS | Boyer-Lindquist |
| **Novikov & Thorne 1973** (Les Houches) | Relativistic thin-disk structure; flux F(r); ISCO zero-torque BC | Boyer-Lindquist |
| **Page & Thorne 1974** (ApJ 191) | Closed-form, viscosity-independent disk flux F(r) | Boyer-Lindquist |
| **Chan, Psaltis & Özel 2013** (GRay) | GPU geodesic integrator; pole fix; shadow fitting formulae | Boyer-Lindquist |
| **James, von Tunzelmann, Franklin & Thorne 2015** (DNGR/*Interstellar*) | Ray-bundle anti-aliasing; super-Hamiltonian integration; motion blur; disk models | Boyer-Lindquist |
| **Luminet 2019** | Historical/qualitative validation benchmarks (no new equations) | Boyer-Lindquist |

**Universal coordinate caveat (both models, all six papers):** every source works in **Boyer-Lindquist**; the active renderer uses **Cartesian Kerr-Schild (CKS)**. The CKS radius `r` *equals* the BL radial coordinate, so radius-indexed disk quantities (F(r), T(r), Ω(r)) port directly; tetrads, momenta, and geodesic RHS must be re-expressed in CKS per `SKILL.md` Part II.

---

## 2. Cross-Validated Constants at a = 0.999

Both models agree on these (sources noted); use as **regression anchors**, not hardcoded values (config lives in `configs/render.yaml`).

| Quantity | Value (a = 0.999, prograde) | Source / note |
|---|---|---|
| Outer horizon r₊ | ≈ **1.0447 M** | GRay; r₊ = 1 + √(1−a²) |
| ISCO r_ms | ≈ **1.18 M** | BPT72 & NT agree (Flash variants 1.16–1.18 M from series vs. exact Z₁,Z₂ cubic) |
| Photon orbit r_ph | → 1 M as a→M (≈1.18 M near-extremal) | BPT72 |
| LNRF orbital speed v^(φ) at ISCO | **½ c** (a→M limit) | BPT72 — both models stress: *not* ultrarelativistic |
| Radiative efficiency η = 1 − Ẽ(r_ms) | **≈ 0.30** at a=0.999 (use this); 0.42 = a→M ceiling; 0.057 at a=0 | Resolved (§7). Not in SKILL.md → put in `configs/render.yaml` if/when used. |

---

## 3. Geodesic Integrator (photon tracing)

**Constants of motion (BPT72, both models):** energy E = −p_t, axial angular momentum L = p_φ, Carter constant Q. Impact parameter b = −p_φ/p_t. These are conserved-quantity checks at every step.

**Two integration strategies, cross-validated:**
- **First-order (Carter-reduced) RHS** — BPT72 Σ dr/dλ = ±√V_r, etc. Kerr-specific, exact conservation, no drift. Best when committed to Kerr.
- **Super-Hamiltonian form** (DNGR) — dx^α/dζ = ∂H/∂p_α; numerically well-behaved **at radial turning points** where ±√R loses sign/precision. Preferred for robustness. GRay instead integrates the **second-order** geodesic equations (RK4) to stay metric-agnostic (non-Kerr extensibility) — a deliberate trade-off, not an error.

**Numerical hygiene for near-extremal a = 0.999 (GRay, both models):**
- Adaptive affine step Δλ ∝ (r − r_bh)/r, with accuracy parameter Δ ≈ 1/32 — shrinks near the horizon.
- Terminate at r ≤ r_bh + δ, δ ≈ 10⁻⁶ (don't cross the horizon).
- Constraint monitor ξ = g_μν k^μ k^ν as an in-flight accuracy guard.
- **BL pole fix:** if the normalization drifts near θ = 0/π, roll back and take one forward-Euler step at Δλ/9. ⚠ **Lower priority in CKS** — the BL polar coordinate singularity does *not* exist in Cartesian Kerr-Schild (both models note this), so this trick is informative but largely moot for our path.

**Validation targets (GRay fitting formulae, eqs. 11–14) — VERIFIED 2026-06-11** against the full paper text (`paper/paper.md` lines 84–92; the earlier `1303.5057v1.md` had these `formula-not-decoded`/image-dropped). ⟨R(a,i)⟩ and asymmetry A(a,i) give ground-truth shadow radius/asymmetry; **all trig arguments are in degrees**. The actual published forms are:

$$R \simeq R_0 + R_1\cos(2.14\,i - 22.9°)$$
$$R_0 = (5.2 - 0.209a + 0.445a^2 - 0.567a^3)\,M,\quad R_1 = \left[0.24 - \frac{3.3}{(a-0.9017)^2+0.059}\right]\times10^{-3}\,M$$
$$A \simeq A_0\,\sin^{n} i,\quad A_0 = (0.332a^3 + 0.176a^{21.7} + 0.0756a^{195})\,M,\quad n = 1.55(1-a)^{-0.022} + 1.3(1-a)^{0.98}$$

Note the actual structure: R is a **single cosine** in inclination (not a power series in i); R₁ is a **Lorentzian** in a (not a polynomial); A is a **power law** `sinⁿ i` with large fractional exponents. The Flash pass had earlier *recalled* a completely different (polynomial-in-i, ×10⁻⁴/⁻⁸/⁻⁵/⁻⁹) parametrization — **that recalled set was WRONG and has been discarded.**

Worked check against the verified formula: a=0.999, i=60° ⟹ R₀≈4.870, R₁≈−0.048, cos(105.5°)≈−0.267 ⟹ ⟨R⟩ ≈ **4.88 M** (the recalled ⟨R⟩≈4.88 coincidentally matched); A₀≈0.565, n≈1.806 ⟹ A ≈ **0.44 M** (the recalled A≈0.045 was wrong by ~10×). These are now a usable numeric regression gate. Note (both models): significant D-shape asymmetry appears only for a ≳ 0.99 **and** i ≳ 60°.

---

## 4. Accretion-Disk Model (Novikov–Thorne / Page–Thorne)

The disk = geometrically thin, optically thick gas on **circular equatorial geodesics**, inner edge at the ISCO, radiating as a local blackbody. Page–Thorne and Novikov–Thorne give the **same** time-averaged flux; both models confirm equivalence.

**Keplerian orbital field (cross-validated, directly usable):**
$$\Omega = \frac{1}{M(r_*^{3/2}+a_*)}, \quad \tilde{E}=\frac{\mathcal{G}}{\mathcal{C}^{1/2}}, \quad \tilde{L}=Mr_*^{1/2}\frac{\mathcal{F}}{\mathcal{C}^{1/2}}$$

with NT relativistic correction functions (r_* = r/M, a_* = a/M):
$$\mathcal{B}=1+a_*r_*^{-3/2},\ \mathcal{C}=1-3r_*^{-1}+2a_*r_*^{-3/2},\ \mathcal{D}=1-2r_*^{-1}+a_*^2r_*^{-2},$$
$$\mathcal{F}=1-2a_*r_*^{-3/2}+a_*^2r_*^{-2},\ \mathcal{G}=1-2r_*^{-1}+a_*r_*^{-3/2}$$

**Emitted surface flux (NT closed form, Flash-extracted, cross-checks Page–Thorne integral):**
$$F(r)=\frac{3M\dot{M}_0}{8\pi r^3}\,\frac{\mathcal{Q}(r)}{\mathcal{B}\,\mathcal{C}^{1/2}\,\mathcal{D}^{1/2}}, \qquad \mathcal{Q}(r)=\mathcal{B}\mathcal{C}^{1/2}\!\int_{r_{ms}}^{r}\frac{\tilde{E}\,\tilde{L}'-\tilde{L}\,\tilde{E}'}{\mathcal{B}\mathcal{C}^2\mathcal{D}^{1/2}}\,dr$$

**Both models agree on the key boundary behavior:**
- **Zero-torque inner BC:** W(r_ms) = 0 ⟹ F(r_ms) = 0. This is what produces the disk's sharp dark inner edge. Inside r_ms gas plunges on geodesics and does not radiate (set emission = 0).
- F(r) rises to a sharp peak just outside r_ms, then falls as ~r⁻³.
- Result is **independent of viscosity microphysics** — the α-parameter cancels.

**Temperature & spectrum:** T_eff(r) = (F(r)/σ)^{1/4}; locally Planck B_ν(T_eff). From F ∝ r⁻³ the large-r law is **T_eff ∝ r^(−3/4)**, which is exactly the renderer's SKILL.md **Decision B** T = T₀(6/r)^0.75. (Sonnet's NT structural detail of a T ∝ r^(−17/9) "middle region" is a Shakura–Sunyaev vertical-structure regime, *not* the bolometric T_eff law — do not conflate; Decision B remains the right rendering approximation.)

**Implementation (both models):** precompute the dimensionless f_NT(r) = Q/(B·C^{1/2}·D^{1/2}) as a **1-D CPU lookup table** indexed by r for fixed a = 0.999; the on-GPU shader reads the LUT instead of integrating Q per pixel.

---

## 5. Relativistic Radiative Transfer (observed intensity)

**Unanimous across Page–Thorne, NT, Luminet, and DNGR** — the single most-agreed result in the corpus:

1. Disk 4-velocity at intersection: u^μ = u^t(1,0,0,Ω), with
   $$u^t=\left(-g_{tt}-2\Omega g_{t\phi}-\Omega^2 g_{\phi\phi}\right)^{-1/2}.$$
2. Redshift factor (observer at rest at infinity):
   $$g=\frac{\nu_{obs}}{\nu_{emit}}=\frac{1}{u^t(1-\Omega b)}, \quad b=-p_\phi/p_t.$$
3. **Liouville invariant** I_ν/ν³ conserved ⟹ observed intensity
   $$\boxed{I_{\nu,obs}=g^3\,B_\nu\!\big(T_{eff}(r)\big)}$$
   (bolometric scales as g⁴). This captures gravitational redshift **and** Doppler beaming together.

**Visual signature to verify (Luminet, DNGR):** the approaching limb (left, for the standard sense) is blue-boosted and bright (g≈1.5×, v≈0.55c at the relevant radius); the receding limb is red and dim (g≈0.4×). The rear of the disk is lensed into view above *and* below the shadow (never occulted). At a = 0.999 + high inclination the shadow is a **D-shape** — if the renderer yields a symmetric circle, frame-dragging in the integrator is wrong.

---

## 6. Anti-Aliasing & Production Quality (DNGR)

For flicker-free, IMAX-grade output (relevant to our offline renderer, not real-time):
- Propagate **finite-width ray bundles** via geodesic-deviation (Sachs optical-scalar) ODEs — five variables {u,v,g,h,χ} integrated alongside the central ray — giving each pixel an explicit ellipse (δ₊, δ₋, μ) on the source. Enables analytic spatial filtering and **mipmap-level selection** for volumetric disks.
- **Pineault–Roeder denominator modification** (p̂_t − p̂_r, not +) is required for ingoing-bundle stability — both models flag this as must-preserve.
- Beam diameter = 2× pixel separation + truncated-Gaussian weighting prevents background-star streaking near critical curves (note: relates to this project's prior EWA star-streaking fix).
- Analytic motion blur via camera-velocity auto-differentiation (~2× cost vs ~4× for 4-sample Monte Carlo).

---

## 7. Cross-Model Agreement & Flagged Discrepancies

**Strong agreement (Flash ≡ Sonnet):** all critical-radius formulae and a=0.999 values; the geodesic constants E/L/Q and RHS; the NT/Page–Thorne flux profile and zero-torque ISCO BC; the I_obs = g³B_ν redshift law; the BL→CKS coordinate caveat; the LUT implementation strategy.

**Flagged discrepancies — RESOLVED 2026-06-11** (human-reviewed; resolutions baked into §2–§4 above):
- ✅ **Radiative efficiency at a = 0.999.** Sonnet/BPT72 quoted the **extremal (a=M) 42.3%**; Flash's NT pass gave **≈30% at a=0.999**. Both correct at their stated spin. **Decision: η ≈ 0.30 for a=0.999**; 0.42 is the a→M asymptotic ceiling, not the working value. SKILL.md currently specifies no η — if/when efficiency enters the renderer it belongs in `configs/render.yaml`, not hardcoded (config-driven policy); extend SKILL.md only with explicit human sign-off.
- ✅ **GRay shadow-fit coefficients** (⟨R⟩, A) — **RESOLVED & VERIFIED.** The full paper text was loaded (`paper/paper.md`, eqs. 11–14, lines 84–92) and the actual published forms transcribed into §3. The coefficients Flash had *recalled* earlier turned out to be **entirely wrong** — a different algebraic structure (polynomial-in-i with ×10⁻⁴/⁻⁸/⁻⁵/⁻⁹ scalings) that matches no formula in this paper; they have been **discarded**. The real fits: R is a single cosine in inclination with a Lorentzian R₁(a); A is a `sinⁿ i` power law. Verified worked values: a=0.999,i=60° → ⟨R⟩≈4.88 M (recalled value coincidentally right), A≈0.44 M (recalled 0.045 was wrong ~10×). **Decision: these verified fits are now a usable regression gate.** (Vindicates CLAUDE.md: recalled formulae are not authoritative — only 1 of the recalled numbers survived contact with the source.)
- ✅ **Temperature exponent.** **Decision: r^(−3/4)** — the bolometric T_eff law, which *is* SKILL.md **Decision B** `T = T₀(6/r)^0.75` (verified against SKILL.md line 942). The r^(−17/9) figure is a Shakura–Sunyaev *vertical-structure* regional law, **not** the bolometric T_eff — do not conflate (see §4).
- ✅ **Luminet spin.** Luminet's r_in = 1.24 M is at a = **0.998**, fully consistent with our **1.18 M at a = 0.999** (SKILL.md Formula 2) — just a different spin. **Decision: keep the two spins distinct; don't cross-wire.**

---

## 8. Recommended Implementation Order (for the CKS GPU path)

1. **Geodesic kernel:** CKS metric + super-Hamiltonian (or Carter-reduced) RHS from `SKILL.md`; adaptive step ∝ (r−r₊)/r; stop at r₊+δ; ξ-constraint monitor. (BPT72 + GRay + DNGR.)
2. **Disk shading:** CPU-precompute f_NT(r) LUT for a=0.999 → T_eff(r) → B_ν; emission = 0 inside r_ms. (NT + Page–Thorne.)
3. **Relativistic transfer:** evaluate u^μ, g, apply I_obs = g³B_ν(T_eff). (Unanimous.)
4. **Anti-aliasing pass:** ray-bundle deviation for star/disk filtering + analytic motion blur. (DNGR.)
5. **Validation gates:** D-shaped shadow at high i; v^(φ)→½ at ISCO; F(r_ms)=0; blue/red limb asymmetry; GRay ⟨R⟩/A fits (eqs. 11–14, verified — §3). (BPT72 + GRay + Luminet.)

> All numeric formulas above are reproduced for orientation. Before they enter any kernel, confirm the exact form in `skills/kerr-physics/SKILL.md`; if a paper's formula appears to conflict with the skill file, **flag for human review rather than substituting it** (CLAUDE.md policy).
