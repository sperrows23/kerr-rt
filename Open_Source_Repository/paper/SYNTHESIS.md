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
| Radiative efficiency η = 1 − Ẽ(r_ms) | **≈ 0.30** at a=0.999; **0.42** extremal limit; 0.057 at a=0 | ⚠ see §7 discrepancy |

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

**Validation targets (GRay fitting formulae):** ⟨R(a,i)⟩ and asymmetry A(a,i) give ground-truth shadow radius/asymmetry. Both models flag the exact coefficients were partly undecoded in source — **re-fetch GRay §4 from the original before using them as a numeric gate.**

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

**Flagged discrepancies / cautions:**
- ⚠ **Radiative efficiency at a = 0.999.** Sonnet's NT pass and the BPT72 passes quote the **extremal (a=M) 42.3%**; Flash's NT pass gives **≈30% at a=0.999** specifically. Both are right at their stated spin — 0.42 is the a→M limit, ~0.30 is the actual a=0.999 value. Use **η ≈ 0.30 for a=0.999**; treat 0.42 as the asymptotic ceiling. Confirm against `SKILL.md` before coding.
- ⚠ **GRay shadow-fit coefficients** (⟨R⟩, A) were partly `formula-not-decoded` in both extractions — re-fetch from the published GRay before using as numeric tests. Flash's quoted coefficient values are therefore **unverified**.
- ⚠ **Temperature exponent.** r^(−3/4) (bolometric T_eff, = Decision B) vs r^(−17/9) (a regional vertical-structure law). Use r^(−3/4); see §4.
- ⚠ Luminet quotes r_in = 1.24 M at a = **0.998** (not 0.999) — consistent with our ≈1.18 M at 0.999, just a different spin; don't cross-wire the two.

---

## 8. Recommended Implementation Order (for the CKS GPU path)

1. **Geodesic kernel:** CKS metric + super-Hamiltonian (or Carter-reduced) RHS from `SKILL.md`; adaptive step ∝ (r−r₊)/r; stop at r₊+δ; ξ-constraint monitor. (BPT72 + GRay + DNGR.)
2. **Disk shading:** CPU-precompute f_NT(r) LUT for a=0.999 → T_eff(r) → B_ν; emission = 0 inside r_ms. (NT + Page–Thorne.)
3. **Relativistic transfer:** evaluate u^μ, g, apply I_obs = g³B_ν(T_eff). (Unanimous.)
4. **Anti-aliasing pass:** ray-bundle deviation for star/disk filtering + analytic motion blur. (DNGR.)
5. **Validation gates:** D-shaped shadow at high i; v^(φ)→½ at ISCO; F(r_ms)=0; blue/red limb asymmetry; (after re-fetch) GRay ⟨R⟩/A fits. (BPT72 + GRay + Luminet.)

> All numeric formulas above are reproduced for orientation. Before they enter any kernel, confirm the exact form in `skills/kerr-physics/SKILL.md`; if a paper's formula appears to conflict with the skill file, **flag for human review rather than substituting it** (CLAUDE.md policy).
