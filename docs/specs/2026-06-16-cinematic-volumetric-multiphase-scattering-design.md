# Cinematic Volumetric Kerr Renderer — Multi-Phase & Scattering Roadmap

> **Status (2026-06-16): DESIGN / PLANNING — owner directed "plans and formulas first,
> no code." Governance: STRICT (the two RTE pillars get human-reviewed SKILL formulas
> before any kernel). SKILL changes permitted for this planning instance.**
>
> Source brief: `tip.md` (ARCHITECTURE SPECIFICATION: CINEMATIC VOLUMETRIC
> KERR-SCHILD RENDERER). This document maps that brief onto the *actual* renderer,
> separates genuine new physics from refinements of subsystems we already ship, and
> fixes the formal mathematics + sequencing. Pillar 2 is specified in full; Pillars
> 3/1/4/5 carry formal equations + open decisions, to be ratified per-pillar.
>
> Decisions locked this session:
> - Scope: full roadmap + Pillar-2 deep-dive first; per-pillar spec→plan→implement after.
> - Governance: PHYSICS-class (P2, P3) get new SKILL formulas (CKS-19, CKS-20).
> - Phase coupling (P2): **hybrid tunable** — one correlation coefficient χ sweeping
>   anti-correlated ↔ independent.
> - Extinction (P2): **grey now, chromatic-ready** — κ structured as a 3-vector, R=G=B today.

---

## 1. The scientific problems, mapped to *this* codebase

`tip.md` frames five "paradigm shifts." Read against the live code, three of them are
**refinements of subsystems we already ship**, and only two are genuinely new physics.
That reframing is the most important output of this analysis.

| Pillar | tip.md problem | Live-code reality | True gap |
|---|---|---|---|
| **P1** Turbulent cascade vs. laminar shear | Shear Ω∝r^−1.5 winds patterns into infinite spirals | **Already fixed**: CKS-12 §2 dual-phase reset crossfade + CKS-18 §2 curl-flow boiling | *Scale-dependent* shear (protect high-freq octaves) — the one new idea |
| **P2** Decoupled multi-phase media | One ρ forces j∝ρ and κ∝ρ ⇒ no dark dust | **True**: `_disk_density_cks` returns one ρ feeding both (CKS-12: `emission∝ρ`, `dτ=absb·ρ`) | A decoupled cold field driving κ only — **new physics (CKS-19)** |
| **P3** In-scattering + HG phase | Pure `dI=j−κI`; clouds can't catch rim-light | **True**: no σ_s, no phase function anywhere | Single-scatter source term + Henyey-Greenstein — **new physics (CKS-20)** |
| **P4** Kelvin-Helmholtz erosion | Smoothstep edge is a synthetic clean rim | **Partly done**: CKS-12 §3 ragged `r_out_eff` smoothstep already ships | Aggressive noise *threshold/clip* shredding — refinement (CKS-21) |
| **P5** Fractal LOD cascade | Fixed octaves alias far / blur near | **Partly done**: screen-space Jacobian exists for the *background* (F10 v1.4) | Wire J → *disk* noise octave count + step dλ — refinement (CKS-22) |

The two headline subsystems are **P2 (multi-phase)** and **P3 (scattering)**; the other
three reuse machinery we already built (dual-phase reset, curl boil, ragged edges, the
screen-space Jacobian).

## 2. Governance classification

The CRITICAL physics policy forces a split `tip.md` blurs:

| Pillar | Touches j_ν / κ_ν / RTE / extinction? | Class | SKILL artifact |
|---|---|---|---|
| P2 | **Yes** — splits j from κ | **PHYSICS** | **CKS-19** (full, this doc) |
| P3 | **Yes** — adds σ_s extinction + in-scatter source | **PHYSICS** | **CKS-20** (full, this doc) |
| P1 | No — warps noise coords | VISUALIZATION | CKS-21 (reserved) |
| P4 | No — thresholds the density window | VISUALIZATION | CKS-22 (reserved) |
| P5 | No — anti-aliasing / sampling rate | SAMPLING | CKS-23 (reserved) |

PHYSICS-class formulas are authored in SKILL.md and human-reviewed **before** kernel
code, then ported verbatim to the CPU twins (`noise.py`, `disk*.py`) and the GPU twin
(`taichi_renderer.py`). No re-derivation in code (CLAUDE.md CRITICAL RULE).

## 3. Dependency graph & sequencing

```
P2 (ρ_hot/ρ_cold) ──► P3 (single-scatter)     P3 needs ρ_cold/κ AND reuses the CKS-17 illuminant ray
P5 (LOD cascade)  ── independent ──► unblocks the V4 free camera
P1 (shear cascade) ── independent (extends CKS-12 §2 / CKS-18)
P4 (KH erosion)    ── independent (extends CKS-12 §3)
```

**Recommended order: P2 → P3 → P5 → {P1, P4}.** P2 has the biggest payoff (dust lanes)
*and* unblocks P3; P3 is the cinematic headliner but is the most expensive and depends
on P2; P5 is the free-camera prerequisite; P1/P4 are low-risk polish.

## 4. Risk / effort / payoff

| Pillar | Effort | Risk | Payoff | Principal risk |
|---|---|---|---|---|
| P2 | Med | Med | **High** | density-fn return + transmittance scalar→vec3 ripples into self-shadow & step-cap |
| P3 | High | **High** | **High** | per-sample illuminant + phase eval → render-time cost; single-scatter scoping |
| P5 | Med-High | Med | High (future) | Jacobian→octave plumbing; perf; temporal stability of a varying octave count |
| P1 | Low-Med | Low | Med | per-octave shear must stay C0 at resets |
| P4 | Low | Low | Med | hard threshold vs. step-cap (sharp edges alias) |

## 5. Cross-cutting invariants (every pillar honors these)

1. New behavior defaults **OFF**; the disabled path is **bit-identical** to current
   goldens (the constraint-6 discipline that CKS-11…18 all follow).
2. PHYSICS dials never double-count g (the g⁴-not-g⁸ rule); VISUALIZATION/SAMPLING
   dials never touch `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT`/chroma-form.
3. CPU/GPU twins stay byte-aligned; every formula is ported, never re-derived.
4. All params live in `configs/render.yaml`; derived values go through the CKS-13
   resolver — never store a derived literal.
5. Docs-sync: each landed change updates SKILL.md + PROJECT.md §6/§7 in the same task.
6. A genuine fluid/MHD continuity solve is still out of scope — these are texturing +
   radiative-transfer approximations, not a CFD step.

---

# PART B — Pillar 2 design (multi-phase media, CKS-19)

## B.1 Goal & acceptance

Decouple emission from absorption so the disk can contain **optically thick, non-luminous
matter** — dark dust lanes that carve high-contrast black silhouettes across the glowing
plasma. Today this is mathematically impossible (one ρ forces j∝ρ and κ∝ρ).

**Defining acceptance test:** a region with `ρ_cold > 0` and `ρ_hot ≈ 0`, placed in front
of bright gas, must read **darker than the background behind it** (a true silhouette) —
not merely dimmer emission.

## B.2 The new physics (summary; formal statement = SKILL CKS-19)

Two decoupled scalar fields replace the single ρ:

```
ρ_hot   → emission   (ionized plasma; the existing density field, renamed)
ρ_cold  → absorption (dust/neutral gas; a NEW decoupled field, pure absorber)

emission = emis_c · ρ_hot · [f_PT] · g_eff⁴ · chroma · ds      # unchanged form, ρ→ρ_hot
dτ⃗      = κ⃗ · ρ_cold · ds                                      # κ⃗ a 3-vector (grey: R=G=B)
```

Transmittance and the front-to-back update become **per-channel**:

```
disk_col += T⃗ ⊙ (w · S)         # ⊙ = componentwise; w,S as in CKS-14
T⃗        *= exp(−dτ⃗)            # vec3 transmittance
```

**Hybrid tunable correlation.** ρ_cold's log-density modulator is correlated to the hot
one by a single coefficient χ ∈ [−1, 1] (config `dust_correlation`):

```
m_cold = χ · m_hot + √(1−χ²) · m_dust          # m_dust = independent fBm (own seed)
ρ_cold = gauss(ζ; σ_cold) · exp(clamp(a_cold · m_cold, ±m_max)) · edge_window
```

- χ = −1 → dust dense where plasma is thin (clean anti-correlated lanes; the default look ≈ −0.6).
- χ =  0 → fully independent dust (chaotic, can occlude bright regions).
- χ = +1 → dust tracks plasma (completeness; rarely useful).
- The √(1−χ²) construction is **variance-preserving**: `Var(m_cold)` is constant for all χ,
  so sweeping the dial does not breathe contrast.

**Grey now, chromatic-ready.** κ⃗ is a 3-vector; today R=G=B (`absorption_coeff` broadcast),
so dust darkens neutrally. The later reddening upgrade (κ_R < κ_G < κ_B) is a **data-only**
change — the RTE structure here already carries it.

## B.3 Architecture changes (no code yet — touch-list for the plan)

1. **`_disk_density_cks` return** widens from `vec2(density, temp_factor)` to
   `vec3(ρ_hot, ρ_cold, temp_factor)` (or a small struct). When `multiphase.enabled==0`
   it sets `ρ_cold = ρ_hot` so every downstream consumer is bit-identical.
2. **`_disk_emit_cks`** emission uses `ρ_hot`; builds `dτ⃗ = κ⃗·ρ_cold·ds` (returns a
   vec3 dτ instead of scalar — or returns ρ_cold and lets the march form dτ⃗).
3. **Beauty-march accumulation** (`render_beauty_physics`) carries `transm` as **vec3**;
   `w = 1 − exp(−dτ⃗)` and `T⃗ *= exp(−dτ⃗)` go componentwise. With R=G=B and ρ_cold=ρ_hot
   this is the scalar legacy march bit-for-bit. `disk_buf` channel 3 (transmittance)
   becomes 3 channels (or stores luminance for the depth key; decide in the plan).
4. **Self-shadow bake (CKS-15/17)** must measure **absorption** optical depth ⇒ when
   multiphase is on it bakes from `ρ_cold`, not `ρ_hot` (a correctness coupling — the
   deep-shadow-map is a τ field, τ is built from κ·ρ_cold). Off ⇒ ρ_cold=ρ_hot ⇒ unchanged.
5. **Vertical step cap** must resolve the **thinner** of σ_hot, σ_cold (worst case) so
   neither slab aliases. Default σ_cold = σ_hot ⇒ unchanged.
6. **CPU twin** (`noise.py`): add the dust modulator `m_dust` (new seed offset
   `NSEED_DUST`) and the χ-mix; a numpy reference so the GPU dust field can be parity-tested.

## B.4 Config schema (additions to `disk:` in `render.yaml`)

```yaml
disk:
  absorption_coeff: 0.8        # EXISTING — becomes the grey κ; broadcast to κ⃗=(.8,.8,.8).
                               # (chromatic-ready: a future [kR,kG,kB] list is accepted here)
  multiphase:                  # CKS-19 — decoupled hot/cold media. Default OFF ⇒ bit-identical.
    enabled: false             # false ⇒ ρ_cold≡ρ_hot, κ⃗ grey ⇒ legacy scalar march exactly.
    dust_correlation: -0.6     # χ ∈ [-1,1]: -1 anti-correlated lanes, 0 independent, +1 tracks plasma.
    dust_amp: 1.0              # a_cold — dust log-density gain (clamped by noise.m_max).
    dust_sigma_frac: 1.0       # σ_cold / σ_hot — dust slab thickness vs the emitting slab.
    # (chromatic upgrade later: extinction_rgb: [1.0, 0.7, 0.45] multiplying κ⃗)
```

`enabled: false` (or an absent block) ⇒ every new term collapses to the current single-phase
march — **zero golden-frame movement**.

## B.5 Test / golden strategy

- **CPU parity:** `tests/test_noise.py` — statistical check that the sampled Pearson
  correlation between `m_cold` and `m_hot` matches χ across a coordinate grid (±tol), and
  that `Var(m_cold)` is χ-invariant (variance preservation).
- **GPU twin parity:** `tests/test_noise_gpu.py` — `ρ_cold` GPU vs CPU within `_SATOL`.
- **Silhouette acceptance:** new `tests/test_disk_multiphase.py` —
  `test_dust_carves_silhouette`: a `ρ_cold>0, ρ_hot≈0` slab in front of emission reads
  darker than the bare background (the B.1 criterion).
- **Bit-identity regression:** unchanged `tests/test_gpu_regression.py` with
  `multiphase.enabled=false` ⇒ goldens bit-identical (the constraint-6 guard).
- **New golden:** one dust-lane beauty frame with multiphase on, χ≈−0.6.

## B.6 Phasing (to expand in the writing-plans step)

1. Author CKS-19 in SKILL.md (this doc) + human review.
2. CPU twin: `m_dust` + χ-mix + ρ_cold in `noise.py` + parity tests (TDD).
3. GPU twin: widen `_disk_density_cks`; vec3 dτ/transmittance in the march; param buffer.
4. Wire self-shadow bake + step cap to ρ_cold.
5. Config + resolver (σ_cold derive if needed); silhouette golden; regression check.

---

# PART C — Forward designs (formal equations, open decisions)

These are ratified per-pillar later; the math is fixed here so the foundation is complete.

## C.1 Pillar 3 — Single-scattering + Henyey-Greenstein (CKS-20, PHYSICS)

Full RTE: `dI/ds = j − (κ+σ_s)I + σ_s ∫_{4π} I(ŝ′)P(ŝ′,ŝ)dΩ′`. The 4π integral is
intractable in one front-to-back march, so we adopt **single-scattering from the dominant
illuminant** (the hot inner edge), reusing the CKS-17 inner-edge-ray geometry:

```
σ_s        = ϖ · κ                                  # single-scatter albedo ϖ ∈ [0,1) (one dial)
dτ_ext⃗     = (κ⃗ + σ_s⃗) · ρ_cold · ds                # scattering also removes forward light
J_scat(x,ŝ) = σ_s · P(cosθ_s) · I_src(x) · e^{−τ_src(x)}
cosθ_s     = ŝ_src · ŝ_view                          # illuminant dir · view dir
P(cosθ)    = (1 − g_HG²) / [4π (1 + g_HG² − 2 g_HG cosθ)^{3/2}]   # Henyey-Greenstein
```

- `τ_src(x)` = absorption optical depth illuminant→x along the inner-edge ray — **already
  baked** in the CKS-15/17 deep-shadow-map. `e^{−τ_src}` is exactly today's `shadow_atten`.
- `ŝ_src` = the inner-edge ray direction at x (the CKS-17 tilted ray); store/derive at bake.
- `I_src(x)` = inner-edge illuminant radiance (its own emission color, CKS-11 inner-edge T).
- March term: `disk_col += T⃗ ⊙ (J_scat · ds)`, attenuated by the running `T⃗` like emission.
- `g_HG > 0.5` (forward) deflects inner-edge light through optically-thin cloud edges to the
  camera — the rim-light / silver-lining of `tip.md` §3.2.

**Reuse, not rebuild** — like CKS-14, CKS-20 is largely an assembly: τ_src and ŝ_src from
CKS-17, ρ_cold/κ from CKS-19, plus the standard HG phase.

**Open decisions (ratify in the P3 spec):**
- `I_src` model: (a) physical inner-edge ring radiance (recommended; ties to CKS-11) vs
  (b) a free `inner_glow` intensity dial.
- σ_s parametrization: `σ_s = ϖ·κ` (recommended, one albedo) vs an independent field.
- `g_HG` default (~0.6 forward) and whether to allow a two-term HG (forward+back lobe).

**Governance / guards:** PHYSICS (modifies extinction + adds a source); single-scatter is a
flagged approximation of the physical in-scatter, like CKS-15. Never touches
`p_μ/u^μ/g/g⁴/f_PT`. `disk.scatter.enabled=false` ⇒ σ_s=0, no in-scatter ⇒ reduces to
CKS-19 exactly. Depends on CKS-19 (ρ_cold/κ) + CKS-17 (τ_src). Default OFF, bit-identical.

```yaml
disk:
  scatter:                # CKS-20 — single-scatter from the inner-edge illuminant. Default OFF.
    enabled: false
    albedo: 0.5           # ϖ = σ_s/(σ_s+κ); 0 ⇒ pure absorption (CKS-19), →1 ⇒ scattering-dominated.
    hg_g: 0.6             # Henyey-Greenstein anisotropy g_HG ∈ (-1,1); >0.5 forward (rim-light).
    inner_glow: 1.0       # I_src amplitude (1.0 = derive from the CKS-11 inner-edge radiance).
```

## C.2 Pillar 1 — Scale-dependent shear cascade (CKS-21, VISUALIZATION)

Amends CKS-12 §2. Today the shear `φ′ = φ − Ω·t` is applied uniformly to the whole fBm.
The cascade applies a **frequency-dependent shear transfer** per octave o (frequency
`f_o = freq·lac^o`):

```
S(f) = 1 / (1 + (f / f_c)^p)                        # shear transfer: low f →1, high f →0
φ′_o = φ − S(f_o) · dynamism · Ω(r) · a_k · T        # per-octave shear amount
n    = Σ_o gain^o · N(u·lac^o, φ′_o · lac^o, ζ; …)   # fBm with per-octave φ shear
```

Large scales shear into filaments (S→1); high-frequency micro-vortices are protected
(S→0 above cutoff `f_c`), regenerating before the bulk flow smears them (Kolmogorov-like).
C0 at resets survives (each octave's reset is the §2 reset; `ω_k→0` regardless of S).
Dials: `shear_cutoff f_c`, `shear_falloff p`. `f_c=∞` (or block absent) ⇒ S≡1 ⇒ current
uniform shear bit-for-bit.

## C.3 Pillar 4 — Kelvin-Helmholtz threshold erosion (CKS-22, VISUALIZATION)

Amends CKS-12 §3. Replaces the *multiplicative* outer smoothstep with a *noise-thresholded
clip* near the boundary band:

```
ρ ← ρ · H_soft( ρ_env(r) − τ_KH · N_KH(u, φ, ζ; t) )
```

- `N_KH` = high-frequency 3D noise (its own seed), advected with the §2 shear.
- `ρ_env(r)` = the smooth boundary envelope (the current window).
- `τ_KH` = erosion strength; `H_soft` = a *narrow* smoothstep Heaviside (shred edges).
- Produces tearing/fraying into vacuum instead of a clean rim (`tip.md` §4.2).
- **Step-cap coupling:** `H_soft`'s width has a floor tied to `max_step_vfrac` so the
  sharpened edges never alias under the vertical step cap. `τ_KH=0` ⇒ current ragged edge.

## C.4 Pillar 5 — Fractal LOD octave modulation (CKS-23, SAMPLING)

Amends Formula 10. Extends the existing screen-space exit Jacobian (F10 v1.4) to a
**noise-domain footprint** J at each disk sample (or maps camera distance → local
noise-coord footprint), then modulates octaves and step:

```
J        = ‖∂(u,φ,ζ)/∂(pixel)‖                       # noise-domain footprint per sample
N_oct    = clamp(N_max − log₂(J / J_0), N_min, N_max)
w_frac   = fract(N_max − log₂(J / J_0))               # fractional top-octave weight (anti-pop)
dλ       ∝ J                                          # coarser steps where the footprint is large
```

The highest partial octave is **crossfaded** by `w_frac` (no integer popping → temporal
stability). Macro views cull sub-pixel octaves (kills moiré/shimmer); close-ups inject
sub-octaves (resolves micro-wisps) — the constant-detail-density goal of `tip.md` §5.2.
**Prerequisite for the V4 free camera.** `enabled=false` ⇒ fixed `octaves` (current path).

---

## 6. SKILL.md changes made this session

- **CKS-19** authored in full (multi-phase RTE) — PHYSICS, ready for review.
- **CKS-20** authored in full (single-scatter + HG) — PHYSICS, with open knobs flagged.
- **CKS-21 / CKS-22 / CKS-23** reserved with formal equations above (VISUALIZATION/SAMPLING),
  to be promoted to full SKILL entries when each pillar is spec'd.
- "When to use" list + Revision history updated.

## 7. Next step

Per the brainstorming flow: review this doc + the CKS-19/20 SKILL entries, then we move to
`writing-plans` for **Pillar 2** (the TDD task breakdown of Part B). No code until the plan
is approved.
