# DNGR / Star-Ingest Artifact Remediation — Root-Cause Investigation & Plan

> **Historical design record (snapshot of 2026-06-06).** Current status lives in
> `PROJECT.md` §6/§7 (S4–S7). Notably, the star counts below predate the
> 2026-06-07 starless Layer-B swap (`mag_limit` 8.0→11.0; 41 410→115 324 stars).

**Date:** 2026-06-06
**Branch:** `feat/point-star-ingest`
**Author:** Claude (Opus 4.8) under CLAUDE.md physics-formula policy
**Status:** Investigation complete. Pass 2 (2026-06-06): **Artifact A (smear)
RESOLVED** (dngr default + mag-gap closed). Pass 3 (2026-06-06): **Artifact B
(dngr seam) RESOLVED** — the R2 splat-placement rule (SKILL.md Formula 13 guard
(b′)) was **approved + landed**; the field is seam-free (coarse center-column
metric 15×→2.06×; masked-field stripe-z ≈14, clean range). The dedicated
location-agnostic stripe test stays `xfail` only because a bright lensed star
confounds it in frame-0's thin sky band (not a residual seam). The Formula-10
`texture`-LOD blocky stripe remains a **deferred** follow-up. See
[§7.3](#73-pass-3-2026-06-06--artifact-b-r2-landed--validated).
**⚠ SUPERSEDED IN PART — see [§6 Update](#6-update-2026-06-06-visual-verification-overturns-the-fixed-verdict).**
A later disk-off visual check proved the `emission_coeff` fix only un-blacked the
*disk*; the user-visible "dark noise" is a **separate, shared background artifact**
(star-smear + seam) that the green coarse tests do not detect. Read §6 first.

---

## 0. TL;DR

The reported "overwhelming number of bugs / visual artifacts" on this branch
resolves to **two** distinct causes, found by empirical bisect against the
committed golden regression:

1. **`disk.emission_coeff: 0`** (uncommitted debug edit) — renders the accretion
   disk as a pure black absorbing silhouette. **Dominant artifact.** This single
   edit breaks 3 of the 4 committed GPU-regression assertions (disk peak, Doppler
   asymmetry, *and* — by darkening the disk so the background dominates — the seam
   guard). **FIXED:** restored to `8.0` (the committed value).

2. **DNGR spin-axis seam star pileup** (uncommitted feature, `starfield.mode:
   dngr`) — a *real* Layer-A artifact: catalog stars in the polar gather cells are
   deposited onto the spin-axis meridian at near-full weight via an *invalid*
   `det J`. The center image column is a 100th-percentile luminance outlier (~2×
   the brightest off-center window). **FLAGGED** — the fix touches Formula-13 seam
   semantics that SKILL.md does not fully specify (physics policy: do not silently
   substitute; ask the owner to extend the skill first).

The rest of the DNGR path (Layers A+B, magnification μ, EWA diffuse, catalog
ingest) is **finite, NaN-free, and matches SKILL.md Formula 13 verbatim**. The
codebase is fundamentally sound; the "bug storm" was one debug edit plus a newly
wired, not-yet-validated feature with one genuine seam bug.

---

## 1. Method

### 1.1 Reference-repo architecture comparison (`Open_Source_Repository/`)

Four reputable open-source renderers were supplied as ground truth. The key
structural finding:

| Repo | Physics | Integrator | Relevance to our Kerr pipeline |
|------|---------|-----------|-------------------------------|
| **Gargantua** (`interstellar_renderer.py`) | Schwarzschild "Newtonian-fantasy" `a⃗ = −1.5·h²·r⃗/r⁵` (Cartesian) | Dormand–Prince 5(4) | Methodology only — **not Kerr** |
| **Blackhole** (`shader/*.frag`) | Same Cartesian post-Newtonian central force | RK / plane-crossing disk | Methodology only — **not Kerr** |
| **starless** (`blackbody.py`, `bloom.py`) | Schwarzschild, LUT blackbody, Airy bloom | fixed-step | Blackbody/bloom recipes only — **not Kerr** |
| **tika** (`schwarzschild.frag`, *misnamed*) | **Full Kerr**, Hamiltonian `dp_μ/dλ = −½ ∂g^{αβ}/∂xᵘ pₐp_β`, BL coords | adaptive | **The only true Kerr comparison** |

**Conclusion:** 3 of 4 references are Schwarzschild and serve as *methodology
guides* (tonemap, bloom, blackbody, disk shading), **not** physics-transplant
sources. Only **tika** is comparable Kerr. Our separated 2nd-order
R(r)/Θ(θ) integrator with constraint projection (`renderer.geodesic._project`,
ported into `taichi_renderer._rk4_step`) is a *different and valid* scheme from
tika's Hamiltonian; it is not a bug. No reference motivated a physics change to
our core geodesic path.

### 1.2 Empirical bisect (the decisive evidence)

Rendered camera frame 0 under three configs, measuring disk emission and
background statistics (`scripts/_bisect_diag.py`, throwaway):

| Config | Disk max | Disk sum | BG star peak | BG peak/median | NaN/Inf |
|--------|---------:|---------:|-------------:|---------------:|:-------:|
| **emission=0, dngr** (working tree) | **0** | −22.8 (absorb only) | 2.41 | 2021× | 0 / 0 |
| emission=8, dngr | 37.36 | 1.61e5 | 2.41 | 2021× | 0 / 0 |
| emission=8, texture (golden) | 37.36 | 1.61e5 | 0.28 | 183× | 0 / 0 |

Cross-checked against the committed `tests/test_gpu_regression.py` (1920×1080,
reads config directly):

| Config | disk_peak (ref 12.77) | doppler (ref 7–8.5) | seam (limit 6) | result |
|--------|----------------------:|--------------------:|---------------:|:------:|
| emission=0, dngr (working tree) | 0.75 ✗ | 2.00 ✗ | 138× ✗ | **3 fail** |
| emission=8, dngr | 12.77 ✓ | 7.78 ✓ | 15.4× ✗ | 1 fail |
| emission=8, texture (committed golden) | 12.77 ✓ | 7.78 ✓ | 1.9× ✓ | **all pass** |

---

## 2. Root causes & resolutions

### 2.1 [FIXED] `disk.emission_coeff: 0` — black disk

`_disk_emit` (taichi_renderer.py:674): `emission = emis_c · density · g⁴ · ds`.
With `emis_c = 0` the disk emits nothing and only *absorbs* the background
(`absorption_coeff: 1.5`) → a black silhouette. The Doppler asymmetry guard also
collapses (2.0 vs 7.78) because the left/right g⁴ split *is* the disk emission.

This is an uncommitted debug edit (committed value = `8.0`); restoring it returns
disk_peak and Doppler to byte-exact golden values and is identical between dngr
and texture modes (1.61e5 both), proving the star-ingest work never touched disk
physics.

**Resolution (applied):** `configs/render.yaml` `emission_coeff: 0 → 8.0`.

### 2.2 [FLAGGED] DNGR spin-axis seam star pileup (`_dngr_shade`, Layer A)

**Symptom:** under `mode: dngr`, the spin-axis meridian (image center column for
the edge-on frame-0 framing) is a genuine luminance outlier — its 8-column max
jump (0.0098) exceeds the *max* over all other interior 8-column windows (0.0047):
**100th percentile, ~2×**. Not point-star variance; a real vertical seam.

**Mechanism** (taichi_renderer.py:1207–1234): on the seam, neighbor pixels
straddle the pole, so `Δφ′ ≈ ±π` ⇒ `detJ = dthx·dphy − dthy·dphx` is large ⇒
`inv_det = 1/detJ` is small. The Layer-A gather is gated on `valid` (not
`usable`) by design — the code comment documents this as the fix for a prior
"polar rope" star-less band. But it then projects every polar-cell catalog star
to screen space via `dpx,dpy = J⁻¹·(Δθ′,Δφ′)`. With tiny `inv_det`, **all** polar
stars collapse to `d ≈ 0` and are deposited at near-full weight `exp(0)·μ`
(μ=1 on the seam). Dozens of stars pile onto the seam pixels → the spike.

**Why this is flagged, not fixed:** SKILL.md Formula 13 guard (b) states that on
the seam `det J` is **invalid** and clamps `μ = 1`. The Layer-A *splat geometry*
when `det J` is invalid is **not specified** by Formula 13. Using the invalid
`J⁻¹` to place the splat (current code) contradicts the skill's own statement
that `det J` is invalid there, but the skill gives no alternative placement rule.
Per CLAUDE.md physics policy ("if the skill file is incomplete, ask the user to
extend it before writing code"), this requires an **owner decision**.

**Candidate resolutions to choose from (owner picks one, then SKILL.md is
extended before coding):**
- **(R1)** On `not usable` (invalid `det J`): suppress the Layer-A gather (accept
  a thin star-less seam — the original "polar rope" behavior). Simplest; matches
  "det J invalid" literally.
- **(R2)** On `not usable`: place stars with a **fallback projection** that does
  not use the degenerate `det J` — e.g. the analytic undeflected pixel-footprint
  used for the μ normalization (`d_omega`), so polar stars splat at their true
  angular separation instead of collapsing to `d≈0`.
- **(R3)** Cap the per-pixel Layer-A contribution (energy clamp) on the seam so a
  fold cannot pile unbounded star flux, independent of placement.

Recommended: **R2** (keeps stars, removes the pileup, reuses an already-approved
quantity). Requires a one-line Formula-13 guard-(b) amendment specifying the
Layer-A fallback projection.

### 2.3 [FLAGGED] μ normalization: analytic `dΩ` vs the approved finite-difference estimator

SKILL.md Formula 13 §2(a) (APPROVED 2026-06-05) says to compute the flat-space
footprint `det J₀·sinθ′₀` with the **same finite-difference estimator** applied
to undeflected camera-ray directions. The code (taichi_renderer.py:1257–1267)
instead uses the **closed-form analytic** per-pixel solid angle `dΩ`
(perspective pinhole). The host test `test_magnification_normalizes_to_one_in_flat_space`
shows analytic-`dΩ` / FD-`src` → 1.0 to **< 0.3%** — they are the continuum limit
of each other. Not a divergence; a **method deviation from the skill's letter**.

**Resolution (owner decision):** either (i) amend Formula 13 §2(a) to bless the
analytic closed form (it is provably the FD continuum limit and is what ships), or
(ii) switch the code to the literal FD estimator. Recommend **(i)** — document the
analytic form as the approved implementation, with the test as its guard.

### 2.4 [FLAGGED] PSF σ vs the paper's anti-flicker target

SKILL.md Formula 13 §3 specifies the splat **verbatim** as
`I_pixel(d) = I_final·exp(−d²/2σ²)` — a peak-1.0 (NOT Σ-normalized) Gaussian; the
code matches this exactly. "Energy conservation" in the skill refers to
`I_final = I_base·μ·g⁴`, *not* splat-sum normalization. **Therefore the splat math
must not be changed** (doing so would be a silent substitution against the source
of truth). The skill's anti-flicker mechanism is σ-sizing: it cites the paper's
"beam radius ≈ 2× pixel separation, targeting ≤2% peak-to-trough flicker." Config
`star_psf_px: 1.3` is below that guidance and may exceed 2% temporal flicker in an
animation. σ is explicitly config-driven, so this is a **tuning** note, not a code
bug.

**Resolution (owner decision):** consider `star_psf_px ≈ 2.0` to honor the cited
≤2% flicker target; verify with a short multi-frame render before committing.

### 2.5 [NOTE] DNGR background ~10× brighter than the legacy texture

BG peak under dngr (2.41) vs texture (0.28). This is *by design* (point-star
energy gathers spike far above a baked texture — the test asserts
`lum.max() > 20×median`), but it shifts overall background exposure. If the
Blender compositor / tonemap was tuned on the texture path, the dngr background
may bloom. Calibration via `mag_zero_point`, not a bug.

---

## 3. Changes applied in this pass

| File | Change | Rationale |
|------|--------|-----------|
| `configs/render.yaml` | `disk.emission_coeff: 0 → 8.0` | Revert debug edit; restores disk + Doppler golden (§2.1) |
| `configs/render.yaml` | `starfield.mode: dngr → texture` | Revert to committed green baseline; dngr is opt-in until the §2.2 seam is resolved |

Both edits revert uncommitted debug changes back to the committed golden
configuration. With them, the **full suite is green**. The DNGR feature (Layers
A+B, +454 uncommitted renderer lines, catalog ingest) remains in the tree as an
opt-in pending the §2.2 owner decision.

## 4. Decisions required from the owner (gates further work)

1. **§2.2 seam fix** — **DONE (R2 approved + landed 2026-06-06, pass 3).** SKILL.md
   Formula 13 guard (b′) applied (v1.7); `_dngr_shade` splat placement uses the
   undeflected proper-separation footprint on the invalid-`det J` branch. Validated:
   coarse center-column seam 15×→2.06× (marker removed → live guard); dedicated
   stripe-z field-clean (≈14 masked). See [§7.3](#73-pass-3-2026-06-06--artifact-b-r2-landed--validated).
   **Still deferred:** porting the same regularization to the Formula-10 `texture`
   LOD (a separate Formula-10 change), and the bright-point recalibration of the
   dedicated stripe detector.
2. **§2.3 μ normalization** — bless analytic `dΩ` in SKILL.md, or switch to FD.
3. **§2.4 PSF σ** — keep 1.3 or move to ≈2.0 (needs a flicker check).
4. **Default mode** — **RESOLVED 2026-06-06: `dngr` promoted to default** (kills
   the dominant smear, Artifact A). See [§7.1](#71-artifact-a-smear--resolved-dngr-promoted).

## 5. Workflow improvements

- **Regression metric robustness:** `test_no_spin_axis_seam` compares *max-of-8*
  at center to the *median* jump elsewhere — unstable for a sparse point-star
  background (median ≈ 0 between stars). Once §2.2 is fixed, recalibrate it to
  compare the center window against a high percentile (e.g. p99) of *all*
  sliding windows, so it catches a real seam without false-positiving on stars.
- **Add a `dngr`-mode golden:** the GPU regression only pins the `texture` path.
  Add a parallel dngr assertion set so the feature has its own guard.
- **Config hygiene:** debug edits like `emission_coeff: 0` should never reach a
  branch tip silently. Consider a pre-commit check that the committed config
  passes `test_gpu_regression.py` (or a fast proxy), so a darkened disk can't slip
  in unnoticed.
- **Don't transplant Schwarzschild references:** 3 of 4 repos are Newtonian-
  fantasy; only tika is Kerr. Future "fix it like the reference" requests should
  cite **tika** for geodesic/g-factor questions and the others only for
  tonemap/bloom/blackbody recipes.

---

## 6. UPDATE 2026-06-06 — visual verification overturns the "fixed" verdict

The owner correctly distrusted the green suite ("tests passing ≠ artifact gone")
and asked to **disable the disk and inspect the isolated starfield**. Doing so
(`scripts/gpu_test.py --no-disk`, plus a strong exposure stretch) proved the
"dark noise" the owner sees is **not** the disk and **not** `emission_coeff` — it
is a background artifact present in **both** `texture` and `dngr`, independent of
the §2.1 fix. §0's "codebase fundamentally sound / suite green" framing was
premature: the coarse metrics are simply insensitive to it.

### 6.1 Two confirmed background artifacts (disk off, controlled A/B)

Same camera/frame, only the background method differs:

| Artifact | `texture` (shipped default) | `dngr` | Root cause |
|----------|-----------------------------|--------|------------|
| **A. Radial star-smear** | Stars smeared into tangential streaks field-wide (the dominant "dark noise") | **Gone** — sharp points on a clean field | Formula-10 uses an **isotropic scalar** mip-LOD; near the ring the lensed footprint is strongly **anisotropic**, so one mip can't represent it → smear. `dngr`'s anisotropic EWA (Layer B) + point gather (Layer A) fixes it |
| **B. Spin-axis seam** | **Blocky coarse-mip vertical stripe** (the `j_fold` "collapse to coarsest mip" — *the cure is the artifact*) | **Bright star-pileup rope** (= §2.2) | Same coordinate singularity on the meridian (Δφ′→±π, footprint/`detJ`→∞); both modes paper over it differently and both band-aids are visible |

The controlled A/B (smear vanishes in `dngr` on the identical frame) isolates
Artifact A to the Formula-10 isotropic-LOD background path; it is not motion blur
or the EXR asset.

### 6.2 Why the green tests lied (validates the owner's premise)

`test_gpu_regression.py` seam ratio passed for `texture` at **1.9× vs the 6.0
limit** while the blocky stripe is plainly visible — a uniformly-blocky
*low-frequency* stripe has small local jumps, so it sails under a max-jump/median
metric. The validation criterion "seam ratio < 6 ⇒ no seam" is simply wrong. This
is a **test-sensitivity defect**, exactly as suspected.

### 6.3 Lock-in: new artifact-sensitive tests (owner-chosen "tests first")

`tests/test_starfield_artifacts.py` (new) encodes the *visual* correctness the
coarse metrics lack. Pure image statistics — **no GR formula evaluated**, so not
governed by SKILL.md. Calibrated on real CUDA renders (disk off, 1280×720):

| Metric | clean (synth) | neg control | inject | `dngr` | `texture` | threshold | catches |
|--------|-------------:|-------:|-------:|-------:|----------:|----------:|---------|
| Smear coherence (structure-tensor, bright features) | 0.01–0.03 | iso-blur 0.07–0.10 | dir-blur 0.75 | 0.257 | 0.50 | < 0.36 | A |
| Vertical-stripe z (MAD-z over columns; location-agnostic) | 9.3–14 | h-band 10–14 | v-stripe 68+ | 27.8 | 138 | **< 20** | A+B |

Both are `xfail(strict=True)` against the shipped default: they record today's
known-broken state and flip the suite **red on xpass**, forcing a marker +
threshold review the moment an artifact is fixed. They stage the fix correctly —
killing the smear (e.g. promoting `dngr`) makes the smear test xpass while the
stripe test stays xfailed (27.8 > 20) until the seam is genuinely fixed.

**Validation (2026-06-06).** Before trusting these detectors, each was checked on
synthetic *ground truth* (artifact-free field, 5 seeds) for both sensitivity and
specificity — not merely fit to the two real renders:
- **Smear — validated.** Rises only for a *directional* blur (0.75); a true
  isotropic Gaussian blur leaves it ~0.08. 0.36 sits in the real 0.10→0.50 gap.
- **Seam — metric sound, threshold was mis-set.** The metric is orientation-
  selective (a vertical stripe → 68+, an equal-energy horizontal band stays at
  clean ~12). But a clean field scores **9–14**, so the original `< 12` sat
  *inside* the clean distribution (would false-positive). Recalibrated to **20**,
  in the empty 14→27.8 gap. Known limitation: a perfectly *uniform* column is
  removed by the per-column demean — accepted, since real seams are textured.
- **Cross-mode corroboration.** Switching the default to `dngr` and running the
  full suite confirmed the staging: the smear test xpasses (red, as designed)
  while both seam tests still fail. Notably the *old coarse* `test_no_spin_axis_seam`
  — which passed `texture` at 1.9× — **fails under `dngr` at 15.4×**, because the
  `dngr` seam is a sharp center-column star-pileup (caught by the coarse metric)
  whereas `texture`'s is a low-frequency blocky stripe (missed). Neither mode is
  seam-free; the morphologies differ.

### 6.4 Revised root-cause map & recommendation

- The §0 "dominant artifact = `emission_coeff`" holds **only for the disk**. For
  the **starfield**, the dominant artifact is **A (smear)**, which `dngr` already
  eliminates — so the earlier "keep `texture` as default" recommendation is
  **reversed**: `dngr` is visually better (localized seam vs field-wide smear).
- **B (seam)** is the one shared, real remaining background bug. Both the
  `texture` `j_fold` blocky-collapse and the `dngr` Layer-A pileup are band-aids
  over the same meridian coordinate singularity. Fixing it touches Formula-10 LOD
  and Formula-13 guard (b) → **physics-policy gated** (owner decision + SKILL.md
  extension before code), same as §2.2.

### 6.5 Status of the four §4 decisions after this pass

1. **§2.2/B seam** — still open; now confirmed to affect **both** modes, not just
   `dngr`. Needs the R1/R2/R3 pick + SKILL.md guard-(b) extension.
2. **§2.3 μ normalization** — unchanged (doc-sync only).
3. **§2.4 PSF σ** — unchanged.
4. **Default mode** — recommendation now **reversed toward `dngr`** (kills the
   dominant smear); blocked only by the shared seam (B).
5. **NEW — Artifact A (smear) fix path** — either promote `dngr` to default
   (config; already-blessed anisotropic path) **or** port anisotropic filtering
   into the Formula-10 `texture` path. Owner choice; the smear test guards either.

---

## 7. UPDATE 2026-06-06 (pass 2) — Artifact A resolved, Artifact B R2 proposal

Three independent sub-agents re-reviewed (a) the new artifact tests, (b) the
default-mode switch, and (c) the shared seam root cause before any change. Their
findings drove the two owner decisions executed below. The test review confirmed
the smear detector **trustworthy** and the seam detector **trustworthy with
bounded caveats** (prominence- not presence-detector; orientation-locked to the
current vertical-axis camera framing; pixel-absolute kernels assume 1280×720) —
i.e. the tests are sound to *guide* this fix, which was the owner's gating premise.

### 7.1 Artifact A (smear) — RESOLVED (`dngr` promoted)

> **⚠ Superseded in part (2026-06-07):** the premise below that `milkyway_2020_16k`
> "omits everything brighter than ~8.0" was later **measured false** — the milkyway
> plate carried the full baked star field (sharp-spike density identical to
> `starmap_2020`). The `mag_limit: 6.5 → 8.0` "A/B energy boundary" reasoning is
> therefore void. Layer B was swapped to the StarNet2-starless `starmap_final.exr`
> and `mag_limit` raised to **11.0** (Layer A now carries the entire star field;
> `assets/stars.npy` re-ingested to 115 324 stars). See PROJECT.md §6 (2026-06-07).
> The §7.1 smear resolution itself (anisotropic EWA, `dngr` default) still stands.

Owner decision: **promote `dngr` to default AND close the magnitude-coverage gap.**

The default-switch review found a real gap a naïve flip would have shipped:
Layer A was capped at `mag_limit: 6.5` while the Layer-B diffuse map
(`milkyway_2020_16k`) **omits everything brighter than ~8.0** (Hipparcos/Tycho
stripped). Stars in **6.5–8.0 would have fallen out of BOTH layers** — the legacy
`texture` bake (complete) contains them, so a bare flip would *lose* ~80% of the
resolved point-star population vs `texture`. Fix: match Layer A's cutoff to the
diffuse map's, then re-ingest.

| File / asset | Change | Why |
|------|--------|-----|
| `configs/render.yaml` | `starfield.mode: texture → dngr` | Anisotropic EWA kills the field-wide smear (coherence 0.50 → 0.257 < 0.36) |
| `configs/render.yaml` | `starfield.mag_limit: 6.5 → 8.0` | Butt against the diffuse map's bright cutoff; close the 6.5–8.0 gap; set the A/B energy boundary (do not exceed ~8.0 or Layer A re-adds Tycho stars the diffuse map already has → double-count) |
| `assets/stars.npy` | re-ingested (`scripts/ingest_stars.py`) | **8 877 → 41 410 stars** at V ≤ 8.0; the gap stars are now in Layer A |
| `tests/test_starfield_artifacts.py` | smear test: dropped `xfail(strict)` → **live PASS guard** | Smear is fixed under the new default; the test now guards against regression (revert-to-texture / broken EWA) |
| `tests/test_gpu_regression.py` | `test_no_spin_axis_seam`: added `xfail(strict)` | Under the `dngr` default this coarse center-column metric now also catches the meridian pileup (≈15×, Artifact B). Known/owner-gated → recorded, not silenced; flips RED when R2 lands |

**Frame consistency confirmed** (review b): Layer-A catalog (`ingest_stars.py`:
`θ′ = π/2 − Dec`, `φ′ = RA`) and Layer-B diffuse (`u = φ′/2π`, `v = θ′/π`) share
the one celestial→BL equirect frame — no misregistration. **Energy:** with the
matched cutoffs the A (≤8.0) and B (>8.0) populations stay disjoint, so the
straight `diffuse + stars` sum does not double-count (no de-dup needed). Cross-layer
absolute brightness (`mag_zero_point`) remains an open *look-tuning* note (§2.5),
not a correctness blocker.

**Suite after the change: 41 passed, 2 xfailed** (the two xfails are now the two
seam detectors — coarse `test_no_spin_axis_seam` and the dedicated stripe test;
the smear test passes live).

### 7.2 Artifact B (seam) — R2 proposal (spec-only; no code or SKILL.md edits)

Owner decision: **write the R2 proposal only** — draft the rationale and the exact
SKILL.md amendment for approval; touch **no** `taichi_renderer.py` and **no**
`SKILL.md` this pass.

**Shared-root-cause confirmed at code level** (review c): both the `texture`
Formula-10 LOD (`_screen_jacobian_lod`) and the `dngr` Formula-13 Layer-A gather
(`_dngr_shade`) read the **identical `exit_buf` +x/+y neighbor stencil** and blow
up on the **same `Δφ′≈±π` meridian trigger**. `J = max(Jx,Jy)` (scalar) and
`detJ`/`δ⁻` (2×2) are the scalar-reduction vs full-matrix form of *one* screen-space
Jacobian (SKILL.md states this verbatim). So it is **one root cause**, but **three
code touch-points** (texture LOD at the shade kernel, its inline offset-ray twin,
and the dngr splat) — not two independent bugs.

**Why R2** (over R1/R3): R1 (suppress the polar gather) reverts to the original
star-less "polar rope" — visibly wrong. R3 (energy clamp) caps the spike but still
*places* stars with the degenerate `J⁻¹`, so the pileup geometry remains. R2 fixes
the **placement**: when `detJ` is invalid on the seam, project Layer-A stars with
the **undeflected analytic footprint already computed for the μ normalization**
(`det J₀·sinθ′₀`, an *already-owner-approved* quantity — guard (a)), so polar stars
splat at their true angular separation instead of collapsing to `d≈0`. The deeper
principle (PROJECT.md §8): `δ⁻→0` is the caustic marker that is the principled
replacement for the `j_fold` heuristic — the same regularization retires the
`texture` blocky-collapse too.

**Governance gate (why this is owner-only):** SKILL.md Formula 13 guard (b)
currently governs **μ (brightness) only** — it declares `detJ` *invalid* on the
seam and clamps `μ=1`, but says **nothing about Layer-A splat placement** when
`detJ` is invalid. The shipped code nonetheless uses that same invalid `J⁻¹` to
*position* the splat, contradicting the skill's own "detJ invalid here" with no
sanctioned alternative. Per CLAUDE.md ("if the skill file is incomplete, ask the
user to extend it before writing code"), the owner must approve the amendment
below **before** any renderer code is written.

**PROPOSED SKILL.md Formula 13 guard-(b) amendment (NOT YET APPLIED — for owner
approval).** Append one sentence to the existing guard (b) "Resolution"
(SKILL.md §F13.2(b), after the `δ⁻ < caustic_delta_min` clause):

```diff
   `δ⁻ < caustic_delta_min` as on-caustic and clamp `μ = min(μ, mag_clip)` so a
   critical curve cannot produce an unbounded splat.
+  **(b′) Layer-A splat placement when `det J` is invalid (R2, PROPOSED 2026-06-06).**
+  When guard (b) marks `det J` invalid (non-ESCAPED neighbour or `J > j_fold`), the
+  star's screen-space offset must NOT be computed from `J⁻¹` (the degenerate
+  inverse collapses all polar-cell stars to `d≈0`, piling them onto the meridian —
+  the observed seam). Instead place the splat using the **flat-space undeflected
+  footprint** `det J₀·sinθ′₀` already computed for the guard-(a) μ normalization:
+  offset the star by its true angular separation under the undeflected exit map, so
+  on-axis stars keep their real angular spacing. This makes the polar gather
+  degenerate gracefully to the no-lens geometry exactly where the lensed Jacobian
+  is unusable, and is the principled replacement for the Formula-10 `j_fold`
+  coarse-mip collapse (PROJECT.md §8 `δ⁻→0` caustic marker).
```

**On approval, the code work is:** (1) implement (b′) in `_dngr_shade` (replace the
`J⁻¹` placement on the invalid-`detJ` branch with the undeflected-footprint offset);
(2) apply the matching meridian rule to the Formula-10 LOD + its offset-ray twin so
the `texture` blocky stripe is removed by the same regularization; (3) recalibrate
both seam tests (`test_no_spin_axis_seam` and `test_background_has_no_vertical_seam_stripe`)
and remove their `xfail` markers (they will xpass when the seam is gone). None of
this is done in this pass.

---

## 7.3 Pass 3 (2026-06-06) — Artifact B R2 LANDED + validated

Owner approved the guard-(b′) amendment and directed: implement R2, validate with
the existing seam tests, stop+report if validation fails or new artifacts appear.

**Applied (touch-point 1 of 3 — the dngr default path):**

| File | Change |
|------|--------|
| `skills/kerr-physics/SKILL.md` | Formula 13 guard **(b′)** added + revision **v1.7**: on the invalid-`det J` branch, place the Layer-A splat by `d² = (Δθ′²+sin²θ′·Δφ′²)/dΩ` (undeflected proper-separation footprint = the guard-(a) quantity), not the degenerate `J⁻¹`. |
| `src/renderer/taichi_renderer.py` | `_dngr_shade`: the Layer-A gather now runs for every escaped pixel; placement is lensed `J⁻¹` when `usable`, else the guard-(b′) undeflected footprint. (Removes the old `valid`-only gate that also left a star-free band around the shadow.) |
| `tests/test_gpu_regression.py` | `test_no_spin_axis_seam`: **xfail marker removed** → live PASS guard (center ratio 2.06× < 6.0). |
| `tests/test_starfield_artifacts.py` | `test_background_has_no_vertical_seam_stripe`: **kept `xfail`**, reason updated to the bright-star-confound finding (owner choice: clear coarse only). |

**Validation (CUDA, RTX 5060; full suite 42 passed / 1 xfailed):**

| Evidence | Pre-R2 (`J⁻¹`) | Post-R2 (b′) | Reading |
|----------|---------------:|-------------:|---------|
| Coarse center-column seam ratio (1920×1080, disk on) | ~15× | **2.06×** | central meridian seam gone (cf. legacy texture ~1.9×) |
| Center-column stripe-z (1280×720) | 20.9 | **11.8** | seam removed at the meridian |
| Dedicated stripe-z, **raw** | 27.3 | 28.1 | unchanged — dominated by a bright star, not the seam |
| Dedicated stripe-z, **brightest-star masked** | — | **14.4** | underlying field is seam-free (clean range 9.3–14) |
| Dedicated stripe-z, **99th-pct clip** | — | **13.9** | confirms a single bright point source is the sole driver |
| dngr finite / sharp-stars / Doppler / disk-peak / smear | pass | **pass** | no regressions, smear still absent (0.275) |

**Root-cause of the dedicated-test confound (systematic-debugging, A/B isolated):**
frame-0's cinematic framing leaves only an ~80-row sky band; in it a single bright
lensed star (row 12, col ~1075) dominates a column's amplitude, so the
location-agnostic stripe metric reads ≈28. This is the "prominence-not-presence /
framing-dependent / pixel-absolute kernels assume 1280×720" limitation the pass-2
sub-agents flagged — **not** a residual seam (the pre-R2 `J⁻¹` render peaks at the
same star/column at z≈27, i.e. the score is unaffected by the seam fix). Masking
that one star, or a 99th-percentile clip, returns the field to the clean range.

**Still deferred (owner-confirmed, pass 3):**
- **Texture-LOD (touch-points 2–3):** porting the (b′) regularization into the
  Formula-10 `texture` LOD + its offset-ray twin to remove the blocky stripe. This
  is a **Formula-10** change (needs its own SKILL.md sanction) on a **non-default**
  path with no current test coverage; it would need a new texture-mode seam test to
  validate. Not a mirror of the dngr one-liner — an anisotropic-LOD rework.
- **Dedicated stripe-detector recalibration:** make `seam_stripe_z` bright-point-
  robust (e.g. clip the top ~1% before the column autocorrelation, honoring its own
  "random point stars score low" spec) so it measures seams not stars; then it reads
  ≈14 and its `xfail` can be removed. Kept as a documented `xfail` for now.
