# P4 + P5 — Kelvin-Helmholtz edge erosion (CKS-22) & fractal LOD cascade (CKS-23)

**Status:** RATIFIED (owner, 2026-06-20) — all 8 open decisions resolved with the
recommended options (see §D.2); ready for SKILL ratification (CKS-22 / CKS-23) + the
implementation plan (`docs/specs/2026-06-20-P4-P5-edge-erosion-lod-cascade-plan.md`). Not
yet SKILL.md formulas; no kernel code until CKS-22 / CKS-23 are written and approved. This
document frames the design for the two remaining **refinement** pillars of the 2026-06-16
roadmap
(`docs/specs/2026-06-16-cinematic-volumetric-multiphase-scattering-design.md`, Part C.3 /
C.4) and surfaces the owner decisions to be resolved one-variable-at-a-time **before** any
code is written. The two headline PHYSICS pillars are already shipped: **P2 (multi-phase,
CKS-19)** and **P3 (single-scatter + HG, CKS-20)**. P4 and P5 are lower-risk
VISUALIZATION / SAMPLING refinements that reuse machinery we already ship.

**Governance class:** both are **non-PHYSICS** (P4 = VISUALIZATION, like CKS-12 §3; P5 =
SAMPLING, like Formula 10). Neither touches `j_ν`, `κ_ν`, the RTE, or extinction. Per the
roadmap's split they get **CKS-22 / CKS-23** promoted from "reserved" to full SKILL
entries, owner-approved, **before** kernel code — but as amplitude/sampling rules, not new
radiative transfer.

**Math of record (proposed):** P4 is an extension of **CKS-12 §3** (the ragged outer
smoothstep edge); P5 is an extension of **Formula 10** (the screen-space LOD Jacobian) and
the disk-noise octave loop. Both are *candidates* here — each must be ratified into SKILL.md
and owner-approved before implementation, exactly as CKS-18 was before V3.0.

---

# PART A — Shared framing

## A.1 Scope

- **In scope:** P4 (CKS-22, KH threshold erosion) and P5 (CKS-23, fractal LOD octave
  cascade). One combined SCOPE doc; each pillar still gets its own `writing-plans` task
  breakdown after owner sign-off.
- **Out of scope — and a record correction:** **P1 (CKS-21, scale-dependent shear
  cascade) is NOT covered here and was never built.** It is easy to conflate with the
  shipped CKS-12 §2 dual-phase shear reset + CKS-18 §2 curl-flow boil (both long landed and
  GPU-verified) — but CKS-21 is a *different* idea: a **frequency-dependent** shear transfer
  `S(f)=1/(1+(f/f_c)^p)` that protects high-frequency octaves from winding into infinite
  spirals (Kolmogorov-like). It remains *reserved* (roadmap C.2) for a later doc.
- A genuine fluid/MHD continuity solve remains out of scope (roadmap invariant 6). Both
  pillars are texturing / sampling approximations.

## A.2 Governance & guard rails

| Pillar | Touches j_ν / κ_ν / RTE / extinction? | Class | SKILL artifact |
|---|---|---|---|
| **P4** KH erosion | No — thresholds the density envelope | **VISUALIZATION** | **CKS-22** (promote from reserved) |
| **P5** LOD cascade | No — sets the noise sampling rate / octave count | **SAMPLING** | **CKS-23** (promote from reserved) |

Neither pillar may touch `p_μ`, `u^μ` (CKS-8), `g` (CKS-9), the `g⁴` exponent (Formula 9),
the `f_PT` radial shape (CKS-11), or the blackbody chroma form. Amplitude (P4) and
sampling-rate (P5) quantities only — the CKS-12 §3 hard-constraint list applies verbatim.

## A.3 Independence & sequencing

```
P4 (KH erosion)  ── extends CKS-12 §3 ─── independent ──┐
P5 (LOD cascade) ── extends Formula 10 + octave loop ───┴── either order; P5 unblocks V4 free cam
```

P4 and P5 are **independent of each other** — P4 modulates the outer edge envelope; P5
modulates the noise sampling rate. Either can land first. P5 is the **prerequisite for the
V4 free camera** (it kills macro-view shimmer and resolves close-up micro-wisps), so it has
the higher forward value; P4 is the lower-risk, smaller diff. The only external coupling is
P4 ↔ **CKS-19** (shipped): when multiphase is ON the erosion must shred the *shared* edge
envelope so both `ρ_hot` and `ρ_cold` fray consistently (see B.3).

## A.4 Cross-cutting invariants (both pillars honor these)

1. New behavior defaults **OFF**; the disabled path is **bit-identical** to current goldens
   (the constraint-6 discipline every CKS-11…20 increment follows).
2. CPU/GPU twins stay byte-aligned; every formula is ported, never re-derived
   (CLAUDE.md CRITICAL RULE).
3. All params live in `configs/render.yaml`; any derived value goes through the CKS-13
   resolver. **Neither pillar requires a resolver change** under the recommended options
   (P4 base dials; P5 `J_0` a base dial).
4. Docs-sync: each landed change updates SKILL.md (CKS-22 / CKS-23) + PROJECT.md §6/§7 in
   the same task.
5. Determinism: integer-hashed seeds from config, **no `ti.random`**; same seed + same
   `t_disk` ⇒ identical frame.

---

# PART B — P4: Kelvin-Helmholtz threshold erosion (CKS-22, VISUALIZATION)

## B.1 Goal & acceptance

Today the disk's outer boundary is a **multiplicative smoothstep** on `r_out_eff` (CKS-12
§3) — ragged in radius (via `e_out`) but still a *clean, continuous rim*. Real shear layers
suffer Kelvin-Helmholtz instability: the boundary **tears** into fingers and detaches into
wisps. P4 replaces the clean rim with a **noise-thresholded clip** that shreds the edge into
vacuum.

**Defining acceptance:** with erosion ON, the outer edge in a beauty frame must show
**disconnected tendrils / holes torn into vacuum** within the boundary band — not a smooth
falloff. With erosion OFF (or `τ_KH = 0`) the frame is **bit-identical** to the current
ragged-smoothstep golden.

## B.2 The proposed physics (summary; formal statement = SKILL CKS-22)

Amends the CKS-12 §3 outer window. Inside the boundary band only, multiply ρ by a
narrow soft-Heaviside of a noise-thresholded envelope:

```
ρ_env(r)  = smoothstep window on [r_in_eff, r_out_eff]          # the CURRENT CKS-12 §3 envelope
N_KH       = sfbm3( advected (u, φ′, ζ) ; hash(seed_KH) )        # high-freq simplex, own seed, §2 material frame
ρ ← ρ · H_soft( ρ_env(r) − τ_KH · N_KH ,  w_soft )              # threshold clip ⇒ tearing
```

- **`N_KH`** reuses the shipped V1.5 simplex basis (`sfbm3`), at a **higher base frequency**
  than the bulk density layers (fine fingers), with its own seed offset `NSEED_KH`. It is
  advected in the §2 material frame (the same `φ′` the other layers already use) so the
  fingers wind with the bulk orbit.
- **`τ_KH`** = erosion strength (0 ⇒ no erosion). Larger ⇒ the noise punches deeper holes.
- **`H_soft(x, w)`** = a *narrow* smoothstep Heaviside of half-width `w_soft` — narrow enough
  to read as a torn edge, wide enough to not alias (see B.4).
- **Band-limited:** the clip is applied **only where `ρ_env` is in its outer transition band**
  `[r_out_eff·(1−band_frac), r_out_eff]`. Inside the band the smoothstep is partial, so the
  subtraction can drive it to 0 (tearing); in the disk interior `ρ_env ≈ 1 ≫ τ_KH·N_KH`, so
  `H_soft → 1` and the interior is untouched. **Outer edge only** (owner-locked): the inner
  `r_in_eff` zero-torque BC (CKS-12 constraint 3, `r_in_eff ≥ r_isco`) is **not** eroded.

## B.3 Architecture changes (touch-list for the plan — no code yet)

1. **`_disk_noise_mod_fields` / the edge-window assembly** (CKS-12 §3) gains the `N_KH`
   evaluation + the `H_soft` clip, gated behind `edge_erosion.enabled`. CPU twin in
   `noise.py` (`_mod_fbm4` / the envelope builder) mirrors it.
2. **Shared-envelope coupling with CKS-19 (multiphase).** Erosion multiplies the **shared
   edge window** that both `ρ_hot` and `ρ_cold` consume — i.e. the `H_soft` factor is
   applied to `ρ_env` *before* the hot/cold split, so a torn finger removes emission **and**
   absorption together (a silhouette-correct frayed dust lane, not a half-tear). When
   multiphase is OFF this is just the single ρ. **(Open decision B.6-3.)**
3. **Vertical step cap (CKS-5 Pipe-B / `max_step_vfrac`).** Sharpening the edge raises the
   local radial gradient; `w_soft` gets a **floor** tied to `max_step_vfrac` (B.4) so the
   torn edge never aliases under the existing step cap. No new step-cap field; just a floor
   on `w_soft` at resolve / setup time.
4. **Param buffer** `_setup_disk_noise` grows past the current `_NOISE_N` (presently 57 after
   CKS-19; confirm at implementation) by the `edge_erosion` dials.
5. **No GR/RTE field changes** — `ρ_env` and ρ are amplitude quantities; emission/`dτ⃗`
   forms (CKS-12 / CKS-19 / CKS-20) are structurally untouched.

## B.4 Step-cap safety (the `w_soft` floor)

The vertical step cap guarantees ≥ N samples across the modulated scale height. The torn
*radial* edge must likewise be resolved: require the soft-Heaviside transition to span at
least one capped step in the relevant coordinate. Proposed floor (exact constant a B.6
decision):

```
w_soft  ≥  k_soft · max_step_vfrac · σ_θ(r_out)        # k_soft ≈ 1 (≥ one capped step wide)
```

`τ_KH = 0` or `enabled:false` ⇒ `H_soft ≡ 1` ⇒ the CKS-12 §3 ragged smoothstep, **bit-for-bit**.

## B.5 Config schema (additions to `disk:` in `render.yaml`)

```yaml
disk:
  edge_erosion:              # CKS-22 — KH threshold erosion of the OUTER edge. Default OFF ⇒ bit-identical.
    enabled: false           # false (or strength 0) ⇒ current CKS-12 §3 ragged smoothstep, bit-for-bit.
    strength: 0.0            # τ_KH — erosion depth; 0 ⇒ no tearing.
    freq: 8.0               # N_KH base frequency (≫ bulk layers ⇒ fine fingers). Integer in φ (seam-free).
    octaves: 3              # N_KH fBm octaves.
    band_frac: 0.15         # erosion confined to the outer [r_out_eff·(1−band_frac), r_out_eff] band.
    soft_width: 0.0         # w_soft; 0 ⇒ auto = the step-cap floor (B.4). >0 ⇒ explicit (floored).
    seed: 0                 # NSEED_KH offset.
```

All base look dials → **no CKS-13 resolver change**. (`soft_width: 0` ⇒ derive the floor at
setup, which *reads* `max_step_vfrac` but stores no derived literal in the YAML.)

## B.6 Open owner decisions — P4

1. **N_KH advection frame:** (a) **§2 material-frame shear only** *(recommended — minimal,
   matches every other layer)* vs (b) §2 **+ the CKS-18 curl warp** (fingers also boil with
   the eddies — richer, but couples erosion to the curl clock).
2. **`w_soft` floor constant `k_soft`:** confirm `k_soft ≈ 1` (one capped step) vs a more
   conservative `k_soft = 2`. Drives how sharp the tears may get before the step cap.
3. **Shared-envelope erosion (with CKS-19 on):** (a) **erode the shared `ρ_env` before the
   hot/cold split** *(recommended — silhouette-correct)* vs (b) independent `H_soft` per
   phase (decoupled hot/cold tearing — more dials, rarely wanted).
4. **Config home:** (a) **new `disk.edge_erosion` block** *(recommended — distinct
   subsystem)* vs (b) nest under `disk.noise`.

## B.7 Tests (TDD)

- **Bit-identity regression** (`tests/test_gpu_regression.py`): `enabled:false` ⇒ goldens
  bit-identical (constraint 1). The key guard.
- **CPU/GPU twin parity** (`tests/test_noise.py` / `test_noise_gpu.py`): `N_KH` and the
  eroded ρ agree within `_SATOL`.
- **Tearing acceptance** (new `tests/test_disk_edge_erosion.py`): with erosion ON, the outer
  band contains ρ→0 holes **interior to `r_out_eff`** (disconnected support), absent when OFF.
- **Interior-untouched:** ρ in the disk body (well inside the band) is unchanged vs OFF.
- **Step-cap floor:** with `soft_width:0`, the derived `w_soft` ≥ the B.4 floor (no aliasing
  regression on the face-on frame the step cap originally fixed).
- **Multiphase coupling** (if B.6-3a): a torn finger removes emission *and* absorption
  together (the frayed silhouette stays silhouette-correct).

---

# PART C — P5: Fractal LOD octave cascade (CKS-23, SAMPLING)

## C.1 Goal & acceptance

The disk noise runs a **fixed `octaves`** count regardless of how much screen area a sample
covers. Macro views alias (sub-pixel octaves shimmer/moiré); close-ups blur (no sub-octave
detail). Formula 10 already solves this for the **background** starmap via a screen-space
LOD Jacobian — P5 brings the same *constant-detail-density* idea to the **disk volume**.

**Defining acceptance:** the on-screen detail density of the disk noise is ~constant with
camera distance — far views drop high octaves (no shimmer), close-ups gain octaves (crisp
wisps) — **with no temporal popping** as the octave count changes. With LOD OFF the frame is
**bit-identical** to the current fixed-`octaves` golden.

## C.2 The proposed sampling rule (summary; formal statement = SKILL CKS-23)

Amends Formula 10 (LOD) and the disk-noise fBm octave loop. Two pieces: a **per-sample
analytic footprint** `J` (owner-locked), and a **smooth octave gate** that reads it.

**Per-sample footprint (owner-locked: analytic, not screen-space FD).** F10 v1.4's
screen-space finite-difference Jacobian was built for *one* background exit per pixel; a
volumetric disk march takes *many* samples per ray at varying depth, so we compute the
footprint analytically at each sample instead:

```
ε        = fov_y / HEIGHT                                   # pixel cone half-angle (rad)
w_world  = ε · d_sample                                     # world-space footprint at sample distance d (small-angle)
J        = w_world · f_ref                                  # noise-coord footprint (cycles): f_ref = local noise frequency scale
```

`d_sample` is the camera→sample distance (already known in the march). `f_ref` converts the
world footprint into the noise's own coordinate cycles (anchored by `J_0`, C.5). `J` grows
with distance ⇒ far samples have a large footprint ⇒ fewer octaves.

**Smooth octave gate (the GPU-friendly core).** Keep the `ti.static` unrolled fBm loop at
`N_max` octaves; give each octave `o ∈ {0…N_max−1}` a **continuous gate** so no integer
popping occurs:

```
N_oct  = clamp( N_max − log₂(J / J_0),  N_min,  N_max )     # target (fractional) octave count
g_o    = clamp( N_oct − o,  0,  1 )                          # per-octave weight: 1 below cutoff, fractional at top, 0 above
n      = Σ_{o=0}^{N_max−1}  g_o · gain^o · N( coord · lac^o ; … )
```

The top **partial** octave is crossfaded by the fractional part of `N_oct` automatically
(`g_o ∈ (0,1)` for the boundary octave) — the anti-pop crossfade the roadmap calls for.
Octaves above the footprint get `g_o = 0` (culled, no shimmer); octaves below get `g_o = 1`
(full detail).

**Octaves only (v1).** The roadmap also floats coarsening the *step* `dλ ∝ J`. That
interacts with the Pipe-B vertical step cap and the self-shadow bake sampling, so v1
modulates **octave count only**; `dλ` step coarsening is deferred (C.6-2).

## C.3 Bit-identity hook

`enabled:false` ⇒ force `N_oct = N_max` and `N_max = octaves` ⇒ `g_o ≡ 1` for all
`o < octaves` ⇒ the loop sums exactly the current `octaves` terms with current weights ⇒
**bit-for-bit** the present fixed-octave path. Requirement: `N_max ≥ octaves`, and the
default `J_0` / `N_min` chosen so that with LOD off nothing gates out.

## C.4 Architecture changes (touch-list for the plan — no code yet)

1. **The disk-noise fBm `@ti.func`s** (`_disk_noise_m`, `_mod_fbm4`, and the CPU twins
   `sfbm2/3`, `_disk_noise_m`) accept `N_oct` (or `J`) and apply the `g_o` gate inside the
   existing unrolled octave loop. The loop bound becomes `N_max` (a `ti.static` constant ≥
   current `octaves`).
2. **The march** (`render_beauty_physics`) computes `d_sample` (already available) → `J` →
   `N_oct` per sample and threads it into the noise calls. Cheap: a `log₂` + a few clamps,
   no neighbor field, no second ray.
3. **`J_0` anchoring** is a base dial (C.5) → no CKS-13 resolver change.
4. **Param buffer / signature:** `N_max`, `N_min`, `J_0`, `enabled` flow through
   `_setup_disk_noise`; `_NOISE_N` grows.
5. **Optional consistency with Formula 10:** keep the *scalar* `J = max(...)` philosophy of
   F10 (isotropic footprint) for v1 (C.6-1); the background LOD path is unchanged.

## C.5 Config schema (additions to `disk:` in `render.yaml`)

```yaml
disk:
  lod:                       # CKS-23 — fractal LOD octave cascade. Default OFF ⇒ bit-identical.
    enabled: false           # false ⇒ N_oct ≡ N_max ≡ octaves ⇒ current fixed-octave path, bit-for-bit.
    n_max: 0                 # N_max; 0 ⇒ auto = disk.noise.octaves (the bit-identity anchor). >octaves to allow close-up sub-octaves.
    n_min: 1                 # floor octave count for far/macro views.
    j0: 1.0                  # J_0 reference footprint (cycles) at which N_oct = N_max. Look anchor; no resolver.
```

`enabled:false` (or `n_max:0` resolving to `octaves` with `J_0` un-gating) ⇒ zero golden
movement. `n_max > octaves` is what actually *adds* close-up detail; it must be paired with
`enabled:true` to take effect.

## C.6 Open owner decisions — P5

1. **Footprint anisotropy:** (a) **isotropic scalar `J`** *(recommended v1 — matches F10's
   `J = max(Jx,Jy)`, simplest)* vs (b) per-axis `(u,φ,ζ)` footprints feeding anisotropic
   octave counts (sharper but 3× the bookkeeping; defer).
2. **Step coarsening `dλ ∝ J`:** (a) **octaves-only v1, defer `dλ`** *(recommended — `dλ`
   couples to the step cap + self-shadow bake)* vs (b) include `dλ` now (more perf upside,
   higher risk).
3. **`J_0` anchoring:** (a) **base dial** *(recommended — no resolver change, preserves the
   no-resolver-touch property)* vs (b) derive from the smallest disk-noise wavelength at
   `r_inner` via CKS-13 (auto-anchored but touches the resolver).
4. **Temporal-stability test:** confirm the relational golden strategy — render the same
   disk at two camera distances and assert the summed-noise field has no discontinuity as
   `N_oct` crosses an integer (the `g_o` crossfade), since a true multi-frame pop test needs
   the V4 camera path.

## C.7 Tests (TDD)

- **Bit-identity regression** (`tests/test_gpu_regression.py`): `enabled:false` ⇒ goldens
  bit-identical. The key guard.
- **Footprint monotonicity** (`tests/test_disk_lod.py`): `J` increases with `d_sample`;
  `N_oct` decreases with distance (far ⇒ fewer octaves).
- **Crossfade continuity:** sweep `J` across an integer `N_oct` boundary; the summed noise
  `n` is continuous (no jump) — guards the `g_o` anti-pop construction.
- **CPU/GPU twin parity** (`tests/test_noise_gpu.py`): the gated fBm and `J`/`N_oct` agree
  within `_SATOL`.
- **Close-up gain:** with `n_max > octaves` and a near camera, the noise field carries
  octaves absent from the OFF render (resolves micro-wisps).

---

# PART D — Phasing & docs-sync

Each pillar is an independent `writing-plans` task breakdown after owner sign-off; either
can go first (A.3). Per-pillar sequence mirrors every prior increment:

1. **Promote the reserved SKILL entry to full** (CKS-22 for P4, CKS-23 for P5) with the math
   of record above + the resolved open decisions; **owner review before kernel code**
   (governance §). Bump SKILL.md version + revision history + "When to use".
2. **CPU twin first (TDD):** the new primitive (`N_KH` clip for P4; the `g_o` gate for P5)
   in `noise.py`, with parity tests, before touching the GPU.
3. **GPU twin:** port verbatim into `taichi_renderer.py` (the §3 envelope for P4; the fBm
   octave loop + per-sample `J` for P5); param buffer growth.
4. **Config + (no) resolver:** add the `disk.edge_erosion` / `disk.lod` blocks; confirm no
   CKS-13 change under the recommended options.
5. **Goldens:** the bit-identity regression (OFF path) + one acceptance frame per pillar
   (a frayed-edge beauty frame; a near/far LOD pair).
6. **Docs-sync (same task):** SKILL.md (CKS-22 / CKS-23) + PROJECT.md §6/§7 updated with the
   landed change (Docs Sync Policy).

## D.1 Risk / effort / payoff (from roadmap §4, refined)

| Pillar | Effort | Risk | Payoff | Principal risk |
|---|---|---|---|---|
| **P4** KH erosion | Low | Low | Med | hard threshold vs step-cap — sharp edges alias if `w_soft` floor wrong (B.4) |
| **P5** LOD cascade | Med | Med | High (future) | per-sample `J` plumbing + temporal stability of a varying octave count; unblocks V4 |

## D.2 Open owner decisions — RESOLVED (owner, 2026-06-20)

All resolved with the recommended option.

**P4 (B.6):**
1. **N_KH advection → §2 material-frame shear only** (B.6-1a). Minimal; matches every other
   layer. Curl-coupling reserved for later if the fingers read as too rigid.
2. **`w_soft` floor → `k_soft = 1`** (B.6-2), i.e. the soft-Heaviside spans ≥ one capped
   vertical step: `w_soft ≥ max_step_vfrac · σ_θ(r_out)`.
3. **Shared-envelope erosion → erode `ρ_env` before the hot/cold split** (B.6-3a). One
   `H_soft` shreds both `ρ_hot` and `ρ_cold` ⇒ silhouette-correct frayed lanes under CKS-19.
4. **Config home → new `disk.edge_erosion` block** (B.6-4a). Distinct subsystem.

**P5 (C.6):**
1. **Footprint → isotropic scalar `J`** (C.6-1a). Matches Formula 10's `J = max(Jx,Jy)`
   philosophy; anisotropic per-axis reserved for a later increment.
2. **Octaves-only; defer `dλ` step coarsening** (C.6-2a). `dλ` couples to the step cap +
   self-shadow bake — out of scope for v1.
3. **`J_0` → base dial** (C.6-3a). No CKS-13 resolver change (preserves the
   no-resolver-touch property).
4. **Temporal stability → relational two-distance golden** (C.6-4): render the same disk at
   two camera distances + a `J`-sweep continuity unit test across an integer `N_oct`
   boundary (true multi-frame pop test waits for the V4 camera path).

**Next step (DONE 2026-06-20):** implementation plan written
(`2026-06-20-P4-P5-edge-erosion-lod-cascade-plan.md`). Per-pillar Task 1 ratifies
CKS-22 / CKS-23 into SKILL.md for approval **before** any kernel code (governance §).
