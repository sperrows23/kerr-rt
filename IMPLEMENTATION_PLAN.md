# Kerr Renderer Optimization — Implementation Plan

> **For agentic workers:** execute **§C** task-by-task. Two tasks are **GATED** on
> human approval of new `SKILL.md` formula entries (see §A.2) — do **not** write
> the physics for items 1.3 or 2.4 until those entries are approved. All other
> items may proceed.

**Goal:** Apply the 5-phase optimization in `guid.md` (FP32 stability, perf
overhaul, volumetrics/deep compositing, 360° camera, multi-channel EXR) to
`src/renderer/taichi_renderer.py` without violating the physics policy or the
Key Invariants.

**Scope note:** every change in `guid.md` lives in `taichi_renderer.py` plus new
scripts/config keys. The CPU reference modules (`geodesic.py`, `disk.py`,
`metric.py`, `starmap.py`) and the pytest suite are **not** edited (hard
constraint: `disk.py` must stay numerically identical for regression). The real
physics regression guard for these changes is the **manual** `gpu_test.py`
Doppler-asymmetry smoke check, not pytest.

**Preconditions (verified 2026-06-02):** `camera_matrix.json` present,
`star_image/starmap_2020_16k.exr` present (423 MB), RTX 5060 (8151 MiB, sm_120)
available, Taichi 1.7.4 CUDA JIT confirmed. All §C verification commands are
runnable on this box.

---

## A. Compatibility Audit

### A.0 Method-of-record correction (applies to every "run the test" step)

The task spec says to run `python tests/test_geodesic.py` after Phase 1/2
changes. That file has **no `__main__` block** and uses pytest fixtures, so
`python tests/test_geodesic.py` is a silent no-op. The meaningful invocation is:

```
pytest tests/test_geodesic.py -v
```

Also note: `test_geodesic.py`, `test_starmap.py`, and `cuda_smoke_test.py`
exercise the **CPU** modules and the CUDA backend — **none import
`taichi_renderer.py`'s kernels**. So no `guid.md` edit can turn the pytest suite
red by itself. The behavioral guard for these edits is:

```
python scripts/gpu_test.py --no-disk     # Doppler asymmetry must stay ≈ 7–8×
python scripts/gpu_test.py               # (with disk) same check
```

### A.1 Per-phase conflict analysis vs. Key Invariants & physics policy

**Phase 1 — FP32 stability.**

- **1.1 horizon constants:** `r_plus` already exists in config (`0.0447`).
  `k_horizon = √(1−a²)` is *derived from `a`* in Python (like the existing
  `tan_half_fov`/`E_I,L_I` host-side derivations), so it is **not** a hardcoded
  physics literal — consistent with the config-driven rule. No invariant
  conflict. **SKILL formula affected:** Formula 1 (Δ) provenance only.
- **1.2 `_delta_y` factored discriminant:** introduces
  `Δ = y(y+2k)` to kill catastrophic cancellation in `r²−2r+a²` near the
  horizon. This is an **algebraic identity** of Formula 1's Δ, *not* a
  re-derivation of physics — but the hard constraint requires any new factored
  form to be entered in `SKILL.md` with the next number. **→ needs new SKILL
  entry (Formula 11), see §A.2.** Affects: Formula 1, and every call site of
  `_delta(r,a)` (Formula 6 potentials, ZAMO init, disk emit, capture tests).
- **1.3 `u = cosθ` state-vector transform:** **this is the invariant-critical
  change.** State goes `[r,θ,φ,t,v_r,v_θ] → [y,u,φ,t,v_y,v_u]`. See §A.3 for the
  full `v_r = Δ·p_r` migration. The polar `1/sin²θ` term in **Θ(θ)** cancels
  analytically under the substitution → a **new singularity-free potential
  Θ_u(u)** that is a coordinate substitution of Formula 6. Per policy this must
  not be hand-rolled in code; **→ needs new SKILL entry (Formula 12), human
  review required, see §A.2.** **Partial disagreement with guid 1.3:** guid says
  "Remove `_SIN2_MIN` … the singularity is mathematically gone." That is true
  for the **Θ potential**, but `dφ/dλ` and `dt/dλ` (Formula 6) *retain*
  `L_z/sin²θ = L_z/(1−u²)`, which still diverges as `u→±1` for rays that do not
  actually reach the axis. Recommendation: keep a **minimal guard on the
  `dφ/dt` denominator only** (or assert `L_z≈0` on axis-crossing rays); do **not**
  blanket-remove the clamp from those two equations. Flagged for human review.
- **1.4 Kahan summation:** pure numerics (compensated accumulation of the RK4
  state and `ray_length`). No physics, no formula, no invariant conflict.

**Phase 2 — performance.**

- **2.1 delete offset ray:** removes `so/Eo/Lo/Qo/out_o`. The offset ray is
  currently the **mechanism** Formula 10 uses to estimate the Jacobian J. Deleting
  it is only safe **together with 2.4**, which re-supplies J from screen-space
  neighbors. Do not land 2.1 without 2.4 or LOD breaks.
- **2.2 adaptive step:** integration-scheme heuristic
  (`local_h = d_lambda·max(floor, y/(y+2))`). Not physics; the floor constant
  goes to config. No SKILL entry. Risk: too-aggressive stepping shifts the
  Doppler ratio → guarded by the gpu_test check.
- **2.3 hardware `ti.Texture`:** swaps the manual f16 mip pyramid
  (`star_flat/off/w/h` + `_sample_trilinear`) for `ti.Texture` + `sample_lod`.
  The equirect UV convention (`u=φ/2π, v=θ/π`) and Formula 10's **L** formula are
  unchanged, so `starmap.py` (the host single-source-of-truth) and
  `test_starmap.py` are unaffected. **Risk (flagged):** confirm Taichi 1.7.4
  `ti.Texture` supports mip generation + `sample_lod` on the CUDA backend; if mip
  auto-gen is unavailable, upload levels explicitly or keep the manual pyramid.
- **2.4 screen-space Jacobian + kernel split:** Kernel 1 (physics) writes
  `(u_exit, φ_exit, outcome)` to a field; Kernel 2 (shading) computes J by
  finite-differencing neighbor pixels. This computes the **same**
  `J = √(δθ² + sin²θ·δφ²)` as Formula 10 but from neighbors instead of an offset
  ray. SKILL Formula 10 *prescribes the offset-ray method* → **Formula 10 needs an
  amendment note** authorizing the screen-space equivalent, incl. the
  escaped-vs-captured boundary rule (a captured neighbor ⇒ clamp to `_MAX_LOD`,
  mirroring today's `out_o != ESCAPED` branch). **→ SKILL amendment, human
  review, see §A.2.**

**Phase 3 — volumetrics / deep compositing.**

- **3.1/3.2 ship depth + early termination:** needs the Blender Phase-1 Z-depth
  pass. `camera_matrix.json` exists but a **ship depth EXR is not part of the
  current asset set** — 3.1/3.2 require that input to exist (or a stub). Pure
  compositing; no GR formula. **Comparison-space caveat:** `ray_length` is a
  Mino-affine path length, **not** metric distance; comparing it to a Blender
  camera-space Z requires a documented mapping. Flagged — do not assume
  `ray_length` is in the same units as `ship_z`.
- **3.3 disk bounding-box early-out:** `|u| > sin(theta_half)` etc. Coordinate
  geometry, ties to the 1.3 `u` variable. No new physics. Reuses
  `disk.theta_half_width`.
- **3.4 transmittance-weighted Z:** new `depth_pixels` output, weighted by
  `transm·emission`. Compositing bookkeeping; no physics.

**Phase 4 — 360° camera & motion blur.**

- **4.1 equirect ray-gen:** replaces the perspective `tan_half_fov` mapping. This
  is a **camera convention**, not a GR formula (the ZAMO tetrad still consumes the
  local `n=(n^r̂,n^θ̂,n^φ̂)`). **Conflict:** the gpu_test Doppler "left-half vs
  right-half" metric assumes the *perspective edge-on* framing; under equirect the
  7–8× number changes meaning. Keep a `projection_mode` switch (default
  `perspective`) so the regression check stays valid; only flip to `equirect` for
  VR output. No SKILL entry, but document the mapping.
- **4.2 motion blur:** wrap per-pixel physics in an N-sample jitter loop on
  `phi_cam/theta_cam`. Sampling/averaging only; no physics.

**Phase 5 — multi-channel EXR.**

- **5.1/5.2:** `to_numpy()` extraction + OpenImageIO RGBAZ writer in a new script.
  `openimageio` is already a declared dependency. No physics, no invariant
  conflict. New numeric params: none required beyond a depth-infinity sentinel.

### A.2 New / amended `SKILL.md` entries required (GATES)

The hard constraint: *"If a new formula is introduced … add it to `SKILL.md`
with the next available number before using it."* Three additions are needed;
**1.3 and 2.4 are gated on human approval** because they touch physics/method.

**Formula 11 — FP32-stable factored discriminant (algebraic identity, low risk).**
```
r± = 1 ± k,   k = √(1−a²)            (horizon radii; r₊ = config r_plus)
y  = r − r₊                          (horizon-relative radial coordinate)
Δ  = (r−r₊)(r−r₋) = y·(y + 2k)       (≡ r²−2r+a², zero catastrophic cancellation)
```
Verification: `y(y+2k)` expands to `r²−2r+a²` since `r₊+r₋=2`, `r₊r₋=a²`.

**Formula 12 — Singularity-free polar potential (μ = cosθ substitution of F6 Θ).**
*Coordinate substitution of the verbatim Formula 6 Θ(θ); NOT new physics, but
must be human-verified against DNGR App. A / Carter (1968) before merge.*
```
u   = cosθ,   sin²θ = 1 − u²,   v_u ≡ du/dλ = −sinθ·(dθ/dλ)

(du/dλ)² = sin²θ·Θ(θ) = (1−u²)(Q + a²E²u²) − L_z²u²  ≡  Θ_u(u)
         # the 1/sin²θ pole cancels analytically

d²u/dλ² = ½ dΘ_u/du,
  dΘ_u/du = −2u(Q + a²E²u²) + 2a²E²u(1−u²) − 2L_z²u

state recovery:  p_θ (covariant) = v_θ = −v_u / √(1−u²)
```
**Reviewer note:** `dφ/dλ`, `dt/dλ` (Formula 6) still contain `L_z/(1−u²)`;
decide whether to keep a denominator guard or assert `L_z=0` on axis crossings.

**Formula 10 — amendment (method equivalence, human review).**
Add: *"The Jacobian J may equivalently be estimated in screen space by
finite-differencing the stored exit directions of the 4-neighborhood
`[py,px±1]`, `[py±1,px]` instead of integrating an offset ray. J, L and the
2π texel normalization are unchanged. Boundary rule: if any differenced neighbor
did not ESCAPE, clamp L to `max_lod` (the chaotic-edge case)."*

### A.3 State-vector invariant migration: `v_r = Δ·p_r`

Current (Key Invariant, `taichi_renderer.py:270`, `geodesic.py:173`):
`state = [r, θ, φ, t, v_r, v_θ]`, `v_r = Δ·p_r`, `v_θ = p_θ`.

After 1.3: `state = [y, u, φ, t, v_y, v_u]`, `y = r−r₊`, `u = cosθ`.

| Old | New | Relationship | Why preserved |
|-----|-----|--------------|---------------|
| `v_r = Δ·p_r` | `v_y` | `v_y = dy/dλ = dr/dλ = v_r = Δ·p_r` | `dy=dr` ⇒ invariant **renamed, not broken**; Δ now via Formula 11 |
| `v_θ = p_θ` | `v_u` | `v_u = du/dλ = −sinθ·p_θ = −√(1−u²)·p_θ` | genuine change — see recovery below |

**Recoveries that MUST change** (consumed by `_disk_emit` for the Formula 8
g-factor):
```
p_r (covariant) = v_y / Δ          # Δ = _delta_y(y,k); same as old v_r/Δ (NOT /Δ²)
p_θ (covariant) = −v_u / √(1−u²)   # was p_θ = v_θ
```
`_zamo_init` must return `v_y0 = Δ·p_r` (unchanged value) and
`v_u0 = −√(1−u_cam²)·p_θ0`; initial state uses `y0 = r_cam−r₊`, `u0 = cosθ_cam`.
Exit-angle recovery for the starmap UV lookup: `θ_exit = acos(clamp(u,−1,1))`,
which **supersedes** most of the `_normalize_sphere` polar-punch-through folding
(u stays in [−1,1] by construction; only a numerical clamp is needed).

### A.4 `configs/render.yaml` keys to add (no hardcoded literals in source)

```yaml
render:
  horizon_epsilon: 0.05        # replaces hardcoded _DELTA_MIN capture stop
  adaptive_step_floor: 0.005   # guid 2.2 local_h floor
  motion_blur_samples: 4       # guid 4.2
  projection_mode: perspective # perspective | equirect (guid 4.1; default keeps Doppler check valid)
  depth_infinity: 1.0e5        # guid 3.4 empty-pixel depth sentinel
camera:
  shutter_fraction: 0.020833   # 1/48 s shutter for relativistic motion blur (guid 4.2)
disk:
  bounding_sin_theta_half: 0.1494  # = sin(theta_half_width=0.15); guid 3.3 |u| early-out
```
(`bounding_sin_theta_half` may instead be derived host-side from
`theta_half_width`; listed explicitly to keep the early-out branch literal-free.)

---

## B. Risk Table

| Item | File(s) touched | Breaks existing test? | Needs new SKILL.md entry? |
|------|-----------------|-----------------------|---------------------------|
| 1.1 precompute horizon constants | `taichi_renderer.py`, `render.yaml` | No (pytest unaffected) | No (derived from `a`) |
| 1.2 `_delta_y` factored Δ | `taichi_renderer.py` | No | **Yes — Formula 11** |
| 1.3 `y,u` state transform | `taichi_renderer.py`, `render.yaml` | No (pytest); **Doppler guard** | **Yes — Formula 12 (GATED)** |
| 1.4 Kahan summation | `taichi_renderer.py` | No | No |
| 2.1 delete offset ray | `taichi_renderer.py` | No (pytest); LOD guard | No |
| 2.2 adaptive step | `taichi_renderer.py`, `render.yaml` | No (pytest); **Doppler guard** | No (scheme) |
| 2.3 `ti.Texture` starmap | `taichi_renderer.py` | No (`test_starmap` is CPU) | No (UV unchanged); **API risk** |
| 2.4 screen-space Jacobian + kernel split | `taichi_renderer.py`, `render.yaml` | No (pytest); LOD guard | **Yes — Formula 10 amend (GATED)** |
| 3.1 ship depth input | `taichi_renderer.py` | No | No (needs ship EXR asset) |
| 3.2 early ray termination | `taichi_renderer.py` | No | No (units caveat A.1) |
| 3.3 disk bounding-box early-out | `taichi_renderer.py`, `render.yaml` | No (pytest); **Doppler guard** | No |
| 3.4 transmittance-weighted Z | `taichi_renderer.py`, `render.yaml` | No | No |
| 4.1 equirect ray-gen | `taichi_renderer.py`, `render.yaml` | No (default stays perspective) | No (camera convention) |
| 4.2 motion blur | `taichi_renderer.py`, `render.yaml` | No | No |
| 5.1 extract arrays | new `scripts/export_exr.py` | No | No |
| 5.2 OIIO multi-channel EXR | new `scripts/export_exr.py` | No | No |

"Doppler guard" = no automated test breaks, but a wrong physics edit shifts the
`gpu_test.py` left/right asymmetry away from ≈7–8×; that manual check is the
gate. "LOD guard" = visual LOD/anti-aliasing regression, checked via the
`_gate2_lod_test` / gpu_test outputs.

---

## C. Execution Order (atomic commits)

Each commit = one logical change + its config/SKILL entry + a verification run.
Ordering refines `guid.md §Summary` against the codebase: SKILL/config land
**before** the code that uses them (hard constraint), data structures first,
kernel-split + texture next, features last, I/O last.

1. **Scaffolding — config + SKILL (no behavior change).**
   File: `configs/render.yaml`, `skills/kerr-physics/SKILL.md`.
   Add all §A.4 keys; add Formula 11 text; add Formula 12 and the Formula 10
   amendment **marked DRAFT / pending human review**.
   Run: `pytest -q` (unaffected) + `python -c "import yaml,io; yaml.safe_load(open('configs/render.yaml',encoding='utf-8'))"`.

2. **1.1 — precompute horizon constants.** Host computes `k_horizon=√(1−a²)`,
   reads `r_plus`; thread both into `render_pipe_a`/`render_beauty` signatures
   (used by 1.2/2.2). File: `taichi_renderer.py`.
   Run: `python scripts/gpu_test.py --no-disk` → Doppler ≈7–8× (unchanged).

3. **1.2 — `_delta_y`.** Add `@ti.func _delta_y(y,k)`; reference Formula 11 in a
   comment. Keep `_delta(r,a)` for r-space call sites for now. File:
   `taichi_renderer.py`. Run: `python scripts/gpu_test.py --no-disk` (Doppler
   ≈7–8×).

4. **[GATE] 1.3 — `y,u` state transform.** *Requires Formula 12 approved.*
   Rewrite state to `[y,u,φ,t,v_y,v_u]`; rewrite `_radial_potential*` (r=y+r₊,
   Δ via `_delta_y`), replace `_theta_potential*` with `Θ_u`/`Θ_u'`, update
   `_deriv`, `_project`, `_zamo_init`, and the per-A.3 `_disk_emit` recoveries
   (`p_r=v_y/Δ`, `p_θ=−v_u/√(1−u²)`); exit angle `θ=acos(clamp(u,−1,1))`; drop
   `_SIN2_MIN` from Θ_u (keep guarded in `dφ/dt`). Remove now-redundant
   `_normalize_sphere` θ-folding where `u` makes it moot. File:
   `taichi_renderer.py`. Run: `pytest -q` (must stay green) **and**
   `python scripts/gpu_test.py` + `--no-disk` → Doppler **≈7–8×**, North-pole
   streak gone.

5. **1.4 — Kahan summation.** Compensated accumulation in `_rk4_step` apply and
   `ray_length`. File: `taichi_renderer.py`. Run: `python scripts/gpu_test.py
   --no-disk` (Doppler ≈7–8×).

6. **2.1 — delete offset ray.** Remove `so/Eo/Lo/Qo/out_o` + offset branches from
   both kernels (LOD temporarily falls back to `L=0`). File: `taichi_renderer.py`.
   Run: `python scripts/gpu_test.py --no-disk` (renders; LOD interim).

7. **2.2 — adaptive step.** `local_h = d_lambda·max(adaptive_step_floor, y/(y+2))`.
   File: `taichi_renderer.py`. Run: `python scripts/gpu_test.py --no-disk`
   (Doppler ≈7–8×, faster).

8. **2.3 — `ti.Texture` starmap.** Replace pyramid fields + `_sample_trilinear`
   with `ti.Texture(rgba16f)` + `sample_lod`; do the swap atomically (don't keep
   both resident — VRAM, §D). File: `taichi_renderer.py`, `setup_renderer`.
   Run: `python -m renderer.taichi_renderer` (`_gate2_lod_test`) → non-black,
   NaN=0.

9. **[GATE] 2.4 — kernel split + screen-space Jacobian.** *Requires Formula 10
   amendment approved.* Kernel 1 writes `(u_exit,φ_exit,outcome)` field; Kernel 2
   reads neighbors, computes J→L (captured-neighbor ⇒ `max_lod`), samples
   `sample_lod`. File: `taichi_renderer.py`. Run: `python scripts/gpu_test.py`
   (LOD anti-aliasing restored) + **full `pytest -q`** (end of Phase 2).

10. **3.1 — ship depth input.** Add `ti.Texture(r32f)` arg for the ship Z pass
    (guard for missing asset). File: `taichi_renderer.py`. Run: gpu_test.

11. **3.2 — early ray termination.** `if ray_length > ship_z: break` with the
    A.1 unit-mapping documented. File: `taichi_renderer.py`. Run: gpu_test.

12. **3.3 — disk bounding-box early-out.** `if r<r_inner or r>r_outer or
    |u|>bounding_sin_theta_half: continue`. File: `taichi_renderer.py`. Run:
    `python scripts/gpu_test.py` → Doppler ≈7–8× (unchanged, just faster).

13. **3.4 — transmittance-weighted Z.** New `depth_pixels` field;
    `weighted_depth/total_emission`, else `depth_infinity`. File:
    `taichi_renderer.py`. Run: gpu_test, inspect depth range.

14. **4.1 — equirect ray-gen.** Branch on `projection_mode`; default
    `perspective`. File: `taichi_renderer.py`, `render.yaml`. Run:
    `python scripts/gpu_test.py --no-disk` (perspective path → Doppler ≈7–8×).

15. **4.2 — motion blur.** `for s in range(motion_blur_samples)` jitter loop on
    `phi_cam/theta_cam` from `shutter_fraction`. File: `taichi_renderer.py`. Run:
    gpu_test.

16. **5.1 — extract arrays.** `beauty_rgb`, `depth_z` via `to_numpy()` in new
    `scripts/export_exr.py`. Run: `python scripts/export_exr.py --frame 0`.

17. **5.2 — OIIO multi-channel EXR.** Stack RGBA+Z, channelnames `("R","G","B","Z")`,
    write via OpenImageIO. File: `scripts/export_exr.py`. Run: `python
    scripts/export_exr.py --frame 0` then verify channels with `oiiotool --info`.

---

## D. VRAM Budget

Configured production resolution: **3840×2160** (`render.width/height`); Taichi
pool capped at `device_memory_gb = 6.0`; GPU has 8151 MiB.

| Buffer | Size @ 4K | Notes |
|--------|-----------|-------|
| Starmap **manual** f16 RGB pyramid (current) | ~1.0 GiB | 16384×8192×3×2 = 0.75 GiB base ×~4/3 mips |
| Starmap **`ti.Texture`** rgba16f + mips (after 2.3) | ~2.7 GiB | 16384×8192×4×2 = 2.0 GiB base ×~4/3; +1 channel vs manual |
| `frame_pixels` f32 (H,W,3) | 95 MB | output beauty |
| exit field f32 (H,W,2) + outcome i32 (H,W) (2.4) | 63 + 32 MB | kernel-split hand-off |
| `depth_pixels` f32 (H,W) (3.4) | 32 MB | weighted Z |
| `ship_depth` r32f (H,W) (3.1) | 32 MB | imported Z |

**Peak estimate:** ~2.7 GiB (texture) + ~0.25 GiB (split/depth/ship fields)
≈ **~3.0 GiB**, comfortably under the 6 GiB pool and 8 GiB card.

**Phase 2.4 kernel-split fields do NOT push usage above `device_memory_gb`** —
they add only ~0.13 GiB at 4K; the dominant consumer is the starmap. **The one
caution is the 2.3 swap:** if the manual pyramid (~1.0 GiB) and the new
`ti.Texture` (~2.7 GiB) are both resident transiently (~3.7 GiB), that is still
< 6 GiB, but the swap should free the old fields before/at texture upload to keep
margin. The rgba16f channel bump (3→4) is the only real increase; if margin ever
tightens, drop to a 3-channel texture format or keep the manual pyramid.

---

## Status

- [x] **Step 1 — `IMPLEMENTATION_PLAN.md` written and complete.**
- [x] §A.2 GATES approved by project owner (2026-06-02): SKILL.md Formula 11/12 +
  Formula 10 amendment written; polar guard kept on dφ/dt only.
- [x] **Commit 1 — config + SKILL scaffolding.** Config keys added; SKILL.md F11/F12 +
  F10 amendment landed (rev v1.4). pytest 5/5, config parses.
- [x] **Phase 1 — FP32 stability (1.1–1.4).** Horizon constants derived in Python
  (`_horizon_constants`); `_delta_y` (F11); `[y,u,…]` state transform with Θ_u (F12)
  and the `v_r=Δ·p_r → v_y` / `p_θ=−v_u/√(1−u²)` migration; Kahan summation.
  **Verified:** pytest 5/5; Doppler **7.78×** (== baseline), disk max=14.11 exact.
- [x] **Phase 2 core — perf (2.1, 2.2, 2.4).** Offset ray eliminated; adaptive Mino
  step (with the `ds=local_h` Riemann-sum fix); `render_beauty` split into
  `render_beauty_physics` + `render_beauty_shade` with the screen-space-Jacobian
  LOD (F10 amendment). **Verified:** full pytest **12/12**; Doppler **7.77×**;
  ~40% faster (1.8s→1.1s @ FHD).
- [~] **2.3 — hardware `ti.Texture`: DEFERRED (Taichi 1.7.4 limitation).** Probe
  confirmed `ti.Texture` exposes no mipmap upload/`from_numpy` API; `sample_lod`
  would sample base-level only and reintroduce starmap flicker. Kept the correct
  manual f16 mip pyramid (within VRAM budget §D). Revisit only if Taichi is
  upgraded (CLAUDE.md pins 1.7.4 for the RTX 5060).
- [x] **Phase 3 (3.3, 3.4 done; 3.1/3.2 blocked).** 3.3 disk bbox early-out in
  `u`-space (`|u|<bound_sin_half`, radial band in `y`); 3.4 transmittance-weighted
  Mino-affine Z → `depth_pixels` (+∞ sentinel for empty pixels), returned via
  `render_beauty_frame(..., return_depth=True)`. **Verified:** perspective Doppler
  unchanged 7.77×; depth range sane. 3.1/3.2 (ship occlusion) remain blocked on a
  Blender ship **Z-depth EXR** asset that does not yet exist.
- [x] **Phase 4 (4.1, 4.2).** 4.1 equirect 360° ray-gen behind `projection_mode`
  (default perspective → Doppler check preserved); 4.2 motion blur as host-side
  averaging of camera-rotated sub-frames (`render_beauty_frame_mb`), compatible
  with the 2.4 split. **Verified:** equirect renders; blur averages 69% of
  background pixels, disk correctly φ-rotation-invariant.
- [x] **Phase 5 (5.1, 5.2).** `scripts/export_exr.py`: extracts beauty + depth via
  `to_numpy()` and writes a 4-channel `(R,G,B,Z)` EXR via OpenImageIO to
  `render_blackhole/bh_####.exr`. **Verified:** channels + depth confirmed with
  `oiio` read-back.

**Note (config bug surfaced):** `configs/render.yaml black_hole.r_plus = 0.0447`
is mislabeled — it is `k=√(1−a²)`, not the outer horizon (true r₊=1.0447), and
its comment (`= 1 − √(1−a²)` = 0.955) is also wrong. It is consumed by
`scripts/thumb.py` as an `r_floor`, so it was **left untouched**; the true r₊ is
derived in `_horizon_constants(a)`. Recommend a separate fix to rename/clarify
that key.

render_pipe_a (the 256² dev LOD test kernel used by `_gate2_lod_test`) was
migrated to the `[y,u,…]` state but intentionally **retains its offset ray** —
it is not on the 4K production path, and it remains the offset-ray LOD reference.
