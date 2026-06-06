# Kerr Black Hole Renderer — Project Guide

Single reference for the whole project: what it is, how the pipeline fits
together, every file in the codebase, what has shipped, what remains, and the
one forward-looking proposal. This file supersedes the former `PROJECT_MAP.md`,
`IMPLEMENTATION_PLAN.md`, `REMAINING_WORK_PLAN.md`, `PIPELINE_OVERVIEW.md`, and
`BACKGROUND_DNGR_PLAN.md` (all merged here).

> **Policies in force everywhere (from `CLAUDE.md`):**
> - **Physics:** every GR/Kerr formula comes verbatim from
>   `skills/kerr-physics/SKILL.md` — never re-derived. A formula that looks wrong
>   is flagged for human review, not silently replaced.
> - **Config:** every numeric parameter lives in `configs/render.yaml` — no
>   hardcoded physics or render literals in source.
> - **Backend:** GPU is locked to `ti.init(arch=ti.cuda)` — never `ti.gpu`.
> - **Units/coords:** geometric `G = M = c = 1`; **Cartesian Kerr-Schild `(t, x, y, z)`**
>   (active renderer path, SKILL.md PART II; spin axis = +z, CKS `r` is the BL radius,
>   `z = r cosθ`). Boyer-Lindquist `(t, r, θ, φ)` is retired/history only.
>   Signature `(− + + +)`; spin `a = 0.999` (near-extremal).
> - **Encoding:** read text/config with `utf-8` / `utf-8-sig` (Windows cp949 box).

---

## Contents

1. [Overview](#1-overview)
2. [End-to-end pipeline (Stages 1–4)](#2-end-to-end-pipeline-stages-14)
3. [Codebase map](#3-codebase-map)
4. [Configuration reference (`render.yaml`)](#4-configuration-reference-renderyaml)
5. [Physics formula index (`SKILL.md`)](#5-physics-formula-index-skillmd)
6. [Implementation status — shipped](#6-implementation-status--shipped)
7. [Remaining work & known issues](#7-remaining-work--known-issues)
8. [Future proposal — DNGR background rearchitecture (gated)](#8-future-proposal--dngr-background-rearchitecture-gated)
9. [Reference material & related files](#9-reference-material--related-files)

---

## 1. Overview

Offline renderer for a Kerr black hole, built around a Taichi/CUDA photon-geodesic
tracer. The repository's focus is **Phase 2** (the GPU renderer); Phases 1 and 3
are Blender wrappers and Phase 4 is a Resolve grade.

The GPU renderer runs two photon sub-pipelines per pixel inside split kernels:

- **Pipe A** — trace a photon *backward* through curved spacetime; on escape, look
  up the gravitationally lensed 16K starmap with Formula-10 differential-mip
  anti-aliasing (screen-space Jacobian LOD).
- **Pipe B** — accumulate volumetric accretion-disk emission along the same path
  (g⁴ relativistic beaming, Formula 9), composited in front of the background.

---

## 2. End-to-end pipeline (Stages 1–4)

```
Stage 1  Blender         scene + ship animation, camera export, ship render
Stage 2  Taichi (CUDA)   black-hole / accretion-disk render → multi-channel EXR
Stage 3  Blender         compositing: deep merge, bloom, lens distortion, aberration
Stage 4  DaVinci Resolve conform, color grade, finishing, delivery
```

Stages 1–2 are **implemented in this repo**. Stages 3–4 are the **intended,
documented workflow** — no compositor `.blend` or Resolve project is checked in
yet, so those sections describe the recommended setup and flag assumptions.

**Assumptions flagged up front**

- **A1 — Frame rate = 24 fps.** Explicit in config (`render.fps = 24.0`). The
  motion-blur shutter arc is `Δφ · fps · shutter_fraction`
  (`camera.shutter_fraction = 1/48 s`), which equals the 180° shutter (`Δφ·0.5`)
  at 24 fps and scales correctly otherwise.
- **A2 — Stage 3 (Blender compositor) is not yet built.** The node graph below is
  the recommended construction.
- **A3 — Stage 4 (DaVinci Resolve) is a finishing recommendation only.**
- **A4 — The black-hole EXR is scene-linear HDR.** `export_exr.py` writes the raw
  un-tonemapped beauty buffer; tone mapping happens later (Stage 3/4).
- **A5 — Ship Z-depth occlusion is not wired yet** (blocked on a Blender ship Z
  pass; see §7, items 3.1/3.2). Until then, composite by layer order, not Z.

### Stage 1 — Blender: scene, camera export, ship render

Produces the two things downstream consumes: a per-frame camera track
(`camera_matrix.json`) and the spaceship EXR sequence
(`render_spaceship/ship_####.exr`).

- **Scene:** the spaceship is the only animated hero element; the black hole and
  disk come from Stage 2. The renderer interprets the camera world position in
  geometric units (M): camera sits at `r = √(x²+y²+z²)`; production framing uses
  `camera.default_radius ≈ 6.03 M`, `default_fov_deg = 90°`.
- **Camera export** (`src/blender/export_camera.py`, run inside Blender): writes one
  JSON record per frame — `{frame, pos[xyz], fwd(−Z), up(+Y), right(+X), fov}`.
  Output `camera_matrix.json` (gitignored), read downstream with `utf-8-sig`.
- **Ship render:** beauty pass to `render_spaceship/ship_####.exr` (OpenEXR,
  scene-linear, with alpha). Keep linear — no baked tone curve.
- **Ship Z-depth pass** *(recommended, not yet produced)*: a `Z` channel (or
  sidecar `shipz_####.exr`) for depth-correct occlusion. This is the asset that
  unblocks renderer items 3.1/3.2. **Unit caveat:** the black-hole Z is a
  Mino-affine path length, *not* metric distance and *not* Blender camera-space Z;
  document the mapping before any Z-based merge.

### Stage 2 — Taichi CUDA renderer (the focus of this repo)

`scripts/export_exr.py` is the production entry point. Per frame it: loads config
(UTF-8) + `camera_matrix.json` (`utf-8-sig`); `tr.setup_renderer(cfg)` does
`ti.init(arch=ti.cuda)`, loads the 16K starmap and uploads the f16 mip pyramid;
converts the Blender world camera basis into the local Boyer-Lindquist triad
(r̂, θ̂, φ̂) feeding the ZAMO tetrad (Formula 7); runs the split kernels; writes the
EXR.

- **Kernel split (Formula 10 v1.4 amendment):** `render_beauty_physics` traces the
  geodesic in the horizon-stable `[y, u, φ, t, v_y, v_u]` state (y = r − r₊,
  u = cosθ), accumulates Pipe B disk RGBA, and writes exit dir / outcome + a
  transmittance-weighted Z. `render_beauty_shade` reads the 4-neighbourhood exit
  directions, computes the screen-space Jacobian → LOD, samples the lensed sky, and
  composites it behind the disk.
- **Numerical stability (shipped):** factored Δ = y(y+2k) (Formula 11) kills
  catastrophic cancellation near the horizon; singularity-free Θ_u(u) (Formula 12)
  removes the 1/sin²θ pole (a `sin2_min` guard stays only on dφ/dλ, dt/dλ);
  Kahan-compensated RK4; adaptive Mino step; the spin-axis "static" seam fix (§6).
- **Optional features:** motion blur (`--motion-blur`, host-side averaging of N
  camera-rotated sub-frames over the shutter arc); 360°/VR
  (`render.projection_mode: equirect`; default stays `perspective` so the Doppler
  regression check remains valid).
- **Output:** a 4-channel EXR `render_blackhole/bh_####.exr` with named channels
  `R, G, B, Z` — RGB scene-linear HDR beauty (un-tonemapped, A4), Z
  transmittance-weighted Mino-affine depth (sentinel `depth_infinity = 1e5` where
  there is no disk emission).
- **Verification (the real guard):** `python scripts/gpu_test.py` → right/left
  Doppler asymmetry ≈ 7–8× (≈ 7.77× baseline) for the a = 0.999 edge-on camera;
  `oiiotool --info` to confirm channels.

### Stage 3 — Blender compositor: deep merge + optical FX *(recommended, A2)*

Build once as a compositor template `.blend`; keep the working space scene-linear.
Recommended node order:

1. **Image inputs** — two EXR-sequence nodes read as linear/non-color; pull `Z`.
2. **Depth-correct merge** — combine ship over black hole by Z. *Current limit
   (A5):* BH Z is Mino-affine and the ship Z may not exist yet → **composite by
   layer order** (ship `Alpha Over` BH) until the unit mapping is established. Do
   not Z-compare across mismatched units.
3. **Bloom / glow** — `Glare` (Fog Glow/Bloom) on the combined HDR-linear image;
   threshold above the star field so only the ring and beamed disk limb bloom.
4. **Chromatic aberration** — radial per-channel offset (Lens Distortion
   *Dispersion*).
5. **Lens distortion** — small barrel/pincushion + the dispersion, after the merge.
6. **Post** — gentle vignette / grain / exposure trim (heavy color → Stage 4).
7. **Output** — finishing master as **linear EXR** (most headroom) or a log/wide-
   gamut intermediate; avoid baking a display tone curve here.

### Stage 4 — DaVinci Resolve: grade + delivery *(recommended, A3)*

Conform the Stage-3 sequence under a managed pipeline (Resolve Color Management or
ACES); tag inputs by their true space. Grade order: normalize/tone-map the linear
HDR → primaries/balance → secondaries (warm the receding limb, push the beamed
approaching limb, control the photon-ring glow) → creative look. Finish (NR,
sharpening, vignette, grain), re-check bloom interaction, deliver to the target
codec/space at 24 fps (A1).

### End-to-end data flow

```
Blender scene ─┬─▶ export_camera.py ─▶ camera_matrix.json ──────────────┐
               │                                                         │
               └─▶ ship render ─▶ render_spaceship/ship_####.exr ──┐     │
                   (+ optional ship Z)                             │     │
                                                                   │     ▼
                            configs/render.yaml ─▶ export_exr.py ◀──┴── (camera track)
                            skills/.../SKILL.md ─▶ (Taichi CUDA: Pipe A + Pipe B)
                                                          │
                                                          ▼
                                       render_blackhole/bh_####.exr  (RGBA-linear + Z)
                                                          │
                       ship EXR ─────────────────────────┤
                                                          ▼
                              Blender compositor (Stage 3): deep merge → bloom →
                              chromatic aberration → lens distortion → post
                                                          │
                                                          ▼
                              DaVinci Resolve (Stage 4): conform → grade →
                              finishing → delivery → final graded clip
```

### What's real vs. recommended

| Stage | In repo? | Entry point / note |
|-------|----------|--------------------|
| 1. Camera export | ✅ Yes | `src/blender/export_camera.py` (run in Blender) |
| 1. Ship beauty render | ⚠️ Process exists; assets gitignored | → `render_spaceship/` |
| 1. Ship **Z** pass | ❌ Not yet produced | unblocks 3.1/3.2 (unit caveat) |
| 2. Black-hole render | ✅ Yes | `scripts/export_exr.py` → `render_blackhole/bh_####.exr` |
| 2. Motion blur / 360° | ✅ Yes (opt-in) | `--motion-blur`; `projection_mode: equirect` |
| 3. Blender compositor | ❌ Recommended (A2) | build node graph above as a template `.blend` |
| 4. DaVinci Resolve grade | ❌ Recommended (A3) | conform → grade → deliver |

---

## 3. Codebase map

### Directory tree

```
Black/
├── configs/
│   └── render.yaml              ← single source of truth for all parameters
├── assets/
│   └── stars.npy                ← point-star catalog [N,5]=θ',φ',flux_rgb (ingest output; gitignored)
├── scripts/
│   ├── thumb.py                 ← CPU preview renderer (development / QA)
│   ├── gpu_test.py              ← FHD GPU beauty render smoke test
│   ├── ingest_stars.py          ← offline ingest: HYG/ATHYG csv (or BSC5) → point-star {θ',φ',flux_rgb}.npy (Formula 13 / §8 Layer A)
│   └── export_exr.py            ← Phase 5: multi-channel RGBAZ EXR writer (OpenImageIO)
├── skills/
│   └── kerr-physics/
│       └── SKILL.md             ← physics formula reference (mandatory, never re-derived)
├── src/
│   ├── blender/
│   │   ├── __init__.py
│   │   └── export_camera.py     ← Blender script: exports camera_matrix.json
│   └── renderer/
│       ├── __init__.py
│       ├── metric.py            ← CKS Kerr metric + exact inverse + analytic derivs (Formulas CKS-1..4)
│       ├── geodesic.py          ← CKS Hamiltonian RK4 null geodesic integrator (Formulas CKS-5/6/7)
│       ├── disk.py              ← Accretion disk gas physics (Formulas CKS-8/9 + F9 chroma)
│       ├── starmap.py           ← 16K HDRI loader, mip pyramid, CKS-10 celestial→UV (reused for the diffuse map)
│       └── taichi_renderer.py   ← GPU renderer: Pipe A + Pipe B + Formula-13 DNGR background, split kernels
├── tests/
│   ├── cuda_smoke_test.py       ← confirms CUDA backend JIT on RTX 5060
│   ├── test_geodesic.py         ← conservation law tests (E, Lz, Q, null norm)
│   ├── test_starmap.py          ← CKS-10 celestial-direction → UV mapping tests
│   ├── test_ingest_stars.py     ← point-star ingest transforms + HYG/BSC5 parsers (Layer A ingest)
│   ├── test_starfield_dngr.py   ← Formula-13 DNGR: μ→1 flat-space, cell-grid CSR, CUDA dngr smoke
│   ├── test_gpu_regression.py   ← automated GPU Doppler / NaN / disk-peak guard (CUDA-gated)
│   └── test_geodesic/
│       └── test_conserved_quantities_regression.csv   ← golden values
│
├── render_blackhole/            ← output EXR sequence (bh_####.exr) — gitignored
├── render_spaceship/            ← Blender ship EXR sequence — gitignored
├── star_image/                  ← 16K HDRI inputs (starmap + milkyway diffuse EXR, HYG csv) — gitignored
│
├── PROJECT.md                   ← this file (the single project reference)
├── CLAUDE.md                    ← project instructions and physics policy
├── AGENTS.md                    ← mirror of CLAUDE.md for the Codex/Agents harness
├── REFERENCE_dngr_paper.md      ← James et al. 2015 (DNGR / Interstellar) — academic source
├── .codex/config.toml           ← Codex MCP config (Context7)
├── pyproject.toml               ← Python deps + uv/pytest config
├── uv.lock                      ← locked dependency versions
└── .gitignore                   ← excludes large assets and render outputs
```

### File reference

**`configs/render.yaml`** — single source of truth for all numerical parameters
(see §4 for the field table). No file in `src/` or `scripts/` hardcodes physics or
render values.

**`skills/kerr-physics/SKILL.md`** — the physics formula reference (see §5). All GR
formulas are copied verbatim from here, referenced by number throughout the code.

**`src/renderer/metric.py`** — CKS Kerr metric (Formulas CKS-1..4): `kerr_radius`,
`null_vector_cks`, `metric_cks`, the **exact** closed-form `inverse_metric_cks`
(`l` is η-null, no numerical inverse), and `dmetric_inv_cks` (analytic spatial
derivatives for the geodesic force). Coordinate order `(t, x, y, z)`. Used by
`geodesic.py`, `disk.py`, `thumb.py`, `test_geodesic.py`. The GPU renderer inlines
the same math as `@ti.func` instead of importing this.

**`src/renderer/geodesic.py`** — CPU null-geodesic integrator (Formulas CKS-5/6/7):
RK4 on the **8-vector** `[t, x, y, z, p_t, p_x, p_y, p_z]` (covariant momenta),
with the adaptive affine step `h = dλ·max(floor, (r−r₊)/r)` and the CKS-6
capture/escape stops. Key: `photon_momentum_cks` (CKS-7 ZAMO + projected ray),
`make_null_initial_conditions`, `integrate_null_geodesic`, `_horizon_radius`, and
the conserved-quantity helpers `energy`, `axial_angular_momentum`, `null_norm`,
`carter_Q` (CKS→BL diagnostic). The legacy BL `radial_turning_point` /
Mino-`R(r)`/`Θ(θ)` API is **removed**. Used by `thumb.py`, `test_geodesic.py`.

**`src/renderer/disk.py`** — CPU accretion-disk gas physics; the CKS reference the
GPU `_disk_emit_cks` / `_gas_four_velocity_cks` `@ti.func` must match. Key:
`gas_four_velocity_cks` (CKS-8 rigid +z rotation; no frozen ISCO constants — the
plunging branch is never sampled since `r_inner = r_isco`), `g_factor` (CKS-9
Cartesian dot product, `p` already covariant — no Δ-divide bug possible),
`blackbody_rgb` (F9 chromaticity, no T⁴). Used by `thumb.py`.

**`src/renderer/starmap.py`** — host-side 16K equirect starmap: `load_equirect`,
`build_mip_pyramid` (box-filter, f16 levels), `Starmap.load/.sample` (trilinear
ground truth), and `celestial_to_uv` (Formula CKS-10: an escaped ray's Cartesian
direction → equirect UV; the BL `normalize_sphere_angles` punch-through fold is
**removed** — CKS is regular on the spin axis). Equirect convention `u = φ'/2π`
(col), `v = θ'/π` (row, north pole at v=0). Used by `taichi_renderer.py` (GPU
upload) and `test_starmap.py`.

**`src/renderer/taichi_renderer.py`** — the GPU renderer (~1084 lines). Ports the CPU
physics to Taichi and runs both pipes on CUDA in the horizon-stable
`[y, u, φ, t, v_y, v_u]` state. Backend locked to `ti.init(arch=ti.cuda)`.

- *Module fields:* `star_flat/off/w/h` (f16 mip pyramid + metadata); `pixels`
  (square `render_pipe_a` output); `frame_pixels` (non-square `render_beauty_*`
  output); `exit_buf`, `disk_buf`, `depth_pixels` (kernel-split hand-off). *DNGR
  (mode=dngr only):* `cat_theta/phi/flux` + `cell_start/cell_count` (Layer-A
  point-star CSR cell grid) and `mw_flat/off/w/h` (Layer-B diffuse Milky-Way mip
  pyramid); size-1 dummies in texture mode.
- *Physics `@ti.func`:* `_horizon_constants(a)` (host, F11 — derives k=√(1−a²),
  r₊=1+k); `_delta_y(y,k)` (F11); `_radial_potential_y` / `…_deriv_y` (F6/11);
  `_theta_potential` Θ_u / deriv (F6/12); `_deriv` (F6/12 ds/dλ); `_project` (F6);
  `_rk4_step` (F6, Kahan + project); `_zamo_init` (F7); `_gas_four_velocity` (F3/5,
  plunging branch uses factored Δ); `_blackbody_rgb` (F9); `_disk_emit` (F8/9,
  recovers p_r=v_y/Δ, p_θ=−v_u/√(1−u²)).
- *Starmap `@ti.func`:* `_texel`, `_sample_level` (bilinear, φ-wrap + θ-clamp),
  `_sample_trilinear`. *(The formerly-unreferenced `_normalize_sphere` helper was
  deleted — §7 C2, resolved 2026-06-04.)*
- *DNGR `@ti.func` (Formula 13):* `_wrap_pi`; `_mw_texel/_mw_sample_level/
  _mw_sample_trilinear` (Layer-B diffuse pyramid samplers, mirror the starmap
  trio); **`_dngr_shade`** — builds the screen-space 2×2 beam Jacobian from the
  +x/+y neighbour exit dirs, magnification `μ = dΩ_pixel/|det J·sinθ′|` (normalized
  by the analytic per-pixel solid angle ⇒ μ→1 flat; clamp `mag_clip`; μ=1 on
  non-ESCAPED/`J>j_fold`), anisotropic-EWA diffuse fetch along the ellipse major
  axis, and the point-star energy gather `Σ flux·μ·g⁴·exp(−d²/2σ²)` over the
  overlapping catalog cells (`d` = screen offset via `J⁻¹`).
- *Kernels:* `render_pipe_a` (Pipe A only, square; retains its offset ray as the
  dev LOD reference); **`render_beauty_physics`** (production K1 — traces the CKS
  8-vector, writes the CKS-10 exit direction `(cosθ′, φ′)` + outcome + weighted Z;
  no per-step φ-wrap needed — the exit direction is a genuine Cartesian unit
  vector); **`render_beauty_shade`** (production K2 — `mode=texture`: the legacy
  screen-space-Jacobian LOD + single starmap fetch, byte-for-byte unchanged;
  `mode=dngr`: `_dngr_shade` two-layer background. `_screen_jacobian_lod` saturates
  LOD to `_MAX_LOD` when J > `render.j_fold`).
- *Host:* `load_config` (UTF-8), `setup_renderer` (cuda init + starmap upload;
  `_setup_dngr` uploads the Layer-A cell grid via `_build_star_grid` + the Layer-B
  pyramid via `_pack_pyramid` when `starfield.mode=dngr`),
  `_alloc_output`/`_alloc_frame`, `render_pipe_a_image`, `_camera_basis`,
  **`render_beauty_frame`** (main entry — Blender world basis = CKS directly,
  re-orthonormalized, runs K1+K2, optional NaN-guarded Z), `render_beauty_frame_mb`
  (motion blur, masked depth averaging), `tonemap` (Reinhard + gamma). **Consumed
  by:** `gpu_test.py`, `export_exr.py`, `test_gpu_regression.py`.

Camera conversion in `render_beauty_frame`: world Cartesian **is** CKS, so the
Blender `pos/fwd/up/right` basis is used directly (Gram-Schmidt re-orthonormalized
`up` against `fwd`); per-pixel `n = normalize(fwd + sx·right + sy·up)` feeds the
CKS-7 ZAMO + projected-ray photon init. No BL spherical embedding / triad.

**`src/blender/export_camera.py`** — Blender Python (Phase 1). Writes
`camera_matrix.json` (one record/frame: `{frame, pos, fwd, up, right, fov}`). `fwd`
= world −Z; `fov = cam.angle` *(labeled "vertical FOV" — see §7 A1)*. Needs `bpy`.

**`scripts/thumb.py`** — CPU preview renderer (single-threaded NumPy; slow but
self-contained) — the CKS reference twin of `taichi_renderer`. `--disk` enables
`march_disk`; uses config `thumb.*` framing overrides. Flow: `render()` (places
the camera in CKS Cartesian, `camera_basis()`) → `pixel_direction()` →
`make_null_initial_conditions()` (CKS-7) → `integrate_null_geodesic()` (CKS-5/6) →
`march_disk()` (CKS-8/9) → `trace_pixel()` → tonemap.

**`scripts/gpu_test.py`** — FHD GPU beauty smoke test. Reads a frame from
`camera_matrix.json` (`utf-8-sig`), runs `render_beauty_frame` at 1920×1080,
reports the Doppler asymmetry (`right_lum/left_lum` ≈ 4.3× under CKS — the affine
emission measure reweights it down from the BL Mino value). `--no-disk`,
`--exposure` flags.

**`scripts/export_exr.py`** — Phase 5 production entry. Extracts beauty + depth via
`to_numpy()` and writes a 4-channel `(R,G,B,Z)` EXR via OpenImageIO to
`render_blackhole/bh_####.exr`. `_shutter_arc(frames, idx, shutter_fraction, fps)`
returns `Δφ·fps·shutter_fraction` (F2). `--motion-blur` opt-in.

**`tests/test_geodesic.py`** — CPU CKS geodesic conservation (E, Lz drift < 1e-4;
null condition `|H|` < 1e-6 along the integrated 8-vector) + a golden-CSV regression.

**`tests/test_starmap.py`** — CKS-10 celestial-direction → UV mapping unit tests
(`celestial_to_uv`).

**`tests/test_gpu_regression.py`** — automated GPU beauty regression (the pytest form
of the manual `gpu_test.py` check). Drives production `render_beauty_frame` (frame
0, FHD, disk on); asserts NaN==0, right/left Doppler ∈ [7.0, 8.5] (right brighter),
disk peak ≈ `_DISK_MAX_REF = 12.7707` ±5%. `pytest.mark.gpu`; **skips cleanly**
without CUDA (Taichi init deferred into a module-scoped fixture).

**`tests/cuda_smoke_test.py`** — confirms the Taichi CUDA backend JITs on this
machine; fails explicitly if `arch` is not `Arch.cuda`.

**`pyproject.toml`** — `taichi==1.7.4` pinned (RTX 5060 / sm_120 / Blackwell);
`openimageio`; `[tool.uv] package = false`; `[tool.pytest] pythonpath = ["src"]`.

**`.gitignore`** — excludes `star_image/`, `render_blackhole/`, `render_spaceship/`,
`camera_matrix.json`, `*.exr`, `scripts/*.png`, `__pycache__/`.

### Data flow

```
configs/render.yaml ──▶ taichi_renderer.py · thumb.py · gpu_test.py · test_geodesic.py
skills/.../SKILL.md  ──▶ metric.py(CKS-1..4) · geodesic.py(CKS-5/6/7) · disk.py(CKS-8/9) ·
                          starmap.py(CKS-10 UV) · taichi_renderer.py(all, GPU port)
metric.py    ──▶ geodesic.py · disk.py · thumb.py · test_geodesic.py
geodesic.py  ──▶ thumb.py · test_geodesic.py
disk.py      ──▶ thumb.py(march_disk)
starmap.py   ──▶ taichi_renderer.py(setup_renderer upload) · test_starmap.py
taichi_renderer.py ──▶ gpu_test.py · export_exr.py · test_gpu_regression.py
export_camera.py(in Blender) ──▶ camera_matrix.json ──▶ gpu_test.py · export_exr.py
```

### Key invariants

| Invariant | Where enforced |
|-----------|---------------|
| GPU backend = `ti.cuda`, never `ti.gpu` | `taichi_renderer.py`, `cuda_smoke_test.py`, `CLAUDE.md` |
| All formulas from `SKILL.md`, no re-derivation | `CLAUDE.md` physics policy |
| All parameters from `configs/render.yaml` | all source; no physics literals |
| CKS coords `(t,x,y,z)`, 8-vector `[t,x,y,z,p_t,p_x,p_y,p_z]` | `geodesic.py`, `taichi_renderer.py` |
| Exact inverse `g^αβ = η − f l^α l^β` (no matrix inverse) | `metric.py:inverse_metric_cks` |
| `g`-factor is a Cartesian dot product (`p` already covariant — no Δ-divide) | `disk.py:g_factor`, `_disk_emit_cks` |
| `blackbody_rgb` chromaticity-only | `disk.py`, `_blackbody_rgb` |
| g⁴ beaming correct, not double-counted | `disk.py`, `_disk_emit_cks` |
| Escaped-ray celestial dir = normalized contravariant `(p^x,p^y,p^z)` (CKS-10) | `starmap.py:celestial_to_uv`, `_exit_cos_phi` |
| Camera file encoding = utf-8-sig | `gpu_test.py` |
| Config files read with utf-8 | `taichi_renderer.py`, `thumb.py`, `test_geodesic.py` |

---

## 4. Configuration reference (`render.yaml`)

| Section | Key fields |
|---------|-----------|
| `black_hole` | `spin` (a=0.999), `r_isco` (1.182 M), `r_plus` (1.0447 M — true outer horizon r₊=1+√(1−a²); now documentation-only, both the renderer and `thumb.py` derive r₊ from `spin` via `_horizon_radius`, CKS-6) |
| `render` | `width`/`height` (4K), `thumb_width/height` (256), `max_steps_pipe_a` (800 — Pipe B shares this same trace loop / step cap; raised from 250 because the CKS affine λ advances ~1 coord-unit/step vs BL Mino ~r²/step), `d_lambda_pipe_a` (0.25 — CKS affine step; far-field h≈dλ, shrunk near the horizon), `r_max` (50 M), `device_memory_gb` (6), `horizon_epsilon` (0.05 — CKS-6 capture margin, cost bound only), `adaptive_step_floor` (0.02), `j_fold` (0.15 — background LOD fold-saturation; under CKS this only guards the equirect texture poles, the BH spin-axis seam is gone), `fps` (24.0 — shutter arc = Δφ·fps·shutter_fraction), `projection_mode` (perspective\|equirect), `depth_infinity` (1e5 no-disk Z sentinel). *(The BL `sin2_min` 1/sin²θ polar guard was removed — CKS has no spin-axis coordinate singularity.)* |
| `disk` | `r_inner`, `r_outer`, `theta_half_width`, `T_0`, `emission_coeff`, `absorption_coeff`, `vertical_sigma_frac` (the bbox `|u|` early-out bound is now **derived** as `sin(theta_half_width)` in code — the old `bounding_sin_theta_half` literal was removed, §7 S2) |
| `starmap` | `path` (relative to repo root), `width` (16384 — used to compute LOD) |
| `starfield` | **DNGR background (Formula 13 / §8).** `mode` (`texture`\|`dngr`; texture default keeps the legacy F10 path + golden frames). *Ingest:* `format` (auto\|hyg\|bsc5), `source_catalog` (HYG/ATHYG csv or `bsc5.dat`), `catalog_path` (`assets/stars.npy`), `mag_limit` (6.5), `mag_zero_point` (0.0). *Layer A (point stars):* `star_grid_cols/rows` (candidate cell grid), `star_cell_radius`, `star_psf_px` (PSF σ), `psf_trunc_sigma`, `mag_clip` (μ cap), `caustic_delta_min` (δ⁻ floor), `g_beaming` (g⁴ hook, default off). *Layer B (diffuse):* `diffuse_map` (Milky-Way EXR), `diffuse_width`, `ewa_max_taps`, `jacobian` (`finite_diff`). The Layer-A/B fields load only in `mode=dngr` |
| `camera` | `default_radius` (6.03 M), `default_fov_deg` (90°), `shutter_fraction` (1/48 s — `export_exr._shutter_arc` reads this with `render.fps` as `arc = Δφ·fps·shutter_fraction`) |
| `thumb` | preview-only framing overrides (camera radius/fov/theta, background, glow, exposure, gamma) |
| `output` | directory names and filename prefixes for the EXR sequences |

---

## 5. Physics formula index (`SKILL.md`)

**Active path = PART II (Cartesian Kerr-Schild).** As of the 2026-06 CKS migration
the renderer, CPU core, disk, and starmap UV all use Formulas **CKS-1…CKS-10**
(SKILL.md PART II). The PART I BL formulas **1/6/7/11/12** are *superseded for the
renderer* (kept for history / the retired BL reference); **2/3/4** are reused
unchanged (BL-radius quantities — CKS `r` is the BL radius); **8/9/10/13** are
reused, acting on coordinate-agnostic quantities (the CKS-9 g-factor / CKS-10
celestial direction). CKS PART II index:

| Formula | Content |
|---------|---------|
| CKS-1 | Implicit Kerr radius r(x,y,z) (= BL radial coord; explicit positive root) |
| CKS-2 | Cartesian Kerr-Schild metric g_αβ = η_αβ + f·l_α l_β (regular on axis + horizon) |
| CKS-3 | **Exact** inverse g^αβ = η^αβ − f·l^α l^β (l is η-null; no matrix inverse) |
| CKS-4 | Analytic coordinate derivatives ∂r, ∂f, ∂l_α (geodesic force term) |
| CKS-5 | Hamiltonian null-geodesic EOM; 8-vector RK4; E=−p_t, L_z=x p_y−y p_x conserved |
| CKS-6 | Horizon capture (r ≤ r₊+ε_h) / escape (ρ ≥ r_max) |
| CKS-7 | Photon init: ZAMO observer (from g^αβ) + g-orthogonal projected ray |
| CKS-8 | Equatorial disk gas 4-velocity: rigid +z rotation at Ω (no BL→KS Jacobian) |
| CKS-9 | g-factor = −E/(p_t u^t + p_x u^x + p_y u^y + p_z u^z) (Cartesian dot product) |
| CKS-10 | Escaped-ray celestial dir = normalized contravariant (p^x,p^y,p^z) → (θ′,φ′) |

PART I (retired/reused) formula index:

| Formula | Content |
|---------|---------|
| 1 | Kerr metric g_{μν} in Boyer-Lindquist coordinates |
| 3 | Circular-orbit 4-velocity u^μ for r ≥ r_isco (Bardeen 1970) |
| 4 | ISCO conserved quantities E_I, L_I (Cunningham 1975), frozen at r_isco |
| 5 | Plunging-region 4-velocity with frozen E_I, L_I; u^r must be negative |
| 6 | Mino-time RK4 null geodesic; radial potential R(r) and angular potential Θ(θ) |
| 7 | ZAMO tetrad photon momentum init; exact A = (r²+a²)² − a²Δsin²θ |
| 8 | g-factor = −1/(p_t·u^t + p_r·u^r + p_θ·u^θ + p_φ·u^φ); p_r covariant |
| 9 | g⁴ volumetric beaming (3D emitter); `blackbody_rgb` chromaticity-only (no T⁴) |
| 10 | Differential-ray mip LOD: J = √(δθ² + sin²θ·δφ²); L = log₂(W·J/2π). **v1.4 amendment:** J may be estimated in screen space from the 4-neighbourhood exit directions (kernel-split LOD) instead of an offset ray |
| 11 | FP32-stable factored discriminant: Δ = (r−r₊)(r−r₋) = y(y+2k), y=r−r₊, k=√(1−a²) |
| 12 | Singularity-free polar potential under u=cosθ: Θ_u(u) = (1−u²)(Q+a²E²u²) − L_z²u²; the 1/sin²θ pole cancels (dφ/dλ, dt/dλ keep a `sin2_min` guard) |
| 13 | Hybrid DNGR (rev v1.6): screen-space 2×2 ray-bundle Jacobian J → point-star magnification μ = \|det J₀·sinθ′₀\|/\|det J·sinθ′\| → energy-conserving flux `I = I_base·μ·g⁴` with truncated-Gaussian PSF. Point stars **brighten**, don't smear. **3 guards owner-approved 2026-06-05** and the two-layer render path **shipped** (§8 Phases 2–5): `taichi_renderer._dngr_shade` + `starfield.mode: texture\|dngr` |

Decisions of record (from `CLAUDE.md`): Decision A = ZAMO tetrad (F7); Decision B =
simple temperature model T = T₀·(6/r)^0.75.

---

## 6. Implementation status — shipped

**CKS migration (2026-06, headline).** The entire renderer was refactored from
Boyer-Lindquist to **Cartesian Kerr-Schild** to eliminate the gray polar-axis
artifact at its source (the BL 1/sin²θ spin-axis pole + Δ→0 horizon singularity).
SKILL.md PART II (CKS-1…10) is the contract; `metric.py`/`geodesic.py` (8-vector
RK4), `taichi_renderer.py` (GPU), `disk.py`, `starmap.py`, `thumb.py`, and
`configs/render.yaml` are all on CKS. The BL band-aids — `sin2_min`, `Θ_u`,
per-step φ-wrap, `j_fold` *meridian* collapse, `normalize_sphere_angles`
punch-through, and the four `scripts/seam_*`/`_uv_sweep`/`test_pipe_a` diagnostics
— are **deleted**. *Validated:* `test_geodesic` green (E conserved ~1e-8, |H|<1e-6);
GPU `--no-disk` shows a clean shadow with **no polar line**; disk render shows the
photon ring + g⁴-beamed approaching edge. The CKS affine-vs-Mino emission measure
reweights the Doppler half-frame ratio to ≈4.3× (was 7.77× under BL) and required
recalibrating `emission_coeff`/`absorption_coeff` ~1/30 (see §4 + render.yaml).

The full 5-phase optimization from `guid.md` (the now-superseded source spec) is
**complete** and committed. Condensed history (the Phase-1 `[y,u,…]` / Θ_u / φ-wrap
work below predates and is **superseded by** the CKS migration above):

- **Gates approved (2026-06-02):** SKILL.md Formula 11/12 + the Formula 10 amendment
  landed (rev v1.4); polar guard kept on dφ/dt only.
- **Phase 1 — FP32 stability (1.1–1.4):** horizon constants in Python; `_delta_y`
  (F11); `[y,u,…]` state transform with Θ_u (F12) and the `v_r=Δ·p_r → v_y` /
  `p_θ=−v_u/√(1−u²)` migration; Kahan summation. *Verified:* pytest 5/5; Doppler
  7.78×; disk max 14.11 exact.
- **Phase 2 — perf (2.1, 2.2, 2.4):** offset ray removed; adaptive Mino step; beauty
  kernel split into `render_beauty_physics` + `render_beauty_shade` with the
  screen-space-Jacobian LOD (F10 amendment). *Verified:* pytest 12/12; Doppler
  7.77×; ~40% faster (1.8s→1.1s FHD). **2.3 (`ti.Texture`) deferred** — Taichi 1.7.4
  exposes no mip-upload API; the correct manual f16 pyramid is retained.
- **Phase 3 (3.3, 3.4):** disk bbox early-out in u-space; transmittance-weighted
  Mino-affine Z → `depth_pixels` (+∞ sentinel). **3.1/3.2 (ship occlusion) blocked**
  on a Blender ship Z-depth asset that does not exist yet.
- **Phase 4 (4.1, 4.2):** equirect 360° ray-gen behind `projection_mode` (default
  perspective); motion blur as host-side averaging of camera-rotated sub-frames.
- **Phase 5 (5.1, 5.2):** `export_exr.py` writes the 4-channel `(R,G,B,Z)` EXR via
  OpenImageIO.
- **Review-fix commit `c45d24b`:** motion-blur Z corruption (masked averaging);
  `r_plus` mislabel `0.0447→1.0447`; depth NaN guard; split-brain Δ in the plunging
  branch. *Verified:* pytest 12/12; Doppler 7.77×.
- **F3 — GPU regression harness (committed):** `tests/test_gpu_regression.py`
  (CUDA-gated, skips without a GPU).
- **Center "static" seam fix + F2 shutter (committed `cab7cbb`):** per-step φ-wrap
  into (−π,π], shortest-arc escape interpolation, `_screen_jacobian_lod`
  fold-saturation when `J > render.j_fold` (new `j_fold: 0.15`); F2 implemented via
  **Option A** — `render.fps` + `arc = Δφ·fps·shutter_fraction` (byte-identical to
  the legacy `Δφ·0.5` at 24 fps). *Verified:* pytest green incl. F3; Doppler 7.77×;
  the static band renders as a smooth faint line; j_fold saturates ~1.2% of escaped
  pixels.
- **Code-review cleanup (2026-06-04):** landed the trivial/safe review findings —
  **A3** (host `acos(z/r_cam)` clamped to [−1,1], NaN-safe at the poles), **C2**
  (dead `_normalize_sphere` deleted), **C1** (seam diagnostics moved out to
  `scripts/seam_diagnostics.py`; production module ~1358→1084 lines), and **F4**
  (dead `render.max_steps_pipe_b` key removed; the shared-cap note now lives on
  `max_steps_pipe_a`). No physics or render output changed. Remaining open findings:
  F5, A1, A4 (see §7).
- **Formula 13 — Hybrid DNGR (SKILL.md, 2026-06-04→05):** the screen-space ray-bundle
  Jacobian / point-star magnification / truncated-Gaussian-PSF formulation was
  verified against `REFERENCE_dngr_paper.md` and merged into `SKILL.md`. The **three
  guards were owner-approved 2026-06-05** (rev v1.6): (a) μ normalized by the FD
  undeflected-reference footprint so μ→1 flat; (b) boundary clamp μ=1 on
  non-ESCAPED / `J>j_fold`, plus `δ⁻<caustic_delta_min ⇒ μ=min(μ,mag_clip)`;
  (c) volumetric g⁴ as a `starfield.g_beaming` hook (default g≡1).
- **DNGR background render path — §8 Phases 2–5 (2026-06-05):** the baked-texture star
  field is replaced (behind `starfield.mode: dngr` — promoted to default 2026-06-06,
  §7 S6; `texture` still reproduces the golden frames) by the two-layer Formula-13 background in
  `taichi_renderer._dngr_shade`. **Layer A** point stars: a HYG/ATHYG (or BSC5)
  catalog binned into an equirect CSR cell grid is gathered per escaped pixel and
  brightened by the screen-space-Jacobian magnification `μ = dΩ_pixel/|det J·sinθ′|`
  (`flux·μ·g⁴·Gaussian-PSF`; stars stay sharp). **Layer B** diffuse: the
  `milkyway_2020_16k.exr` band is fetched with an anisotropic (EWA) filter along the
  beam-ellipse major axis. New `starfield` config (mode/grid/PSF/mag/EWA keys);
  `tests/test_starfield_dngr.py` pins μ→1 flat-space (guard a), the cell-grid CSR, and
  a CUDA dngr smoke render. *Verified (RTX 5060):* full suite 41 passed; texture-path
  GPU regression unchanged (Doppler 7.77×, disk peak 12.77); dngr frame NaN-free with
  sharp point stars (peak > 20× diffuse median); flat-space μ mean 0.99973 (σ 0.2%).
- **Point-star ingest — §8 Phase 1 (2026-06-04, retargeted 2026-06-05):**
  `scripts/ingest_stars.py` ingests a bright-star catalogue into `assets/stars.npy` —
  `float32 [N,5] = (θ′, φ′, flux_r, flux_g, flux_b)` in the integrator's BL exit frame
  (`θ′=π/2−Dec`, `φ′=RA`, matching the equirect `u=φ′/2π`, `v=θ′/π` lookup). Flux is
  built as energy, not a texture sample: Pogson `10^(−0.4·Vmag)` brightness ×
  `blackbody_rgb` chromaticity (Formula 9 helper **reused** from `disk.py`) with
  B−V→T via Ballesteros (2012). **Now reads the HYG/ATHYG v3.2 csv** the owner
  supplied (`ra`/`dec`/`mag`/`ci`; Sun skipped) as well as the original V/50
  `bsc5.dat`, dispatched by `starfield.format` (auto = by extension). Unit tests in
  `tests/test_ingest_stars.py` (25; both parsers, no catalogue file needed). The HYG
  ingest yielded 8 877 stars at V ≤ 6.5 (raised to V ≤ 8.0 → 41 410 stars when `dngr`
  became the default 2026-06-06, to match the diffuse map's bright-star cutoff; §7 S6).
- **Codebase modernization + README rewrite (2026-06-07):** non-behavioral tooling
  and docs pass. Added a `ruff` lint/format config (`[tool.ruff]`, line-length 100,
  rules `E/F/I/W/UP/B/C4/SIM`; `E741` ignored — `l` is the Kerr-Schild null covector;
  `extend-exclude` covers `Open_Source_Repository/` and the frozen `metric.py`/
  `geodesic.py`/`disk.py`) and a `[dependency-groups] dev = ["ruff>=0.6"]` group;
  filled in `[project]` metadata (`readme`, `license`, classifiers). Applied `ruff`
  fixes + format and added type hints to the pure-Python (non-Taichi, non-frozen)
  functions in `scripts/thumb.py`, `export_exr.py`, `gpu_test.py`, `ingest_stars.py`.
  Rewrote `README.md` for the CKS migration (BL→Cartesian Kerr-Schild header,
  CKS-1..10 physics table, `dngr` as the default background). **No physics, numeric
  values, or render output changed; frozen files untouched.** *Verified:* `ruff check`
  / `ruff format --check` clean; pytest 40 passed (GPU deselected).

*Note — `render_pipe_a`* (the 256² dev LOD kernel for `_gate2_lod_test`) was
migrated to `[y,u,…]` but **intentionally keeps its offset ray** as the offset-ray
LOD reference; it is not on the 4K production path.

---

## 7. Remaining work & known issues

Physics policy unchanged: any formula touch must cite a `SKILL.md` number — never
re-derive. **`disk.py`, `geodesic.py`/`metric.py` CPU references, and any
`render.yaml` physics value are out of scope** (frozen). GPU backend stays
`ti.cuda`.

### Backlog (tracked items)

| ID | Item | State | Class |
|----|------|-------|-------|
| **F4** | `render.max_steps_pipe_b` declared but read by no kernel (Pipe B shares the Pipe A loop) | ✅ **Resolved (2026-06-04)** — dead key removed from `render.yaml`; `max_steps_pipe_a`'s comment now records that Pipe B shares the same trace loop / step cap | Config |
| **F5** | Docstring / cross-ref drift after the `[y,u,…]` migration (GPU side only; the CPU `[r,θ,…]` docstring in `geodesic.py` is correct, leave it) | Open — comments-only pass over `taichi_renderer.py`; copy from §3 of this file. Do **not** let it drift into a `disk.py`/physics edit | Docs |
| **3.1/3.2** | Ship depth occlusion (early ray termination vs. Blender ship Z) | **Blocked** on a Blender ship Z-depth EXR asset. Sequence when unblocked: produce the asset → derive & document the Mino-affine ↔ camera-Z mapping → wire `ti.Texture(r32f)` + early-out behind an off-by-default flag → validate on a synthetic plane. **Biggest correctness trap:** `ray_length` is Mino-affine, not metric/Blender-Z | Asset + code |
| **2.3** | Hardware `ti.Texture` starmap + `sample_lod` | **Deferred (external)** — Taichi 1.7.4 has no mip-upload API; revisit only after a Taichi upgrade is independently justified and re-validated on sm_120 (CLAUDE.md pins 1.7.4) | External |
| **T3** | Moving-camera observer model (camera peculiar velocity, not just ZAMO) | **Roadmap, gated** — needs a new `SKILL.md` tetrad-boost formula approved (human review) before any code; high risk if rushed (sign/normalization) | Physics (gated) |

### Code-review findings (verified against current code)

From a comprehensive review. A3, C1 and C2 are **resolved** (2026-06-04); the rest
are documented, not yet applied — confirmed present on inspection:

- **A1 — camera FOV axis label.** `export_camera.py:33` writes `"fov": cam.angle`
  commented "vertical FOV in radians," but Blender's `cam.angle` is the
  **larger-dimension** FOV (horizontal for a landscape sensor). If the renderer
  treats it as vertical, the framing/scale is off. **Verify** against Blender's
  sensor-fit before changing; if confirmed, derive the vertical FOV explicitly.
- **A3 — host `acos` domain.** ✅ **RESOLVED (2026-06-04).**
  `taichi_renderer.py:941` now reads
  `theta_cam = math.acos(min(1.0, max(-1.0, z / r_cam)))` — the input is clamped to
  [−1, 1], so fp rounding at the poles can no longer yield NaN. (Matches the
  in-kernel `acos(clamp(...))` pattern used everywhere else.)
- **A4 — automated seam regression.** ✅ **RESOLVED (2026-06-06).**
  `test_gpu_regression.py::test_no_spin_axis_seam` is now a live PASS guard pinning the
  spin-axis-meridian smoothness (center-column jump ≤ 6× the off-seam median; measured
  2.06× on the dngr R2 path, ~1.9× on legacy texture) — a future LOD/placement change
  that reintroduces the static band trips it. The dedicated location-agnostic stripe
  detector (`test_starfield_artifacts.py`) complements it (currently `xfail` on a
  bright-star confound; see §7 S4/S7).
- **C1 — diagnostics in the production module.** ✅ **RESOLVED (2026-06-04).**
  `render_starmap_raw`, `render_fixed_lod`, `dump_phi_exit` (and the `__main__`
  block) were extracted from `taichi_renderer.py` to `scripts/seam_diagnostics.py`,
  which imports the production `@ti.func` helpers by name (no physics re-implemented).
  The production module is now ~1084 lines.
- **C2 — dead code.** ✅ **RESOLVED (2026-06-04).** `_normalize_sphere` has been
  deleted from `taichi_renderer.py` (no call sites remained — the `[y,u,…]`
  migration made `acos(clamp(u))` the exit path).
- **S1 — DNGR polar "rope" = guard-(b) deviation.** ✅ **RESOLVED (2026-06-05).**
  `_dngr_shade` gated the Layer-A star gather on `usable` (= `valid and δ⁺≤j_fold`),
  so along the spin-axis seam stars were *suppressed entirely*, collapsing the
  meridian to a flat, star-less band. SKILL.md Formula 13 guard (b) specifies
  μ=1 (no brightening) there but the stars are **still drawn**. Fixed by gating the
  gather on `valid` only (`mu` is already 1.0 when not `usable`) — code now matches
  the approved formula; no formula change. Texture-mode (`mode: texture`) is
  unaffected (different shade branch) — its flat band is the inherent baked-texture
  limitation §8 was built to remove (use `mode: dngr`).
- **S2 — config-sync: derived literals.** ✅ **RESOLVED (2026-06-05).**
  `disk.bounding_sin_theta_half` (= sin(θ_half)) is now **derived** in
  `render_beauty_frame` from `disk.theta_half_width`; the duplicated literal
  (which desynced when θ_half was edited) was removed from `render.yaml`.
  `black_hole.r_plus`/`r_isco` are likewise derived-value duplicates of `spin`;
  the GPU already derives `r_plus` (`_horizon_radius`) — its stale docstring was
  corrected. ✅ **RESOLVED by the CKS migration (2026-06):** the CPU preview
  (`thumb.py`) now also derives `r₊` from `spin` via `_horizon_radius` (the BL
  `radial_turning_point` r_floor it used is gone), so `r_plus` is documentation-only
  and the desync hazard is removed.
- **S3 — GPU kernel consistency.** ✅ **RESOLVED (2026-06-05).** The per-step φ-wrap
  and `_J_FOLD` fold-saturation that the production beauty kernels carry were
  propagated into `render_pipe_a` (it had neither, so its raw exit-φ collapsed to
  noise near the pole). All production GPU kernels now share identical polar/φ
  handling. (Note: `dump_phi_exit` in `scripts/seam_diagnostics.py` intentionally
  keeps raw φ — that is its branch-cut probe — and is left unchanged.)
- **S4 — DNGR seam star *pileup* (the S1 trade-off, now visible).** 🚩 **FLAGGED
  (2026-06-06) — needs an owner decision + SKILL.md extension before code.** The
  S1 fix (gate the Layer-A gather on `valid`, not `usable`) cured the star-less
  "rope" but introduced the opposite artifact: on the spin-axis seam the neighbour
  pixels straddle the pole, so `Δφ′≈±π` ⇒ `detJ` is large ⇒ `inv_det` is tiny ⇒
  every polar-cell catalog star projects to `d≈0` (`dpx,dpy = J⁻¹·(Δθ′,Δφ′)`) and
  is splatted at near-full weight `exp(0)·μ` (μ=1 there). Measured: the center
  image column is a **100th-percentile** luminance outlier, ~2× the brightest
  off-center 8-col window; `test_gpu_regression.py::test_no_spin_axis_seam` fails
  15.4× (limit 6) under `mode: dngr` (passes under `mode: texture`). Formula 13
  guard (b) says `detJ` is *invalid* on the seam but does **not** specify the
  Layer-A splat geometry there — so the fix is gated on extending the skill.
  Candidates: (R1) suppress the gather on `not usable`; (R2, recommended) place
  seam stars with the undeflected analytic footprint instead of the degenerate
  `J⁻¹`; (R3) energy-clamp the per-pixel Layer-A contribution. See
  `docs/specs/2026-06-06-dngr-artifact-remediation.md`. **UPDATE (2026-06-06,
  later):** disk-off visual verification showed the seam is **shared by both
  modes** — `texture` renders a *blocky coarse-mip vertical stripe* (the `j_fold`
  collapse, i.e. the cure is itself the artifact), `dngr` the star-pileup rope.
  So this is the one real *shared* background bug, not a `dngr`-only trade-off; the
  fix touches both Formula-10 LOD and Formula-13 guard (b). See spec §6.
  **UPDATE (2026-06-06, pass 2):** owner chose **R2** (place seam stars by the
  undeflected analytic footprint, not the degenerate `J⁻¹`). A code-level sub-agent
  review confirmed *one* shared root cause (both modes read the identical `exit_buf`
  neighbour stencil and blow up on the same `Δφ′≈±π` trigger; `J` and `detJ` are the
  scalar vs 2×2 form of one Jacobian) with **three** code touch-points. Still gated:
  the *proposed* SKILL.md guard-(b′) amendment is drafted in spec §7.2 but **not
  applied**, and **no renderer code is written** — owner must approve the guard text
  first. Under the new `dngr` default, `test_no_spin_axis_seam` now also catches this
  seam (≈15×) and is marked `xfail(strict=True)` alongside the dedicated stripe test.
  **UPDATE (2026-06-06, pass 3) — ✅ RESOLVED (dngr).** Owner approved the guard-(b′)
  amendment; it landed (SKILL.md **v1.7**) and the `_dngr_shade` splat now uses the
  undeflected proper-separation footprint `d²=(Δθ′²+sin²θ′·Δφ′²)/dΩ` on the invalid-
  `detJ` branch (the gather also runs for every escaped pixel, removing the old star-
  free band around the shadow). Validated: coarse center-column seam **15×→2.06×**
  (cf. legacy texture ~1.9×), so `test_no_spin_axis_seam`'s `xfail` was **removed → live
  PASS guard**; the field is seam-free (masking the brightest star drops the dedicated
  stripe-z 28→14, the clean range). The dedicated `test_background_has_no_vertical_seam_stripe`
  **stays `xfail`** — its raw z≈28 is a *bright-lensed-star confound* in frame-0's ~80-row
  sky band (A/B-proven: pre-R2 `J⁻¹` peaks at the same star/col z≈27), not a residual
  seam; its bright-point recalibration is deferred. The **texture** Formula-10 LOD
  blocky stripe (touch-points 2–3) is also **deferred** (separate Formula-10 change,
  non-default, needs its own test). See spec §7.3.
- **S6 — Background star-smear under `mode: texture` (the dominant "dark noise").**
  ✅ **RESOLVED (2026-06-06) — `dngr` promoted to default.** Formula-10 uses an
  *isotropic scalar* mip-LOD; near the photon ring the lensed footprint is strongly
  *anisotropic*, so a single mip level smears background stars into tangential
  streaks across the whole field. Controlled A/B (same frame, disk off) confirmed
  `dngr`'s anisotropic EWA (Layer B) + point gather (Layer A) **eliminates** it
  (smear coherence 0.50→0.257). Owner chose to promote `dngr`:
  `render.yaml starfield.mode: texture → dngr`. A sub-agent review caught a coverage
  gap — Layer A was capped at `mag_limit 6.5` while the Layer-B diffuse map omits
  stars *brighter* than ~8.0, so 6.5–8.0 stars fell out of both layers — fixed by
  `mag_limit: 6.5 → 8.0` and re-ingesting (`assets/stars.npy`: 8 877 → 41 410 stars).
  The smear test is now a *live* PASS guard (xfail dropped). See spec §7.1.
- **S7 — Coarse GPU-regression metrics are insensitive to the visible artifacts.**
  🚩 **FLAGGED (2026-06-06).** `test_no_spin_axis_seam` passed `texture` at 1.9×
  (limit 6) *while the blocky seam stripe is plainly visible* — a low-frequency
  stripe has small local jumps and evades a max-jump/median metric. Mitigation
  landed: `tests/test_starfield_artifacts.py` adds two *visual* artifact detectors
  (structure-tensor smear coherence < 0.36; location-agnostic vertical-stripe MAD-z
  < 20), pure image statistics (no GR formula), calibrated on real CUDA renders and
  marked `xfail(strict=True)` so they lock in today's broken state and turn the
  suite **red on xpass** when an artifact is fixed. **Validated 2026-06-06** against
  synthetic ground truth (5 seeds) for sensitivity *and* specificity: smear fires
  only on directional (not isotropic) blur; seam fires only on vertical (not
  horizontal) structure. The seam threshold was found mis-set — a clean field
  scores 9–14, so the original `< 12` sat inside the clean distribution; raised to
  **20** (empty gap between clean ceiling ~14 and the real seam 27.8). Cross-mode
  run confirmed staging and revealed the `dngr` seam is a *sharp* center pileup that
  the old coarse `test_no_spin_axis_seam` catches (15.4×) while `texture`'s blocky
  stripe evades it (1.9×) — neither mode is seam-free. See spec §6.2/§6.3.
  **UPDATE (2026-06-06, pass 2):** with `dngr` now the default, the staging settled:
  the smear test is a live PASS guard (xfail dropped) and the two seam detectors are
  the suite's `xfail(strict=True)` pair — the dedicated stripe test *and*
  `test_no_spin_axis_seam` (newly xfailed, since under `dngr` it catches the ≈15×
  pileup). Full suite: **41 passed, 2 xfailed.** An independent re-review flagged the
  seam detector's bounded limits (prominence- not presence-detector; orientation-
  locked to the current vertical-axis camera framing; pixel-absolute kernels assume
  1280×720) — sound to guide the fix, worth revisiting if the camera framing changes.
  **UPDATE (2026-06-06, pass 3):** the R2 fix (S4) landed and validated those bounded-
  limits flags exactly — the dedicated stripe-z is dominated by a single bright lensed
  star in the thin sky band, not a seam (masked field ≈14, clean). Owner chose to clear
  the coarse marker only: `test_no_spin_axis_seam` is now a **live PASS guard** (2.06×),
  and the dedicated stripe test remains the suite's lone `xfail` (documented confound;
  bright-point recalibration deferred). Full suite now: **42 passed, 1 xfailed.**
- **S5 — μ normalization uses analytic `dΩ`, skill says FD.** 🚩 **FLAGGED
  (2026-06-06).** SKILL.md Formula 13 §2(a) (approved) prescribes the *same FD
  estimator* for the flat-space footprint `detJ₀·sinθ′₀`; the code uses the
  closed-form analytic per-pixel `dΩ`. The host test shows the two agree to
  <0.3% (analytic = FD continuum limit). Doc/code-sync only: bless the analytic
  form in SKILL.md, or switch the code to FD. See the spec doc §2.3.

### Recommended order (when approved)

1. ~~**A3** acos clamp · **C2** delete dead `_normalize_sphere` · **C1** move
   diagnostics to `scripts/seam_diagnostics.py` · **F4** remove `max_steps_pipe_b`~~
   ✅ **done (2026-06-04).**
2. **F5** docstring refresh (zero risk, do while context is fresh).
3. **A1** confirm + fix the FOV axis (needs a Blender check first).
4. **A4** add the seam regression assertion.
5. **3.1/3.2** ship occlusion (blocked on the asset + unit mapping).
6. **2.3** `ti.Texture` (external; after a justified Taichi upgrade).
7. **T3** moving camera (gated on an approved SKILL formula).

```
A3 · C2 · C1 · F4 (done) ────────────────────────────────────
F3(done) ─┬─▶ F5 · A1 · A4
          └─▶ 3.1/3.2  (also needs: ship-Z asset + unit-mapping note)
2.3  ◀── (external) Taichi > 1.7.4 re-validated on sm_120
T3   ◀── (gated)   new SKILL.md tetrad-boost formula approved
```

---

## 8. DNGR background rearchitecture (SHIPPED — `starfield.mode: dngr`)

**Status: complete (2026-06-05; seam guard (b′) added 2026-06-06).** Phases 0–5 are
landed: Formula 13 is approved (rev v1.6, all three guards signed off 2026-06-05; rev
**v1.7** adds the seam splat-placement guard **(b′)**, approved 2026-06-06 — §7 S4) and
the two-layer render path is implemented in `taichi_renderer._dngr_shade`, gated by
`starfield.mode`
(`dngr` is the default since 2026-06-06 — see §7 S6 — and activates the rebuild;
`texture` remains selectable and reproduces the legacy F10 golden frames). It
replaces the baked-texture star field with the *Interstellar* DNGR
treatment — **point stars stay sharp; gravitational lensing changes their brightness,
not their size.** Every new lensing formula was routed through `SKILL.md` for human
approval **first** (Formula 13), exactly as Formula 10 was. The `j_fold` seam fix
remains the `texture`-mode stopgap (its blocky-stripe replacement by the (b′)
regularization is deferred — §7 S4); in `dngr` mode Layer A is a point gather (not a
trilinear fetch) and the spin-axis seam star-pileup is resolved by guard (b′)
(undeflected proper-separation splat placement, 2026-06-06). Only the optional Phase 6
(geodesic-deviation Jacobian) remains as a future fidelity upgrade. The subsections below are retained as the design
record.

### Why the texture approach fails

| Symptom | Root cause | Status |
|---|---|---|
| Stars smear into streaks under lensing | a baked texture has fixed angular resolution; magnification µ>1 stretches a few texels over many pixels | architectural — this plan |
| Stars dim where they should brighten | texture energy is mip-averaged radiance; magnification is not converted to brightness | architectural — this plan |
| Center "static" seam at the spin-axis meridian | BL φ folds by π across the meridian caustic; a scalar-LOD trilinear fetch lands on unrelated coarse texels | **mitigated now** by `j_fold`; the rebuild removes the cause entirely |

### Target model — two decoupled sky layers

- **Layer A — point-star catalog** (sharp, lensed-brightness): a star list
  `{(θ′, φ′), flux_rgb}` in BL celestial coords (the same exit `{θ′,φ′}` the
  integrator already produces). Source: Yale Bright Star / Hipparcos / Tycho-2 /
  Gaia subset; apparent magnitude → linear flux; B−V → RGB via blackbody (**reuse**
  `_blackbody_rgb`). Rendering is an **energy gather, not a texture fetch**: a star
  contributes total flux scaled by the pixel's ray-bundle **magnification**
  (µ>1 → brighter, demagnified → dimmer); its image stays a sub-pixel point.
  Multi-imaging falls out for free (each pixel is one image sheet).
- **Layer B — diffuse galaxy/nebula** (low-frequency): keep an equirect texture
  **only** for the smooth Milky-Way band; replace the isotropic scalar-LOD fetch
  with an **anisotropic (EWA-style)** filter driven by the ray-bundle ellipse
  `(µ, δ⁺, δ⁻)`.

### The one new piece of physics (gated)

We already compute the scalar footprint `J = max(Jx, Jy)` (F10). DNGR needs the full
2×2 beam Jacobian `Jac = ∂(θ′,φ′)/∂(x_pix,y_pix)`, from which: the ellipse axes
`δ⁺,δ⁻` and orientation are the singular values/vectors; the **magnification** is
`mag = Ω_pixel/Ω_beam = 1/|det Jac · sinθ′|` (Layer-A brightness); `δ⁻→0` marks a
caustic/critical curve (the principled replacement for the `j_fold` heuristic).

> **Physics-policy gate.** `mag = Ω_pixel/Ω_beam` and the ellipse extraction are new
> formulas → must be added to `SKILL.md` and approved before coding. `Jac` can be
> obtained two ways, decided at approval: (1) **finite-difference** the existing
> per-pixel exit map (cheap, reuses `exit_buf`; accuracy limited near caustics);
> (2) **geodesic deviation** integrated alongside the central ray (exact DNGR;
> needs the deviation ODE in SKILL.md). Recommendation: ship FD first, offer
> geodesic-deviation as a flagged fidelity upgrade.

### GPU architecture (fits the existing 2.4 split)

```
render_beauty_physics (unchanged): trace primary ray → exit_buf{θ′,φ′,out}, disk_buf, depth
        ├─ build beam Jacobian Jac(py,px) from exit_buf neighbours (FD, in-kernel)
render_beauty_shade (rewritten background half):
        ├─ Layer B: anisotropic EWA fetch of the diffuse map using (δ⁺,δ⁻,µ)
        └─ Layer A: gather stars in the beam ellipse, add flux·mag·PSF
        → frame = disk + transm·(diffuse + stars)
```

Star gather: bin the catalog into an equirect-cell (or HEALPix) grid uploaded as
Taichi fields; per pixel query the few overlapping cells. Bright-star catalogs are
~10⁴–10⁵ stars → candidate counts stay O(1–10) off the galactic plane. Gate the
whole path behind `starfield.mode: texture | dngr` for frame-by-frame A/B.

### Config (as shipped — see §4 for the full list)

```yaml
starfield:
  mode: dngr                 # texture | dngr  (dngr default since 2026-06-06; texture = legacy F10)
  format: auto               # auto | hyg | bsc5
  source_catalog: star_image/hyglike_from_athyg_v32.csv
  catalog_path: assets/stars.npy            # {θ',φ',flux_rgb}
  diffuse_map: star_image/milkyway_2020_16k.exr   # Layer B only (low-freq)
  star_grid_cols: 720; star_grid_rows: 360  # Layer-A candidate cell grid
  star_psf_px: 1.3           # gaussian PSF σ (px)
  mag_clip: 50.0             # cap on lensing brightness gain (caustic safety)
  caustic_delta_min: 1.0e-3  # δ⁻ below this ⇒ on a caustic
  ewa_max_taps: 8            # Layer-B anisotropic taps
  jacobian: finite_diff      # finite_diff | geodesic_deviation (Phase 6)
  g_beaming: false           # volumetric g⁴ star hook (g≡1 until moving-observer g)
```
`j_fold` stays for the `texture` fallback; unused in `dngr` mode.

### Validation & phasing

Validation extends the GPU regression harness: flux conservation (lensing off);
Einstein-ring brightness (single antipode star → ring of correct radius, brightness
rising toward δ⁻→0); FD-vs-geodesic-deviation `mag` spot-check; "seam gone, not
hidden" (sharp split images, no `j_fold`); `mode: texture` reproduces today's golden
frames bit-for-bit; keep `pytest` green + a new `test_starfield_dngr.py`.

| Phase | Deliverable | Risk |
|---|---|---|
| 0 | SKILL.md: `mag` + Jacobian formulas, get approval | ✅ **Formula 13**; 3 guards **approved 2026-06-05** (rev v1.6) |
| 1 | Catalog ingest → `{θ′,φ′,flux_rgb}.npy`; B−V→RGB reuse | ✅ **shipped** — `scripts/ingest_stars.py`; HYG/ATHYG csv **+** BSC5; blackbody-from-B−V |
| 2 | FD beam Jacobian + `mag`; ellipse `(δ⁺,δ⁻,µ)` | ✅ **shipped** — `_dngr_shade` (μ→1 flat verified) |
| 3 | Layer A star gather (cell grid) + PSF splat | ✅ **shipped** — CSR cell grid + Gaussian PSF |
| 4 | Layer B anisotropic EWA diffuse fetch | ✅ **shipped** — `milkyway_2020_16k.exr`, EWA major-axis taps |
| 5 | Config gate, A/B harness, validation suite | ✅ **shipped** — `mode` gate + `tests/test_starfield_dngr.py` |
| 6 (opt) | Geodesic-deviation Jacobian upgrade | **open** — future fidelity upgrade (new ODE) |

**Decisions of record (resolved 2026-06-05):** catalog scope = **HYG/ATHYG v3.2** (the
owner-supplied csv; BSC5 still supported) with **blackbody-from-B−V** colour; Jacobian
method = **finite-difference** (FD-first, per the recommendation; geodesic deviation
deferred to Phase 6); diffuse map = **keep the Milky-Way equirect**
(`milkyway_2020_16k.exr`) with anisotropic EWA. **Still open:** Phase 6 geodesic-
deviation Jacobian (gated on a new SKILL.md deviation-ODE formula); a quantitative
Einstein-ring brightness regression (the current suite asserts finiteness + point-star
sharpness + flat-space μ→1, not ring photometry).

---

## 9. Reference material & related files

- **`skills/kerr-physics/SKILL.md`** — the authoritative physics formula reference
  (§5). Mandatory; never re-derive.
- **`CLAUDE.md`** — project instructions and policy for the assistant (the source of
  the policy box at the top of this file). **`AGENTS.md`** mirrors it for the
  Codex/Agents harness.
- **`REFERENCE_dngr_paper.md`** — James, von Tunzelmann, Franklin & Thorne (2015),
  *Gravitational Lensing by Spinning Black Holes… and in the movie Interstellar*
  (Class. Quantum Grav. 32 065001). The academic source for §8 (DNGR ray-bundle
  technique, magnification, caustics).
