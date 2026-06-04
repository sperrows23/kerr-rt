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
> - **Units/coords:** geometric `G = M = c = 1`; Boyer-Lindquist `(t, r, θ, φ)`;
>   signature `(− + + +)`; spin `a = 0.999` (near-extremal).
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
│   ├── bsc5.dat                 ← raw Yale Bright Star Catalogue (V/50) — gitignored input
│   └── stars_bsc.npy            ← point-star catalog [N,5]=θ',φ',flux_rgb (ingest output; gitignored)
├── scripts/
│   ├── thumb.py                 ← CPU preview renderer (development / QA)
│   ├── gpu_test.py              ← FHD GPU beauty render smoke test
│   ├── ingest_stars.py          ← Part A offline ingest: BSC5 → point-star {θ',φ',flux_rgb}.npy (Formula 13 / §8)
│   ├── seam_diagnostics.py      ← spin-axis seam isolation tools (off the render path; ex-§7 C1)
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
│       ├── metric.py            ← Kerr metric (Formula 1)
│       ├── geodesic.py          ← Mino-time RK4 null geodesic integrator (Formula 6)
│       ├── disk.py              ← Accretion disk gas physics (Formulas 3/4/5/8/9) — FROZEN
│       ├── starmap.py           ← 16K HDRI loader, mip pyramid, UV mapping
│       └── taichi_renderer.py   ← GPU renderer: Pipe A + Pipe B, split kernels (~1084 lines)
├── tests/
│   ├── cuda_smoke_test.py       ← confirms CUDA backend JIT on RTX 5060
│   ├── test_geodesic.py         ← conservation law tests (E, Lz, Q, null norm)
│   ├── test_starmap.py          ← polar punch-through / UV normalization tests
│   ├── test_ingest_stars.py     ← point-star ingest transforms + BSC5 parser (Part A)
│   ├── test_gpu_regression.py   ← automated GPU Doppler / NaN / disk-peak guard (CUDA-gated)
│   └── test_geodesic/
│       └── test_conserved_quantities_regression.csv   ← golden values
│
├── render_blackhole/            ← output EXR sequence (bh_####.exr) — gitignored
├── render_spaceship/            ← Blender ship EXR sequence — gitignored
├── star_image/                  ← 16K HDRI starmap EXR — gitignored
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

**`src/renderer/metric.py`** — Kerr metric (Formula 1): `metric_bl(r, theta, a)` and
the numerical inverse `inverse_metric_bl`. Used by `thumb.py` and
`test_geodesic.py`. The GPU renderer inlines the metric analytically instead of
importing this.

**`src/renderer/geodesic.py`** — CPU null-geodesic integrator (Formula 6): Mino-time
RK4 with a projection step re-imposing `(dr/dλ)²=R`, `(dθ/dλ)²=Θ`. Key:
`integrate_null_geodesic`, `make_null_initial_conditions`, `carter_Q`,
`radial_turning_point`, `_DELTA_MIN = 0.05`. **CPU state vector**
`[r, θ, φ, t, v_r, v_θ]` with `v_r = Δ·p_r`, `v_θ = p_θ`. *(This CPU `[r,θ,…]`
docstring is correct and intentional — only the GPU side migrated to `[y,u,…]`.)*
Used by `thumb.py`, `test_geodesic.py`.

**`src/renderer/disk.py`** — CPU accretion-disk gas physics; the reference the GPU
port must match numerically (verified at three test points). Key:
`isco_conserved_quantities` (F4), `gas_four_velocity` (F3/5), `g_factor` (F8;
`p_cov[R]` already covariant — do not divide by Δ again), `blackbody_rgb` (F9
chromaticity, no T⁴). **FROZEN — do not edit** (numerical regression). Used by
`thumb.py` and `taichi_renderer.py` (imports `isco_conserved_quantities`).

**`src/renderer/starmap.py`** — host-side 16K equirect starmap: `load_equirect`,
`build_mip_pyramid` (box-filter, f16 levels), `Starmap.load/.sample` (trilinear
ground truth), `normalize_sphere_angles` (polar punch-through fold),
`direction_to_uv`. Equirect convention `u = φ/2π` (col), `v = θ/π` (row, north pole
at v=0). Used by `taichi_renderer.py` (GPU upload) and `test_starmap.py`.

**`src/renderer/taichi_renderer.py`** — the GPU renderer (~1084 lines). Ports the CPU
physics to Taichi and runs both pipes on CUDA in the horizon-stable
`[y, u, φ, t, v_y, v_u]` state. Backend locked to `ti.init(arch=ti.cuda)`.

- *Module fields:* `star_flat/off/w/h` (f16 mip pyramid + metadata); `pixels`
  (square `render_pipe_a` output); `frame_pixels` (non-square `render_beauty_*`
  output); `exit_buf`, `disk_buf`, `depth_pixels` (kernel-split hand-off).
- *Physics `@ti.func`:* `_horizon_constants(a)` (host, F11 — derives k=√(1−a²),
  r₊=1+k); `_delta_y(y,k)` (F11); `_radial_potential_y` / `…_deriv_y` (F6/11);
  `_theta_potential` Θ_u / deriv (F6/12); `_deriv` (F6/12 ds/dλ); `_project` (F6);
  `_rk4_step` (F6, Kahan + project); `_zamo_init` (F7); `_gas_four_velocity` (F3/5,
  plunging branch uses factored Δ); `_blackbody_rgb` (F9); `_disk_emit` (F8/9,
  recovers p_r=v_y/Δ, p_θ=−v_u/√(1−u²)).
- *Starmap `@ti.func`:* `_texel`, `_sample_level` (bilinear, φ-wrap + θ-clamp),
  `_sample_trilinear`. *(The formerly-unreferenced `_normalize_sphere` helper was
  deleted — §7 C2, resolved 2026-06-04.)*
- *Kernels:* `render_pipe_a` (Pipe A only, square; retains its offset ray as the
  dev LOD reference); **`render_beauty_physics`** (production K1 — traces, writes
  exit/outcome + weighted Z, wraps φ into (−π,π] each step, shortest-arc exit
  interp); **`render_beauty_shade`** (production K2 — screen-space Jacobian → LOD,
  composites; `_screen_jacobian_lod` saturates LOD to `_MAX_LOD` when J >
  `render.j_fold`). *(The `render_starmap_raw` / `render_fixed_lod` / `dump_phi_exit`
  seam diagnostics now live in `scripts/seam_diagnostics.py` — §7 C1, resolved.)*
- *Host:* `load_config` (UTF-8), `setup_renderer` (cuda init + starmap upload),
  `_alloc_output`/`_alloc_frame`, `render_pipe_a_image`,
  **`render_beauty_frame`** (main entry — Blender world→BL, camera triad, runs
  K1+K2, optional NaN-guarded Z), `render_beauty_frame_mb` (motion blur, masked
  depth averaging), `tonemap` (Reinhard + gamma). **Consumed by:** `gpu_test.py`,
  `export_exr.py`, `test_gpu_regression.py`.

Camera conversion in `render_beauty_frame`: Blender `pos/fwd/up/right` →
spherical embedding (`r_cam, θ_cam, φ_cam`) → local triad (r̂, θ̂, φ̂) → dot the
camera axes onto the triad → feed the three local components to the ZAMO tetrad.

**`src/blender/export_camera.py`** — Blender Python (Phase 1). Writes
`camera_matrix.json` (one record/frame: `{frame, pos, fwd, up, right, fov}`). `fwd`
= world −Z; `fov = cam.angle` *(labeled "vertical FOV" — see §7 A1)*. Needs `bpy`.

**`scripts/thumb.py`** — CPU preview renderer (single-threaded NumPy; slow but
self-contained). `--disk` enables `march_disk`; uses config `thumb.*` framing
overrides; `zamo_photon_momentum` is the CPU reference for `_zamo_init`. Flow:
`render()` → `camera_ray_direction()` → `zamo_photon_momentum()` →
`integrate_null_geodesic()` → `march_disk()` → `trace_pixel()` → tonemap.

**`scripts/gpu_test.py`** — FHD GPU beauty smoke test. Reads a frame from
`camera_matrix.json` (`utf-8-sig`), runs `render_beauty_frame` at 1920×1080,
reports the Doppler asymmetry (`right_lum/left_lum` ≈ 7–8× expected). `--no-disk`,
`--exposure` flags.

**`scripts/export_exr.py`** — Phase 5 production entry. Extracts beauty + depth via
`to_numpy()` and writes a 4-channel `(R,G,B,Z)` EXR via OpenImageIO to
`render_blackhole/bh_####.exr`. `_shutter_arc(frames, idx, shutter_fraction, fps)`
returns `Δφ·fps·shutter_fraction` (F2). `--motion-blur` opt-in.

**`tests/test_geodesic.py`** — CPU geodesic conservation (E, Lz, Q drift < 1e-4;
null condition < 1e-6 over 4000 steps) + a golden-CSV regression.

**`tests/test_starmap.py`** — polar punch-through fix unit tests
(`normalize_sphere_angles`, `direction_to_uv`).

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
skills/.../SKILL.md  ──▶ metric.py(F1) · geodesic.py(F6) · disk.py(F3/4/5/8/9) ·
                          starmap.py(F10 UV) · taichi_renderer.py(all, GPU port)
metric.py    ──▶ thumb.py · test_geodesic.py
geodesic.py  ──▶ thumb.py · test_geodesic.py
disk.py      ──▶ thumb.py(march_disk) · taichi_renderer.py(isco_conserved_quantities)
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
| `v_r = Δ·p_r` (CPU) renamed `v_y = dy/dλ = Δ·p_r` (GPU `[y,u,…]`) | `geodesic.py` / `_zamo_init`,`_disk_emit` |
| `p_r` covariant recovery = `v_r/Δ` (not `/Δ²`) | `disk.py`, `_disk_emit` |
| `blackbody_rgb` chromaticity-only | `disk.py`, `_blackbody_rgb` |
| g⁴ beaming correct, not double-counted | `disk.py`, `_disk_emit` |
| θ ∈ [0, π] before UV lookup | `starmap.py:normalize_sphere_angles` (GPU clamps `acos(u)` inline) |
| Camera file encoding = utf-8-sig | `gpu_test.py` |
| Config files read with utf-8 | `taichi_renderer.py`, `thumb.py`, `test_geodesic.py` |

---

## 4. Configuration reference (`render.yaml`)

| Section | Key fields |
|---------|-----------|
| `black_hole` | `spin` (a=0.999), `r_isco` (1.182 M), `r_plus` (1.0447 M — true outer horizon r₊=1+√(1−a²); consumed only by `thumb.py`, the renderer derives r₊ in `_horizon_constants`) |
| `render` | `width`/`height` (4K), `thumb_width/height` (256), `max_steps_pipe_a` (250 — Pipe B shares this same trace loop / step cap; the dead `max_steps_pipe_b` key was removed, §7 F4), `d_lambda_pipe_a` (0.01), `r_max` (50 M), `device_memory_gb` (6), `horizon_epsilon` (0.05), `adaptive_step_floor` (0.005), `sin2_min` (1e-10 polar guard), `j_fold` (0.15 — background LOD fold-saturation; kills the center "static" seam), `fps` (24.0 — shutter arc = Δφ·fps·shutter_fraction), `projection_mode` (perspective\|equirect), `depth_infinity` (1e5 no-disk Z sentinel) |
| `disk` | `r_inner`, `r_outer`, `theta_half_width`, `T_0`, `emission_coeff`, `absorption_coeff`, `vertical_sigma_frac`, `bounding_sin_theta_half` (=sin(theta_half_width); bbox early-out) |
| `starmap` | `path` (relative to repo root), `width` (16384 — used to compute LOD) |
| `starfield` | **Part A / §8 Phase 1 point-star ingest** (`scripts/ingest_stars.py`): `source_catalog` (raw BSC5 `bsc5.dat`), `catalog_path` (output `.npy`), `mag_limit` (6.5 — drop fainter stars), `mag_zero_point` (0.0 — Pogson flux scale). Consumed only by the offline ingest; the DNGR render path (§8 Phases 2–5) is not wired yet |
| `camera` | `default_radius` (6.03 M), `default_fov_deg` (90°), `shutter_fraction` (1/48 s — `export_exr._shutter_arc` reads this with `render.fps` as `arc = Δφ·fps·shutter_fraction`) |
| `thumb` | preview-only framing overrides (camera radius/fov/theta, background, glow, exposure, gamma) |
| `output` | directory names and filename prefixes for the EXR sequences |

---

## 5. Physics formula index (`SKILL.md`)

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
| 13 | Hybrid DNGR (rev v1.5): screen-space 2×2 ray-bundle Jacobian J → point-star magnification μ = \|det J₀·sinθ′₀\|/\|det J·sinθ′\| → energy-conserving flux `I = I_base·μ·g⁴` with truncated-Gaussian PSF. Point stars **brighten**, don't smear. **Phase-0 only** (§8); 3 guards flagged pending owner approval; no renderer code yet |

Decisions of record (from `CLAUDE.md`): Decision A = ZAMO tetrad (F7); Decision B =
simple temperature model T = T₀·(6/r)^0.75.

---

## 6. Implementation status — shipped

The full 5-phase optimization from `guid.md` (the now-superseded source spec) is
**complete** and committed. Condensed history:

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
- **Formula 13 — Hybrid DNGR (SKILL.md, 2026-06-04):** the screen-space ray-bundle
  Jacobian / point-star magnification / truncated-Gaussian-PSF formulation was
  verified against `REFERENCE_dngr_paper.md` and merged into `SKILL.md` (rev v1.5).
  This is the Phase-0 physics deliverable of the §8 rearchitecture; **no renderer
  code yet**, and three guards (μ normalization, boundary inheritance, g⁴ exponent)
  remain flagged for owner approval before implementation.
- **Point-star ingest — Part A / §8 Phase 1 (2026-06-04):** `scripts/ingest_stars.py`
  ingests the Yale Bright Star Catalogue (V/50 `bsc5.dat`) into the render-ready
  point-star table `assets/stars_bsc.npy` — `float32 [N,5] = (θ′, φ′, flux_r, flux_g,
  flux_b)` in the integrator's BL exit frame (`θ′=π/2−Dec`, `φ′=RA`, matching the
  equirect `u=φ′/2π`, `v=θ′/π` lookup). Flux is built as energy, not a texture
  sample: Pogson `10^(−0.4·Vmag)` brightness × `blackbody_rgb` chromaticity (Formula 9
  helper **reused** from `disk.py`) with B−V→T via Ballesteros (2012). New config
  block `starfield:` (`source_catalog`, `catalog_path`, `mag_limit`, `mag_zero_point`);
  unit tests in `tests/test_ingest_stars.py` (15, no catalogue file needed). **Offline
  data tool only** — touches no renderer code; the GPU star-gather path (§8 Phases 2–5)
  is still not wired up, and the three Formula 13 guards remain pending.

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
- **A4 — no automated seam regression.** The center-seam fix has no test pinning it;
  a future LOD change could silently reintroduce the static band. Consider extending
  `test_gpu_regression.py` with a spin-axis-meridian smoothness assertion.
- **C1 — diagnostics in the production module.** ✅ **RESOLVED (2026-06-04).**
  `render_starmap_raw`, `render_fixed_lod`, `dump_phi_exit` (and the `__main__`
  block) were extracted from `taichi_renderer.py` to `scripts/seam_diagnostics.py`,
  which imports the production `@ti.func` helpers by name (no physics re-implemented).
  The production module is now ~1084 lines.
- **C2 — dead code.** ✅ **RESOLVED (2026-06-04).** `_normalize_sphere` has been
  deleted from `taichi_renderer.py` (no call sites remained — the `[y,u,…]`
  migration made `acos(clamp(u))` the exit path).

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

## 8. Future proposal — DNGR background rearchitecture (gated)

**Status: Phase 0 formula merged (SKILL.md Formula 13, rev v1.5, 2026-06-04); no
renderer code yet.** Replaces the baked-texture star field with the *Interstellar*
DNGR treatment — **point stars stay sharp; gravitational lensing changes their
brightness, not their size.** Every new lensing formula it introduces (magnification,
ray-bundle Jacobian) is routed through `SKILL.md` for human approval **first**,
exactly as Formula 10 was — the screen-space-Jacobian magnification + PSF are now
**Formula 13** there, with three guards (μ normalization, boundary inheritance, g⁴
exponent) still flagged for owner sign-off before Phase 1 coding begins. The just-landed `j_fold` seam fix is a
**stopgap** that restores the "smooth faint line" look; it does not make the lensed
star field correct — that is this proposal's job.

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

### Config additions (proposed)

```yaml
starfield:
  mode: dngr                 # texture | dngr
  catalog_path: assets/stars_bsc.npy        # {θ',φ',flux_rgb}
  diffuse_map: assets/milkyway_diffuse.exr  # Layer B only (low-freq)
  jacobian: finite_diff      # finite_diff | geodesic_deviation
  star_psf_px: 1.3           # gaussian splat radius
  mag_clip: 50.0             # cap on lensing brightness gain (caustic safety)
  caustic_delta_min: 1.0e-3  # δ⁻ below this ⇒ on a caustic
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
| 0 | SKILL.md: `mag` + Jacobian formulas, get approval | ✅ **merged as Formula 13** (v1.5); 3 guards pending owner sign-off |
| 1 | Catalog ingest → `{θ′,φ′,flux_rgb}.npy`; B−V→RGB reuse | ✅ **shipped (2026-06-04)** — `scripts/ingest_stars.py` + `tests/test_ingest_stars.py`; chose bright-star (BSC5) scope + blackbody-from-B−V (see §6) |
| 2 | FD beam Jacobian + `mag`; ellipse `(δ⁺,δ⁻,µ)` | med |
| 3 | Layer A star gather (cell grid) + PSF splat | med |
| 4 | Layer B anisotropic EWA diffuse fetch | med |
| 5 | Config gate, A/B harness, validation suite | low |
| 6 (opt) | Geodesic-deviation Jacobian upgrade | higher (new ODE) |

**Open decisions for the human:** Phase 1 settled two of these — **catalog scope =
bright-star only (BSC5)** and **color fidelity = blackbody-from-B−V** (both realized in
`scripts/ingest_stars.py`; revisit only if + faint Gaia is wanted later). Still open
before the **render** path (Phase 2+): Jacobian method (FD-first vs. straight to
geodesic deviation); diffuse map (keep a small Milky-Way equirect vs. point stars only
first). No renderer code is wired until those settle and the Phase-0 Formula 13 guards
are approved.

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
