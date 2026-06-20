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
8. [DNGR background rearchitecture (SHIPPED)](#8-dngr-background-rearchitecture-shipped--starfieldmode-dngr)
9. [Reference material & related files](#9-reference-material--related-files)
10. [Accretion-disk procedural turbulence (PLANNED — D2)](#10-accretion-disk-procedural-turbulence-planned--d2)

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
  Doppler asymmetry for the a = 0.999 edge-on camera. At full physics
  (`doppler_strength: 1.0`, simple model) this is ≈5.15× post-CKS-13 (was 4.32× pre-D3,
  7.77× on the retired BL Mino path); the `doppler_strength = s` knob scales it down
  (g_eff = g^s), so the YAML's current s = 0.1 reads ≈1.9×. `oiiotool --info` to
  confirm channels.

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
│   ├── README_ingest_stars.md   ← standalone guide for the ingest step (API, config, tests)
│   ├── seam_diagnostics.py      ← spin-axis seam isolation tools (off the render path; §7 C1)
│   ├── check_starless_map.py    ← DNGR Layer-B starless acceptance gate (§6 2026-06-07)
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
├── docs/
│   └── specs/
│       └── 2026-06-06-dngr-artifact-remediation.md  ← dated design record (§7 S4–S7)
├── PROJECT.md                   ← this file (the single project reference)
├── README.md                    ← public-facing overview (points here for depth)
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
0, FHD, disk on); asserts NaN==0 and no spin-axis seam on the YAML-configured frame.
The Doppler / disk-peak guards are **dynamic in `doppler_strength`** (g_eff = g^s):
rather than pin one brittle band, they render frame 0 at forced s ∈ {0, 0.5, 1.0}
(simple model, disk-only `disk_buf` metrics) and assert the beaming *response* —
near-symmetric at s=0 (< 1.5), monotone non-decreasing in s, and matching the
re-anchored s=1.0 goldens `_DOPPLER_RATIO_REF = 5.15` (±10%) / `_DISK_MAX_REF = 14.45`
(±8%). **Re-anchored 2026-06-13:** the old CKS-era 4.32× / 6.1667 references held
through the doppler_strength knob (4.317× / 6.1665) but were silently invalidated by
D3 / CKS-13 (commit 30f8511), which redefined the simple-model amplitude from the old
`T_0: 5500` *inner-reference* temperature (peak T_eff ≈ 18,600 K) to
`target_peak_temperature: 5500` (peak T_eff = 5500 K) — the warm-peak chroma shift
moved both goldens. (The v1.14 changelog's "GPU regression bit-identical" claim was
inaccurate; corrected in SKILL.md.) Plus flag-path guards on deepcopied configs:
`page_thorne` renders NaN-free / beamed (> 2×, forced s=1.0 so the YAML's s=0.1 can't
suppress it) / emitting (D1), and `doppler_strength: 0` collapses the disk-only
left/right ratio to < 1.5 (knob guard). `pytest.mark.gpu`; **skips cleanly**
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
| `black_hole` | `spin` (a=0.999) — the only stored key. `r_isco`/`r_plus` are **derived at load** by `src/renderer/kerr_params.resolve_config` (Formula CKS-13: BPT Formula 2 / CKS-6) and injected into the config dict; the old YAML literals (1.182 / 1.0447) were removed 2026-06-13 because they silently desynced when `spin` was edited |
| `render` | `width`/`height` (4K), `thumb_width/height` (256), `max_steps_pipe_a` (800 — Pipe B shares this same trace loop / step cap; raised from 250 because the CKS affine λ advances ~1 coord-unit/step vs BL Mino ~r²/step), `d_lambda_pipe_a` (0.25 — CKS affine step; far-field h≈dλ, shrunk near the horizon), `r_max` (50 M), `device_memory_gb` (6), `horizon_epsilon` (0.05 — CKS-6 capture margin, cost bound only), `adaptive_step_floor` (0.02), `j_fold` (0.15 — background LOD fold-saturation; under CKS this only guards the equirect texture poles, the BH spin-axis seam is gone), `fps` (24.0 — shutter arc = Δφ·fps·shutter_fraction), `projection_mode` (perspective\|equirect), `depth_infinity` (1e5 no-disk Z sentinel). *(The BL `sin2_min` 1/sin²θ polar guard was removed — CKS has no spin-axis coordinate singularity.)* |
| `disk` | `r_inner` (`auto` → r_isco(spin), the zero-torque inner edge; a number is an artistic override, clamped ≥ r_isco — CKS-13), `r_outer`, `theta_half_width`, `target_peak_temperature` (5500 K — **base** look parameter; the model amplitude `T_0` is derived from it at load: page_thorne `T_0 = target` since the f_PT LUT is max-normalized, simple `T_0 = target·(r_inner/6)^¾`; an explicit `T_0` key is a legacy override), `dynamics.inner_lap_seconds`/`dynamics.shear_wrap_budget` (base time-mapping look targets → derived `omega_inner/outer`, `period_inner_M/period_outer_M`, `wrap_time_M`, `time_scale` (M of disk time per footage second), `shear_period_M` (CKS-12 reset period) — all rescale automatically with spin/extent), `emission_coeff`, `absorption_coeff`, `vertical_sigma_frac` (the bbox `|u|` early-out bound is now **derived** as `sin(theta_half_width)` in code — the old `bounding_sin_theta_half` literal was removed, §7 S2), `max_step_vfrac` (0.5 — disk-thickness step cap: inside the slab the geodesic step is shrunk so the vertical displacement `|dz/dλ|·h ≤ vfrac·σ_z`, σ_z=`r·theta_half_width·vertical_sigma_frac`, stopping a steep equatorial crossing from striding over the thin emitting layer and aliasing it into a moiré; no-op for in-plane/edge-on grazers and for the current thick disk, so golden frames are unchanged — `tests/test_disk_step_convergence.py`), `temperature_model` (`simple`\|`page_thorne` — D1 radial-profile selector, default `simple`), `flux_lut_samples` (256 — page_thorne f_PT LUT resolution), `doppler_strength` (1.0 — **visualization-only** exponent `g_eff = g^s` on the CKS-9 shift, feeding both the g⁴ beaming and the blackbody color shift; 1.0 = full physics / golden frames, 0.0 = shift off, the Interstellar/DNGR artistic treatment) |
| `starmap` | `path` (relative to repo root), `width` (16384 — used to compute LOD) |
| `starfield` | **DNGR background (Formula 13 / §8).** `mode` (`texture`\|`dngr`; texture default keeps the legacy F10 path + golden frames). *Ingest:* `format` (auto\|hyg\|bsc5), `source_catalog` (HYG/ATHYG csv or `bsc5.dat`), `catalog_path` (`assets/stars.npy`), `mag_limit` (11.0), `mag_zero_point` (3.5 — **ingest-time**; baked into `stars.npy`, the renderer never reads it, so re-ingest after editing). *Layer A (point stars):* `star_grid_cols/rows` (candidate cell grid), `star_cell_radius`, `star_psf_px` (PSF σ), `psf_trunc_sigma`, `mag_clip` (μ cap), `caustic_delta_min` (δ⁻ floor), `g_beaming` (g⁴ hook, default off). *Layer B (diffuse):* `diffuse_map` (Milky-Way EXR), `diffuse_width`, `diffuse_gain` (render-time band brightness ×, default 1.0; no re-ingest), `ewa_max_taps`, `jacobian` (`finite_diff`). The Layer-A/B fields load only in `mode=dngr` |
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
| CKS-11 | Page-Thorne flux shape f_PT(r) (cubic roots + three-log bracket, zero-torque BC) — **wired** behind `disk.temperature_model: page_thorne` (D1) |
| CKS-12 | Disk procedural turbulence (**fully wired** through D2.4, 2026-06-13 — density + T/edges/σ_θ; noise ON by default): noise coords (u=ln r/r_inner, φ, ζ), Keplerian shear advection φ′=φ−Ω(r)·t_disk with dual-phase reset blend (Ω = Formula 3 verbatim), modulation bookkeeping (amplitudes only — density/T_emit/edges/σ_θ; never p_μ/u^μ/g/g⁴/f_PT). §3 modulates four advected [0,1] fBm envelopes: T_emit pre-g (g⁴-not-g⁸), smoothstep edges (`r_in_eff≥r_isco`), lumpy σ_θ with worst-case-σ_z step cap. Non-physical `disk.noise.dynamism` viz gain (φ′=φ−dynamism·Ω·a·T, default 1.0=bit-identical) emphasises per-frame swirl — same dial spirit as `doppler_strength` (SKILL.md v1.20) |
| CKS-13 | Derived-parameter config resolver (**wired**, D3, 2026-06-13): `kerr_params.resolve_config` injects r_plus (CKS-6), r_isco (Formula 2), `disk.r_inner` (auto→r_isco, override clamped), `disk.T_0` from `target_peak_temperature` (per temperature model), and `disk.dynamics` time mapping (T_orb=2π(r^{3/2}+a), t_wrap=2π/ΔΩ → `time_scale`, `shear_period_M`). No new physics — pinned formulas + trivial inverses; literature anchors in `tests/test_kerr_params.py` |

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
simple temperature model T = T₀·(6/r)^0.75 **(ACTIVE default)**. The physically-correct
Page-Thorne flux profile (`SKILL.md` Formula **CKS-11**, guarded by `tests/test_disk_flux.py`,
SKILL.md rev v1.11) is now **wired behind the config flag `disk.temperature_model: page_thorne`**
(2026-06-12; CPU-precomputed `f_PT(r)` LUT in `src/renderer/disk_flux.py`, GPU guard in
`tests/test_gpu_regression.py`) — the simple model stays the default, so golden frames /
the pinned GPU regression are unchanged; see §7 backlog **D1**.

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

**Starless Layer-B + deeper catalog (2026-06-07).** Root-caused the wide-FOV
*edge star-elongation*: the DNGR Layer-B diffuse plate (`milkyway_2020_16k.exr`) still
carried the **full baked point-star field** — measured ~2% of lit pixels were sharp
local-max spikes (>10× neighbourhood), identical to `starmap_2020_16k` (the config's old
claim that milkyway "omits stars brighter than ~8.0" was **empirically false**). The
anisotropic EWA footprint, which grows to ~2.3× at the 90° screen corner, stretches every
such point source into a directional streak (Layer A's J⁻¹ point splats stay round — the
streaks are 100% Layer B; verified by isolated-layer corner renders, BH out of frame so
zero lensing). **Fix:** Layer B now points at `starmap_final.exr`, an owner-produced
StarNet2-cleaned plate (workflow in `star_image/starmap_workflow.md`: StarNet2 + a
gnomonic-reprojection polar pass for the ERP-stretched poles). Sharp >10× spikes drop **46×
to 0.045%**; corner diffuse mean drops 124× and bright-pixel count 2957→0. With Layer B now
starless, **every** visible star must come from Layer A, so `mag_limit` 8.0 → **11.0**
(~Tycho-2) and `assets/stars.npy` re-ingested **41 410 → 115 324 stars** to repopulate the
faint field as sharp, correctly-lensed point sources (the *Interstellar*/DNGR intent).
*Verified:* full suite 44 passed; corner render shows round catalog stars, no streaks.
The acceptance metric is now a permanent tool — `scripts/check_starless_map.py`
(luminance → separable equirect box-mean → local-contrast spike count; floor =
band median, i.e. median over the *positive* pixels, since a cleaned plate is
mostly hard zeros). It exits non-zero unless sharp >10× spikes ≤ 0.05% of lit
pixels; `starmap_final` passes at 0.031%, `starmap_2020`/`milkyway_2020` fail at
~2.1% (the disproof, reproduced). Run it on any candidate Layer-B plate.
**Background brightness (2026-06-07, follow-up).** The starless plate reads dim,
and an owner `--no-disk` test bumping `mag_zero_point` 0→10 changed *nothing* —
root-caused to two facts: (1) `mag_zero_point` is an **ingest-time** parameter
(baked into `assets/stars.npy` by `ingest_stars.py:261`); the renderer never reads
it, so editing `render.yaml` is a no-op until you re-ingest. (2) Even re-ingested,
it only scales **Layer A point stars** — the **Layer B diffuse band had no
brightness control at all** (`_dngr_shade` returned the raw EXR sample). **Fix:**
added `starfield.diffuse_gain` (default 1.0 = identity), applied at render time as
`diffuse *= _MW_GAIN` in `_dngr_shade` — tunable per render with **no re-ingest**.
Live config: `diffuse_gain: 3.0` (the StarNet2 plate is ~1.75× dimmer mean than the
old milkyway band, so 1.75 restores parity, 3.0 brightens past it) and
`mag_zero_point: 3.5` (≈25× stars: clearly visible but kept **below** the disk-edge
peak — ~6.17 at the time; now ~14.45 post-CKS-13, so stars sit even further below it —
so the disk stays the dominant feature — 5.0/100× made a lensed star
outshine the disk, 10.0/10000× blew it out), re-ingested. Both are look-tuning
starting points, not correctness values. **Test follow-on:** brightening Layer A
exposed a latent assumption in the disk-peak guard — it used the
**global frame max** (`hdr.max()`) as a disk-peak proxy, which only holds while the
disk is the brightest pixel. Fixed it to read the disk emission straight from
`tr.disk_buf[..., :3]` (the final pixel is `disk_rgb + transm·bg`, so this is the
beamed disk edge independent of the sky). *(Reference was 6.1667 here; later
re-anchored to 14.45 and made dynamic in `doppler_strength` — see the
`test_gpu_regression.py` entry above and SKILL.md v1.16.)*

**Disk-thickness step cap (2026-06-12).** Fixed a concentric **moiré band** around
the disk on face-on / steep-crossing views. Root cause: the Pipe-B march sizes its
geodesic step by horizon distance only (`h = dλ·max(floor, (r−r₊)/r)`, CKS-5) and is
blind to the disk's vertical extent, so when a ray crosses the equatorial plane
steeply and the emitting layer is thin, one step strides over the Gaussian density
(Formula 9) and the layer is sampled 0/1/2× almost at random across the frame —
aliasing (the user's "shrink the step and it renders normally" is the classic
under-resolved-quadrature signature). **Fix:** while inside the slab, cap the step's
**vertical** displacement `|dz/dλ|·h ≤ max_step_vfrac·σ_z`, σ_z=`r·θ_half·σ_frac`
(`disk.max_step_vfrac`, default 0.5 ≈ 2 samples/σ; `dz/dλ` read for free by hoisting
RK4's `k1` out via `_rk4_step_k1`, bit-identical to `_rk4_step`). It only bites for
steep crossings — near-in-plane / edge-on grazers (dz/dλ→0) keep the full step, so it
adds no cost there and **cannot** push them into `max_steps` (a path-length cap was
tried and rejected: it throttled edge-on grazers into 76% ray truncation). *Validated:*
out-of-band pixels and the production thick-disk golden frame 0 are bit-identical (cap
ON≡OFF to 4 dp, both disk models — so golden frames / pinned regression are unchanged);
on a thin disk seen face-on the cap cuts divergence-from-ground-truth 0.123→0.002 (56×)
and the moiré vanishes. Guard: `tests/test_disk_step_convergence.py` (thin slab, top-
down camera, non-truncated fine reference).

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

- **`disk.doppler_strength` artistic shift dial (2026-06-12):** visualization-only
  exponent on the CKS-9 relativistic shift, `g_eff = g^s`, applied once in
  `_disk_emit_cks` and feeding **both** the Formula-9 g⁴ beaming amplitude and the
  blackbody color shift (g⁴-not-g⁸ bookkeeping unchanged). `s=1` (default) skips the
  `ti.pow` branch entirely — verified bit-equivalent to the pre-knob kernel (Doppler
  4.317×, disk peak 6.1665 vs goldens 4.32× / 6.1667). `s=0` ⇒ g_eff≡1: no beaming,
  no color shift — the Interstellar/DNGR treatment (the film suppressed the disk's
  Doppler asymmetry); lensing geometry is untouched. Runtime kernel arg, no re-JIT.
  NOT physics: scales the TOTAL CKS-9 g (orbital + gravitational, inseparable there).
  *Verified:* new GPU guard `test_doppler_strength_zero_symmetrizes_disk` (s=0
  disk-only L/R ratio < 1.5, NaN-free, still emits) green.

- **V epoch — volumetric disk self-shadow + source function (2026-06-13, V1.0–V1.2):**
  the 2.5D-slab path toward the Interstellar glowing-gas-with-voids look (config alone
  can't reach it). **V1.0** extracted the density stack into the shared `@ti.func
  _disk_density_cks` (single source for the emit march AND the shadow bake; bit-identical
  refactor). **V1.1 (Formula CKS-14)** added the radiative-transfer source-function march
  `dI=(S−I)dτ`, `S=emission/dτ`, behind `disk.volumetric.source_function` — the SAME
  continuum integral as the legacy emission sum but the exact per-step quadrature, and it
  *materialises* `S`. **V1.2 (Formula CKS-15)** added the radial deep-shadow-map: a
  per-frame baked `τ_shadow[NU,NPHI,NZ]` on the CKS-12 noise coords (integrated outward
  from `r_inner` off the shared density), trilinearly sampled to dim emissivity
  `j → j·e^{−strength·τ_s}` behind `disk.volumetric.self_shadow.enabled` — VISUALIZATION
  occlusion (straight radial ray, amplitude-only, flagged like `doppler_strength`). The
  void look needs **both** flags (CKS-14 materialises `S`, CKS-15 carves it). Both default
  **off** ⇒ goldens bit-identical. *Verified:* `tests/test_disk_source_function.py` (3) +
  `tests/test_disk_self_shadow.py` (4, incl. the GPU bake vs an analytic Gaussian column)
  green; `test_gpu_regression.py` unchanged. (V1.3 showcase flags / V1.4 golden + this
  §-rewrite / V1.5 Simplex follow — spec `docs/specs/2026-06-13-V1-self-shadow-source-function.md`.)
- **V epoch — V2 flared 3D density (2026-06-14, Formula CKS-16):** the constant angular
  scale height becomes radius-flared `σ_θ(r)=σ0·(r/r_inner)^β` (`σ0=theta_half_width·
  vertical_sigma_frac`), giving the slab real vertical bulk so the `ridged3`/`fbm3` stack's
  `ζ=dz_ang/σ_eff` coordinate stops being squashed — **genuine 3D with no new noise
  primitive**. Lives once in the shared `_disk_density_cks` (so the emit march AND the
  CKS-15 shadow bake inherit it); `β=0` skips `ti.pow` ⇒ the V1 slab bit-for-bit. Two
  knock-on fixes: (A) the CKS-13 resolver derives a separate `theta_half_bound ≥
  band_sigma·σ_θ(r_outer)` (default 3σ) so the flared outer envelope isn't hard-clipped,
  with `theta_half_width` left as the un-mutated σ0 anchor (idempotent); (B) the Pipe-B
  vertical step cap is unchanged — flare thickens *outward* so the inner edge σ0 is still
  the worst case, *verified* by the convergence test. Gated by `disk.volumetric.flare`
  (default `enabled:false`; `enabled:true,β=0` also bit-identical) ⇒ all V1/D2 goldens stay
  green. GEOMETRY/TEXTURE only — no `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT` touched. *Verified:*
  `tests/test_disk_flare.py` (7 resolver/CPU + 2 GPU) green; `test_gpu_regression.py` /
  `test_disk_step_convergence.py` unchanged. Spec
  `docs/specs/2026-06-14-V2-flared-3d-density.md`; SKILL.md rev v1.24.
- **V epoch — vertical self-shadow / 3D inner-edge ray (2026-06-14, Formula CKS-17):**
  generalises the CKS-15 radial deep-shadow-map to a **3D shadow ray** from the inner edge
  *in the midplane* `(u=0,ζ=0)` to the sample `(u_s,φ,ζ_s)` at fixed φ (`ζ(u)=(u/u_s)·ζ_s`),
  so an off-midplane parcel is shadowed by the dense midplane gas between it and the hot
  inner edge — the *vertical* self-shadow V2's bulk makes physical. Bake accumulates
  `Σ_{j<i_u} absb_c·ρ(u_j,φ,ζ_j)·ds_j` with the **tilted** sample and 3D arc length
  `ds=√((r·du)²+ΔZ²)`. **CKS-15 is the ζ=0 limit** (`ΔZ≡0 ⇒ ds=r·du`, ρ at midplane —
  bit-exact). Only `bake_disk_shadow`'s ray geometry changed: the `disk_shadow_tau` field,
  `_sample_shadow_tau` lookup, and `emission *= e^{−strength·τ_s}` application (emissivity
  only; κ/dτ untouched) are unchanged — **no new config/field/flag**, same
  `disk.volumetric.self_shadow.enabled` (default `false` ⇒ goldens bit-identical). Cost
  `O(NU²·NPHI·NZ)` (~NU/2× CKS-15, no prefix sum), parallel over cells. VISUALIZATION /
  amplitude-only — straight CKS ray, single inner-edge illuminator, single-scatter; no
  `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT` touched. *Verified:* `tests/test_disk_self_shadow.py`
  (flag-off bit-identity + outward-steepening dimming + contrast-rise carry over;
  `test_bake_matches_analytic_3d_ray_integral` re-derived to the tilted-ray line integral);
  `test_gpu_regression.py` unchanged. Spec
  `docs/specs/2026-06-14-V2-vertical-self-shadow.md`; SKILL.md rev v1.25.
- **V epoch — curl-flow domain warp / V3.0 (2026-06-14, Formula CKS-18):** adds
  divergence-free turbulent structure (eddies/billows) that the laminar §2 Keplerian
  shear can't produce, by warping the noise **sample coordinate** `(u,φ)` with the 2-D
  curl of a scalar potential `ψ = sfbm3(cosφ·ρ_c, sinφ·ρ_c, u·k_u)` on the **V1.5
  isotropic simplex basis**: `δu=+∂ψ/∂φ`, `δφ=−∂ψ/∂u` (central FD), `u'=u+A·δu`,
  `φ'=φ+A·δφ`. **Divergence-free** by construction (curl of a scalar) and **seamless
  across φ=0** because `δu`/`δφ` are built on `cos φ`/`sin φ` (CKS-12 constraint 5 holds
  even though simplex is not lattice-periodic — seamlessness from the cylinder
  *embedding*, so `ρ_c`/`k_u` may be any real). Owner decisions: stage **static warp
  (V3.0) → curl-flow advection (V3.1)** (the D2.2→D2.3 split); V3.0 displacement is
  **in-plane `(u,φ)`** only (ζ untouched). Applied at the entry of `_disk_noise_m` /
  `_mod_fbm4` on the already-sheared per-phase `φ′_k` (material-frame — eddies freeze
  into the gas, §2 winds them into filaments; density + §3 modulation share one warp),
  using a **fixed `curl.seed`** (not the per-cycle reseed) so V3.0 is genuinely static —
  only the §2 winding animates. VISUALIZATION / texturing — relocates the noise
  coordinate only, never `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT`. New code: `noise.py`
  `curl_warp`/`curl_warp_ti`; `taichi_renderer.py` warp at the two stack entries +
  `disk.noise.curl` dials through `_setup_disk_noise` (`_NOISE_N` 43→52); **no CKS-13
  resolver change** (all base look dials). Gated by `disk.noise.curl.enabled` (default
  `false`, `amp=0` ⇒ identity) ⇒ bit-identical to V2. *Verified:* `tests/test_noise.py`
  (divergence-free, seamlessness, determinism, zero-amp identity, moves-density) +
  `tests/test_noise_gpu.py` (`curl_warp_ti` twin parity to derived `amp·_SATOL/fd_eps`
  — the warp is a derivative, so the ~1e-5 `sfbm3` twin gap is amplified ×1/(2ε); obs
  ~6.5e-5); `test_gpu_regression.py` default-off goldens unshifted. *GPU-verified
  2026-06-14:* noise_gpu 15, disk_noise + gpu_regression pass, noise 44 (CPU). Spec
  `docs/specs/2026-06-14-V3-curl-domain-warp.md`; SKILL.md rev v1.26.

**Multi-phase disk media — emission/absorption density split (P2, shipped 2026-06-16,
  GPU-verified 2026-06-18, Formula CKS-19).** Decouples disk **emission** density
  (`ρ_hot`) from **absorption** density (`ρ_cold`) so a cold dust phase carves dark
  **silhouettes** into the glow instead of only dimming it. `_disk_density_cks` now
  returns `vec3(ρ_hot, ρ_cold, temp_factor)`: emission integrates `ρ_hot`, while the
  optical-depth `dτ` *and* the CKS-15/17 self-shadow bake read `ρ_cold` (per-channel
  chromatic `dτ⃗` shipped as Task 7 — see below). `ρ_cold =
  exp(clamp(m_cold))` with the **variance-preserving Pearson mix** `m_cold = χ·m_hot
  + √(1−χ²)·m_dust` (χ = `dust_correlation`; dust field drawn at `seed+NSEED_DUST`,
  `NSEED_DUST=911`) so the cold modulator keeps `m_hot`'s variance for any χ; χ=−1
  anti-correlates fully (dust fills the hot voids). The Pipe-B vertical step cap
  tightens to the thinner cold slab (`σ_cold = σ_hot·dust_sigma_frac`). VISUALIZATION
  / texturing — never touches `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT`. New code: `noise.py`
  `dust_density_mult` + `_advected_m` (CPU twin); `taichi_renderer.py`
  `_disk_blended_m`/`_disk_cold_mult_from_hot` + the `vec3` `_disk_density_cks` +
  `disk.multiphase` dials through `_setup_disk_noise` (`_NI_MP_{EN,CHI,AMP,SIGFRAC}`,
  `_NOISE_N` 53→57). **JIT:** the dust branch is emitted only when enabled
  (`ti.static(_MP_COMPILE)` gate set at setup) — toggling `disk.multiphase.enabled`
  forces a one-time recompile, but the OFF default keeps the original fast JIT (a
  *runtime* `if` blew the mega-kernel compile past 2 h). Gated by
  `disk.multiphase.enabled` (default `false` ⇒ `ρ_cold ≡ ρ_hot` ⇒ legacy march
  bit-identical). *Verified:* `tests/test_noise.py` (CPU correlation/variance/χ=+1
  parity), `tests/test_disk_noise.py::test_rho_cold_gpu_matches_cpu` (GPU↔CPU `ρ_cold`
  parity) + `::test_multiphase_off_bit_identical` (strict OFF bit-identity with
  non-default cold params), `tests/test_disk_multiphase.py::test_dust_carves_silhouette`
  (MP-on darkens >2% of disk pixels and lowers the floor); `test_gpu_regression.py`
  6/6 goldens unshifted. Plan `docs/plans/2026-06-16-pillar2-multiphase-implementation.md`;
  SKILL.md Formula CKS-19 (DESIGN→ACTIVE, rev v1.30).

**Chromatic extinction — CKS-19 Task 7 (wired 2026-06-18, rev v1.31).** Completes P2:
  the grey scalar κ becomes a 3-vector `κ⃗ = absb_c·extinction_rgb`, so cold dust can
  **redden** (not just dim) the light it absorbs. `disk.extinction_rgb` (default grey
  `[1,1,1]`) is the per-channel multiplier; `κ_B>κ_R` (e.g. `[0.6,1.0,1.6]`) absorbs
  blue more than red — astrophysical dust reddening. Task 5 left the beauty march
  scalar, so Task 7 promoted the running `transm` to a **vec3 `T⃗`**, carries per-channel
  `dτ⃗`, and widened `disk_buf` (H,W,4)→(H,W,6) so `T⃗` survives into the composite
  (`col = disk_col + T⃗ ⊙ bg` — the background starlight reddens through dust too). The
  per-channel CKS-14 source factor is `(1−e^{−dτ_c})/dτ_c`; the depth proxy keeps the
  `transm[0]·Σadded` form so **grey is bit-exact** (goldens unshifted). `extinction_rgb`
  is a runtime kernel arg (3 floats `ext_r/g/b` on `render_beauty_physics`), not a
  compile gate, so it tunes per render; works on the MP-off path too (orthogonal to the
  ρ_hot/ρ_cold split). Self-shadow bake stays grey (chromatic τ_s not required by the
  formula). *Verified:* `tests/test_disk_multiphase.py::test_chromatic_extinction_reddens`
  (blue-weighted κ⃗ lets proportionally more red than blue survive), `test_gpu_regression.py`
  goldens unshifted. SKILL.md Formula CKS-19 (rev v1.31).

**Single-scattering + Henyey-Greenstein rim-light — Pillar 3, CKS-20 (wired & validated
  2026-06-19, rev v1.32).** Cold dust (`ρ_cold`) now scatters, not just absorbs: the beauty
  march adds `σ_s = ϖ·κ` to the extinction (`dτ_ext⃗ = (κ⃗+σ_s)·ρ_cold·ds`) and an in-scatter
  source from the hot inner edge `J_scat·ds = σ_s·ρ_cold·P(cosθ_s)·I_src·e^{−τ_src}·ds`, with a
  forward Henyey-Greenstein phase `P` (`disk.scatter.hg_g`, default 0.6). New `@ti.func`
  `_hg_phase` (GPU) + `renderer.disk.hg_phase` (CPU twin, parity-tested) and `_disk_scatter_cks`
  (returns `vec4(J_rgb, σ_s·ρ_cold·ds)`), both behind a `ti.static(_SCATTER_COMPILE)` gate so
  `disk.scatter.enabled:false` (default) ⇒ the CKS-19 march is **bit-identical**
  (`test_scatter_albedo_zero_identical`, atol 1e-6). `I_src = blackbody_chroma(T_inner)·inner_glow`,
  `T_inner = T_0·(6/r_inner)^0.75` (SIMPLE model regardless of `temperature_model`); straight-CKS-ray
  scattering geometry — no geodesic/Doppler contamination (constraint 3). **Compile:**
  `_disk_scatter_cks` reuses the emission march's `grey_dtau` instead of re-inlining the expensive
  `_disk_density_cks` (the double-inline exploded LLVM compile to 15 h / >50 GB; `σ_s·ρ_cold·ds =
  albedo·grey_dtau` byte-for-byte), and new `render.advanced_optimization`/`cfg_optimization` JIT
  knobs (default `true` = Taichi defaults ⇒ unchanged; `false` skips the super-linear IR/CFG passes
  for tests/look-dev — `showcase_disk.py --fast-compile`; offline cache keys on the flags). **Scene
  finding (empirical):** for the canonical edge-on camera the visible dust is mostly front-lit
  (`cosθ_s<0`), so back-scatter (`g<0`) dominates the *aggregate* rim brightening ≈25× over forward
  and scatter-ON **net-darkens** the frame (σ_s adds opacity; single-scatter re-injects only the
  inner-glow bounce, to single-scatter order) — the forward "silver-lining" is a localized limb, not
  a net-brighten (SKILL.md CKS-20 'Aggregate vs limb' note). *Verified:* `tests/test_disk_scatter.py`
  (6 — HG normalisation/forward-dominance, GPU↔CPU parity, analytic assembly, albedo-0 identity, and
  end-to-end rim-light asserting the *true* observables: rim light on dim edges + directional lobe +
  σ_s removes forward light), `test_gpu_regression.py` goldens unshifted. Plan
  `docs/plans/2026-06-19-pillar3-scattering-implementation.md`; SKILL.md Formula **CKS-20**
  (DESIGN→ACTIVE, rev v1.32). Look-dev `scripts/showcase_disk.py --scatter/--albedo/--hg-g/--inner-glow`.

**Kelvin-Helmholtz edge erosion — Pillar 4, CKS-22 (authored 2026-06-20, rev v1.33).**
  Replaces the clean CKS-12 §3 outer smoothstep rim with a noise-thresholded soft-Heaviside
  clip so the outer edge TEARS into vacuum (fingers/holes) instead of fading smoothly:
  inside the §3 modulation branch of `_disk_density_cks`,
  `win_out → smoothstep(0, w_soft, win_out − τ_KH·N_KH)`. `N_KH ∈ [0,1]` is a high-freq
  simplex (`_kh_field` GPU / `noise.kh_field` CPU twin, seed `NSEED_KH=1009`) advected by the
  SAME §2 dual-phase shear as the density; its φ axis uses the CKS-18 **cylinder embedding**
  `(cos φ, sin φ)·freq_phi` so it is seamless across φ=±π (classic simplex is not
  φ-periodic — constraint 5). The clip multiplies the SHARED `win` before the hot/cold split,
  so under CKS-19 emission AND absorption fray together (silhouette-correct lanes). Interior
  immunity via the load-time clamp `τ_KH ≤ 1−w_soft`; step-cap floor
  `w_soft ≥ max_step_vfrac·σ0·r_outer/edge_softness` (k_soft=1, clamped [0.02,0.5]). New slots
  `_NI_EROS_{EN,STR,FU,FP,FZ,OCT,WSOFT}` (`_NOISE_N` 57→64). **JIT:** the clip is emitted only
  under a `ti.static(_EROS_COMPILE)` gate (like `_MP_COMPILE`/`_SCATTER_COMPILE`), so
  `disk.edge_erosion.enabled:false` (default) ⇒ the original mega-kernel compiles unchanged and
  golden frames are **bit-identical**. REQUIRES `disk.noise.modulation.enabled` (the only
  producer of a soft `win_out`). Config `disk.edge_erosion` (base dials, no CKS-13 change).
  Guards: `tests/test_noise.py` (CPU `kh_field`/`kh_erode_winout` range/seam/clip),
  `tests/test_noise_gpu.py::test_kh_field_gpu_matches_cpu` (twin parity),
  `tests/test_disk_edge_erosion.py` (OFF bit-identity + outer-band tearing), unchanged
  `test_gpu_regression.py`. SKILL.md Formula **CKS-22** (rev v1.33).

**Fractal LOD octave cascade — Pillar 5, CKS-23 (authored 2026-06-20, rev v1.34).**
  Anti-aliases the disk turbulence so it survives a moving/zooming camera (the V4
  prerequisite): each disk sample picks an octave count from its pixel footprint
  `n_oct = clamp(N_max − log₂(ε·d / J₀), N_min, N_max)` (ε = vertical_fov/HEIGHT,
  d = camera→sample distance, `_lod_noct_ti` / `noise.lod_noct`), and the L0/L2/L1-mask
  fBm octaves are weighted by the smooth gate `g_o = clamp(n_oct − o, 0, 1)` — far views
  shed the shimmering sub-pixel octaves, close-ups keep the full stack, and the top partial
  octave crossfades so there is no integer popping. The new gated primitive
  `fbm2_lod_ti` / `noise.fbm2_lod` gates BOTH the fBm numerator AND denominator, so the
  normalization is exact and at `n_oct ≥ octaves` every `g_o = 1` ⇒ the result is `fbm2`
  **byte-for-byte**. `n_oct` is threaded `render_beauty_physics → _disk_emit_cks →
  _disk_density_cks → _disk_blended_m / _disk_cold_mult_from_hot → _disk_noise_m`; every
  hop defaults the arg to the `_LOD_OFF` sentinel (1e9), so the shadow bake and the
  parity-test callers stay exact with no edit. New slots `_NI_LOD_{EN,NMAX,NMIN,J0,EPS}`
  (`_NOISE_N` 64→69). **JIT:** there is **no** `ti.static` recompile gate — the gated fBm
  is bit-exact at the sentinel (×1.0 is exact), so it is always compiled and
  `disk.lod.enabled:false` (default) ⇒ golden frames **bit-identical**. ε is refreshed
  per frame in `render_beauty_frame` from the actual cam fov / render height (one f32
  upload, no re-JIT). Config `disk.lod` (base dials `n_max`/`n_min`/`j0`, no CKS-13 change).
  v1 scope: octaves-only (no `dλ`), isotropic scalar `J`. Guards: `tests/test_noise.py`
  (CPU `fbm2_lod`/`lod_noct`/`lod_octave_weight`: full-octave identity, integer-truncation,
  crossfade, distance clamp), `tests/test_noise_gpu.py` (`fbm2_lod_ti` twin parity +
  LOD-off = `fbm2_ti`), `tests/test_disk_lod.py` (OFF bit-identity + ON re-textures the
  disk-only buffer at a face-on camera, with sampling-only containment), unchanged
  `test_gpu_regression.py`. The cascade *math* (octave-drop = fBm truncation, exact
  renorm, monotone `n_oct`, anti-pop crossfade) is proven by the CPU/GPU twins above, not
  a render-level high-frequency metric: the gated fBm renormalizes (numerator AND
  denominator), so culling the already-sub-pixel octaves re-weights the surviving coarse
  octave UP and does not lower (empirically slightly raises) the rendered disk's
  |Laplacian| — the LOD benefit is anti-aliasing vs a supersampled reference, not raw
  curvature. SKILL.md Formula **CKS-23** (rev v1.34).

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
| **D1** | Decision-B physical disk: Page-Thorne flux profile (replaces simple `(6/r)^0.75`) | ✅ **Resolved (2026-06-12)** — wired behind `disk.temperature_model: page_thorne` (default `simple`); LUT in `src/renderer/disk_flux.py`; GPU guard in `test_gpu_regression.py`. Closed form is `SKILL.md` Formula **CKS-11** (numerically reproduces the conservation-law flux integral; guard `tests/test_disk_flux.py`; SKILL.md rev v1.11). The "wire the LUT" path: CPU-precompute `f_PT(r)` shape LUT → `T_eff=T₀·f_PT^{1/4}`, amplitude ×f_PT → sampled in the disk kernel **behind the config flag** (`T₀` stays the amplitude knob; g⁴-not-g⁸ respected, Formula 9 / CKS-11 Piece 3). Default disk stays Decision-B-simple, so golden frames / pinned GPU regression are unchanged. | Physics |
| **D2** | Disk procedural turbulence: layered noise (Interstellar-base fBm + MRI clump/tear ridged-MF/Voronoi accents) with full Keplerian shear advection, modulating density, temperature, edges, and scale height | **D2.1–D2.4 SHIPPED 2026-06-13; noise ON by default.** Spec: `docs/specs/2026-06-13-disk-noise-turbulence.md`; math: SKILL.md Formula **CKS-12** (rev v1.20; Ω = Formula 3 verbatim, all noise amplitude-only). Wired: noise lib + static density (D2.2) + Keplerian shear advection (D2.3) + T/edge/scale-height modulation (D2.4) — `disk.noise` block (`enabled: true`, `modulation.enabled: true`), `_setup_disk_noise` buffer (`_NOISE_N=43`), GPU `_disk_noise_density_mult`/`_disk_noise_mod_fields` + CPU twins, `thumb.py` look-dev. `disk.noise.enabled: false` is a verified bit-identical branch; GR/calibration guards force noise off so goldens are unshifted. **Remaining: D2.5** (MB `t_disk` jitter, perf pass, noise-on golden, owner sign-off) | Visualization |
| **V2** | Flared 3D volumetric density: radius-varying scale height so the noise stack gets real vertical bulk | ✅ **Resolved (2026-06-14)** — `σ_θ(r)=σ0·(r/r_inner)^β` in the shared `_disk_density_cks` (SKILL.md Formula **CKS-16**, rev v1.24); CKS-13 resolver derives `flare_beta` + `theta_half_bound`; gated by `disk.volumetric.flare.enabled` (default `false` ⇒ bit-identical). GEOMETRY/TEXTURE only. Guard `tests/test_disk_flare.py` (7 CPU + 2 GPU); `test_gpu_regression.py`/`test_disk_step_convergence.py` unchanged. Spec `docs/specs/2026-06-14-V2-flared-3d-density.md`. | Visualization |
| **V2b** | Vertical self-shadow: top/inner gas shadows the midplane (the V1 deep-shadow-map was in-plane/radial only) | ✅ **Resolved (2026-06-14)** — `bake_disk_shadow` generalised from the CKS-15 radial column scan to a **3D inner-edge ray** (SKILL.md Formula **CKS-17**, rev v1.25): ray from the inner edge in the midplane `(u=0,ζ=0)` to the sample `(u_s,φ,ζ_s)`, tilted ρ sample + 3D arc length `ds=√((r·du)²+ΔZ²)`. CKS-15 is the ζ=0 limit (bit-exact). **No new config/field/flag** — same `disk.volumetric.self_shadow.enabled` (default `false` ⇒ goldens bit-identical); only the bake's ray geometry changed. VISUALIZATION/amplitude-only. Guard `tests/test_disk_self_shadow.py` (relational checks carry over; `test_bake_matches_analytic_3d_ray_integral` re-derived to the tilted-ray line integral); `test_gpu_regression.py` unchanged. Spec `docs/specs/2026-06-14-V2-vertical-self-shadow.md`. | Visualization |
| **V3.0** | Curl-flow **domain warp**: divergence-free turbulent eddies the laminar §2 shear can't make (the static half of V3; advection is V3.1) | ✅ **Resolved (2026-06-14)** — in-plane curl warp of the noise coord `(u,φ)` = 2-D curl of an `sfbm3` scalar potential on the `(cosφ,sinφ,u)` cylinder embedding (SKILL.md Formula **CKS-18**, rev v1.26): `δu=+∂ψ/∂φ`, `δφ=−∂ψ/∂u` (central FD); divergence-free + seamless across φ=0 (embedding, not lattice period ⇒ freqs may be any real). Applied at `_disk_noise_m`/`_mod_fbm4` entry on the already-sheared `φ′_k` (material-frame; fixed `curl.seed` ⇒ static, only §2 animates). Consumes the parked V1.5 simplex basis. `noise.py` `curl_warp`/`curl_warp_ti`; `disk.noise.curl` dials through `_setup_disk_noise` (`_NOISE_N` 43→52); **no CKS-13 change**. Gated by `disk.noise.curl.enabled` (default `false`, `amp=0` ⇒ identity ⇒ bit-identical). VISUALIZATION/texturing-only. Guards `tests/test_noise.py` (divergence-free / seamless / determinism / identity / moves-density) + `tests/test_noise_gpu.py` (twin parity); `test_gpu_regression.py` unchanged. Spec `docs/specs/2026-06-14-V3-curl-domain-warp.md`. **Next V-epoch increment: V3.1 curl-flow advection** (animate ψ + dual-phase reset), then 3D-curl / V4 free cam. | Visualization |
| **D3** | Dynamic derived parameters: editing base config (spin, target temperature, disk extent) must rescale every dependent quantity automatically | ✅ **Resolved (2026-06-13)** — `src/renderer/kerr_params.resolve_config` (SKILL.md Formula **CKS-13**, rev v1.14) runs inside every config loader (`taichi_renderer.load_config`, `thumb.py`); derives `r_plus`/`r_isco`/`disk.r_inner`/`disk.T_0` (from new base `disk.target_peak_temperature`) + `disk.dynamics` time mapping (`time_scale`, `shear_period_M` for D2). Desync-prone YAML literals removed. Closed forms (BPT 1972 — exact, beats any LUT; only CKS-11 f_PT needs tabulation), literature anchors pinned in `tests/test_kerr_params.py` (11 tests). Render impact: r_inner 1.182→1.181765 (exact ISCO). ⚠️ The original "GPU regression bit-identical except Doppler Δ5e-6" claim was wrong: re-keying `T_0`→`target_peak_temperature` dropped the simple-model peak T_eff 18,600→5,500 K, moving the disk peak 6.17→14.45 and Doppler ratio 4.32→5.15 — the `test_gpu_regression.py` goldens were re-anchored + made dynamic in `doppler_strength` 2026-06-13 (see the test entry above + SKILL.md v1.16) | Config |
| **P2** | Multi-phase disk media: decouple emission density (`ρ_hot`) from absorption density (`ρ_cold`) so cold dust carves dark **silhouettes** into the glow rather than only dimming it | ✅ **Resolved (2026-06-16; GPU-verified 2026-06-18)** — `_disk_density_cks` returns `vec3(ρ_hot, ρ_cold, temp_factor)`; emission←`ρ_hot`, absorption (`dτ`) + the CKS-15 self-shadow bake←`ρ_cold`; step cap resolves the thinner cold slab (`σ_cold=σ_hot·dust_sigma_frac`). `ρ_cold=exp(clamp(m_cold))`, `m_cold=χ·m_hot+√(1−χ²)·m_dust` (variance-preserving Pearson mix; dust field at `seed+NSEED_DUST=911`). SKILL.md Formula **CKS-19** (DESIGN→ACTIVE, rev v1.31); **chromatic per-channel `dτ⃗` shipped 2026-06-18 (Task 7)** — `κ⃗ = absb_c·extinction_rgb` (`disk.extinction_rgb`, default grey `[1,1,1]` ⇒ bit-identical), vec3 transmittance `T⃗`, `disk_buf` widened (H,W,4)→(H,W,6) so the background reddens through dust; guard `test_disk_multiphase.py::test_chromatic_extinction_reddens`. Config `disk.multiphase` (sibling of `disk.noise`, default `enabled:false` ⇒ `ρ_cold≡ρ_hot` ⇒ legacy march bit-identical). **JIT:** dust branch emitted only when enabled (`ti.static` `_MP_COMPILE` gate) ⇒ toggling `enabled` forces a one-time recompile; OFF default keeps the original fast JIT (a runtime `if` blew compile past 2 h). New slots `_NI_MP_{EN,CHI,AMP,SIGFRAC}` (`_NOISE_N` 53→57). Guards: `test_noise.py` (CPU correlation/variance parity), `test_disk_noise.py::{test_rho_cold_gpu_matches_cpu,test_multiphase_off_bit_identical}`, `test_disk_multiphase.py::test_dust_carves_silhouette`, unchanged `test_gpu_regression.py`. Plan `docs/plans/2026-06-16-pillar2-multiphase-implementation.md`. | Visualization/Physics |
| **P3** | Volumetric single-scattering + Henyey-Greenstein rim-light: cold dust catches forward-scattered inner-edge light (the "silver-lining") instead of only absorbing | ✅ **Resolved (2026-06-19)** — `σ_s = ϖ·κ` added to the extinction + single-scatter source `J_scat = σ_s·ρ_cold·P(cosθ_s)·I_src·e^{−τ_src}` from the hot inner edge, forward HG phase (`disk.scatter.hg_g`, default 0.6). SKILL.md Formula **CKS-20** (DESIGN→ACTIVE, rev v1.32); `_hg_phase`/`_disk_scatter_cks` behind the `ti.static` `_SCATTER_COMPILE` gate; `disk.scatter` config (default `enabled:false` ⇒ CKS-19 march bit-identical). **Compile:** scatter reuses the march's `grey_dtau` (no `_disk_density_cks` double-inline) + new `render.advanced_optimization`/`cfg_optimization` JIT knobs (default true; `--fast-compile` for tests/look-dev). **Empirical scene note:** the canonical edge-on camera is back-scatter-dominated (forward silver-lining is a localized limb; scatter-ON net-darkens) — `test_scatter_rim_light` asserts the true observables, not a net-brighten. Guards `tests/test_disk_scatter.py` (6), `test_gpu_regression.py` unshifted. Plan `docs/plans/2026-06-19-pillar3-scattering-implementation.md`. | Visualization/Physics |
| **P4** | Kelvin-Helmholtz edge erosion: shred the clean outer rim into vacuum (fingers/holes) instead of a smooth falloff | ✅ **Resolved (2026-06-20)** — inside the §3 modulation branch of `_disk_density_cks`, `win_out → smoothstep(0, w_soft, win_out − τ_KH·N_KH)`; `N_KH` = high-freq simplex (`_kh_field` GPU / `noise.kh_field` CPU, seed `NSEED_KH=1009`) §2-advected, φ via the CKS-18 cylinder embedding (seamless, constraint 5). Clips the SHARED `win` ⇒ with CKS-19 emission & absorption fray together. Interior immunity (`τ_KH ≤ 1−w_soft` clamp), step-cap floor on `w_soft` (k_soft=1). New slots `_NI_EROS_{EN,STR,FU,FP,FZ,OCT,WSOFT}` (`_NOISE_N` 57→64). **JIT:** clip emitted only under `ti.static(_EROS_COMPILE)` ⇒ `disk.edge_erosion.enabled:false` (default) keeps the original JIT + **bit-identical** goldens. REQUIRES `disk.noise.modulation.enabled`. SKILL.md Formula **CKS-22** (rev v1.33). Guards `tests/test_noise.py` (CPU twins), `tests/test_noise_gpu.py::test_kh_field_gpu_matches_cpu`, `tests/test_disk_edge_erosion.py` (OFF bit-identity + tearing), unchanged `test_gpu_regression.py`. Plan `docs/specs/2026-06-20-P4-P5-edge-erosion-lod-cascade-plan.md`. | Visualization |
| **P5** | Fractal LOD octave cascade: anti-alias the disk turbulence so it survives a moving/zooming camera (the V4 free-camera prerequisite) | ✅ **Resolved (2026-06-20)** — per disk sample `n_oct = clamp(N_max − log₂(ε·d / J₀), N_min, N_max)` (ε = vertical_fov/HEIGHT, d = camera distance; `_lod_noct_ti` / `noise.lod_noct`) gates the L0/L2/L1-mask fBm octaves by `g_o = clamp(n_oct − o, 0, 1)` via the new `fbm2_lod_ti` / `noise.fbm2_lod` (gates BOTH numerator and denominator ⇒ exact renorm; top octave crossfades ⇒ no integer popping). `n_oct` threaded `render_beauty_physics → _disk_emit_cks → _disk_density_cks → _disk_blended_m/_disk_cold_mult_from_hot → _disk_noise_m`, every hop defaulting to the `_LOD_OFF` sentinel (1e9) ⇒ shadow bake + parity-test callers exact with no edit. New slots `_NI_LOD_{EN,NMAX,NMIN,J0,EPS}` (`_NOISE_N` 64→69). **JIT:** NO `ti.static` recompile gate — the gated fBm is bit-exact at the sentinel (×1.0 is exact), always compiled ⇒ `disk.lod.enabled:false` (default) keeps goldens **bit-identical**; ε refreshed per frame in `render_beauty_frame` (one f32 upload, no re-JIT). Config `disk.lod` (base dials, no CKS-13 change); v1 scope octaves-only / isotropic scalar `J`. SKILL.md Formula **CKS-23** (rev v1.34). Guards `tests/test_noise.py` (CPU `fbm2_lod`/`lod_noct`/`lod_octave_weight`), `tests/test_noise_gpu.py` (`fbm2_lod_ti` twin + LOD-off = `fbm2_ti`), `tests/test_disk_lod.py` (OFF bit-identity + octave dropping), unchanged `test_gpu_regression.py`. Plan `docs/specs/2026-06-20-P4-P5-edge-erosion-lod-cascade-plan.md`. | Sampling |

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
  **SUPERSEDED (2026-06-07):** the "diffuse map omits stars brighter than ~8.0"
  premise was **measured false** — `milkyway_2020_16k` carried the full baked star
  field (sharp-spike density identical to `starmap_2020`). That field is what EWA
  smeared into *edge* streaks at wide FOV (distinct from the photon-ring smear S6
  fixed). Resolved by swapping Layer B to the StarNet2-starless `starmap_final.exr`
  and, since Layer B is now starless, raising `mag_limit 8.0 → 11.0` so Layer A
  alone carries the star field (`assets/stars.npy`: 41 410 → 115 324). See §6
  2026-06-07.
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
  mag_limit: 11.0            # Layer A catalog depth (~Tycho-2); was 8.0 — see §6 2026-06-07
  mag_zero_point: 3.5        # Layer-A star flux scale (INGEST-time → re-ingest after change); see §6
  diffuse_map: star_image/starmap_final.exr   # Layer B only — STARLESS (low-freq); was milkyway_2020_16k.exr
  diffuse_gain: 3.0          # Layer-B band brightness × (RENDER-time, no re-ingest); 1.0 = raw EXR
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
| 4 | Layer B anisotropic EWA diffuse fetch | ✅ **shipped** — `starfield.diffuse_map` (STARLESS `starmap_final.exr` since 2026-06-07; was `milkyway_2020_16k.exr`), EWA major-axis taps; render-time `diffuse_gain` brightness × (2026-06-07) |
| 5 | Config gate, A/B harness, validation suite | ✅ **shipped** — `mode` gate + `tests/test_starfield_dngr.py`; Layer-B starless gate = `scripts/check_starless_map.py` |
| 6 (opt) | Geodesic-deviation Jacobian upgrade | **open** — future fidelity upgrade (new ODE) |

**Decisions of record (resolved 2026-06-05):** catalog scope = **HYG/ATHYG v3.2** (the
owner-supplied csv; BSC5 still supported) with **blackbody-from-B−V** colour; Jacobian
method = **finite-difference** (FD-first, per the recommendation; geodesic deviation
deferred to Phase 6); diffuse map = **a STARLESS Milky-Way equirect** with anisotropic
EWA — originally `milkyway_2020_16k.exr`, **superseded 2026-06-07** by `starmap_final.exr`
(StarNet2-cleaned) because the milkyway plate still carried the full baked point-star
field, which EWA smears into edge streaks (§6 2026-06-07). **Still open:** Phase 6 geodesic-
deviation Jacobian (gated on a new SKILL.md deviation-ODE formula); a quantitative
Einstein-ring brightness regression (the current suite asserts finiteness + point-star
sharpness + flat-space μ→1, not ring photometry); **Layer-B/Layer-A brightness grade** —
the starless plate halves the diffuse mean, so the background reads dim (owner `--no-disk`
test confirmed); tune `starfield.mag_zero_point` (currently 0.0) against a graded BH frame.

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
- **`research/accretion-disk/`** — scouted + (sub-domain B) validated source corpus
  for the disk-turbulence work (§10): `sources-procedural-noise.md` +
  `validation-procedural-noise.md`. Validator caveat: only VERIFIED-rated excerpts
  are trustworthy (several sources carry scout paraphrases presented as quotes).

---

## 10. Accretion-disk procedural turbulence (PLANNED — D2)

**Status: design approved 2026-06-13; docs landed (SKILL.md CKS-12, spec below).
The full D2 turbulence backlog (D2.1 primitives → D2.2 static density → D2.3 Keplerian
shear advection → D2.4 temperature/edge/scale-height modulation) shipped 2026-06-13
and procedural noise is now ON by default in `configs/render.yaml`
(`disk.noise.enabled: true`, `modulation.enabled: true`).** `src/renderer/noise.py`
(primitives + `@ti.func` twins + the combined `noise_density_mult` stack with the §2
dual-phase reset blend + the §3 `noise_modulation_fields` envelopes), wired into the GPU
beauty path (`disk.noise` config block, `_setup_disk_noise` param buffer now
`_NOISE_N=43`, `_disk_noise_density_mult` + `_disk_noise_mod_fields` kernels,
`_smoothstep_ti` edge windows) and the CPU `thumb.py` look-dev path. The renderer's
first time variable `t_disk = frame/fps·time_scale` is threaded through
`render_beauty_frame{,_mb}` → `_disk_emit_cks`, `export_exr.py`, and
`thumb.py --frame/--t-disk`; Ω is Formula 3 per disk sample. **D2.4** modulates emission
*amplitudes* with four advected [0,1] fBm envelopes (`n_T, n_e_in, n_e_out, n_h`): emitted
temperature pre-g-shift (constraint 2), inner/outer edges via smoothstep windows with
`r_in_eff ≥ r_isco` (constraint 3), and a lumpy scale height σ_θ whose worst-case
`σ_z·(1−h/2)` re-arms the Pipe-B step cap (constraint 4). `disk.noise.enabled: false`
(and `modulation.enabled: false`) keep the legacy bit-identical branch (constraint 6);
the GR/calibration guards (`test_gpu_regression.py`, base `test_disk_step_convergence`)
force noise OFF so the global enable does not shift the pinned goldens. Tests:
`test_noise.py` (31 CPU, incl. §2 advection + §3 modulation: disabled-is-½ / unit-range /
decorrelation / advect+determinism), `test_noise_gpu.py` (10 CUDA), `test_disk_noise.py`
(8 CUDA: bit-identity, enabled-changes-disk, determinism, seed, static + advected GPU↔CPU
stack agreement, dynamism-gain, **mod-fields GPU↔CPU**), `test_disk_step_convergence.py`
(2 CUDA: smooth-slab + §3 lumpy-slab constraint-4 cap). A non-physical
`disk.noise.dynamism` viz gain (default 1.0 = bit-identical) scales the shear amount
(`φ′=φ−dynamism·Ω·a·T`) to emphasise the per-frame swirl. **D2 turbulence complete;
remaining D2.5 = MB `t_disk` jitter / perf pass / noise-on golden / owner sign-off.**
Backlog row **D2** (§7).

**Spec of record:** `docs/specs/2026-06-13-disk-noise-turbulence.md` — layer stack,
noise primitives, config draft, perf budget, test plan.
**Math of record:** SKILL.md **Formula CKS-12** — noise coordinates, shear
advection, modulation bookkeeping (hard constraints 1–7).

### What it is

The disk's radial-only profile (rings) gains fluid-like structure from **layered
procedural noise evaluated in-kernel** at disk-natural coordinates
`(u = ln r/r_inner, φ = atan2(y,x), ζ = Δθ/σ_θ)` and **advected by Keplerian shear**
`φ′ = φ − Ω(r)·t_disk` (Ω = Formula 3 verbatim; dual-phase reset blend caps the
shear stretching). Three layers: L0 anisotropic fBm (Interstellar filaments along
orbits), L1 ridged-multifractal × Voronoi-billow under a slow coverage mask (MRI
clumping/tearing), L2 low-frequency fBm (large-scale patchiness). Noise multiplies
**amplitudes only** — volumetric density (emission *and* absorption → clumps
self-shadow), emitted temperature (pre-g-shift, ≤ ~±15%), smoothstep edge windows
(`r_in_eff ≥ r_isco` always), and scale height σ_θ. The geodesic path, g-factor,
g⁴ bookkeeping, and f_PT shape are untouched.

### Key plumbing facts (why this isn't a pure-kernel change)

- The renderer gets its **first time variable**: `render_beauty_frame(..., t_disk)`
  with `t_disk = frame_index / render.fps × disk.dynamics.time_scale`. Since D3
  (2026-06-13) `time_scale` and the reset period `shear_period_M` are **derived**
  (CKS-13) from the base look targets `disk.dynamics.inner_lap_seconds` /
  `shear_wrap_budget` — they rescale automatically with spin and disk extent, so
  the old open shot-length question reduces to picking `inner_lap_seconds`.
- **Motion blur:** `render_beauty_frame_mb` must jitter `t_disk` across the shutter
  alongside the camera `dphi` jitter — the rotate-the-camera trick alone is only
  valid while the disk is axisymmetric.
- **No re-JIT look-dev:** layer parameters go in a small uploaded `ti.field` param
  buffer (`_setup_disk_noise`, following the D1 `_setup_disk_flux` / ti.init
  re-setup pattern); `t_disk`/`enabled`/seed are kernel args.
- **Step-cap interaction:** the CKS-5 vertical step cap must use the worst-case
  modulated scale height `σ_z·(1 − h_amp/2)` or the face-on moiré returns.

### Build order (each phase lands with tests + docs sync)

| Phase | Scope | Gate |
|---|---|---|
| D2.0 | Docs (CKS-12, spec, this section) | ✅ done 2026-06-13 |
| D2.1 | `src/renderer/noise.py` CPU reference + `@ti.func` twins + `tests/test_noise.py` (CPU) + `tests/test_noise_gpu.py` (CPU↔GPU agreement, φ-periodicity, determinism) | ✅ done 2026-06-13 — no renderer change; 26 tests green |
| D2.2 | Static structure: `(u,φ,ζ)` mapping + 3 layers on density only at `t_disk=0`; `disk.noise` config block; off-branch bit-identity vs goldens; **thumb.py look-dev loop** | ✅ done 2026-06-13 — GPU `_disk_noise_density_mult` (reads `_setup_disk_noise` param buffer) + CPU `noise.noise_density_mult` twin; `test_disk_noise.py` (bit-identity / determinism / GPU↔CPU stack agreement). Perf noise-on ≈ 2.66× off at 960×540 (above the 2× target but sub-100 ms; offline render, tunable via config-only octave dials — defaults are look-dev placeholders) |
| D2.3 | Shear advection: `t_disk` plumbing, dual-phase reset + per-cycle reseed, temporal-continuity test | ✅ done 2026-06-13 — `t_disk=frame/fps·time_scale` through `render_beauty_frame{,_mb}`/`export_exr.py`/`thumb.py`; §2 blend wraps the `m`-stack in `noise_density_mult` + GPU `_disk_noise_density_mult` (Ω = Formula 3/sample); `variance_preserve` dial; `shear_period≤0` ⇒ static D2.2 bit-identical. Tests: `test_noise.py §2` (5: static fallback/evolution/determinism/reset-continuity/variance-preserve) + `test_disk_noise.py::test_advected_stack_matches_cpu_reference` |
| D2.4 | Temperature + edges + scale height (incl. step-cap σ_z fix, extended convergence test) | ✅ done 2026-06-13 — four advected [0,1] fBm envelopes (CPU `noise_modulation_fields` + GPU `_disk_noise_mod_fields` vec4 twin, seed offsets 503/601/701/809, no variance-preserve); `T_emit·(1+τ(n_T−½))` pre-g, smoothstep edges (`r_in_eff≥r_isco`), σ_θ·(1+h(n_h−½)) with worst-case σ_z step cap; `_NOISE_N` 32→43; **noise enabled globally** in `render.yaml` (guards force it off). Tests: `test_noise.py §3` (4) + `test_disk_noise.py::test_mod_fields_match_cpu_reference` + `test_disk_step_convergence.py::test_disk_emission_resolves_lumpy_slab` (≤0.06) |
| D2.5 | MB `t_disk` jitter, perf pass, noise-on golden (fixed seed/t), docs/memory sync | owner sign-off; `enabled` flipped to `true` 2026-06-13 per owner ("apply it globally") |

Known confound: the pre-existing `doppler_strength=0.1` regression failure
(2026-06-12) predates D2 — do not attribute it to this work.

---

## 11. Volumetric disk: self-shadow + source function (V epoch)

**Status: V1 (the 2.5D-slab phase) shipped 2026-06-13 — V1.0 → V1.4 all green;
V1.5 isotropic simplex basis shipped 2026-06-14 (library only, not wired).**
The goal is the Interstellar close-up accretion-gas look — **glowing turbulent gas
with deep black voids** (`whatiwant1.png`/`whatiwant2.png`) — which config alone
cannot reach. Two flag-gated radiative-transfer terms, both **bit-identical to the
D2 goldens when off**, supply the missing physics; the void look needs **both
together** (CKS-14 materialises the source function `S`; CKS-15 dims it to carve the
voids — neither alone suffices).

**Math of record:** SKILL.md **Formula CKS-14** (volumetric RTE source-function march,
NO new GR — assembled from CKS-9 g / CKS-11 f_PT / Formula-9 chroma·g⁴) and **Formula
CKS-15** (radial deep-shadow-map self-shadow — VISUALIZATION, amplitude-only, flagged
like `doppler_strength`).
**Spec of record:** `docs/specs/2026-06-13-V1-self-shadow-source-function.md` (parent
roadmap `2026-06-13-volumetric-disk-and-gas-flow.md`).

### Key facts (the corrected motivation)

- **CKS-14 is quadrature, not a brightness fix.** The legacy emission march and the
  CKS-14 source-function march integrate the **same** continuum integral
  `I = ∫ S e^{−τ} dτ` (since `transm·j·ds = transm·S·dτ`); they differ only in
  *quadrature*. Legacy over-counts thick steps by `dτ/(1−e^{−dτ})`, so CKS-14 is
  *dimmer and more accurate* in the thick regime — its standalone value is removing
  that over-count and **materialising `S`** for the self-shadow to attenuate.
- **CKS-15 dims emissivity only** (`j → j·e^{−strength·τ_s}`); the absorption `κ`/`dτ`
  is untouched (the gas still occludes regardless of how lit it is). The shadow ray
  is a straight radial CKS line (not a geodesic) — a viz approximation, like
  `doppler_strength`.
- **Shared density** `@ti.func _disk_density_cks` (V1.0) is the single source for both
  the emission march and the shadow bake, so the two can never drift.

### Build order (each phase landed with tests + docs sync)

| Phase | Scope | Gate |
|---|---|---|
| V1.0 | Extract `_disk_density_cks` (shared density) | ✅ done 2026-06-13 — bit-identical refactor; 14/14 GPU tests |
| V1.1 | CKS-14 source-function march behind `disk.volumetric.source_function` | ✅ done 2026-06-13 — march-loop only (`_RTE_TAU_EPS` guard); `test_disk_source_function.py` (3) + flag-off goldens bit-identical |
| V1.2 | CKS-15 radial deep-shadow-map behind `disk.volumetric.self_shadow.enabled` | ✅ done 2026-06-13 — `disk_shadow_tau` field + `bake_disk_shadow` (radial scan off the shared density) + `_sample_shadow_tau` (trilinear, φ-periodic); `test_disk_self_shadow.py` incl. GPU bake vs an analytic Gaussian column |
| V1.3 | `showcase_disk.py` look-dev flags + LOD gate | ✅ done 2026-06-13 — `--source-function`/`--self-shadow`/`--shadow-strength`/`--shadow-grid`; `self_shadow.lod_max_camera_radius` skips the bake for wide/mid shots (0 = off, default); `test_lod_gate_drops_self_shadow_for_distant_camera` |
| V1.4 | Docs sync (SKILL CKS-14/15, this §, render.yaml, memory) + combined golden | ✅ done 2026-06-13 — `test_combined_source_function_and_self_shadow_golden` (relational: composition active, NaN-free, dimmer than source-fn-only) |
| V1.5 | Isotropic **simplex** basis `snoise2/3` + `sfbm2/3` (+ GPU twins) for the V3 curl potential | ✅ done 2026-06-14 — Perlin/Gustavson skewed-simplex (SKILL §3.6 / v1.23); library addition only, NOT wired (classic simplex isn't φ-periodic), so all goldens bit-identical; `test_noise.py` (8 CPU incl. the m=4 isotropy guard: ~12× less axis bias than Perlin) + `test_noise_gpu.py` (4 CUDA twin-parity) |

**Out of V1 scope (the V2–V5 roadmap):** ~~V1.5 Simplex noise~~ ✅ shipped 2026-06-14;
~~V2 3D flared density~~ ✅ shipped 2026-06-14 (below); ~~vertical self-shadowing (top gas
shadowing the midplane)~~ ✅ shipped 2026-06-14 (Formula CKS-17, below — the V1
deep-shadow-map was in-plane/radial only; CKS-17 generalises it to a 3D inner-edge ray now
that V2 supplies the bulk); V3 curl-flow advection + domain warp (consumes the V1.5 simplex
basis); V4 free camera.

### V2 — flared 3D volumetric density (shipped 2026-06-14)

**Status: shipped 2026-06-14, default OFF (bit-identical to V1).** Gives the disk real
vertical bulk so the noise stack's `ζ` coordinate stops being squashed by the constant
thin slab. **Math of record:** SKILL.md **Formula CKS-16** (flared scale height
`σ_θ(r) = σ0·(r/r_inner)^β`, GEOMETRY/TEXTURE — amplitude/geometry only, no GR) + the
**CKS-13 θ-band addendum** (derived `theta_half_bound`). **Spec of record:**
`docs/specs/2026-06-14-V2-flared-3d-density.md`.

- **One-point change.** The flare lives in the shared `_disk_density_cks`, so the
  emission march and the CKS-15 shadow bake inherit `σ_θ(r)` automatically; `β=0` skips
  `ti.pow` ⇒ the V1 constant slab bit-for-bit.
- **Genuine 3D for free.** The existing `ridged3`/`fbm3` stack already consumes
  `ζ=dz_ang/σ_eff`; a real radius-varying thickness un-squashes it — no new noise
  primitive (V1.5 simplex stays parked for V3).
- **Knock-on A — θ band.** The CKS-13 resolver derives a separate
  `theta_half_bound ≥ band_sigma·σ_θ(r_outer)` (default `band_sigma=3.0`) as the photon
  trace band, leaving `theta_half_width` as the un-mutated σ0 anchor (⇒ idempotent).
- **Knock-on B — step cap.** Unchanged: flare thickens *outward*, so the thinnest slab is
  still the inner edge σ0, the existing worst case — verified by the convergence test, not
  assumed.
- **Gating.** `disk.volumetric.flare.{enabled,beta,band_sigma}`; default `enabled:false`
  (and `enabled:true,β=0`) are both bit-identical, so every V1/D2 golden stays green.
- **Guards.** `tests/test_disk_flare.py` (7 resolver/CPU no-op/widen/monotone/idempotent/
  validation + 2 GPU flag-off bit-identity & β>0 thickens the silhouette — flare is
  geometry, *not* a brightness boost: it read ~1.6% dimmer in total because the added
  cold outer bulk self-absorbs the unchanged hot inner edge more than it emits) + the unchanged
  `test_gpu_regression.py` / `test_disk_step_convergence.py`.

**Out of V2 scope (one-variable-at-a-time):** ~~vertical self-shadow (CKS-15 extension)~~
✅ shipped 2026-06-14 (Formula CKS-17, below); a dedicated V2 volumetric golden (deferred to
V5 sign-off); ~~V3 curl-flow~~ → V3.0 static curl domain warp ✅ shipped 2026-06-14 (Formula
CKS-18, below); V3.1 curl-flow advection next.

### Vertical self-shadow — 3D inner-edge ray (shipped 2026-06-14, Formula CKS-17)

**Status: shipped 2026-06-14, default OFF (bit-identical to V1.2).** Generalises the
CKS-15 radial deep-shadow-map to a **3D inner-edge shadow ray**, so an off-midplane parcel
is shadowed by the dense midplane gas between it and the hot inner edge — the *vertical*
self-shadow the V2 3D bulk makes physical. **Math of record:** SKILL.md **Formula CKS-17**
(VISUALIZATION, same governance class as CKS-15 — straight CKS ray, single inner-edge
illuminator, single-scatter, amplitude-only). **Spec of record:**
`docs/specs/2026-06-14-V2-vertical-self-shadow.md`.

- **The ray.** From the illuminator at the inner edge **in the midplane** `(u=0,ζ=0)` to
  the sample `(u_s,φ,ζ_s)` at fixed φ, with `ζ(u)=(u/u_s)·ζ_s`. The bake accumulates
  `Σ_{j<i_u} absb_c·ρ_j·ds_j` with the **tilted** sample `ρ_j=ρ(u_j,φ,ζ_j)` and 3D arc
  length `ds_j=√((r_j·du)²+ΔZ_j²)`, `ΔZ_j` from `Z(u)=r·ζ(u)·σ_θ(r)` (CKS-16 flared σ).
- **CKS-15 is the ζ=0 limit.** On the midplane the ray is flat (`ΔZ≡0 ⇒ ds=r·du`, ρ at the
  midplane) ⇒ the old radial column term-for-term. The radial element keeps `dr=r·du`
  precisely so this reduction is bit-exact; CKS-15 is not a separate code path.
- **One-point change.** Only `bake_disk_shadow`'s ray geometry changed — the
  `disk_shadow_tau` field, the `_sample_shadow_tau` trilinear lookup, and the
  `emission *= exp(−strength·τ_s)` application (emissivity only; κ/dτ untouched; composes
  with CKS-14) are all unchanged. **No new config, field, or flag** — same
  `disk.volumetric.self_shadow.enabled` (default `false` ⇒ no bake, goldens bit-identical).
- **Cost.** Each target ζ_s tilts its own ray (no prefix sum) ⇒ `O(NU²·NPHI·NZ)`, ~NU/2× the
  CKS-15 evals, parallel over all cells — accepted for the offline bake.
- **Guards.** `tests/test_disk_self_shadow.py` — flag-off bit-identity, outward-steepening
  dimming, and noise-on contrast-rise carry over unchanged; `test_bake_matches_analytic_
  3d_ray_integral` re-derives the analytic column to the 3D-ray line integral (the
  constant-ζ radial closed form was the CKS-15 model, superseded off-midplane). Unchanged
  `test_gpu_regression.py`. SKILL.md rev v1.25.

### Curl-flow domain warp — V3.0 static (shipped 2026-06-14, Formula CKS-18)

**Status: shipped 2026-06-14, default OFF (bit-identical to V2).** Adds divergence-free
turbulent structure (eddies/billows) that the laminar §2 Keplerian shear cannot produce,
by warping the noise **sample coordinate** `(u,φ)` with the 2-D curl of a scalar potential
on the **V1.5 isotropic simplex basis** (finally consumed). **Math of record:** SKILL.md
**Formula CKS-18** (VISUALIZATION, same governance class as CKS-12 §2/§3 — texturing only,
never `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT`). **Spec of record:**
`docs/specs/2026-06-14-V3-curl-domain-warp.md`. **Owner decisions (2026-06-14):** stage
*static warp (V3.0) → curl-flow advection (V3.1)* (the D2.2→D2.3 split); V3.0 displacement
is *in-plane `(u,φ)`* only (ζ untouched).

- **The warp.** `ψ(u,φ) = sfbm3(cosφ·ρ_c, sinφ·ρ_c, u·k_u)` on the cylinder embedding;
  `δu=+∂ψ/∂φ`, `δφ=−∂ψ/∂u` (central finite difference — simplex has no analytic gradient
  here); `u'=u+A·δu`, `φ'=φ+A·δφ`.
- **Divergence-free** by construction (the curl of a scalar ⇒ the incompressible-flow
  look). **Seamless across φ=0**: `δu`/`δφ` are built on `cos φ`/`sin φ` ⇒ exactly
  2π-periodic, so CKS-12 constraint 5 holds even though classic simplex is not
  lattice-periodic — seamlessness comes from the *embedding*, not a lattice period, so
  `ρ_c`/`k_u` may be any real (no integer-period restriction the density stack has).
- **Material-frame integration.** Applied at the entry of `_disk_noise_m` (density) and
  `_mod_fbm4` (§3 envelopes) on the already-sheared per-phase `φ′_k`, so eddies freeze into
  the gas and the §2 shear winds them into filaments; density + modulation share one warp ⇒
  swirl coherently. A **fixed `curl.seed`** (not the per-cycle reseed) keeps V3.0 genuinely
  static — only the §2 winding animates over `t_disk`. V3.1 makes ψ itself time-dependent.
- **Surface.** `noise.py` `curl_warp` (CPU source of truth) + `curl_warp_ti` (`@ti.func`
  twin); `taichi_renderer.py` warp at the two stack entries + `disk.noise.curl` dials
  through `_setup_disk_noise` (`_NOISE_N` 43→52). **No CKS-13 resolver change** — all base
  look dials, nothing derived. Cost: central-diff gradient = 4 `sfbm3` evals × octaves per
  phase, for density AND modulation (offline-acceptable; analytic-gradient / 3-eval forward
  diff deferred).
- **Gating.** `disk.noise.curl.enabled` (default `false`, and `amp=0` ⇒ identity) ⇒
  bit-identical to V2; every V1/V2/D2 golden stays green.
- **Guards.** `tests/test_noise.py` (divergence-free via the curl-vs-gradient divergence
  ratio; 2π-periodic seamlessness; determinism/seed-sensitivity; zero-amp identity;
  moves-the-density-stack + enabled:false bit-identity) + `tests/test_noise_gpu.py`
  (`curl_warp_ti` twin parity to derived `amp·_SATOL/fd_eps` — derivative amplifies the
  ~1e-5 `sfbm3` twin gap ×1/(2ε), obs ~6.5e-5); unchanged `test_gpu_regression.py`.
  GPU-verified 2026-06-14 (noise_gpu 15, disk_noise + gpu_regression). SKILL.md rev v1.26.
- **Followed by:** V3.1 curl-flow advection (animate ψ — shipped 2026-06-15, below); then
  3D curl (vector potential, churn ζ); analytic simplex gradient; a curl-on golden
  (V5 sign-off).

### Curl-flow advection — V3.1 (shipped 2026-06-15, Formula CKS-18 §2)

**Status: shipped 2026-06-15, default OFF (bit-identical to V3.0).** Makes the V3.0 curl
potential `ψ` **time-dependent** so the eddies *boil* (form/stretch/merge) over `t_disk`,
composed additively on top of the §2 Keplerian shear (§2 = laminar bulk winding, V3.1 =
in-place turbulent evolution). **Math of record:** SKILL.md **Formula CKS-18 §2** (same
VISUALIZATION class as §1). **Spec of record:** `docs/specs/2026-06-14-V3.1-curl-flow-advection.md`.
**Owner decisions (2026-06-14): Option A** (reuse the CKS-12 §2 dual-phase reset blend,
applied to ψ) + **clock B1** (new independent dial `flow_period_M`, no CKS-13 change).

- **The mechanism.** `ψ = ω_0·ψ_0 + ω_1·ψ_1` with triangle weights `ω_k = 1−|2α_k−1|`,
  `α_k = frac(s_c + k/2)`, `s_c = t_disk/T_c`; each phase reseeded
  `seed + k·NCYC_PHASE + γ_k·NCYC_CYCLE` (`γ_k = floor(s_c + k/2)`) — reusing the §2 reseed
  strides. The central difference runs over the blended ψ.
- **All three V3.0 invariants survive.** Divergence-free (curl is linear ⇒ a convex
  combination of div-free fields is div-free); seamless in φ (per-phase, from the cylinder
  embedding); **C0-continuous through reseeds** (`ω_k → 0` exactly at each reset — the §2
  property, now on the time axis).
- **Clock B1.** `T_c = disk.noise.curl.flow_period_M` is a NEW independent base dial, NOT
  derived from `disk.dynamics.shear_period_M` (eddy turnover and bulk winding are separate
  timescales) ⇒ **no CKS-13 resolver change**. Decoupled from the shear, so the curl can
  boil even on the static-shear path.
- **No `flow_dynamism` (flagged, deferred).** The §2 `dynamism` scales a *continuous* shear
  displacement independently of the cadence; a pure reset-blend has no such separable
  displacement (its only rate `1/T_c` sets boil speed and reseed cadence together), so a
  C0-safe boil-rate-vs-cadence dial is not possible in Option A — it needs the Option-B
  continuous 4-D time axis. Surfaced rather than shipping a dial that doesn't match its
  §2 namesake.
- **Surface.** `noise.py` `curl_warp`/`curl_warp_ti` gained `(t_disk, flow_period)` +
  `_curl_psi_ti` (the dual-phase potential); `t_disk` threaded through `_disk_noise_m` /
  `_mod_fbm4` → `_disk_curl_warp`; new `_NI_CURL_FLOWP` slot (`_NOISE_N` 52→53) in
  `_setup_disk_noise`. Cost: 2× the §1 curl evals at `T_c>0` (the two phases), offline-OK.
- **Gating.** `flow_period_M ≤ 0` (default / absent) ⇒ the V3.0 static warp **bit-for-bit**
  (the regression hook; mirror of §2's `shear_period ≤ 0`). Every V1/V2/V3.0 golden stays
  green.
- **Guards.** `tests/test_noise.py` §2 (divergence-free at several `t`, seamless at each `t`,
  C0-continuity through resets via a no-spike-vs-median step sweep, static-fallback
  bit-identity, evolution+determinism) + `tests/test_noise_gpu.py`
  (`curl_flow_twin_matches_reference` at several `t` to the derived `amp·_SATOL/fd_eps`
  bound) + `tests/test_disk_noise.py` (`curl_flow_advection_matches_cpu_and_animates` —
  end-to-end `t_disk → _disk_curl_warp` threading + actually animates); unchanged
  `test_gpu_regression.py` (default-off ⇒ goldens bit-identical). SKILL.md rev v1.28.
- **Deferred:** 3D curl (vector potential, churn ζ); analytic simplex gradient; the
  Option-B 4-D `sfbm4` basis (would re-enable a true `flow_dynamism`); a curl-on golden
  (V5 sign-off).

### Look pipeline — warm-amber color grade + render-path ergonomics (shipped 2026-06-14)

**Status: shipped 2026-06-14, identity-default (bit-identical to before).** Two
look-development additions in service of the *Interstellar* reference look
(`whatiwant1/2.png`, `wantblackhole.png`) — a warm-amber gas with organic color and
switchable structure cost. Neither touches GR/physics.

- **Non-physical color grade (the amber).** `renderer.tonemap` gained optional
  `saturation` + `tint` (per-channel linear gain), applied in **linear HDR before** the
  Reinhard compressor. This is a **VISUALIZATION-class dial, same governance as
  `disk.doppler_strength`** — it reaches the warm amber the **Formula 9** blackbody
  chromaticity (deliberately desaturated: `1−exp(−T/{3500,5500,9500})` keeps a strong blue
  channel ⇒ trends sepia) cannot reach on its own. Config: `render.color_grade.{saturation,
  tint}` (identity `1.0` / `[1,1,1]` ⇒ tonemap bit-for-bit the ungraded result, guarded).
  `scripts/showcase_disk.py`: `--saturation`, `--tint r,g,b`, and the `--amber` preset
  (`saturation 1.6`, `tint 1.18,1.0,0.74`). **Why a grade, not a chroma change:** the owner
  chose the viz dial over revising Formula 9 (which would shift goldens) — color is
  art-directed downstream, the rendered radiance is untouched.
- **`--peak-temp` look dial.** `showcase_disk.py --peak-temp K` re-anchors disk hue by
  overriding `disk.target_peak_temperature` and **re-running the CKS-13 resolver** (drops the
  derived `T_0` first — never edits a derived literal). Lower ⇒ the working band sits in the
  saturated-amber part of the chromaticity curve.
- **`--quality` render-path preset (fast↔volumetric switch + self-shadow cost knob).**
  `showcase_disk.py --quality {fast,balanced,hero}`. `fast` = noise+bloom only (~12s/4K, the
  **video path**). `balanced`/`hero` = volumetric void-carving (CKS-14 source function +
  CKS-17 3D self-shadow) at deep-shadow-map grid `64,192,12` / `128,512,24`. **The cost/
  accuracy knob is the grid resolution itself** — the CKS-17 bake is `O(NU²·NPHI·NZ)` and
  `grid_nu` is simultaneously the shadow-map detail *and* the radial march step count
  (`du=u_max/NU`), so there is no separate "step size" to add: `--shadow-grid NU,NPHI,NZ`
  (and the preset) *is* the manual time/quality balance. Explicit
  `--source-function`/`--self-shadow`/`--shadow-grid` override the preset; the existing
  `self_shadow.lod_max_camera_radius` auto-disables the bake on wide shots.
- **Constraint honored:** the fast path stays the video path (efficient); the volumetric
  richness is opt-in per shot via `--quality`.
- **Deferred / next:** V3.1 curl-flow advection (formula CKS-18 §2 drafted, owner-approved
  mechanism = §2 dual-phase reset blend + independent `flow_period_M` clock).
