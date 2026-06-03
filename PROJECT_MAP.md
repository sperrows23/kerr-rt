# Kerr Black Hole Renderer — Project Map

Single reference document for the complete codebase. Every file is listed with
its location, purpose, key interfaces, and relationships to other files.

---

## Three-Phase Pipeline Overview

```
Phase 1 (Blender)              Phase 2 (Taichi GPU)            Phase 3 (Blender)
─────────────────────          ─────────────────────────────   ──────────────────
Animate spaceship          →   Trace photon geodesics in    →  Composite both EXR
Export camera matrices         Kerr spacetime                   sequences with glow,
Render ship EXRs               Produce black-hole EXRs          aberration, lens FX
camera_matrix.json                 bh_####.exr
ship_####.exr
```

The Taichi GPU renderer (Phase 2) is the focus of this repository. It has two
sub-pipelines that run inside a single kernel per pixel:

- **Pipe A** — trace a photon backward through curved spacetime; when it
  escapes, look up the gravitationally lensed starmap sky with Formula 10
  differential-mip anti-aliasing.
- **Pipe B** — accumulate volumetric emission from the accretion disk along the
  same photon path, compositing it in front of the background.

---

## Directory Tree

```
Black/
│
├── configs/
│   └── render.yaml              ← single source of truth for all parameters
│
├── scripts/
│   ├── thumb.py                 ← CPU preview renderer (development / QA)
│   └── gpu_test.py              ← FHD GPU beauty render smoke test
│
├── skills/
│   └── kerr-physics/
│       └── SKILL.md             ← physics formula reference (mandatory, never re-derived)
│
├── src/
│   ├── blender/
│   │   ├── __init__.py
│   │   └── export_camera.py     ← Blender script: exports camera_matrix.json
│   │
│   └── renderer/
│       ├── __init__.py
│       ├── metric.py            ← Kerr metric (Formula 1)
│       ├── geodesic.py          ← Mino-time RK4 null geodesic integrator (Formula 6)
│       ├── disk.py              ← Accretion disk gas physics (Formulas 3/4/5/8/9)
│       ├── starmap.py           ← 16K HDRI loader, mip pyramid, UV mapping
│       └── taichi_renderer.py   ← GPU renderer: Pipe A + Pipe B, split kernels (1297 lines)
│
├── scripts/
│   └── export_exr.py            ← Phase 5: multi-channel RGBAZ EXR writer (OpenImageIO)
│
├── IMPLEMENTATION_PLAN.md       ← 5-phase optimization plan + execution status
├── AGENTS.md                    ← mirror of CLAUDE.md for the Codex/Agents harness
├── .codex/config.toml           ← Codex MCP config (Context7)
│
├── tests/
│   ├── cuda_smoke_test.py       ← confirms CUDA backend JIT on RTX 5060
│   ├── test_geodesic.py         ← conservation law tests (E, Lz, Q, null norm)
│   ├── test_starmap.py          ← polar punch-through / UV normalization tests
│   └── test_geodesic/
│       └── test_conserved_quantities_regression.csv   ← golden values
│
├── render_blackhole/            ← output EXR sequence (bh_####.exr) — gitignored
├── render_spaceship/            ← Blender ship EXR sequence — gitignored
├── star_image/                  ← 16K HDRI starmap EXR — gitignored
│
├── CLAUDE.md                    ← project instructions and physics policy
├── PROJECT_MAP.md               ← this file
├── pyproject.toml               ← Python deps + uv/pytest config
├── uv.lock                      ← locked dependency versions
└── .gitignore                   ← excludes large assets and render outputs
```

---

## File Reference

### `configs/render.yaml`

**Role:** Single source of truth for all numerical parameters. No file in `src/`
or `scripts/` contains hardcoded physics or render values — everything reads from
here via `yaml.safe_load`.

| Section | Key fields |
|---------|-----------|
| `black_hole` | `spin` (a=0.999), `r_isco` (1.182 M), `r_plus` (1.0447 M — true outer horizon r₊=1+√(1−a²); consumed only by `thumb.py`, the renderer derives r₊ in `_horizon_constants`) |
| `render` | `width`/`height` (4K target), `thumb_width/height` (256), `max_steps_pipe_a` (250), `max_steps_pipe_b` (200 — **declared but currently unused**, see taichi_renderer note), `d_lambda_pipe_a` (0.01), `r_max` (50 M), `device_memory_gb` (6), `horizon_epsilon` (0.05 Δ-capture stop), `adaptive_step_floor` (0.005), `sin2_min` (1e-10 polar guard), `motion_blur_samples` (4), `projection_mode` (perspective\|equirect), `depth_infinity` (1e5 no-disk Z sentinel) |
| `disk` | `r_inner`, `r_outer`, `theta_half_width`, `T_0`, `emission_coeff`, `absorption_coeff`, `vertical_sigma_frac`, `bounding_sin_theta_half` (=sin(theta_half_width); bbox early-out) |
| `starmap` | `path` (relative to repo root), `width` (16384 — used to compute LOD) |
| `camera` | `default_radius` (6.03 M), `default_fov_deg` (90°), `shutter_fraction` (1/48 s — motion-blur shutter; **note: `export_exr._shutter_arc` hardcodes a 0.5 factor and does not yet read this key**) |
| `thumb` | Preview-only framing overrides: `camera_radius`, `fov_deg`, `camera_theta_deg`, background colors, ring glow, exposure, gamma |
| `output` | Directory names and filename prefixes for EXR sequences |

---

### `skills/kerr-physics/SKILL.md`

**Role:** The physics formula reference. Per `CLAUDE.md`, ALL general-relativity
formulas must be copied verbatim from here — never re-derived. Every formula has
a number and is referenced by that number throughout the codebase.

| Formula | Content |
|---------|---------|
| 1 | Kerr metric g_{μν} in Boyer-Lindquist coordinates |
| 3 | Circular-orbit 4-velocity u^μ for r ≥ r_isco (Bardeen 1970) |
| 4 | ISCO conserved quantities E_I, L_I (Cunningham 1975), frozen at r_isco |
| 5 | Plunging-region 4-velocity with frozen E_I, L_I; u^r must be negative |
| 6 | Mino-time RK4 null geodesic integration; radial potential R(r) and angular potential Θ(θ) |
| 7 | ZAMO tetrad photon momentum initialization; exact A = (r²+a²)² − a²Δsin²θ |
| 8 | g-factor = −1/(p_t·u^t + p_r·u^r + p_θ·u^θ + p_φ·u^φ); p_r is covariant |
| 9 | g⁴ volumetric beaming (3D emitter); `blackbody_rgb` is chromaticity-only (no T⁴) |
| 10 | Differential-ray mip LOD: J = √(δθ² + sin²θ·δφ²); L = log₂(W·J / 2π). **Amendment (v1.4):** J may equivalently be estimated in screen space from the 4-neighbourhood exit directions (kernel-split LOD) instead of an offset ray |
| 11 | FP32-stable factored discriminant: Δ = (r−r₊)(r−r₋) = y(y+2k), y=r−r₊, k=√(1−a²) (kills catastrophic cancellation near the horizon) |
| 12 | Singularity-free polar potential under u=cosθ: Θ_u(u) = (1−u²)(Q+a²E²u²) − L_z²u²; the 1/sin²θ pole cancels analytically (dφ/dλ, dt/dλ keep a `sin2_min` guard) |

---

### `src/renderer/metric.py`

**Role:** Kerr metric (Formula 1). Provides the covariant metric tensor
`metric_bl(r, theta, a)` and its numerical inverse `inverse_metric_bl`. Used by
`scripts/thumb.py` (to lower photon momenta) and by `tests/test_geodesic.py` (to
check the null condition). The GPU renderer inlines the metric analytically rather
than importing this module.

---

### `src/renderer/geodesic.py`

**Role:** CPU null-geodesic integrator (Formula 6). Integrates the Mino-time
second-order equations of motion with RK4 and a projection step that re-imposes
`(dr/dλ)² = R` and `(dθ/dλ)² = Θ` after each step to prevent drift.

Key functions:
- `integrate_null_geodesic(x0, p0, a, n_steps, d_lambda)` — returns `(x, p_cov)` arrays
- `make_null_initial_conditions(...)` — sets up the state vector from E, Lz, Q
- `carter_Q(theta, p_theta, E, Lz, a)` — null Carter constant
- `radial_turning_point(E, Lz, Q, a, r_start, r_floor)` — analytic perihelion root via bisection
- `_DELTA_MIN = 0.05` — integration stops before the horizon where BL momenta diverge

State vector layout: `[r, θ, φ, t, v_r, v_θ]` where `v_r = Δ·p_r` and `v_θ = p_θ`.

Used by: `scripts/thumb.py`, `tests/test_geodesic.py`.

---

### `src/renderer/disk.py`

**Role:** CPU accretion disk gas physics. The reference implementation for all
disk formulas. The GPU port in `taichi_renderer.py` must produce identical
numbers — verified numerically at three test points.

Key functions:
- `isco_conserved_quantities(r_isco, a)` — Formula 4; called once, result frozen
- `gas_four_velocity(r, theta, a, r_isco, E_I, L_I)` — Formulas 3/5; returns `[u^t, u^r, u^θ, u^φ]`
- `g_factor(p_cov, u_con)` — Formula 8; `p_cov[R]` is already covariant, do NOT divide by Δ again
- `blackbody_rgb(temperature)` — Formula 9 chromaticity helper; no T⁴ amplitude

Used by: `scripts/thumb.py` (`march_disk`), `src/renderer/taichi_renderer.py`
(`isco_conserved_quantities` import for frozen E_I, L_I).

---

### `src/renderer/starmap.py`

**Role:** Host-side 16K equirectangular starmap management. Loads the HDRI,
builds a box-filtered mip pyramid (stored as float16 to fit VRAM), and provides
the reference trilinear sampler that the GPU sampler mirrors.

Key functions:
- `load_equirect(path)` — loads via OpenImageIO; returns `(H, W, 3)` float32
- `build_mip_pyramid(base_rgb)` — box-filter halving, float32→float16 each level
- `Starmap.load(path)` — combined load + pyramid; returns `Starmap` dataclass
- `Starmap.sample(u, v, lod)` — trilinear reference sampler (ground truth for GPU)
- `normalize_sphere_angles(theta, phi)` — folds raw integrator `(θ, φ)` onto `[0, π]`; handles polar punch-through
- `direction_to_uv(theta, phi)` — calls normalize then maps to equirect UV

Equirect convention: `u = φ / 2π` (column), `v = θ / π` (row, north pole at v=0).

Used by: `src/renderer/taichi_renderer.py` (upload to GPU), `tests/test_starmap.py`.

---

### `src/renderer/taichi_renderer.py`

**Role:** The GPU renderer — Phase 2 core (1297 lines). Ports the CPU physics
to Taichi `@ti.func` / `@ti.kernel` functions and runs both pipes on CUDA. The
production beauty path is split into a **physics kernel** + a **shading kernel**
(Formula 10 screen-space-Jacobian LOD) and integrates in the horizon-stable
`[y, u, φ, t, v_y, v_u]` state (y=r−r₊, u=cosθ).

**Backend:** Locked to `ti.init(arch=ti.cuda)` — never `ti.gpu`.

#### Module-level state
| Field | Purpose |
|-------|---------|
| `star_flat`, `star_off`, `star_w`, `star_h` | Mip pyramid packed as flat f16 buffer + metadata |
| `pixels` | Square output buffer for `render_pipe_a` |
| `frame_pixels` | Non-square output buffer for `render_beauty_*` |
| `exit_buf`, `disk_buf`, `depth_pixels` | Kernel-split hand-off: physics kernel writes exit dirs/outcome + accumulated disk RGBA + transmittance-weighted Z; shade kernel reads them |

#### Physics `@ti.func` functions

| Function | Formula | Notes |
|----------|---------|-------|
| `_horizon_constants(a)` (host) | 11 | Derives k=√(1−a²), r₊=1+k in Python (not a hardcoded literal) |
| `_delta_y(y, k)` | 11 | Δ = y(y+2k) factored form — FP32-stable near horizon |
| `_radial_potential_y` / `_radial_potential_deriv_y` | 6/11 | R(y), ½R′ in horizon-relative y |
| `_theta_potential` (Θ_u) / its deriv | 6/12 | Singularity-free polar potential Θ_u(u), ½Θ_u′ |
| `_deriv(s, E, Lz, Q, a, k, r_plus)` | 6/12 | Mino-time ds/dλ in `[y,u,…]` state |
| `_project(s, …)` | 6 | Re-impose (dy/dλ)²=R, (du/dλ)²=Θ_u after RK4 |
| `_rk4_step(s, …, h)` | 6 | One adaptive RK4 step (Kahan-compensated) + project |
| `_zamo_init(r, theta, a, k, r_plus, n_r, n_th, n_ph)` | 7 | ZAMO tetrad → (E, Lz, Q, v_y0, v_u0) |
| `_gas_four_velocity(r, theta, a, r_isco, E_I, L_I)` | 3/5 | GPU port of `renderer.disk.gas_four_velocity` (plunging branch uses factored Δ) |
| `_blackbody_rgb(temp)` | 9 | Chromaticity only, no T⁴ |
| `_disk_emit(y, u, vy, vu, E, Lz, a, k, r_plus, r_isco, E_I, L_I, …)` | 8/9 | One volumetric disk sample → `vec4(emitRGB, dτ)`; recovers p_r=v_y/Δ, p_θ=−v_u/√(1−u²) |

#### Starmap `@ti.func` functions

| Function | Notes |
|----------|-------|
| `_texel(level, x, y)` | Index into flat f16 buffer |
| `_sample_level(level, u, v)` | Bilinear with φ-wrap and θ-clamp |
| `_normalize_sphere(theta, phi)` | Polar punch-through fix: θ→\|θ\|, φ→φ+π |
| `_sample_trilinear(u, v, lod)` | Trilinear across mip levels |

#### Kernels

| Kernel | Purpose |
|--------|---------|
| `render_pipe_a(res, ...)` | Pipe A only (square, ZAMO-aligned camera). **Retains the offset ray** as the LOD reference (dev/`_gate2_lod_test` path, not 4K production). |
| `render_beauty_physics(width, height, ...)` | **Production kernel 1.** Arbitrary camera basis via ZAMO triad. Traces the geodesic, accumulates Pipe B disk RGBA, writes exit dir/outcome + transmittance-weighted Z to `exit_buf`/`disk_buf`/`depth_pixels`. |
| `render_beauty_shade(width, height, lod_enabled)` | **Production kernel 2.** Reads the 4-neighbourhood exit dirs, computes the Formula-10 screen-space Jacobian → LOD, samples the lensed starmap, composites it behind the disk. |
| `render_starmap_raw` | Diagnostic 1: equirect sky dump at fixed LOD, no geodesic |
| `render_fixed_lod` | Diagnostic 2: geodesic lensing, LOD pinned (no Jacobian) |
| `dump_phi_exit` | Diagnostic 3: per-column raw φ exit dump for seam root-cause analysis |

#### Host functions

| Function | Purpose |
|----------|---------|
| `load_config(path)` | YAML load with explicit UTF-8 (avoid Windows cp949) |
| `setup_renderer(cfg)` | `ti.init(cuda)` + load starmap + upload mip pyramid to GPU |
| `_alloc_output(res)` | Allocate square `pixels` field if size changed |
| `_alloc_frame(width, height)` | Allocate `frame_pixels` field for non-square renders |
| `render_pipe_a_image(cfg, res, lod_enabled)` | Render square Pipe A frame, return float32 HDR |
| `render_beauty_frame(cfg, cam_frame, width, height, with_disk, lod_enabled, return_depth)` | **Main entry point.** Converts Blender world Cartesian → BL, projects camera axes onto local (r̂, θ̂, φ̂) triad, runs the physics+shade kernels; optionally returns the (NaN-guarded) Z pass. |
| `render_beauty_frame_mb(cfg, cam_frame, width, height, shutter_arc, ...)` | Temporal motion-blur variant: averages N camera-rotated sub-frames; depth uses masked per-pixel averaging (sentinel-safe). |
| `tonemap(hdr, exposure, gamma)` | Reinhard tonemap + gamma → uint8 |

**Consumed by:** `scripts/gpu_test.py` (smoke render), `scripts/export_exr.py`
(Phase 5 RGBAZ writer).

**Note — `max_steps_pipe_b`:** declared in `render.yaml` but not read by any
kernel; disk marching is tied to the Pipe A geodesic loop (`max_steps_pipe_a`).
The key is currently inert (see fix plan).

#### Camera conversion in `render_beauty_frame`

```
Blender camera_matrix.json entry
  pos  = [x, y, z]     world Cartesian position
  fwd  = [fx, fy, fz]  forward vector (-Z in Blender local)
  up   = [ux, uy, uz]  up vector (Y in Blender local)
  right= [rx, ry, rz]  right vector (X in Blender local)
  fov                  vertical FOV (radians)

                    ↓ spherical embedding

  r_cam   = √(x²+y²+z²)
  θ_cam   = acos(z / r_cam)
  φ_cam   = atan2(y, x)

  r̂  = [sin θ cos φ,  sin θ sin φ,  cos θ]
  θ̂  = [cos θ cos φ,  cos θ sin φ, -sin θ]
  φ̂  = [-sin φ,        cos φ,         0 ]

  fwd_local = (fwd·r̂, fwd·θ̂, fwd·φ̂)    ← these three components
  rgt_local = (rgt·r̂, rgt·θ̂, rgt·φ̂)    ← feed directly into the
  up_local  = (up·r̂,  up·θ̂,  up·φ̂)     ← ZAMO tetrad (Formula 7)
```

---

### `src/blender/export_camera.py`

**Role:** Blender Python script (Phase 1). Run inside Blender to export per-frame
camera data. Writes `camera_matrix.json` in the project root as a JSON array with
one entry per frame: `{frame, pos, fwd, up, right, fov}`.

- `fwd` = world −Z axis of the camera object (Blender convention)
- `fov` = `cam.angle` = vertical FOV in radians
- Must be run with `bpy` available (i.e., inside Blender's Python environment)

Output is gitignored (`camera_matrix.json`).

---

### `scripts/thumb.py`

**Role:** CPU preview renderer. Single-threaded NumPy ray tracer; slow but
self-contained. Used during development to verify physics without the GPU stack.

- `--disk` enables Pipe B via `march_disk` (CPU version of the disk volume march)
- Camera uses config `thumb.*` overrides (pulled back, narrower FOV) so the full
  shadow + photon ring fit in frame
- Outputs `scripts/thumb_output.png` or `scripts/thumb_disk.png` (gitignored)
- `zamo_photon_momentum` here is the CPU reference for `_zamo_init` in the GPU kernel

**Flow:** `render()` → per-pixel `camera_ray_direction()` → `zamo_photon_momentum()`
→ `integrate_null_geodesic()` → `march_disk()` (if `--disk`) → `trace_pixel()`
→ Reinhard tonemap.

---

### `scripts/gpu_test.py`

**Role:** FHD GPU beauty render smoke test. Reads frame 0 (or any `--frame N`)
from `camera_matrix.json`, runs `render_beauty_frame` at 1920×1080 on CUDA, saves
the result, and reports the Doppler asymmetry ratio as a physics sanity check.

- Opens `camera_matrix.json` with `encoding="utf-8-sig"` (handles BOM from Blender)
- Default output: `scripts/gpu_test_disk.png` (gitignored)
- `--no-disk` disables Pipe B (Pipe A only)
- `--exposure` overrides config tonemap exposure
- Reports `right_lum / left_lum` asymmetry; expected ≈ 7–8× for a=0.999 edge-on
  camera (g⁴ beaming, approaching limb to the right)

---

### `tests/test_geodesic.py`

**Role:** Conservation law tests for the CPU geodesic integrator. Traces one null
geodesic at off-equatorial initial conditions and checks that E, Lz, Q drift less
than 1e-4 (relative) and the null condition `g^{μν} p_μ p_ν` stays below 1e-6
over 4000 steps.

Also runs a golden-value regression (`pytest-regressions`) that pins sampled
trajectory values to a CSV; any physics change that shifts the numerics breaks this
test.

---

### `tests/test_starmap.py`

**Role:** Unit tests for the polar punch-through fix in `normalize_sphere_angles`
and `direction_to_uv`. Verifies that θ < 0 (from integrator overshoot past the
north pole) is reflected to a genuine UV row instead of being clamped to v=0 (the
old streak bug), and that the normalization preserves the physical direction vector.

---

### `tests/cuda_smoke_test.py`

**Role:** Confirms the Taichi CUDA backend JITs and runs correctly on this machine.
Fills a 1M-element field with `√i` and spot-checks against NumPy. Fails explicitly
if `arch` is not `Arch.cuda`, catching any silent fallback to CPU.

Run standalone: `python tests/cuda_smoke_test.py`

---

### `tests/test_geodesic/test_conserved_quantities_regression.csv`

**Role:** Golden values for `test_conserved_quantities_regression`. Auto-generated
by `pytest-regressions` on the first passing run; compared on all subsequent runs.
Contains sampled `(lambda_index, r, theta, E, L_z, Q)` at 11 points along the
test geodesic.

---

### `pyproject.toml`

**Role:** Python project manifest for `uv`. Declares dependencies and test config.

Key entries:
- `taichi==1.7.4` — pinned; this exact version is confirmed working on RTX 5060 (sm_120/Blackwell)
- `openimageio` — preferred 16K EXR loader
- `[tool.uv] package = false` — src-layout workspace, not a distributable wheel
- `[tool.pytest] pythonpath = ["src"]` — makes `from renderer import ...` work in tests

---

### `.gitignore`

**Role:** Prevents large regenerated assets from entering the repository.

Excluded:
- `star_image/` — 423 MB 16K starmap HDRI (downloaded externally)
- `render_blackhole/`, `render_spaceship/` — EXR render output sequences
- `camera_matrix.json` — Blender camera export (regenerated each production run)
- `*.exr` — all EXR files
- `scripts/*.png` — GPU test and diagnostic images
- `__pycache__/`, `*.pyc` — Python bytecode

---

### `CLAUDE.md`

**Role:** Project instructions and physics policy for the AI assistant. Authoritative
on: unit conventions, coordinate system, GPU backend lock, formula policy (all GR
formulas must come from `SKILL.md`), config-driven development rule, directory
layout, and build commands.

---

## Data Flow Between Files

```
configs/render.yaml
    │
    ├──▶ src/renderer/taichi_renderer.py   (load_config)
    ├──▶ scripts/thumb.py                  (load_config)
    ├──▶ scripts/gpu_test.py               (tr.load_config)
    └──▶ tests/test_geodesic.py            (yaml.safe_load)

skills/kerr-physics/SKILL.md
    │  (formulas copied verbatim into)
    ├──▶ src/renderer/metric.py            (Formula 1)
    ├──▶ src/renderer/geodesic.py          (Formula 6)
    ├──▶ src/renderer/disk.py              (Formulas 3/4/5/8/9)
    ├──▶ src/renderer/starmap.py           (Formula 10 UV convention)
    └──▶ src/renderer/taichi_renderer.py   (all of the above, ported to GPU)

src/renderer/metric.py  ──▶  scripts/thumb.py
                         ──▶  tests/test_geodesic.py

src/renderer/geodesic.py  ──▶  scripts/thumb.py
                           ──▶  tests/test_geodesic.py

src/renderer/disk.py  ──▶  scripts/thumb.py (march_disk)
                       ──▶  src/renderer/taichi_renderer.py (isco_conserved_quantities)

src/renderer/starmap.py  ──▶  src/renderer/taichi_renderer.py (setup_renderer upload)
                          ──▶  tests/test_starmap.py

src/renderer/taichi_renderer.py  ──▶  scripts/gpu_test.py (render_beauty_frame)

src/blender/export_camera.py  (run inside Blender)
    ──▶  camera_matrix.json  ──▶  scripts/gpu_test.py
```

---

## Key Invariants

| Invariant | Where enforced |
|-----------|---------------|
| GPU backend = `ti.cuda`, never `ti.gpu` | `taichi_renderer.py:99`, `cuda_smoke_test.py:21`, `CLAUDE.md` |
| All formulas from `SKILL.md`, no re-derivation | `CLAUDE.md` physics policy |
| All parameters from `configs/render.yaml` | All source files; no numeric literals for physics |
| State vector `v_r = Δ·p_r` (CPU); renamed `v_y = dy/dλ = Δ·p_r` in GPU `[y,u,…]` state | `geodesic.py` (CPU `[r,θ,…]`), `taichi_renderer.py` `_zamo_init`/`_disk_emit` (GPU `[y,u,…]`, value preserved) |
| `p_r` covariant recovery = `v_r / Δ` (not `v_r / Δ²`) | `disk.py:84-87` (Formula-8 known bug note), `_disk_emit:420` |
| `blackbody_rgb` is chromaticity-only | `disk.py:91-102`, `_blackbody_rgb:387-397` |
| g⁴ beaming is correct and not double-counted | `disk.py:277`, `_disk_emit:433` |
| θ ∈ [0, π] before UV lookup (punch-through fix) | `starmap.py:normalize_sphere_angles`, `taichi_renderer.py:_normalize_sphere` |
| Camera file encoding = utf-8-sig | `gpu_test.py:57` |
| Config files read with encoding="utf-8" | `taichi_renderer.py:42`, `thumb.py:65`, `test_geodesic.py:62` |
