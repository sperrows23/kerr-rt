# Kerr Black Hole — End-to-End Production Pipeline

A single, ordered reference for taking a shot from an empty Blender scene to a
finished, color-graded clip. It covers four stages:

```
Stage 1  Blender         scene + ship animation, camera export, ship render
Stage 2  Taichi (CUDA)   black-hole / accretion-disk render → multi-channel EXR
Stage 3  Blender         compositing: deep merge, bloom, lens distortion, aberration
Stage 4  DaVinci Resolve conform, color grade, finishing, delivery
```

Stages 1 and 2 are **implemented in this repository**. Stages 3 and 4 are the
**intended, documented workflow** — there is no compositor `.blend` or Resolve
project checked in yet, so those sections describe the recommended node/grade
setup and flag where assumptions are made.

> **Conventions in force everywhere:** geometric units G = M = c = 1;
> Boyer-Lindquist coordinates; metric signature (− + + +); spin a = 0.999. All GR
> formulas come from `skills/kerr-physics/SKILL.md` (never re-derived). All
> numeric parameters come from `configs/render.yaml`. GPU backend is locked to
> `ti.init(arch=ti.cuda)`.

### Assumptions (flagged up front)

- **A1 — Frame rate = 24 fps.** Now explicit in config: `render.fps = 24.0`. The
  motion-blur shutter arc is `Δφ · fps · shutter_fraction` (`camera.shutter_fraction
  = 1/48 s`), which equals the 180° shutter (`Δφ · 0.5`) at 24 fps and scales
  correctly at other rates. Change `render.fps` if you render at a different rate.
  (Resolves `REMAINING_WORK_PLAN.md` F2.)
- **A2 — Stage 3 (Blender compositor) is not yet built.** The node graph below is
  the recommended construction, consistent with `CLAUDE.md`'s "Phase 3" intent.
- **A3 — Stage 4 (DaVinci Resolve) is entirely a finishing recommendation.** No
  Resolve project exists in the repo.
- **A4 — The black-hole EXR is scene-linear HDR.** `scripts/export_exr.py` writes
  the raw (un-tonemapped) beauty buffer; tone mapping happens later (Stage 3/4),
  not in the render. Keep the working space linear until the grade.
- **A5 — Ship Z-depth occlusion is not wired yet.** The deep-composite step that
  uses ship depth is blocked on a Blender ship Z pass (see Stage 1 §1.4 and
  `REMAINING_WORK_PLAN.md` 3.1/3.2). Until then, composite by layer order, not Z.

---

## Stage 1 — Blender: scene, camera export, ship render

**Goal:** produce two things the renderer and compositor consume — a per-frame
camera track (`camera_matrix.json`) and the spaceship beauty (and, ideally, Z)
EXR sequence (`render_spaceship/ship_####.exr`).

### 1.1 Scene setup

- Build the shot in Blender with the spaceship as the only animated hero element.
  The black hole and accretion disk are **not** modeled in Blender — they come
  from Stage 2.
- Place and animate the camera. The camera's world transform per frame is the
  contract with the renderer, so animate it deliberately.
- **World scale:** the renderer interprets the camera world position in geometric
  units (M). The camera sits at Boyer-Lindquist `r = √(x²+y²+z²)`; production
  framing uses `camera.default_radius ≈ 6.03 M`, `default_fov_deg = 90°`. Keep the
  Blender camera distance consistent with these geometric-unit expectations.

### 1.2 Camera export — `src/blender/export_camera.py`

Run inside Blender (needs `bpy`). For every frame in `[frame_start, frame_end]` it
writes one JSON record:

```json
{ "frame": N,
  "pos":   [x, y, z],        // world Cartesian position
  "fwd":   [fx, fy, fz],     // world  -Z (Blender forward)
  "up":    [ux, uy, uz],     // world  +Y (Blender up)
  "right": [rx, ry, rz],     // world  +X (Blender right)
  "fov":   angle }           // vertical FOV, radians (cam.angle)
```

Output: `camera_matrix.json` in the project root (gitignored). The renderer reads
this with `utf-8-sig` to tolerate Blender's BOM.

### 1.3 Spaceship render

- Render the ship beauty pass to `render_spaceship/ship_####.exr` (OpenEXR,
  scene-linear, with alpha). Prefix/dir come from `configs/render.yaml` `output`.
- Keep it in a linear working space — no baked tone curve — so it composites
  cleanly against the linear black-hole EXR in Stage 3.

### 1.4 Spaceship Z-depth pass *(recommended, not yet produced)*

For true depth-correct occlusion of the ship by the disk / behind the hole, also
output a **Z-depth pass** (a `Z` channel in the ship EXR, or a sidecar
`shipz_####.exr`). This is the asset that unblocks renderer items 3.1/3.2.

> **Unit caveat (critical).** The black-hole render's Z is a **Mino-affine path
> length**, *not* metric distance and *not* Blender camera-space Z. Document the
> ship-Z unit (camera-space metres / Blender units) and the mapping between the
> two before trusting any Z-based merge. See `REMAINING_WORK_PLAN.md` 3.1/3.2.

**Stage 1 outputs:** `camera_matrix.json`, `render_spaceship/ship_####.exr`
(+ optional ship Z).

---

## Stage 2 — Taichi CUDA renderer: the black hole + accretion disk

**Goal:** for each camera frame, trace photon geodesics in Kerr spacetime and emit
a scene-linear multi-channel EXR (`render_blackhole/bh_####.exr`) carrying RGB +
a transmittance-weighted Z.

### 2.1 What runs

`scripts/export_exr.py` is the production entry point. Per frame it:
1. loads `configs/render.yaml` (UTF-8) and `camera_matrix.json` (`utf-8-sig`),
2. `tr.setup_renderer(cfg)` — `ti.init(arch=ti.cuda)`, loads the 16K starmap,
   uploads the f16 mip pyramid,
3. converts the Blender world camera basis into the local Boyer-Lindquist triad
   (r̂, θ̂, φ̂) that feeds the ZAMO tetrad (Formula 7),
4. runs the split production kernels, then writes the EXR.

Two photon sub-pipelines run per pixel inside the kernel:
- **Pipe A** — trace the photon *backward* through curved spacetime; on escape,
  look up the gravitationally lensed 16K starmap with Formula-10 differential-mip
  anti-aliasing (screen-space Jacobian LOD).
- **Pipe B** — accumulate volumetric accretion-disk emission along the same path
  (g⁴ relativistic beaming, Formula 9), composited in front of the background.

### 2.2 Kernel split (Formula 10, v1.4 amendment)

- `render_beauty_physics` — traces the geodesic in the horizon-stable
  `[y, u, φ, t, v_y, v_u]` state (y = r − r₊, u = cosθ), accumulates Pipe B disk
  RGBA, and writes exit direction / outcome and the transmittance-weighted Z.
- `render_beauty_shade` — reads the 4-neighbourhood exit directions, computes the
  screen-space Jacobian → LOD, samples the lensed sky, composites behind the disk.

### 2.3 Numerical stability (Phase 1, shipped)

- Factored discriminant Δ = y(y + 2k) (Formula 11) kills catastrophic cancellation
  near the horizon; horizon constants k = √(1−a²), r₊ = 1+k derived in Python.
- Singularity-free polar potential Θ_u(u) (Formula 12) removes the 1/sin²θ pole;
  a `sin2_min` guard remains only on the dφ/dλ, dt/dλ denominators.
- Kahan-compensated RK4 accumulation; adaptive Mino step.
- Spin-axis "static" seam fix: φ is wrapped into (−π, π] every step (exact
  axisymmetry identity) so a near-pole passage cannot inflate it past f32
  precision, and `render_beauty_shade` saturates the LOD to the coarsest mip when
  the exit footprint `J > render.j_fold` (the meridian lensing caustic renders as
  smooth grey instead of an aliased static band).

### 2.4 Optional features

- **Motion blur** (`--motion-blur`): host-side averaging of N camera-rotated
  sub-frames over the shutter arc `Δφ · render.fps · camera.shutter_fraction`
  (depth uses sentinel-safe masked averaging). *See assumption A1; the shutter
  factor is now config-driven (F2 resolved).*
- **360°/VR** (`render.projection_mode: equirect`): equirectangular ray-gen.
  Default stays `perspective` so the Doppler regression check remains valid.

### 2.5 Output

A 4-channel EXR `render_blackhole/bh_####.exr` with named channels `R, G, B, Z`:
- **RGB** = scene-linear HDR beauty (un-tonemapped — assumption A4), and
- **Z** = transmittance-weighted Mino-affine depth (sentinel `depth_infinity = 1e5`
  for pixels with no disk emission).

**Verification (manual, the real guard):**
`python scripts/gpu_test.py` → right/left Doppler asymmetry ≈ 7–8× (≈ 7.77×
baseline) for the a = 0.999 edge-on camera. `oiiotool --info` to confirm channels.

**Stage 2 output:** `render_blackhole/bh_####.exr` (RGBA-linear + Z).

---

## Stage 3 — Blender compositor: deep merge + optical FX

> **Assumption A2:** this node graph is the recommended construction; it is not yet
> checked into the repo. Build it once as a compositor template `.blend` and reuse
> per shot. Keep the compositor working space **scene-linear**.

**Goal:** merge the ship and black-hole sequences and apply the in-camera optical
effects (bloom, lens distortion, chromatic aberration) before handing a clean
linear (or log) master to the grade.

### 3.1 Inputs

- `render_blackhole/bh_####.exr` — RGBA-linear + Z (Stage 2).
- `render_spaceship/ship_####.exr` — RGBA-linear (+ Z, when available) (Stage 1).

### 3.2 Recommended node graph (in order)

1. **Image inputs.** Two `Image` (EXR sequence) nodes. Set both to read as
   **linear / non-color** data so no double color transform is applied. Pull the
   `Z` channels for the depth merge.
2. **Depth-correct merge (deep composite).** Combine ship over black-hole using
   their Z channels so the disk can occlude the ship and vice-versa.
   - *Current limitation (A5):* the BH Z is Mino-affine, not metric, and the ship Z
     pass may not exist yet. **Until the unit mapping is established, composite by
     layer order** (ship `Alpha Over` black hole) and treat Z-merge as a later
     upgrade. Do **not** Z-compare across mismatched units.
3. **Bloom / glow.** Apply a `Glare` node (Fog Glow or Bloom) to the combined
   linear image to bloom the bright photon ring and the approaching (blue-shifted,
   beamed) disk limb. Because the input is HDR-linear, bloom thresholds key off
   true luminance — keep the threshold above the background star field so only the
   ring and disk bloom.
4. **Chromatic aberration.** Split RGB and offset/scale the channels radially
   (Lens Distortion node's *Dispersion*, or a manual per-channel scale) for a
   subtle lensing fringe toward frame edges.
5. **Lens distortion.** `Lens Distortion` node — a small barrel/pincushion
   `Distort` plus the `Dispersion` above — to seat the CG in a physical lens.
   Apply *after* the merge so both layers share the same distortion.
6. **Post-compositing adjustments.** Optional vignette (radial mask multiply),
   subtle film grain, and a final exposure trim. Keep these gentle — heavy color
   work belongs in Stage 4, not here.
7. **Output.** Write a finishing master sequence. Recommended: **linear EXR** (most
   headroom, defer all tone mapping to Resolve) or a **log/wide-gamut** intermediate
   if Resolve conform prefers it. Avoid baking a display tone curve here.

### 3.3 Ordering rationale

Merge first (so effects act on the unified image), then bloom (needs HDR
luminance), then the lens stack (aberration + distortion as one optical layer),
then gentle post. Color grading is deliberately deferred to Stage 4.

**Stage 3 output:** a composited finishing master (linear EXR or log intermediate)
per frame.

---

## Stage 4 — DaVinci Resolve: grade + finishing + delivery

> **Assumption A3:** entirely a recommended finishing workflow; no Resolve project
> is in the repo.

**Goal:** conform the composited frames, establish the look, and deliver.

### 4.1 Conform & color management

- Import the Stage 3 sequence as a clip; set the project to a managed pipeline
  (**Resolve Color Management** or **ACES**). Tag the incoming clips by their true
  space: scene-linear EXR → linear/ACEScg input; a log intermediate → its log
  curve. Getting input tagging right is what makes the HDR ring/disk grade cleanly.

### 4.2 Grade (node order in the Color page)

1. **Normalize / tone map** the scene-linear HDR into the grading range (an output
   transform or a tone-map node) — this is where the deferred tone mapping from
   Stage 2/3 finally lands.
2. **Primaries / balance** — set black point, white point, overall exposure and
   neutral balance against the star field.
3. **Secondaries** — qualify and shape the hero elements: warm the disk's
   red-shifted receding limb, push the beamed approaching limb brighter/bluer,
   isolate and control the photon-ring glow.
4. **Look** — creative LUT / film emulation, contrast and saturation curves.

### 4.3 Finishing

- Sky/disk **noise reduction** if the volumetric march shows sampling noise; final
  **sharpening**; **vignette** refinement; optional **grain**.
- Re-check **bloom interaction**: if Resolve's tone map crushes or blooms
  differently than Stage 3, rebalance rather than double-blooming.

### 4.4 Delivery

- Render out to the delivery codec/space (e.g. Rec.709 / sRGB for SDR, or an HDR
  delivery if mastering HDR). Match frame rate to the render (A1 — 24 fps assumed).

**Stage 4 output:** the finished, graded clip.

---

## End-to-end data flow

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
                              finishing → delivery
                                                          │
                                                          ▼
                                               final graded clip
```

---

## Quick reference — what's real vs. recommended

| Stage | Implemented in repo? | Entry point / note |
|-------|----------------------|--------------------|
| 1. Camera export | ✅ Yes | `src/blender/export_camera.py` (run in Blender) |
| 1. Ship beauty render | ⚠️ Process exists; assets gitignored | outputs to `render_spaceship/` |
| 1. Ship **Z** pass | ❌ Not yet produced | unblocks renderer 3.1/3.2 (unit caveat) |
| 2. Black-hole render | ✅ Yes | `scripts/export_exr.py` → `render_blackhole/bh_####.exr` |
| 2. Motion blur / 360° | ✅ Yes (opt-in) | `--motion-blur`; `projection_mode: equirect` |
| 3. Blender compositor | ❌ Recommended (A2) | build node graph above as a template `.blend` |
| 4. DaVinci Resolve grade | ❌ Recommended (A3) | conform → grade → deliver |

See `REMAINING_WORK_PLAN.md` for the concrete backlog (F2 shutter, F3 GPU
regression test, F4 dead key, F5 docs, 3.1/3.2 ship occlusion, 2.3 texture, T3
moving camera).
