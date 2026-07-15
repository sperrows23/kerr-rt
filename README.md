# Kerr Black Hole Offline Renderer

A physically-based, GPU-accelerated offline renderer for a near-extremal **Kerr
(rotating) black hole** and its layered accretion disk — built around a Taichi/CUDA
photon **geodesic tracer** that bends light through curved spacetime exactly as
general relativity prescribes, then composited with a Blender-animated spaceship
into a finished shot.

> Geometric units `G = M = c = 1` · Cartesian Kerr-Schild coordinates `(t, x, y, z)`
> (spin axis `+z`) · metric signature `(− + + +)` · spin `a = 0.999` (near-extremal).

---

## Why this exists

Most "black hole" visuals are art, not physics — a glowing ring slapped onto a
sphere. This project takes the opposite stance: **every pixel is the endpoint of a
real null geodesic** integrated backward through the Kerr metric. The goal is an
*Interstellar*-grade image where the lensing, the photon ring, the asymmetric disk
brightness, and the warped star field all emerge from the spacetime itself rather
than from hand-tuned shaders.

That demands a few things ordinary renderers don't provide, which is the reason this
codebase is its own project:

- **Correctness over speed-at-any-cost.** Light paths near a spinning horizon are
  numerically vicious (catastrophic cancellation, coordinate singularities at the
  poles and at the horizon). The renderer works in **Cartesian Kerr-Schild**
  coordinates — a frame that is regular on both the spin axis and the horizon — with
  an exact analytic inverse metric (no matrix inversion) and a Hamiltonian 8-vector
  RK4 integrator, so the geodesics actually conserve their constants of motion. (The
  earlier Boyer–Lindquist path left a gray polar-axis seam; the 2026-06 CKS migration
  removed it at the coordinate level and is now the sole active path — BL is
  retired/history only.)
- **A single physics source of truth.** Every GR formula is copied *verbatim* from
  [`skills/kerr-physics/SKILL.md`](skills/kerr-physics/SKILL.md) and referenced by
  formula ID in the code (`CKS-1` … `CKS-23` and beyond). Formulas are **never
  re-derived ad hoc** — re-derivation is how sign errors and normalization bugs creep
  in. If a formula looks wrong, it gets flagged for human review, not silently
  replaced. This policy is enforced by [`CLAUDE.md`](CLAUDE.md).
- **Config-driven, no magic numbers.** Spin, ISCO radius, resolution, step caps,
  bounding boxes, camera framing, and every accretion-disk feature toggle live in
  [`configs/render.yaml`](configs/render.yaml). No physics or render literal is
  hardcoded in source.

---

## What it does

The renderer traces two photon sub-pipelines **per pixel** inside split GPU kernels:

- **Pipe A — background / lensing.** Trace a photon *backward* from the camera
  through curved spacetime. When it escapes to infinity, look up where it came from
  on the sky. Two background modes are supported:
  - `dngr` (default, since 2026-06-06): the *Interstellar*-style two-layer model — a
    **point-star catalog** that stays sharp and *brightens* under lensing
    (magnification → flux, not smear), plus an anisotropically-filtered diffuse
    Milky-Way band (Formula 13).
  - `texture` (legacy): sample a gravitationally-lensed 16K equirect starmap with
    screen-space–Jacobian mip anti-aliasing (Formula 10).
- **Pipe B — accretion disk.** March a volumetric disk emission/absorption integral
  along the same geodesic, with `g⁴` relativistic Doppler/gravitational beaming so
  the approaching limb of the disk blazes and the receding limb darkens, composited
  in front of the background. This is where most of the project's depth lives — see
  [Accretion disk feature stack](#accretion-disk-feature-stack) below.

The result is a multi-channel scene-linear HDR EXR per frame, ready for compositing.

**Signature verification:** for the `a = 0.999` edge-on camera the right/left disk
Doppler brightness asymmetry comes out to ≈ **4.3×** (under the CKS affine emission
measure; the retired BL Mino path read ≈7.8×). `tests/test_gpu_regression.py` gates
production renders on this physical band (`[3.8, 4.9]`), not on a pixel-perfect
golden image alone.

---

## Three-phase production pipeline

```
Phase 1  Blender          → camera_matrix.json  +  render_spaceship/ship_####.exr
Phase 2  Taichi (CUDA)    → render_blackhole/bh_####.exr   (RGB-linear + Z depth)
Phase 3  Blender compositor → composite: deep merge, bloom, chromatic aberration, lens
(Phase 4  DaVinci Resolve → grade + delivery — recommended finishing, external)
```

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Camera export (`src/blender/export_camera.py`) | **Implemented.** Run inside Blender to write `camera_matrix.json`: per-frame `pos`, `fwd`, `up`, `right`, and vertical `fov`, with explicit handling of Blender's sensor-fit modes. |
| 1 | Spaceship EXR render | **Manual.** Animating the ship and rendering `render_spaceship/ship_####.exr` (scene-linear RGBA, 24 fps) is a Blender task performed by hand — no render script is checked in. |
| 1 | Ship Z-depth pass | **Not implemented.** Blocked on documenting the CKS Mino-affine ↔ Blender camera-space unit mapping (see below). |
| 2 | GPU photon tracer (`scripts/export_exr.py` + `src/renderer/`) | **Implemented — the focus of this repo.** Reads `camera_matrix.json` + `configs/render.yaml`, produces `render_blackhole/bh_####.exr`. |
| 3 | Blender compositor | **Template only.** A recommended node graph is documented in `PROJECT.md` §2.3; no `.blend` file is checked into the repository. |
| 4 | DaVinci Resolve grade | **Recommended, external.** Conform/grade/deliver from the Phase-3 linear EXR. |

Because Phase 3 has no ship Z-depth to composite against, the documented workaround
is to composite by **layer order** (`Alpha Over`, ship always in front) rather than a
true depth merge — the black-hole `Z` channel is a Mino-affine path-length parameter
and the ship's camera-space depth is a different unit entirely, so a direct Z compare
isn't valid until that mapping is written down.

### Phase 2 output format

`render_blackhole/bh_####.exr` is scene-linear HDR with:

- **R, G, B** — composite of Pipe A (lensed background) + Pipe B (disk emission),
  un-tonemapped.
- **Z** — transmittance-weighted Mino-affine depth at the disk-emission transition;
  sentinel `depth_infinity = 1.0e5` for background-only pixels.

---

## Accretion disk feature stack

The disk model is the project's most developed piece of physics: a dozen-plus
layered effects, each traced to a numbered formula in `SKILL.md`, each **gated by a
config flag or Taichi compile-time branch**, and each **default OFF so the render is
bit-identical to the pure-GR golden frames** unless explicitly enabled. Nothing here
touches the conserved quantities (`p_μ`, `u^μ`, `g`, `g⁴`) — turbulence and lighting
layers modulate emission/absorption, not the underlying orbital physics.

**Temperature / spectral**
- **Simple model** (default): `T = T_0·(6/r)^0.75`.
- **Page-Thorne** (`CKS-11`, `disk.temperature_model: page_thorne`): physically
  accurate zero-torque flux profile `F(r)` from a closed-form cubic-root solve,
  precomputed to a LUT, `T_eff = T_0·F(r)^{1/4}`.
- Blackbody chromaticity (`Formula 9`) is layered on separately — chroma has no
  built-in `T⁴` scaling, only `I_obs = g⁴·I_emit` carries the beaming exponent.

**Relativistic Doppler / beaming**
- Rigid `+z` circular-orbit gas 4-velocity at `r ≥ r_isco` (`CKS-8`).
- `g`-factor as a plain Cartesian dot product — no Boyer-Lindquist `Δ` bug (`CKS-9`).
- `g⁴` volumetric intensity beaming (`Formula 9`); `disk.doppler_strength` is a
  *visualization* dial (`g_eff = g^s`), not physics — `1.0` is the physical value.

**Turbulence & procedural noise** (`CKS-12`, master gate `disk.noise.enabled`)
- Three-layer density stack: base streaks (fBm), clump/tear (ridged·billow +
  Voronoi), and large-scale patchiness — combined via a log-sum clamp.
- Keplerian shear advection: dual-phase reset-blend winding at a config-derived
  orbital period, giving filaments that wind up realistically over time.
- Emission-amplitude modulation: advected temperature, inner/outer edge raggedness,
  and scale-height fields — all `[0,1]` fBm envelopes, never touching momentum.
- **V3.0/V3.1 curl domain warp** (`CKS-18`): divergence-free noise-coordinate warp
  (curl of a simplex potential on a seamless cylinder embedding) for textured
  eddies, plus a flow-advection variant so the eddies visibly boil.
- **P1 shear cascade** (`CKS-21`): frequency-dependent shear transfer
  `S(f) = 1/(1+(f/f_c)^p)` with per-octave de-shear add-back, protecting
  high-frequency detail near the photon ring from over-winding.
- All primitives use a deterministic integer PCG hash (no `ti.random`) with a
  CPU source-of-truth (`noise.py`) and Taichi GPU twins held to ~1e-6.

**Volumetric radiative transfer**
- **Source-function march** (`CKS-14`, `disk.volumetric.source_function`):
  `dI = (S−I)dτ` in place of a closed-form integral — same continuum result in the
  thin limit, but materializes `S` for shadowing.
- **Self-shadow** (`CKS-15`/`CKS-17`, `disk.volumetric.self_shadow.enabled`): a
  baked radial deep-shadow-map, generalized to a tilted 3D inner-edge ray, so the
  disk casts shadows on itself (voids emerge behind the near side).
- **Flared 3D scale height** (`CKS-16`, `disk.volumetric.flare.enabled`):
  `σ_θ(r) = σ0·(r/r_inner)^β` — constant at `β=0`, physically-flared outward for
  `β>0`.

**Multi-phase media & scattering**
- **Multi-phase split** (`CKS-19`, `disk.multiphase.enabled`): separate hot plasma
  (`ρ_hot`, emission) from cold dust (`ρ_cold`, absorption) via a
  variance-preserving Pearson-correlated mix, plus chromatic extinction
  (`disk.extinction_rgb`) that reddens the background through dust.
- **Single-scatter + Henyey-Greenstein rim-light** (`CKS-20`,
  `disk.scatter.enabled`): forward/backward anisotropic in-scatter off the cold
  medium, lighting the inner edge from behind.

**Instabilities & fine detail**
- **Kelvin-Helmholtz edge erosion** (`CKS-22`, `disk.edge_erosion.enabled`): a
  soft-Heaviside threshold replaces the plain outer smoothstep, tearing the rim
  into advected fingers and holes instead of a clean edge.
- **Fractal LOD cascade** (`CKS-23`, `disk.lod.enabled`): per-sample octave count
  scales with screen-space footprint, so distant shots drop high-frequency
  shimmering while close-ups keep the full noise stack.

Every one of the flags above defaults to `false` (or a neutral value) in
`configs/render.yaml`; the regression suite specifically checks that flipping a
feature off reproduces the legacy bit-identical output, and that flipping it on
changes only what it's supposed to (see [Testing](#testing)).

---

## Quick start

Requires Python ≥ 3.10, an NVIDIA CUDA GPU (developed/verified on an **RTX 5060**,
sm_120 / Blackwell, via Taichi 1.7.4 CUDA JIT), and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Confirm the CUDA backend JITs on your card
python tests/cuda_smoke_test.py

# Run the full test suite (geodesic conservation, starmap, disk features, GPU regression)
pytest tests/

# Geodesic conservation only (E, L_z, Carter Q, null norm)
pytest tests/test_geodesic.py -v

# Quick CPU thumbnail (256x256, frame 0) - slow but dependency-light, no GPU needed
python scripts/thumb.py --res 256 --frame 0

# Full-HD GPU beauty smoke render (reports the Doppler asymmetry, ~4.3x)
python scripts/gpu_test.py

# DNGR Layer-B starless gate (checks the configured diffuse_map)
python scripts/check_starless_map.py

# Production: write the multi-channel EXR sequence for one frame
python scripts/export_exr.py --frame 0
```

> **GPU backend is locked to `ti.init(arch=ti.cuda)` — never `ti.gpu`.** `ti.gpu`
> selects Metal on macOS and can silently fall back to CPU on Windows.

### Cold compile vs. warm render

The main GPU kernel is large — expect a **long first compile**: roughly 27 minutes
for the default beauty kernel, and upward of 80 minutes if `disk.scatter.enabled`
is on (LLVM IR optimization is super-linear in kernel size, and it's CPU-bound —
the GPU sits near idle while it compiles). After that, Taichi's **offline cache**
reuses the compiled kernel: a warm single-frame render is about **0.15 s**, and a
second process rendering a whole sequence pays no further compile cost. For fast
iteration, set `render.advanced_optimization: false` and `render.cfg_optimization:
false` in `configs/render.yaml` to trade a small runtime cost for a ~20× faster
cold compile (roughly 4 minutes) — not intended for production frames.

---

## Physics rigor

- **Coordinates:** Cartesian Kerr-Schild `(t, x, y, z)`, spin axis `+z`, metric
  signature `(− + + +)`. CKS is regular on the spin axis and across the horizon
  (no `1/sin²θ` pole), has an **exact closed-form inverse metric** (`g⁻¹ = η − f·l⊗l`,
  since the Kerr-Schild null covector `l` is null with respect to flat `η` — no
  numerical matrix inversion is ever needed), and gives every escaped ray a genuine
  Cartesian celestial direction with no coordinate-seam fallback. Boyer-Lindquist is
  retired/history only.
- **Single source of truth:** every GR/Kerr formula used anywhere in the codebase is
  transcribed verbatim from [`skills/kerr-physics/SKILL.md`](skills/kerr-physics/SKILL.md)
  and referenced by formula ID (`CKS-1` … `CKS-23`+). The project's hard rule,
  stated in `CLAUDE.md`: **never re-derive a formula in conversation or code** — if
  something looks wrong, flag it for human review instead of silently "fixing" it.
- **Conservation tests** (`tests/test_geodesic.py`): every integrated null geodesic
  is checked for drift in photon energy `E = −p_t`, axial angular momentum
  `L_z = x·p_y − y·p_x`, the Carter constant `Q` (via a CKS→BL diagnostic
  conversion), and the null condition `H = ½ g^{αβ}p_αp_β`, all held to tight
  tolerances over thousands of RK4 steps.
- **CPU reference + GPU twin discipline:** `metric.py`, `geodesic.py`, and `disk.py`
  are the frozen, numerically-validated CPU ground truth (excluded from ruff
  auto-formatting for exactly this reason). The Taichi GPU kernels in
  `taichi_renderer.py` are twins of that CPU math and are held to the reference to
  within about `1e-6` in dedicated tests (e.g. `test_noise_gpu.py`).

---

## Configuration

All tunable parameters live in [`configs/render.yaml`](configs/render.yaml), in
sections `black_hole`, `render`, `disk` (including `disk.volumetric`, `disk.noise`,
`disk.multiphase`, `disk.scatter`, `disk.edge_erosion`, `disk.lod`), `starmap`,
`starfield` (DNGR), `camera`, `thumb`, and `output`. **The YAML stores base
parameters only** — anything that is a *function* of them (`r_isco`, `r_plus`,
`disk.r_inner`, `disk.T_0`, the `disk.dynamics` time mapping) is derived at load by
[`src/renderer/kerr_params.py`](src/renderer/kerr_params.py)
(`resolve_config`, Formula `CKS-13`) and injected into the config dict. Never write
a derived literal directly into the YAML — it will silently desync the next time
the base parameter (e.g. `black_hole.spin`) is edited. The full field-by-field
reference is documented in `PROJECT.md` §4.

---

## Repository layout

```
Black/
├── configs/
│   └── render.yaml           ← single source of truth for ALL numerical parameters
├── skills/
│   └── kerr-physics/
│       └── SKILL.md          ← physics formula reference (mandatory, never re-derived)
├── src/
│   ├── blender/
│   │   └── export_camera.py  ← Blender script: exports camera_matrix.json (Phase 1)
│   └── renderer/
│       ├── metric.py         ← CKS Kerr metric + exact inverse + derivs (CKS-1..4) — FROZEN
│       ├── geodesic.py       ← CKS Hamiltonian RK4 null geodesic integrator (CKS-5/6/7) — FROZEN
│       ├── disk.py           ← accretion-disk gas physics (CKS-8/9 + F9 chroma) — FROZEN
│       ├── disk_flux.py      ← Page-Thorne flux LUT (Decision B, CKS-11)
│       ├── kerr_params.py    ← config resolver: derives r_isco/r_plus/T_0/dynamics (CKS-13)
│       ├── noise.py          ← procedural noise primitives, CPU truth + GPU twins (CKS-12)
│       ├── starmap.py        ← 16K HDRI loader, mip pyramid, CKS-10 celestial→UV
│       └── taichi_renderer.py ← GPU renderer: Pipe A + Pipe B + DNGR, split kernels
├── scripts/
│   ├── thumb.py               ← CPU preview renderer (development / QA)
│   ├── gpu_test.py            ← FHD GPU beauty smoke test
│   ├── export_exr.py          ← multi-channel RGBAZ EXR writer (Phase 2 entry point)
│   ├── ingest_stars.py        ← offline catalog ingest → point-star {θ',φ',flux_rgb}.npy
│   ├── showcase_disk.py       ← turbulent-disk showcase (inclined camera)
│   ├── check_starless_map.py  ← DNGR Layer-B starless gate
│   └── regrade.py               ← post-hoc EXR tonemap / regrade
├── tests/                    ← conservation laws, disk features, starmap, GPU regression
├── render_blackhole/         ← output EXR sequence (bh_####.exr)      — gitignored
├── render_spaceship/         ← Blender ship EXR sequence              — gitignored
├── star_image/                ← 16K HDRI starmap + Milky-Way + catalog — gitignored
├── PROJECT.md                 ← the single, complete project reference (read this)
├── CLAUDE.md / AGENTS.md      ← project instructions + physics policy
└── REFERENCE_dngr_paper.md    ← James et al. 2015 (DNGR / Interstellar) academic source
```

---

## Testing

```bash
pytest tests/                        # everything (GPU tests auto-skip without CUDA)
pytest tests/test_geodesic.py -v     # E, L_z, Q, null-norm conservation along geodesics
python tests/cuda_smoke_test.py      # confirms the sm_120 / Blackwell CUDA JIT
```

The test suite spans conservation physics, the GPU disk feature stack, and
production regression:

- **Frozen core:** `test_geodesic.py` (conservation), `test_kerr_params.py`
  (config resolver derivation).
- **Background / starfield:** `test_starmap.py`, `test_starfield_dngr.py`,
  `test_starfield_artifacts.py`, `test_ingest_stars.py`.
- **Disk feature stack:** one test module per feature —
  `test_disk_flux.py`, `test_disk_noise.py`, `test_noise_gpu.py`,
  `test_disk_source_function.py`, `test_disk_self_shadow.py`, `test_disk_flare.py`,
  `test_disk_multiphase.py`, `test_disk_scatter.py`, `test_disk_edge_erosion.py`,
  `test_disk_lod.py`, `test_disk_shear_cascade.py`, `test_disk_step_convergence.py`.
  Each asserts the feature is bit-identical when disabled and changes only the
  expected observable when enabled.
- **Production regression** (`test_gpu_regression.py`, marked `gpu`): renders a
  production frame at FHD and asserts no NaNs, right/left Doppler asymmetry in
  `[3.8, 4.9]` (right brighter), and disk peak magnitude within ±5 % of reference —
  golden frames disable `disk.noise.enabled` first, since noise is an art dial, not
  physics. GPU tests auto-skip cleanly on machines without CUDA.

---

## Documentation

- **[`PROJECT.md`](PROJECT.md)** — the single, complete project reference: the full
  pipeline, every file, the field-by-field configuration reference, shipped status,
  remaining work, and the DNGR background design record. **Start here for depth.**
- **[`skills/kerr-physics/SKILL.md`](skills/kerr-physics/SKILL.md)** — the
  authoritative GR/Kerr formula reference (`CKS-1`…`CKS-23`+, `Formula 2/3/9/10/13`).
  Mandatory; never re-derived. If you're extending the disk or geodesic physics,
  this file has to be extended first, not the code.
- **[`CLAUDE.md`](CLAUDE.md)** — project instructions and the physics-formula
  policy that every contributor (human or agent) follows.
- **[`REFERENCE_dngr_paper.md`](REFERENCE_dngr_paper.md)** — James, von Tunzelmann,
  Franklin & Thorne (2015), *Gravitational Lensing by Spinning Black Holes…*
  (Class. Quantum Grav. 32 065001) — the academic source for the DNGR background.

---

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
