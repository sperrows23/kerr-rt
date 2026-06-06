# Kerr Black Hole Offline Renderer

A physically-based, GPU-accelerated offline renderer for a near-extremal **Kerr
(rotating) black hole** and its accretion disk — built around a Taichi/CUDA photon
**geodesic tracer** that bends light through curved spacetime exactly as general
relativity prescribes, then composited with a Blender-animated spaceship into a
finished shot.

> Geometric units `G = M = c = 1` · Boyer–Lindquist coordinates `(t, r, θ, φ)` ·
> metric signature `(− + + +)` · spin `a = 0.999` (near-extremal).

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
  poles and at the horizon). The renderer uses horizon-stable state variables, a
  factored discriminant, a singularity-free polar potential, and Kahan-compensated
  RK4 so the geodesics actually conserve their constants of motion.
- **A single physics source of truth.** Every GR formula is copied *verbatim* from
  [`skills/kerr-physics/SKILL.md`](skills/kerr-physics/SKILL.md) and referenced by
  number in the code. Formulas are **never re-derived ad hoc** — re-derivation is how
  sign errors and normalization bugs creep in. If a formula looks wrong, it gets
  flagged for human review, not silently replaced.
- **Config-driven, no magic numbers.** Spin, ISCO radius, resolution, step caps,
  bounding boxes, camera framing — all of it lives in
  [`configs/render.yaml`](configs/render.yaml). No physics or render literal is
  hardcoded in source.

---

## What it does

The renderer traces two photon sub-pipelines **per pixel** inside split GPU kernels:

- **Pipe A — background / lensing.** Trace a photon *backward* from the camera
  through curved spacetime. When it escapes to infinity, look up where it came from
  on the sky. Two background modes are supported:
  - `texture` (default): sample a gravitationally-lensed 16K equirect starmap with
    screen-space–Jacobian mip anti-aliasing.
  - `dngr`: the *Interstellar*-style two-layer model — a **point-star catalog** that
    stays sharp and *brightens* under lensing (magnification → flux, not smear),
    plus an anisotropically-filtered diffuse Milky-Way band.
- **Pipe B — accretion disk.** Accumulate volumetric disk emission along the same
  geodesic, with `g⁴` relativistic Doppler/gravitational beaming so the approaching
  limb of the disk blazes and the receding limb darkens. Composited in front of the
  background.

The result is a multi-channel scene-linear HDR EXR per frame, ready for compositing.

**Signature verification:** for the `a = 0.999` edge-on camera the right/left disk
Doppler brightness asymmetry comes out to ≈ **7.8×** — the renderer is gated on this
physical number, not on a golden image alone.

### Three-phase production pipeline

```
Phase 1  Blender         → camera_matrix.json  +  render_spaceship/ship_####.exr
Phase 2  Taichi (CUDA)   → render_blackhole/bh_####.exr   (RGBA-linear + Z depth)
Phase 3  Blender         → composite: deep merge, bloom, chromatic aberration, lens
(Phase 4 DaVinci Resolve → grade + delivery — recommended finishing)
```

- **Phase 1 — Blender:** animates the spaceship (the only hero element), exports a
  per-frame camera track, and renders the ship EXR sequence.
- **Phase 2 — Taichi GPU tracer (the focus of this repo):** traces the Kerr photon
  geodesics and produces the black-hole / accretion-disk EXR sequence.
- **Phase 3 — Blender compositor:** merges the two EXR sequences with glow,
  aberration, and lens effects.

> **This repository implements Phases 1–2.** The Phase-3 compositor `.blend` and the
> Phase-4 Resolve grade are documented, recommended workflows rather than checked-in
> assets — see [`PROJECT.md`](PROJECT.md) for the full node-graph and grading recipe.

---

## Quick start

Requires Python ≥ 3.10, an NVIDIA CUDA GPU (developed/verified on an **RTX 5060**,
sm_120 / Blackwell, via Taichi 1.7.4 CUDA JIT), and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Confirm the CUDA backend JITs on your card
python tests/cuda_smoke_test.py

# Run the full test suite (geodesic conservation, starmap, GPU regression)
pytest tests/

# Quick CPU thumbnail (256×256, frame 0) — slow but dependency-light, no GPU needed
python scripts/thumb.py --res 256 --frame 0

# Full-HD GPU beauty smoke render (reports the Doppler asymmetry ≈ 7.8×)
python scripts/gpu_test.py

# Production: write the multi-channel EXR sequence
python scripts/export_exr.py
```

> **GPU backend is locked to `ti.init(arch=ti.cuda)` — never `ti.gpu`.** `ti.gpu`
> selects Metal on macOS and can silently fall back to CPU on Windows.

---

## Repository layout

```
Black/
├── configs/
│   └── render.yaml          ← single source of truth for ALL numerical parameters
├── skills/
│   └── kerr-physics/
│       └── SKILL.md         ← physics formula reference (mandatory, never re-derived)
├── src/
│   ├── blender/
│   │   └── export_camera.py ← Blender script: exports camera_matrix.json (Phase 1)
│   └── renderer/
│       ├── metric.py        ← Kerr metric            (Formula 1)
│       ├── geodesic.py      ← Mino-time RK4 null geodesic integrator (Formula 6)
│       ├── disk.py          ← accretion-disk gas physics (F3/4/5/8/9) — FROZEN
│       ├── starmap.py       ← 16K HDRI loader, mip pyramid, UV mapping
│       └── taichi_renderer.py ← GPU renderer: Pipe A + Pipe B + DNGR, split kernels
├── scripts/
│   ├── thumb.py             ← CPU preview renderer (development / QA)
│   ├── gpu_test.py          ← FHD GPU beauty smoke test
│   ├── ingest_stars.py      ← offline catalog ingest → point-star {θ',φ',flux_rgb}.npy
│   ├── seam_diagnostics.py  ← spin-axis seam isolation tools (off the render path)
│   └── export_exr.py        ← multi-channel RGBAZ EXR writer (OpenImageIO)
├── tests/                   ← conservation laws, starmap, ingest, GPU regression
├── render_blackhole/        ← output EXR sequence (bh_####.exr)      — gitignored
├── render_spaceship/        ← Blender ship EXR sequence              — gitignored
├── star_image/              ← 16K HDRI starmap + Milky-Way + catalog — gitignored
├── PROJECT.md               ← the single, complete project reference (read this)
├── CLAUDE.md / AGENTS.md    ← project instructions + physics policy
└── REFERENCE_dngr_paper.md  ← James et al. 2015 (DNGR / Interstellar) academic source
```

---

## How the physics fits together

| Stage | Formula(s) | What happens |
|-------|-----------|--------------|
| Camera → photon | F7 (ZAMO tetrad) | Convert the Blender camera basis into a local Boyer–Lindquist triad and launch the photon momentum from a zero-angular-momentum observer. |
| Trace | F6 / F11 / F12 | Integrate the null geodesic in Mino time with RK4, in the horizon-stable `[y = r − r₊, u = cos θ, φ, t, …]` state; factored `Δ = y(y+2k)` and the singularity-free `Θ_u(u)` keep it finite at the horizon and poles. |
| Disk emission | F3 / F4 / F5 / F8 / F9 | Gas 4-velocity (circular outside ISCO, plunging inside), `g`-factor, and `g⁴` volumetric beaming with chromaticity-only blackbody color. |
| Background | F10 / F13 | Mip-LOD lensed starmap (`texture`), or the two-layer DNGR point-star + diffuse model (`dngr`). |

The CPU reference path (`metric.py`, `geodesic.py`, `disk.py`) is the numerically
frozen ground truth that the Taichi GPU port is validated against — geodesic tests
assert `E`, `Lz`, `Q`, and the null norm are conserved to tight tolerances over
thousands of integration steps.

---

## Configuration

All tunable parameters live in [`configs/render.yaml`](configs/render.yaml), grouped
into `black_hole`, `render`, `disk`, `starmap`, `starfield` (DNGR), `camera`,
`thumb`, and `output` sections. The full field-by-field reference is documented in
[`PROJECT.md` §4](PROJECT.md). Nothing in `src/` or `scripts/` hardcodes a physics or
render value — change the YAML, not the code.

---

## Testing

```bash
pytest tests/                       # everything (GPU tests auto-skip without CUDA)
pytest tests/test_geodesic.py -v    # E, Lz, Q, null-norm conservation along geodesics
python tests/cuda_smoke_test.py     # confirms the sm_120 / Blackwell CUDA JIT
```

The GPU regression (`tests/test_gpu_regression.py`) drives the production renderer at
FHD and asserts: no NaNs, right/left Doppler asymmetry ∈ [7.0, 8.5] (right brighter),
and disk peak within ±5 % of reference. It is marked `gpu` and skips cleanly on
machines without CUDA.

---

## Documentation

- **[`PROJECT.md`](PROJECT.md)** — the single, complete project reference: the full
  pipeline, every file, configuration reference, shipped status, remaining work, and
  the DNGR background design record. **Start here for depth.**
- **[`skills/kerr-physics/SKILL.md`](skills/kerr-physics/SKILL.md)** — the
  authoritative GR/Kerr formula reference. Mandatory; never re-derived.
- **[`CLAUDE.md`](CLAUDE.md)** — project instructions and the physics-formula policy.
- **[`REFERENCE_dngr_paper.md`](REFERENCE_dngr_paper.md)** — James, von Tunzelmann,
  Franklin & Thorne (2015), *Gravitational Lensing by Spinning Black Holes…*
  (Class. Quantum Grav. 32 065001) — the academic source for the DNGR background.

---

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
