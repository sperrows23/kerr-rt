# Kerr Black Hole Offline Renderer

## Project Overview

Offline renderer for a Kerr black hole. Three-phase pipeline:

```
Phase 1 (Blender)            → camera_matrix.json + ship_####.exr
Phase 2 (Taichi GPU tracer)  → bh_####.exr
Phase 3 (Blender Compositor) → final composited video
```

- **Phase 1**: Blender animates the spaceship, exports camera matrices per frame, renders ship EXR sequence.
- **Phase 2**: Taichi kernel traces photon geodesics in Kerr spacetime, produces black hole / accretion disk EXR sequence.
- **Phase 3**: Blender compositor merges both EXR sequences with glow, aberration, and lens effects.

---

## Unit and Coordinate Conventions

| Item | Value / Convention |
|------|-------------------|
| Units | Geometric: G = M = c = 1 |
| Coordinates | Boyer-Lindquist (t, r, θ, φ) |
| Metric signature | (− + + +) |
| Spin parameter | a = 0.999 (near-extremal) |
| GPU backend | `ti.init(arch=ti.cuda)` — **locked**, never use `ti.gpu` |

---

## CRITICAL: Physics Formula Policy

**All general relativity and Kerr physics formulas must strictly follow `skills/kerr-physics/SKILL.md`.**

- Do NOT re-derive any formula from scratch during any conversation or coding session.
- Re-derivation introduces sign errors, index mismatches, and normalization mistakes.
- If a formula seems wrong, **flag it for human review** — do not silently substitute a re-derived version.
- The skill file is the single source of truth. If the skill file is incomplete, ask the user to extend it before writing code.

---

## Build and Environment Commands

```bash
# Install dependencies
uv sync

# Run all tests
pytest tests/

# Geodesic conservation tests (E, L, Q conserved along geodesic)
pytest tests/test_geodesic.py -v

# Quick thumbnail render (256×256, frame 0)
python scripts/thumb.py --res 256 --frame 0

# CUDA smoke test (confirms sm_120 / Blackwell JIT)
python tests/cuda_smoke_test.py
```

---

## Known Resolved Issues

- **RTX 5060 (sm_120, Blackwell)**: confirmed working with Taichi 1.7.4 via CUDA JIT.
  - Use `ti.init(arch=ti.cuda)`, **not** `ti.gpu`.
  - `ti.gpu` selects Metal on macOS and may silently fall back to CPU on Windows — always use `ti.cuda` explicitly.

---

## Skills (Source-of-Truth Files)

| Skill | Purpose |
|-------|---------|
| `skills/kerr-physics/SKILL.md` | All GR and Kerr metric formulas — **mandatory reference** |
   Decision A: ZAMO tetrad (Formula 7)
   Decision B: Simple temperature model T = T_0·(6/r)^0.75
| `skills/taichi-conventions/` | Taichi kernel patterns, field layout, JIT conventions (to be written) |
| `skills/blender-pipeline/` | Blender headless scripting, camera export, compositor nodes (to be written) |

---

## Config-Driven Development

**All numerical parameters must live in `configs/render.yaml`. No hardcoded values in source.**

Parameters that must come from config (non-exhaustive):

- Spin `a`, ISCO radius `r_isco`, horizon radius `r_plus`
- `WIDTH`, `HEIGHT`, thumb dimensions
- `max_steps` for integrator pipes A and B
- Bounding box ranges (`r_max`, `theta_half_width`)
- Camera `radius`, `fov_deg`
- Output directory paths and filename prefixes

---

## Directory Layout

```
src/
  renderer/     ← GPU ray tracing, geodesic integrator, Taichi kernels
  blender/      ← camera matrix export, headless render scripts
tests/          ← pytest conservation tests, golden image regression
configs/        ← YAML/JSON parameter files
skills/         ← Codex skill files (kerr-physics, taichi-conventions, etc.)
render_blackhole/ ← output EXR sequence from Taichi (bh_####.exr)
render_spaceship/ ← output EXR sequence from Blender (ship_####.exr)
```
