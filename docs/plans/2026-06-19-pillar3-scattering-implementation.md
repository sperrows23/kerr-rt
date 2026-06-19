# Pillar 3 — Volumetric Single-Scattering + Henyey-Greenstein (CKS-20) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the cold dust (`ρ_cold`, from CKS-19) catch forward-scattered light from the hot inner edge, so optically-thin cloud edges between the inner disk and the camera glow with a cinematic rim-light / "silver lining" — without moving any existing golden frame.

**Architecture:** Today the disk march does pure emission–absorption: `disk_col += T⃗⊙(emission)`, `T⃗ *= exp(−dτ⃗)` with `dτ⃗ = κ⃗·ρ_cold·ds`. CKS-20 adds **single-scattering from the dominant illuminant** (the hot inner edge): the extinction grows to `dτ_ext⃗ = (κ⃗ + σ_s)·ρ_cold·ds` (scattering also removes forward light) and a new in-scatter source `J_scat = σ_s·ρ_cold·P(cosθ_s)·I_src·e^{−τ_src}` is injected toward the camera, where `P` is the Henyey-Greenstein phase, `σ_s = ϖ·κ` (grey, one albedo dial), `e^{−τ_src}` is the **already-baked** CKS-17 deep-shadow-map attenuation, and `cosθ_s = ŝ_src·ŝ_view` is a pure geometric dot product of two straight CKS rays (illuminant→sample, sample→camera). Default OFF ⇒ `σ_s = 0` ⇒ no `J_scat` and `dτ_ext⃗ = κ⃗·ρ_cold·ds` ⇒ **exactly CKS-19/14**, bit-identical.

**Tech Stack:** Python, NumPy (CPU twin in `src/renderer/disk.py`), Taichi CUDA (`src/renderer/taichi_renderer.py` — backend LOCKED to `ti.init(arch=ti.cuda)`), pytest (CPU + GPU-probe parity in `tests/`, regression in `tests/test_gpu_regression.py`). Physics source of truth: `skills/kerr-physics/SKILL.md` Formula **CKS-20** (already authored, status DESIGN — Task 1 ratifies its open decisions and flips it ACTIVE in Task 5).

---

## Global Constraints

Copied verbatim from CLAUDE.md / SKILL.md CKS-20 / the project memory. Every task implicitly includes these.

- **Physics is SKILL.md-only.** All GR/Kerr/RTE formulas follow `skills/kerr-physics/SKILL.md` CKS-20. Do **not** re-derive. If a formula seems wrong or under-specified, **flag for human review** — do not silently substitute. (CKS-20's three open decisions are ratified in Task 1; the straight-ray `ŝ` geometry and the `ρ_cold`-in-`J_scat` energy assembly are recorded there too.)
- **Backend locked:** `ti.init(arch=ti.cuda)`, never `ti.gpu`.
- **Coordinates:** Cartesian Kerr-Schild (t,x,y,z); spin axis +z; `r` is the BL radial coord, `z = r cosθ`; geometric units G=M=c=1; a=0.999.
- **No hardcoded numbers in source.** All numerical parameters live in `configs/render.yaml`; derived quantities come from `kerr_params.resolve_config` (CKS-13). The new `disk.scatter` block stores BASE dials only (`enabled`, `albedo`, `hg_g`, `inner_glow`); the inner-edge illuminant reference temperature is derived in code from `T_0` and `r_inner`, never stored.
- **Constraint 6 — default OFF ⇒ bit-identical.** `disk.scatter.enabled=false` (or an absent block) reproduces the CKS-19 march byte-for-byte; `tests/test_gpu_regression.py` goldens must stay identical.
- **Constraint 3 — no geodesic/Doppler contamination.** `J_scat`, `P`, `g_HG`, `ϖ`, `σ_s`, and the `ŝ` directions touch **no** `p_μ` / `u^μ` / `g` / `g⁴` / `f_PT`. The inner-edge illuminant color carries the CKS-11 inner-edge emission spectrum (rest-frame blackbody chroma) but the scattering kernel adds no new g factor.
- **Constraint 2 — energy bookkeeping.** Light removed from the forward beam by `σ_s` (`(κ+σ_s)·ρ_cold·ds` extinction) is the light re-injected by `J_scat·ds = σ_s·ρ_cold·P·I_src·e^{−τ_src}·ds`. `σ_s` must appear in BOTH the extinction and the in-scatter source.
- **Windows / cp949:** open every text/config/source file with `encoding="utf-8"`.
- **Docs-sync policy:** this work updates `skills/kerr-physics/SKILL.md` (CKS-20) and `PROJECT.md` §6/§7 (if present) in the SAME task that lands the code (Task 5).
- **JIT gate discipline (mirror `_MP_COMPILE`):** scattering is gated by a module-level Python bool `_SCATTER_COMPILE` consumed via `ti.static(...)`, NOT a runtime field read. OFF ⇒ the scatter body is never emitted into the mega-kernel ⇒ the default path compiles exactly as before. Toggling `disk.scatter.enabled` therefore costs a one-time recompile (deliberate, same trade-off as multiphase). `setup_renderer` re-runs `ti.init`, clearing the kernel cache so the new value takes effect.

---

## Key design decisions (read before starting)

1. **`_SCATTER_COMPILE` `ti.static` gate, NOT a buffer slot.** Scatter's per-sample body (a density re-eval, a shadow lookup, two normalizes, an HG eval) would bloat the LLVM inliner into a multi-hour JIT if emitted unconditionally — the exact failure `_MP_COMPILE` was created to avoid (taichi_renderer.py:178–187). So scatter mirrors it: a module bool set at setup from `disk.scatter.enabled`, read via `ti.static`. OFF ⇒ zero bytes emitted ⇒ provably bit-identical.

2. **Scatter is a SEPARATE `@ti.func _disk_scatter_cks`, called from the march under the static gate — `_disk_emit_cks` is left byte-for-byte untouched.** This guarantees the emission path's bit-identity (the regression goldens) by construction, and isolates the entire scatter surface in one new func. The cost is that `_disk_scatter_cks` re-evaluates `ρ_cold` (via `_disk_density_cks[1]`) and `τ_src` (via `_sample_shadow_tau`) that `_disk_emit_cks` also computes — a deliberate, scatter-ON-only duplication, chosen over widening `_disk_emit_cks`'s return type (which would risk the emission goldens).

3. **`σ_s = ϖ·κ` is GREY (one albedo dial); the scattered light is colored by the illuminant.** `σ_s = albedo · absorption_coeff` (a scalar). It adds the SAME scalar increment to all three `dτ_ext` channels (`dtau_v[c] += σ_s·ρ_cold·ds`). The scattered radiance gets its color from `I_src` (the inner-edge blackbody chroma), not from `σ_s`. `albedo = 0` ⇒ `σ_s = 0` ⇒ numerically identical to scatter-off even when the gate is compiled in (an energy/zero test).

4. **`I_src` = inner-edge blackbody chroma × `inner_glow` (physical color, amplitude dial).** The reference temperature is `T_inner = T_0·(6/r_inner)^0.75` — the SIMPLE model evaluated at `r_inner`, used regardless of `disk.temperature_model`. Rationale: the Page-Thorne flux `f_PT(r_inner)→0` at the zero-torque ISCO edge would make `T_inner→0` (a black illuminant); the simple form gives a stable hot color. `T_inner` is derived on the host and passed in as `scatter_T_inner`; the per-pixel `src_rgb = _blackbody_rgb(scatter_T_inner)·inner_glow` reuses the existing GPU blackbody (no CPU blackbody twin needed). This is one of CKS-20's three open decisions — **ratify it in Task 1.**

5. **`ŝ_src` and `ŝ_view` are STRAIGHT CKS rays, derived in-march — no direction field is baked.** `ŝ_src = normalize(x_sample − x_inner)` with `x_inner = (r_inner·cosφ, r_inner·sinφ, 0)` (the midplane inner edge at the sample's φ); `ŝ_view = normalize(x_cam − x_sample)` (the camera position `cx,cy,cz` is already a kernel arg). `cosθ_s = ŝ_src·ŝ_view`. Both are straight CKS rays — NOT geodesics — exactly the governance posture CKS-17 already established for the shadow ray, so `cosθ_s` is a pure geometric dot product with no metric and no `p_μ` (honors constraint 3). **Record this definition in Task 1's SKILL ratification** (the SKILL currently says "store/derive at bake"; we derive in-march).

6. **Single-extinction march assembly + `ρ_cold` in `J_scat`.** The per-step `dtau_v` (now `(κ⃗+σ_s)·ρ_cold·ds`) is the single extinction used for BOTH the running transmittance update AND the CKS-14 emission source-function factor `f = (1−e^{−dτ})/dτ` — the literal reading of CKS-20's "extinction uses dτ_ext⃗ (κ+σ_s)". `J_scat` carries an explicit `ρ_cold` (the SKILL's `J_scat` line omits it, but constraint 2's energy balance — re-injecting exactly what `σ_s·ρ_cold·ds` removed — requires it). **Both are recorded in Task 1.** `J_scat` is added as its own source (`disk_col += T⃗⊙J_scat·ds`); it does NOT pass through the emission `f` factor.

7. **`τ_src` reuse + graceful degradation.** `e^{−τ_src}` is `exp(−shadow_strength·_sample_shadow_tau(...))` — the identical expression `_disk_emit_cks` uses for `shadow_atten` (taichi_renderer.py:1647–1653). It requires the CKS-17 bake, so `_disk_scatter_cks` gates the lookup on `self_shadow==1` exactly as the emit path does; with `self_shadow` off, `τ_src=0` ⇒ `e^{−τ_src}=1` (unoccluded illuminant) and scatter still functions. CKS-20 depends on CKS-17 for source-occlusion, not for existence.

---

## File structure (what changes and why)

| File | Change | Responsibility |
|---|---|---|
| `skills/kerr-physics/SKILL.md` | Task 1: ratify CKS-20's 3 open decisions + record straight-ray `ŝ` geometry and the `ρ_cold`-in-`J_scat` / single-extinction assembly (status DESIGN→"ratified, wiring"). Task 5: flip status to ACTIVE + revision-history `v1.32`. | Physics source of truth. |
| `src/renderer/disk.py` | Add CPU `hg_phase(cos_theta, g)`. | CPU source-of-truth twin for the HG phase (parity + normalization tests). |
| `src/renderer/taichi_renderer.py` | Add `_SCATTER_COMPILE` bool; `_hg_phase` `@ti.func`; `_disk_scatter_cks` `@ti.func`; set `_SCATTER_COMPILE` in `_setup_disk_noise`; thread 4 scatter args + per-pixel `src_rgb` + the gated scatter term through `render_beauty_physics`; read `disk.scatter` + derive `T_inner` in `render_beauty_frame` and `render_beauty_frame_mb`. | GPU twin + wiring. |
| `configs/render.yaml` | Add `disk.scatter` block (default `enabled:false`). | Config (CKS-20 dials). |
| `scripts/showcase_disk.py` | Add `--scatter` / `--albedo` / `--hg-g` / `--inner-glow` dials + status line. | Look-dev A/B. |
| `tests/test_disk_scatter.py` | **New.** HG normalization + forward/back (CPU); HG GPU-vs-CPU parity; `_disk_scatter_cks` analytic probe (noise/shadow off); `albedo=0` identity; rim-light acceptance. | Physics correctness + acceptance. |
| `tests/test_gpu_regression.py` | Unchanged; re-run to prove `scatter` default-OFF goldens bit-identical. | Constraint-6 guard. |

---

## Task 1: Ratify CKS-20 in SKILL.md (lock the open decisions + record the assembly)

CKS-20 is authored but carries three **open decisions** and two assembly details the code depends on. Lock them in the source of truth FIRST so the implementation has an unambiguous spec. Docs-only; no kernel change.

**Files:**
- Modify: `skills/kerr-physics/SKILL.md` (CKS-20 block, ~lines 1809–1876)

- [ ] **Step 1: Write the decisions into the "Open decisions" block**

In `skills/kerr-physics/SKILL.md`, replace the **"Open decisions (ratify in the P3 spec — flagged, NOT decided)"** block (~lines 1863–1869) with a ratified block:

```markdown
**Decisions (ratified 2026-06-19, owner-approved):**
- **`I_src` model:** physical inner-edge ring radiance — `I_src = blackbody_chroma(T_inner)·inner_glow`,
  `T_inner = T_0·(6/r_inner)^0.75` (the SIMPLE model at r_inner, used regardless of
  `disk.temperature_model`; the Page-Thorne f_PT(r_inner)→0 at the ISCO edge would blacken the
  illuminant). `inner_glow` is a free amplitude dial; the COLOR is tied to the disk's own inner edge.
- **`σ_s` parametrization:** `σ_s = ϖ·κ` (grey), one albedo dial `ϖ = disk.scatter.albedo ∈ [0,1)`.
  `κ` is the grey `absorption_coeff`. The scattered light is colored by `I_src`, not `σ_s`.
- **`g_HG`:** a SINGLE forward HG lobe, default `g_HG = 0.6`. No two-term (forward+back) lobe — the
  single lobe delivers the rim-light; a second lobe is deferred (revisit only if back-scatter haze
  is wanted).

**Geometry (straight CKS rays, derived in-march — NOT baked):**
- `ŝ_src(x) = normalize(x − x_inner)`, `x_inner = (r_inner·cosφ, r_inner·sinφ, 0)`, `φ = atan2(y,x)` —
  the midplane inner-edge illuminant at the sample's azimuth.
- `ŝ_view(x) = normalize(x_cam − x)` — the camera direction (camera position is a kernel arg).
- `cosθ_s = ŝ_src·ŝ_view`. Both are STRAIGHT CKS rays (not geodesics), the same VISUALIZATION
  posture as the CKS-17 shadow ray, so the scattering angle is a pure geometric dot product — no
  metric, no p_μ (constraint 3).

**March assembly (single extinction; ρ_cold in J_scat — energy-consistent):**
- `dτ_ext⃗ = (κ⃗ + σ_s)·ρ_cold·ds` is the SINGLE per-step extinction, used for both the running
  transmittance T⃗ AND the CKS-14 emission source-function factor f = (1−e^{−dτ})/dτ.
- `J_scat·ds = σ_s·ρ_cold·P(cosθ_s)·I_src·e^{−τ_src}·ds` carries an EXPLICIT ρ_cold (constraint 2:
  re-inject exactly the σ_s·ρ_cold·ds removed from the forward beam). Added as its own source —
  `disk_col += T⃗ ⊙ (J_scat·ds)` — NOT through the emission f factor.
```

- [ ] **Step 2: Update the status line**

Change the CKS-20 header/status (~lines 1809–1818): `DESIGN 2026-06-16` → `DESIGN ratified 2026-06-19 (owner-approved); wiring in progress`. (Task 5 flips it to ACTIVE.) Leave the formula body (`σ_s = ϖ·κ`, `dτ_ext⃗`, `J_scat`, HG) unchanged — it is correct; only the open-decisions/geometry/assembly prose is added.

- [ ] **Step 3: Verify the ratification landed**

Run: `python -c "import io; t=io.open(r'skills/kerr-physics/SKILL.md', encoding='utf-8').read(); assert 'Decisions (ratified 2026-06-19' in t and 'Geometry (straight CKS rays' in t and 'March assembly (single extinction' in t; print('CKS-20 ratified')"`
Expected: prints `CKS-20 ratified`.

- [ ] **Step 4: Commit**

```bash
git add skills/kerr-physics/SKILL.md
git commit -m "docs(CKS-20): ratify P3 scatter decisions (I_src, sigma_s, g_HG) + straight-ray geometry + energy assembly"
```

---

## Task 2: Henyey-Greenstein phase — CPU twin + GPU `@ti.func` + parity

**Files:**
- Modify: `src/renderer/disk.py` (add `hg_phase`)
- Modify: `src/renderer/taichi_renderer.py` (add `_hg_phase` `@ti.func` next to `_blackbody_rgb`, ~line 1018)
- Create: `tests/test_disk_scatter.py`

- [ ] **Step 1: Write the failing CPU tests** (normalization over the sphere + forward/iso/back ordering)

Create `tests/test_disk_scatter.py`:

```python
"""CKS-20 single-scattering + Henyey-Greenstein tests.

CPU tests run anywhere; GPU tests are CUDA-mandatory (backend LOCKED to ti.cuda)
and skip cleanly without it.
"""
# pyright: reportInvalidTypeForm=false
import copy
import math

import numpy as np
import pytest


def test_hg_phase_normalized():
    """∫_{4π} P(cosθ) dΩ = 1 for representative g (HG is a normalized phase function)."""
    from renderer.disk import hg_phase
    th = np.linspace(0.0, math.pi, 4001)
    cos_th = np.cos(th)
    for g in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
        # ∫ P · 2π sinθ dθ over θ∈[0,π]; trapezoid on a fine grid.
        integ = np.trapz(hg_phase(cos_th, g) * 2.0 * math.pi * np.sin(th), th)
        assert abs(integ - 1.0) < 1e-3, f"g={g}: ∮P dΩ={integ}"


def test_hg_phase_forward_dominant():
    """g=0.6 ⇒ forward (cosθ=+1) > isotropic (cosθ=0) > back (cosθ=−1)."""
    from renderer.disk import hg_phase
    fwd = float(hg_phase(1.0, 0.6))
    iso = float(hg_phase(0.0, 0.6))
    bak = float(hg_phase(-1.0, 0.6))
    assert fwd > iso > bak
    assert abs(float(hg_phase(0.0, 0.0)) - 1.0 / (4.0 * math.pi)) < 1e-6  # g=0 ⇒ 1/4π
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_disk_scatter.py -k hg_phase -v`
Expected: FAIL — `ImportError: cannot import name 'hg_phase' from 'renderer.disk'`.

- [ ] **Step 3: Implement the CPU `hg_phase`**

Add to `src/renderer/disk.py` (near the other disk source functions; this is the CKS-20 phase, the CPU source of truth for the GPU twin):

```python
def hg_phase(cos_theta, g):
    """Henyey-Greenstein phase function P(cosθ) (SKILL.md CKS-20) — CPU source of truth.

    P(cosθ) = (1 − g²) / [4π (1 + g² − 2g cosθ)^{3/2}].  g∈(−1,1): g=0 isotropic
    (1/4π), g>0 forward-scattering (the rim-light lobe), g<0 back. Normalized:
    ∫_{4π} P dΩ = 1. Pure optics — no g-factor / metric (constraint 3).
    """
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    g = float(g)
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return ((1.0 - g2) / (4.0 * np.pi * denom ** 1.5)).astype(np.float64)
```

> If `import numpy as np` is not already at the top of `disk.py`, add it.

- [ ] **Step 4: Run to verify the CPU tests pass**

Run: `pytest tests/test_disk_scatter.py -k hg_phase -v`
Expected: PASS (both).

- [ ] **Step 5: Add the GPU `@ti.func` twin**

In `src/renderer/taichi_renderer.py`, immediately after `_blackbody_rgb` (ends ~line 1017), add:

```python
@ti.func
def _hg_phase(cos_theta, g):
    """Henyey-Greenstein phase (SKILL.md CKS-20) — GPU twin of disk.hg_phase.

    P = (1−g²)/[4π·denom^{3/2}], denom = 1+g²−2g·cosθ (>0 for |g|<1). Pure optics:
    no p_μ/u^μ/g/g⁴ (constraint 3). g=0 ⇒ 1/4π isotropic.
    """
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return (1.0 - g2) / (4.0 * math.pi * denom * ti.sqrt(denom))
```

- [ ] **Step 6: Write the failing GPU parity test**

Add to `tests/test_disk_scatter.py` (mirror the CUDA-guard helper used elsewhere — import `taichi_renderer as tr`, init CUDA, skip if unavailable):

```python
pytestmark_gpu = pytest.mark.gpu


def _cuda_or_skip():
    import taichi as ti
    from renderer import taichi_renderer as tr  # noqa: F401
    try:
        ti.init(arch=ti.cuda)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA unavailable: {e}")


@pytest.mark.gpu
def test_hg_phase_gpu_matches_cpu():
    """GPU _hg_phase == CPU hg_phase across cosθ∈[−1,1], g∈{−0.6,0,0.6}."""
    _cuda_or_skip()
    import taichi as ti
    from renderer import taichi_renderer as tr
    from renderer.disk import hg_phase

    cos_vals = np.linspace(-1.0, 1.0, 64).astype(np.float32)
    g_vals = np.array([-0.6, 0.0, 0.6], dtype=np.float32)
    cf = ti.field(ti.f32, shape=cos_vals.size)
    gf = ti.field(ti.f32, shape=g_vals.size)
    out = ti.field(ti.f32, shape=(g_vals.size, cos_vals.size))
    cf.from_numpy(cos_vals)
    gf.from_numpy(g_vals)

    @ti.kernel
    def probe():
        for gi, ci in ti.ndrange(g_vals.size, cos_vals.size):
            out[gi, ci] = tr._hg_phase(cf[ci], gf[gi])

    probe()
    got = out.to_numpy()
    for gi, g in enumerate(g_vals):
        ref = hg_phase(cos_vals, float(g)).astype(np.float32)
        np.testing.assert_allclose(got[gi], ref, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 7: Run to verify the GPU parity passes**

Run: `pytest tests/test_disk_scatter.py -k hg_phase_gpu -v`
Expected: PASS (or SKIP if no CUDA).

- [ ] **Step 8: Commit**

```bash
git add src/renderer/disk.py src/renderer/taichi_renderer.py tests/test_disk_scatter.py
git commit -m "feat(CKS-20): Henyey-Greenstein phase (CPU hg_phase + GPU _hg_phase twin)"
```

---

## Task 3: GPU `_disk_scatter_cks` source func + `_SCATTER_COMPILE` gate

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (add `_SCATTER_COMPILE` near `_MP_COMPILE` ~line 187; set it in `_setup_disk_noise` ~line 623–626; add `_disk_scatter_cks` `@ti.func` after `_disk_emit_cks` ~line 1691)
- Test: `tests/test_disk_scatter.py`

**Interfaces:**
- Consumes: `_disk_density_cks(...)→vec3` (index `[1]`=ρ_cold), `_sample_shadow_tau(u,φ,ζ)`, `_kerr_radius`, `_hg_phase` (Task 2).
- Produces: `_disk_scatter_cks(x, y, z, cx, cy, cz, a, r_inner, r_outer, r_isco, theta_half_bound, sigma_theta0, flare_beta, noise_enabled, noise_seed, t_disk, self_shadow, shadow_strength, absb_c, ds, albedo, hg_g, src_rgb) -> vec4(J_r, J_g, J_b, sigma_dtau)` where `J_*` is `J_scat·ds` per channel and `sigma_dtau = σ_s·ρ_cold·ds` (grey). Module bool `_SCATTER_COMPILE`.

- [ ] **Step 1: Add the `_SCATTER_COMPILE` module bool**

In `src/renderer/taichi_renderer.py`, right after `_MP_COMPILE: bool = False` (line 187), add:

```python
# CKS-20 single-scattering compile gate (same rationale as _MP_COMPILE above): the
# scatter body (a ρ_cold re-eval, a shadow lookup, two normalizes, an HG eval) would
# bloat the JIT if emitted unconditionally, so it is gated by this module bool via
# ti.static. OFF ⇒ no bytes emitted ⇒ the default path compiles exactly as before and
# golden frames are bit-identical. `_setup_disk_noise` sets it from disk.scatter.enabled;
# setup_renderer re-runs ti.init so the new value takes effect on the next compile.
_SCATTER_COMPILE: bool = False
```

- [ ] **Step 2: Set the gate from config in `_setup_disk_noise`**

Find where `_MP_COMPILE` is set (~lines 623–626: `global _MP_COMPILE` / `_MP_COMPILE = mp_enabled`). Immediately after it, add:

```python
    # CKS-20: compile the scatter branch only when disk.scatter.enabled (see _SCATTER_COMPILE).
    global _SCATTER_COMPILE
    _SCATTER_COMPILE = bool((d.get("scatter", {}) or {}).get("enabled", False))
```

> `d` is the `disk` config dict already in scope in this function (the same one `mp = d.get("multiphase", ...)` reads). If the local is named differently, use that name.

- [ ] **Step 3: Write the failing analytic probe test** (noise OFF + shadow OFF ⇒ `ρ_cold`, `τ_src` are analytic)

Add to `tests/test_disk_scatter.py`:

```python
@pytest.mark.gpu
def test_disk_scatter_cks_analytic():
    """With noise OFF (ρ_cold = bare Gaussian) and self_shadow OFF (e^{−τ_src}=1),
    _disk_scatter_cks returns σ_dτ = albedo·κ·ρ·ds and J = σ_dτ·P(cosθ_s)·src_rgb
    for a hand-placed sample — pins the σ_s=ϖ·κ assembly and the HG/geometry."""
    _cuda_or_skip()
    import taichi as ti
    from renderer import taichi_renderer as tr
    from renderer.disk import hg_phase

    # Single-phase scene, noise + shadow OFF ⇒ _disk_density_cks midplane density is
    # the bare Gaussian gauss(dz_ang/σ)=1 at ζ=0, edge-window≈1 well inside the band.
    tr.setup_disk_noise({"disk": {}})           # noise off ⇒ ρ_cold = bare Gaussian

    a = 0.999
    r_inner, r_outer = 4.0, 25.0
    sigma0, beta = 0.15, 0.0
    theta_half = 0.3
    absb_c, albedo, hg_g, ds = 0.8, 0.5, 0.6, 0.1
    src = (1.0, 0.7, 0.4)

    # Sample on the midplane (z=0) at r=8, φ=0.6; camera far on +x so ŝ_view is known.
    r = 8.0
    phi = 0.6
    x, y, z = r * math.cos(phi), r * math.sin(phi), 0.0
    cx, cy, cz = 60.0, 0.0, 0.0

    out = ti.field(ti.f32, shape=4)

    @ti.kernel
    def probe():
        sc = tr._disk_scatter_cks(
            x, y, z, cx, cy, cz, a, r_inner, r_outer, 4.0, theta_half,
            sigma0, beta, 0, 1234, 0.0, 0, 0.0, absb_c, ds, albedo, hg_g,
            tr.vec3(src[0], src[1], src[2]),
        )
        for c in ti.static(range(4)):
            out[c] = sc[c]

    probe()
    got = out.to_numpy()

    # ρ_cold at midplane, noise off ≈ 1.0 (bare Gaussian peak). σ_dτ = albedo·κ·ρ·ds.
    rho = 1.0
    sigma_dtau = albedo * absb_c * rho * ds
    assert abs(got[3] - sigma_dtau) < 1e-4, f"sigma_dtau {got[3]} vs {sigma_dtau}"

    # cosθ_s = ŝ_src·ŝ_view, both straight CKS rays.
    import numpy as _np
    s_src = _np.array([x - r_inner * math.cos(phi), y - r_inner * math.sin(phi), 0.0])
    s_src /= _np.linalg.norm(s_src)
    s_view = _np.array([cx - x, cy - y, cz - z])
    s_view /= _np.linalg.norm(s_view)
    cos_s = float(s_src @ s_view)
    P = float(hg_phase(cos_s, hg_g))
    j_r = sigma_dtau * P * src[0]          # e^{−τ_src}=1 (shadow off)
    assert abs(got[0] - j_r) < 1e-4, f"J_r {got[0]} vs {j_r}"
```

- [ ] **Step 4: Run to verify it fails**

Run: `pytest tests/test_disk_scatter.py::test_disk_scatter_cks_analytic -v`
Expected: FAIL — `AttributeError: module 'renderer.taichi_renderer' has no attribute '_disk_scatter_cks'` (or SKIP without CUDA).

- [ ] **Step 5: Implement `_disk_scatter_cks`**

In `src/renderer/taichi_renderer.py`, immediately after `_disk_emit_cks` (ends ~line 1691, before the Pipe-A kernel comment), add:

```python
@ti.func
def _disk_scatter_cks(
    x, y, z, cx, cy, cz, a, r_inner, r_outer, r_isco, theta_half_bound,
    sigma_theta0, flare_beta, noise_enabled, noise_seed, t_disk,
    self_shadow, shadow_strength, absb_c, ds, albedo, hg_g, src_rgb,
):
    """CKS-20 single-scatter source at one CKS sample → vec4(J_scat·ds RGB, σ_s·ρ_cold·ds).

    Single-scattering from the hot inner edge (the dominant illuminant):
        σ_s        = albedo · absb_c                      # ϖ·κ, grey (Decision 3)
        ρ_cold     = _disk_density_cks(...)[1]            # CKS-19 cold absorber
        ŝ_src      = normalize(x − x_inner), x_inner = (r_inner·cosφ, r_inner·sinφ, 0)
        ŝ_view     = normalize(x_cam − x)
        cosθ_s     = ŝ_src·ŝ_view                          # straight CKS rays (constraint 3)
        e^{−τ_src} = exp(−shadow_strength·τ_shadow)        # CKS-17 deep-shadow-map (=shadow_atten)
        J_scat·ds  = σ_s·ρ_cold·_hg_phase(cosθ_s, hg_g)·src_rgb·e^{−τ_src}·ds

    Returns vec4(J_r, J_g, J_b, σ_s·ρ_cold·ds). The caller adds σ_s·ρ_cold·ds to the
    grey extinction (so scattering removes forward light — constraint 2) and adds
    T⃗⊙(J·ds) to disk_col. Outside the slab band ⇒ zeros. Pure optics: no p_μ/u^μ/g/g⁴.
    Only compiled when _SCATTER_COMPILE (caller gates with ti.static); albedo=0 ⇒ zeros.
    """
    out = vec4(0.0, 0.0, 0.0, 0.0)
    r = _kerr_radius(x, y, z, a)
    cos_th = ti.min(ti.max(z / r, -1.0), 1.0)
    th = ti.acos(cos_th)
    dz_ang = th - 0.5 * math.pi
    if (ti.abs(dz_ang) < theta_half_bound) and (r >= r_inner) and (r <= r_outer):
        sigma_eff = sigma_theta0
        if flare_beta != 0.0:
            sigma_eff = sigma_theta0 * ti.pow(r / r_inner, flare_beta)
        dens = _disk_density_cks(
            x, y, r, dz_ang, sigma_theta0, flare_beta, r_inner, r_outer, r_isco,
            noise_enabled, noise_seed, t_disk, a,
        )
        rho_cold = dens[1]
        # e^{−τ_src}: the CKS-17 inner-edge-ray shadow (graceful: self_shadow==0 ⇒ 1).
        atten = 1.0
        if self_shadow == 1:
            u_n = ti.log(r / r_inner)
            phi_n = ti.atan2(y, x)
            zeta_n = dz_ang / sigma_eff
            atten = ti.exp(-shadow_strength * _sample_shadow_tau(u_n, phi_n, zeta_n))
        # Straight-CKS-ray scattering geometry (Decision 5).
        phi = ti.atan2(y, x)
        sx = x - r_inner * ti.cos(phi)
        sy = y - r_inner * ti.sin(phi)
        sz = z
        inv_s = 1.0 / ti.max(ti.sqrt(sx * sx + sy * sy + sz * sz), 1e-9)
        sx *= inv_s; sy *= inv_s; sz *= inv_s
        vx = cx - x
        vy = cy - y
        vz = cz - z
        inv_v = 1.0 / ti.max(ti.sqrt(vx * vx + vy * vy + vz * vz), 1e-9)
        vx *= inv_v; vy *= inv_v; vz *= inv_v
        cos_s = sx * vx + sy * vy + sz * vz
        phase = _hg_phase(cos_s, hg_g)
        sigma_s = albedo * absb_c                  # σ_s = ϖ·κ (grey)
        sigma_dtau = sigma_s * rho_cold * ds
        j = sigma_dtau * phase * atten             # σ_s·ρ_cold·P·e^{−τ_src}·ds (scalar)
        out = vec4(j * src_rgb[0], j * src_rgb[1], j * src_rgb[2], sigma_dtau)
    return out
```

- [ ] **Step 6: Run to verify the probe passes**

Run: `pytest tests/test_disk_scatter.py::test_disk_scatter_cks_analytic -v`
Expected: PASS (σ_dτ and J_r match the analytic values within 1e-4).

- [ ] **Step 7: Commit**

```bash
git add src/renderer/taichi_renderer.py tests/test_disk_scatter.py
git commit -m "feat(CKS-20): _disk_scatter_cks single-scatter source + _SCATTER_COMPILE gate"
```

---

## Task 4: Wire the march — `dτ_ext += σ_s`, `disk_col += T⃗⊙J_scat`, host plumbing

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (`render_beauty_physics` signature ~line 1838 + body ~1887/2018/2026; `render_beauty_frame` launch ~2591–2643; `render_beauty_frame_mb` launch)
- Test: `tests/test_disk_scatter.py`

**Interfaces:**
- Consumes: `_disk_scatter_cks` + `_SCATTER_COMPILE` (Task 3); `_blackbody_rgb` (existing).
- Produces: `render_beauty_physics(..., scatter_albedo: float, scatter_hg_g: float, scatter_inner_glow: float, scatter_T_inner: float)` — 4 new trailing args after `shadow_strength`.

- [ ] **Step 1: Write the `albedo=0` identity test** (gate compiled, but σ_s=0 ⇒ numerically identical to OFF)

Add to `tests/test_disk_scatter.py`:

```python
@pytest.mark.gpu
def test_scatter_albedo_zero_identical():
    """scatter.enabled:true but albedo:0 ⇒ σ_s=0 ⇒ frame identical to scatter OFF."""
    _cuda_or_skip()
    from renderer import taichi_renderer as tr
    base = _scatter_scene()                       # helper (Step 5 of Task 5 defines it)
    cfg_off = copy.deepcopy(base)
    cfg_off["disk"].pop("scatter", None)
    cfg_on0 = copy.deepcopy(base)
    cfg_on0["disk"]["scatter"] = {"enabled": True, "albedo": 0.0, "hg_g": 0.6, "inner_glow": 1.0}
    img_off = _render_scatter(cfg_off)
    img_on0 = _render_scatter(cfg_on0)
    np.testing.assert_allclose(img_on0, img_off, rtol=0, atol=1e-6)
```

> `_scatter_scene()` / `_render_scatter()` are defined in Task 5 Step 1 (a small edge-on backlit scene + a render helper mirroring `test_disk_noise.py`). If you implement Task 4 before Task 5, stub them by copying `test_disk_noise.py`'s smallest cfg + render helper; Task 5 finalizes them.

- [ ] **Step 2: Run — expect collection/skip until the helpers + config exist**

Run: `pytest tests/test_disk_scatter.py::test_scatter_albedo_zero_identical -v`
Expected: FAIL/ERROR (helper not yet defined) or SKIP (no CUDA). This guards Step 3–4; it must PASS once Task 5's helpers land.

- [ ] **Step 3: Add the 4 scatter args to `render_beauty_physics`**

In the signature (after `shadow_strength: float,` ~line 1837), add:

```python
    scatter_albedo: float,
    scatter_hg_g: float,
    scatter_inner_glow: float,
    scatter_T_inner: float,
```

Compute the per-pixel illuminant color ONCE, before the step loop. After `transm = vec3(1.0, 1.0, 1.0)` (~line 1892), add:

```python
        # CKS-20: inner-edge illuminant radiance I_src = blackbody(T_inner)·inner_glow.
        # Compiled only when scatter is on (ti.static) ⇒ off path bit-identical.
        src_rgb = vec3(0.0, 0.0, 0.0)
        if ti.static(_SCATTER_COMPILE):
            src_rgb = _blackbody_rgb(scatter_T_inner) * scatter_inner_glow
```

- [ ] **Step 4: Inject the scatter term into the march**

In the `if in_band:` block, the current sequence is (~lines 2017–2036):

```python
                    dtau = ev[3]
                    dtau_v = vec3(dtau * ext_r, dtau * ext_g, dtau * ext_b)
                    added = vec3(ev[0], ev[1], ev[2])
                    if source_function == 1 and dtau > _RTE_TAU_EPS:
                        ...
                        added = vec3(ev[0] * f[0], ev[1] * f[1], ev[2] * f[2])
                    disk_col += vec3(transm[0] * added[0], transm[1] * added[1], transm[2] * added[2])
                    ...
                    transm[0] *= ti.exp(-dtau_v[0])
                    transm[1] *= ti.exp(-dtau_v[1])
                    transm[2] *= ti.exp(-dtau_v[2])
```

Make exactly two gated insertions (everything else untouched ⇒ off path byte-identical):

**(a)** Immediately AFTER `dtau_v = vec3(dtau * ext_r, dtau * ext_g, dtau * ext_b)` and BEFORE `added = vec3(ev[0], ev[1], ev[2])`, insert:

```python
                    # CKS-20 single-scatter. σ_s adds to the grey extinction (so it removes
                    # forward light); J_scat is injected below. OFF (ti.static) ⇒ removed
                    # ⇒ dtau_v unchanged, no J term ⇒ exactly CKS-19 (constraint 6).
                    scatter_j = vec3(0.0, 0.0, 0.0)
                    if ti.static(_SCATTER_COMPILE):
                        sc = _disk_scatter_cks(
                            x, y, z, cx, cy, cz, a, r_inner, r_outer, r_isco,
                            theta_half_bound, sigma_theta0, flare_beta,
                            noise_enabled, noise_seed, t_disk, self_shadow, shadow_strength,
                            absb_c, local_h, scatter_albedo, scatter_hg_g, src_rgb,
                        )
                        sig = sc[3]
                        dtau_v = vec3(dtau_v[0] + sig, dtau_v[1] + sig, dtau_v[2] + sig)
                        scatter_j = vec3(sc[0], sc[1], sc[2])
```

> Placing this BEFORE the source-function `f` loop makes `dtau_v` the SINGLE extinction (κ+σ_s) used for both `f` and the transmittance update — Decision 6. `local_h` is the in-band step length (the same `ds` `_disk_emit_cks` received).

**(b)** The existing code is (in order): the `disk_col += vec3(transm[0]*added[0], ...)` line, then `contribution = transm[0] * (added[0] + added[1] + added[2])`, then `weighted_depth += ray_length * contribution`. Insert the gated scatter block on the line BETWEEN the `contribution = ...` assignment and `weighted_depth += ...` (so `contribution` already exists and the depth proxy includes the scattered glow):

```python
                    contribution = transm[0] * (added[0] + added[1] + added[2])
                    # CKS-20: add the in-scattered radiance as its own source (NOT through
                    # the emission f factor), attenuated by the running T⃗ like emission.
                    # OFF (ti.static) ⇒ removed ⇒ disk_col/contribution unchanged (bit-identical).
                    if ti.static(_SCATTER_COMPILE):
                        disk_col += vec3(transm[0] * scatter_j[0],
                                         transm[1] * scatter_j[1],
                                         transm[2] * scatter_j[2])
                        contribution += transm[0] * (scatter_j[0] + scatter_j[1] + scatter_j[2])
                    weighted_depth += ray_length * contribution
```

> Only the gated `if ti.static(_SCATTER_COMPILE):` block is new — the `contribution = ...` and `weighted_depth += ...` lines are the existing ones, shown for placement. When the gate is off the block vanishes and these two lines run exactly as today.

- [ ] **Step 5: Thread the args through `render_beauty_frame`**

In `render_beauty_frame` (~line 2448), after the `ext_rgb` block (~line 2586) and before `_alloc_frame`, add:

```python
    # CKS-20 single-scatter dials. Absent block ⇒ off (and _SCATTER_COMPILE=False ⇒ the
    # kernel has no scatter body). T_inner = simple model at r_inner (Decision 4): always
    # hot, avoids the page_thorne f_PT→0 ISCO-edge zero. Derived, never stored (CKS-13 rule).
    sc_cfg = d.get("scatter", {}) or {}
    scatter_albedo = float(sc_cfg.get("albedo", 0.5))
    scatter_hg_g = float(sc_cfg.get("hg_g", 0.6))
    scatter_inner_glow = float(sc_cfg.get("inner_glow", 1.0))
    scatter_T_inner = float(d["T_0"]) * (6.0 / float(d["r_inner"])) ** 0.75
```

At the `render_beauty_physics(...)` call, append the 4 args after `shadow_strength,` (~line 2642):

```python
        shadow_strength,
        scatter_albedo,
        scatter_hg_g,
        scatter_inner_glow,
        scatter_T_inner,
    )
```

- [ ] **Step 6: Thread the same args through `render_beauty_frame_mb`**

`render_beauty_frame_mb` (~line 2671) calls `render_beauty_physics` per motion-blur sub-sample. Add the identical `sc_cfg`/`scatter_*` derivation (copy the Step-5 block) and append the same 4 args to its `render_beauty_physics(...)` call. (Search within `render_beauty_frame_mb` for `shadow_strength,` in the kernel-launch arg list and append after it.)

> If `render_beauty_frame_mb` factors the arg list through a shared helper, add the 4 args once there. Verify by grepping `render_beauty_physics(` — every call site must pass 4 new trailing floats.

- [ ] **Step 7: Run the off-path regression + the albedo-zero identity**

Run: `pytest tests/test_gpu_regression.py -v`
Expected: PASS — every golden bit-identical (`scatter` absent ⇒ `_SCATTER_COMPILE=False` ⇒ no scatter body emitted).
Run: `pytest tests/test_disk_scatter.py::test_scatter_albedo_zero_identical -v`
Expected: PASS once Task 5's `_scatter_scene`/`_render_scatter` helpers exist (else run after Task 5 Step 1).

- [ ] **Step 8: Commit**

```bash
git add src/renderer/taichi_renderer.py tests/test_disk_scatter.py
git commit -m "feat(CKS-20): march injects sigma_s extinction + J_scat in-scatter (gated, off=bit-identical)"
```

---

## Task 5: Config, showcase dial, rim-light acceptance, regression, docs-sync

**Files:**
- Modify: `configs/render.yaml` (`disk:` block)
- Modify: `scripts/showcase_disk.py` (dials + status line)
- Modify: `tests/test_disk_scatter.py` (scene helpers + acceptance)
- Modify: `skills/kerr-physics/SKILL.md` (CKS-20 status→ACTIVE, revision history); `PROJECT.md` §6/§7 if present

- [ ] **Step 1: Add the scene helpers + the failing rim-light acceptance test**

Add to `tests/test_disk_scatter.py` (mirror `test_disk_noise.py`'s smallest render path — load the canonical edge-on camera, small res, the production beauty path; `_render_scatter` returns the HDR-linear RGB array):

```python
def _scatter_scene():
    """Edge-on, backlit cloudy disk: camera roughly opposite the inner edge through a
    ρ_cold clump, so its near (camera-side) edge faces forward-scatter (cosθ_s→+1).
    Noise ON (structured dust), multiphase ON (a ρ_cold absorber), self_shadow ON
    (τ_src defined). Mirror the cfg dict test_disk_noise.py builds; key bits:"""
    import copy
    from tests._scene_helpers import canonical_small_cfg  # or inline test_disk_noise's loader
    cfg = canonical_small_cfg()
    d = cfg["disk"]
    d.setdefault("noise", {})["enabled"] = True
    d["absorption_coeff"] = 2.0
    d["multiphase"] = {"enabled": True, "dust_correlation": -0.8, "dust_amp": 1.5,
                       "dust_sigma_frac": 1.0}
    d.setdefault("volumetric", {}).setdefault("self_shadow", {})["enabled"] = True
    return cfg


def _render_scatter(cfg):
    from renderer import taichi_renderer as tr
    tr.setup_renderer(cfg)                         # re-inits ti + sets _SCATTER_COMPILE
    # ... build the canonical small cam_frame exactly as test_disk_noise.py does ...
    return tr.render_beauty_frame(cfg, cam_frame, W, H, with_disk=True, lod_enabled=False)


@pytest.mark.gpu
def test_scatter_rim_light():
    """Forward HG (g=0.6) ⇒ scatter ON brightens the backlit (forward-scatter) cloud
    edges vs scatter OFF; the brightening is concentrated where cosθ_s→+1."""
    _cuda_or_skip()
    base = _scatter_scene()
    cfg_off = copy.deepcopy(base); cfg_off["disk"].pop("scatter", None)
    cfg_on = copy.deepcopy(base)
    cfg_on["disk"]["scatter"] = {"enabled": True, "albedo": 0.6, "hg_g": 0.6, "inner_glow": 2.0}
    img_off = _render_scatter(cfg_off)
    img_on = _render_scatter(cfg_on)
    lum_off = img_off.sum(axis=2)
    lum_on = img_on.sum(axis=2)
    brighter = (lum_on > lum_off + 1e-4)
    assert brighter.mean() > 0.01, "scatter did not brighten any appreciable region"
    assert lum_on.max() >= lum_off.max(), "scatter produced no new bright pixel"
    # Energy: scatter only ADDS in-scattered light (it never darkens below OFF by more
    # than the extra σ_s extinction allows); the net brightened area must dominate any
    # darkened area for a forward lobe on backlit edges.
    darker = (lum_on < lum_off - 1e-4)
    assert brighter.sum() > darker.sum(), "forward scatter should net-brighten the frame"
```

> The exact scene/cam loader must mirror what `tests/test_disk_noise.py` already uses (do NOT invent a new camera). If `test_disk_noise.py` exposes a reusable `_load_cfg`/`_render_small`, import those instead of `canonical_small_cfg`. Keep the resolution small (e.g. 128²) — the scatter kernel's first compile is the slow path.

- [ ] **Step 2: Run to verify it fails without the config**

Run: `pytest tests/test_disk_scatter.py::test_scatter_rim_light -v`
Expected: FAIL (no `disk.scatter` semantics yet / scene helper incomplete) or SKIP without CUDA.

- [ ] **Step 3: Add the config block**

In `configs/render.yaml`, under `disk:` (a sibling of `multiphase:`, after the multiphase block ~line 222), add:

```yaml
  scatter:                   # CKS-20 — volumetric single-scattering + Henyey-Greenstein.
                             # The cold dust (ρ_cold) catches forward-scattered light from the
                             # hot inner edge, so optically-thin cloud EDGES between the inner
                             # disk and the camera glow with a rim-light / "silver lining".
                             # Extinction grows to (κ+σ_s)·ρ_cold·ds (σ_s removes forward light);
                             # J_scat = σ_s·ρ_cold·P(cosθ_s)·I_src·e^{−τ_src} is injected toward
                             # the camera, P = Henyey-Greenstein, e^{−τ_src} = the CKS-17 shadow.
                             # enabled:false ⇒ σ_s=0 ⇒ exactly CKS-19, BIT-IDENTICAL (constraint 6).
                             # NOTE: toggling `enabled` changes the compiled kernel (ti.static
                             # _SCATTER_COMPILE gate) ⇒ a one-time recompile; OFF keeps the
                             # original fast JIT untouched. Depends on CKS-19 (ρ_cold) + CKS-17
                             # (τ_src; with self_shadow off, e^{−τ_src}=1 — unoccluded illuminant).
    enabled: false           # master switch. false ⇒ emission–absorption only (golden frames intact).
    albedo: 0.5              # single-scatter albedo ϖ ∈ [0,1): σ_s = ϖ·absorption_coeff (grey).
    hg_g: 0.6                # Henyey-Greenstein anisotropy g ∈ (−1,1): >0 forward (the rim-light
                             #   lobe), 0 isotropic, <0 back. ~0.6 = cinematic forward scatter.
    inner_glow: 1.0          # amplitude of the inner-edge illuminant I_src (its color is the
                             #   blackbody at T_inner = T_0·(6/r_inner)^0.75, derived in code).
```

- [ ] **Step 4: Run the acceptance + the full regression guard**

Run: `pytest tests/test_disk_scatter.py -v`
Expected: PASS — HG normalization/parity, the analytic probe, `albedo=0` identity, and the rim-light acceptance (forward scatter net-brightens the backlit edges).
Run: `pytest tests/test_gpu_regression.py -v`
Expected: PASS — every existing golden bit-identical (`scatter` defaults OFF).

- [ ] **Step 5: Add the `showcase_disk` dials**

In `scripts/showcase_disk.py`, mirror the `--multiphase` / `--extinction` pattern. After the `--extinction` argument (~line 204), add:

```python
    p.add_argument(
        "--scatter", action="store_true",
        help="enable disk.scatter (CKS-20): single-scattering + Henyey-Greenstein rim-light. "
        "Cold dust (rho_cold) catches forward-scattered inner-edge light. Implies multiphase "
        "is worthwhile (needs a rho_cold absorber) and self_shadow on (source occlusion).",
    )
    p.add_argument(
        "--albedo", type=float, default=None,
        help="override disk.scatter.albedo (implies --scatter): single-scatter albedo varpi in [0,1).",
    )
    p.add_argument(
        "--hg-g", type=float, default=None,
        help="override disk.scatter.hg_g (implies --scatter): HG anisotropy in (-1,1), >0 forward.",
    )
    p.add_argument(
        "--inner-glow", type=float, default=None,
        help="override disk.scatter.inner_glow (implies --scatter): inner-edge illuminant amplitude.",
    )
```

After the `--extinction` apply block (~line 303), add the scatter apply (mirroring the multiphase `setdefault` pattern at ~line 284):

```python
    # CKS-20 single-scatter. Default OFF in the YAML; --scatter / any --albedo/--hg-g/--inner-glow
    # turns it on. Recompiles the kernel (ti.static _SCATTER_COMPILE gate).
    if args.scatter or args.albedo is not None or args.hg_g is not None \
            or args.inner_glow is not None:
        sc = cfg["disk"].setdefault("scatter", {})
        sc["enabled"] = True
        if args.albedo is not None:
            sc["albedo"] = args.albedo
        if args.hg_g is not None:
            sc["hg_g"] = args.hg_g
        if args.inner_glow is not None:
            sc["inner_glow"] = args.inner_glow
```

In the status-line block (~line 348–368), add a `scatter=` field mirroring `multiphase=`:

```python
    sc_cfg = cfg["disk"].get("scatter", {})
    sc_status = (
        f"on(varpi={sc_cfg.get('albedo', 0.5)},g={sc_cfg.get('hg_g', 0.6)})"
        if sc_cfg.get("enabled") else "off"
    )
```

and append `scatter={sc_status}` to the printed status f-string (next to `multiphase=`).

- [ ] **Step 6: Smoke the showcase dial** (a tiny render proves the CLI path wires end-to-end)

Run: `python scripts/showcase_disk.py --scatter --width 160 --height 90 --frame 0 --out /tmp/scatter_smoke.exr`
Expected: completes and writes the EXR; the status line shows `scatter=on(...)`. (First run pays the scatter-kernel JIT — minutes; this is expected, not a hang.)

- [ ] **Step 7: Docs-sync (SKILL ACTIVE flip + PROJECT.md) + commit**

In `skills/kerr-physics/SKILL.md`, flip the CKS-20 status from `DESIGN ratified 2026-06-19; wiring in progress` to `ACTIVE (wired 2026-06-19)`, and add a revision-history entry: `v1.32 — CKS-20 wired: single-scatter + HG (σ_s=ϖ·κ grey, I_src=inner-edge blackbody·inner_glow, single forward g=0.6, straight-CKS-ray geometry, _SCATTER_COMPILE gate); default OFF bit-identical`. If `PROJECT.md` exists, update §6 (formula status) and §7 (pillar roadmap: P3 COMPLETE) per the docs-sync policy. Add the CKS-20 file-map line to SKILL.md's "File locations" (`_disk_scatter_cks` + `_hg_phase` in `taichi_renderer.py`; `hg_phase` in `disk.py`).

```bash
git add configs/render.yaml scripts/showcase_disk.py tests/test_disk_scatter.py skills/kerr-physics/SKILL.md PROJECT.md
git commit -m "feat(CKS-20): wire scatter config + showcase dial + rim-light acceptance; SKILL v1.32 ACTIVE"
```

---

## Self-review notes

- **Spec coverage (tip.md Pillar 3 / SKILL CKS-20):** in-scattering coefficient `σ_s` added to the transport equation → Task 4 (`dτ_ext += σ_s`); Henyey-Greenstein `P(cosθ)` → Task 2 (CPU+GPU) + Task 3 (used in `_disk_scatter_cks`); single-scatter from the dominant illuminant (`J_scat`, `τ_src`, `ŝ_src`, `I_src`) → Task 3; forward `g>0.5` rim-light acceptance → Task 5 `test_scatter_rim_light`; the three CKS-20 open decisions → Task 1; config `scatter.{enabled,albedo,hg_g,inner_glow}` → Task 5.
- **Type consistency:** `_disk_scatter_cks` returns `vec4(J_r, J_g, J_b, σ_s·ρ_cold·ds)` in Tasks 3–5; its caller reads `sc[0..2]` as `scatter_j` and `sc[3]` as `sig`. `hg_phase` (CPU, `disk.py`) / `_hg_phase` (GPU, `taichi_renderer.py`) are the paired names. `_SCATTER_COMPILE` (module bool) / `disk.scatter.enabled` (config) are the paired gate. The 4 new kernel args — `scatter_albedo, scatter_hg_g, scatter_inner_glow, scatter_T_inner` — are appended (in that order) at BOTH `render_beauty_physics` call sites (`render_beauty_frame` + `render_beauty_frame_mb`).
- **Bit-identity (constraint 6):** every task keeps `disk.scatter` absent/`enabled:false` ⇒ `_SCATTER_COMPILE=False` ⇒ the scatter body is never emitted ⇒ goldens byte-identical. The dedicated guards are Task 4 Step 7 (regression) and `test_scatter_albedo_zero_identical` (gate compiled, `albedo=0` ⇒ σ_s=0 ⇒ identical). `_disk_emit_cks` is left literally untouched (Decision 2), so the emission path cannot drift.
- **Energy bookkeeping (constraint 2):** `σ_s·ρ_cold·ds` enters BOTH the extinction (`dtau_v += sig`, Task 4a) and the in-scatter source (`J_scat·ds`, the `j = sigma_dtau·phase·atten` line in `_disk_scatter_cks`) — the light removed equals the light re-injected, to single-scatter order.
- **No new physics re-derived:** the formulas come verbatim from SKILL CKS-20; Task 1 ratifies the three knobs and RECORDS (does not invent) the straight-CKS-ray `ŝ` geometry and the `ρ_cold`-in-`J_scat` assembly as the owner-approved reading. The **single-extinction** choice (σ_s enters the CKS-14 emission `f` factor too) — the literal reading of CKS-20's "extinction uses dτ_ext⃗ (κ+σ_s)" — was **confirmed by the physics owner on 2026-06-19**; no open physics items remain. (Recorded in Decision 6 and Task 1.)
- **Open items the worker must resolve from the codebase, not invent:** (1) the exact `disk`-config local name and insertion point in `_setup_disk_noise` (Task 3 Step 2 — search for the `_MP_COMPILE` set); (2) `test_disk_noise.py`'s actual small-scene cfg loader + render helper to mirror in `_scatter_scene`/`_render_scatter` (Task 5 Step 1 — do NOT invent a camera); (3) every `render_beauty_physics(` call site (Task 4 Step 6 — grep to be exhaustive, including any motion-blur helper).
```
