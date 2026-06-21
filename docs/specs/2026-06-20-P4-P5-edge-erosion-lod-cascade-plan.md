# P4 + P5 — KH Edge Erosion (CKS-22) & Fractal LOD Cascade (CKS-23) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two default-OFF disk refinements — a Kelvin-Helmholtz noise-threshold that shreds the outer disk edge into vacuum (CKS-22), and a distance-driven fractal LOD that gates disk-noise octaves to hold detail density ~constant across camera distance (CKS-23).

**Architecture:** Both are amplitude/sampling-only (never touch `p_μ/u^μ/g/g⁴/f_PT`/chroma). P4 replaces the CKS-12 §3 *outer* smoothstep envelope `win_out` in `_disk_density_cks` with a soft-Heaviside clip `H_soft(win_out − τ_KH·N_KH)`, where `N_KH` is a high-freq simplex advected by the SAME §2 dual-phase shear as the density (modeled on `_disk_blended_m`). P5 threads a per-sample octave count `n_oct = clamp(N_max − log₂(J/J₀), N_min, N_max)` (J = pixel-cone footprint at the sample's camera distance) down the density noise chain and applies a smooth per-octave gate `g_o = clamp(n_oct − o, 0, 1)` inside a gated fBm. Each pillar's disabled branch is bit-identical to current goldens.

**Tech Stack:** Python, Taichi 1.7.4 (CUDA backend, `ti.init(arch=ti.cuda)` — LOCKED), NumPy CPU reference twins, pytest. Geometric units G=M=c=1, Cartesian Kerr-Schild coords, spin a=0.999.

## Global Constraints

- **Backend LOCKED:** `ti.init(arch=ti.cuda)`, never `ti.gpu` (CLAUDE.md / SKILL.md).
- **Default-OFF bit-identity (constraint 6):** `edge_erosion.enabled:false` and `lod.enabled:false` must take a branch that is byte-for-byte the current kernel; `tests/test_gpu_regression.py` goldens stay valid.
- **No formula re-derivation:** all physics/sampling formulas come from `skills/kerr-physics/SKILL.md`; CKS-22 / CKS-23 are ratified there (Tasks 1.x) BEFORE kernel code.
- **CPU/GPU twins byte-aligned:** every GPU `@ti.func` has a NumPy reference in `src/renderer/noise.py`; parity within `_SATOL` (`tests/test_noise_gpu.py`).
- **No `ti.random`:** deterministic integer-hashed seeds from config; same seed + same `t_disk` ⇒ identical frame (constraint 7).
- **φ-seam-free (constraint 5):** azimuthal frequencies that index a lattice are integers (2π period).
- **Config-driven (no hardcoded numerics):** all dials in `configs/render.yaml`; no derived literal stored in YAML (CKS-13 resolver owns derived values). Neither pillar changes the resolver.
- **Docs-sync (same task):** every landed change updates `skills/kerr-physics/SKILL.md` + `PROJECT.md` §6/§7.
- **Spin/params come from config:** `a`, `r_inner`, `r_outer`, `max_step_vfrac`, `sigma_theta0` are read, never hardcoded.

**Design source of truth:** `docs/specs/2026-06-20-P4-P5-edge-erosion-lod-cascade.md` (RATIFIED; all 8 decisions resolved). Key resolved decisions: P4 advects with §2 shear only; `w_soft` floor `k_soft=1`; erode the SHARED `win_out` (both ρ_hot & ρ_cold); config home `disk.edge_erosion`. P5 isotropic scalar J; octaves-only (defer `dλ`); `J₀` base dial; relational two-distance golden.

---

# PART P4 — Kelvin-Helmholtz edge erosion (CKS-22)

**File structure (P4):**
- `skills/kerr-physics/SKILL.md` — promote CKS-22 to a full formula entry.
- `src/renderer/noise.py` — add `NSEED_KH`; CPU twins `kh_field`, `kh_erode_winout`.
- `src/renderer/taichi_renderer.py` — add `_NI_EROS_*` indices + `_NSEED_KH`; pack in `_setup_disk_noise`; add `_kh_field` `@ti.func`; apply the clip in `_disk_density_cks`.
- `configs/render.yaml` — add `disk.edge_erosion` block.
- `tests/test_noise.py`, `tests/test_noise_gpu.py`, `tests/test_disk_edge_erosion.py` (new), `tests/test_gpu_regression.py` (unchanged-goldens guard).
- `PROJECT.md` — §6/§7 docs-sync.

---

### Task P4.1: Ratify CKS-22 in SKILL.md

**Files:**
- Modify: `skills/kerr-physics/SKILL.md` (add a full `## Formula CKS-22` section; update the "When to use" list line 22, the file-map block ~line 1926, the version header, and the revision history tail).

**Interfaces:**
- Produces: the authoritative CKS-22 math every later P4 task ports verbatim.

- [ ] **Step 1: Replace the reserved CKS-22 stub with a full entry.** In the "When to use" list (line 22) remove `Kelvin-Helmholtz threshold erosion (CKS-22 — VISUALIZATION)` from the RESERVED line and add a real bullet pointing to the new section. Then insert a new section (after the CKS-20 entry, before the Formula-10 LOD region is unaffected):

````markdown
## Formula CKS-22 — Kelvin-Helmholtz edge erosion (VISUALIZATION)

**Class:** VISUALIZATION (amplitude only). Amends CKS-12 §3. Never touches
`p_μ`, `u^μ` (CKS-8), `g` (CKS-9), `g⁴` (Formula 9), `f_PT` (CKS-11), or chroma.

Replaces the CKS-12 §3 *outer* multiplicative smoothstep envelope with a
noise-thresholded soft-Heaviside clip, shredding the outer rim into vacuum
(tearing/fraying instead of a clean falloff). The inner edge (`r_in_eff ≥ r_isco`,
zero-torque BC, CKS-12 constraint 3) is NOT eroded.

Let `win_out(r) = 1 − smoothstep(r_out_eff − soft, r_out_eff, r) ∈ [0,1]` be the
existing smooth outer envelope (the "ρ_env" of the roadmap). With erosion ON:

```
N_KH(u,φ,ζ;t) ∈ [0,1]   = high-freq simplex (own seed NSEED_KH), advected by the
                          CKS-12 §2 dual-phase shear (material frame), convex
                          triangle weights ⇒ stays in [0,1].
win_out' = smoothstep( 0, w_soft,  win_out − τ_KH·N_KH )      # the clip REPLACES win_out
win      = win_in · win_out'                                   # win_in unchanged (inner edge)
ρ        = gauss · density_mult · win                         # both ρ_hot and ρ_cold share win
```

- `τ_KH ∈ [0, 1 − w_soft]` (clamped). This bound guarantees **interior immunity**:
  where `win_out = 1`, `win_out − τ_KH·N_KH ≥ w_soft ⇒ smoothstep = 1` (untouched);
  only the transition band (`win_out < 1`) tears.
- `w_soft` (soft-Heaviside half-width on the [0,1] envelope) has a step-cap floor
  (constraint 4): the resulting spatial edge width `≈ w_soft·soft` must span ≥ one
  capped vertical step (k_soft = 1):
  `w_soft ≥ max_step_vfrac · σ_θ(r_outer) / soft`,  σ_θ(r_outer) = r_outer·σ0.
- **Shared envelope (with CKS-19):** the clip multiplies the SHARED `win` BEFORE the
  hot/cold split, so a torn finger removes emission AND absorption together
  (silhouette-correct frayed dust lanes). When multiphase is off this is the single ρ.
- **Bit-identity (constraint 6):** `edge_erosion.enabled = false` ⇒ the clip branch is
  skipped, `win_out` is the unmodified CKS-12 §3 smoothstep ⇒ golden frames unchanged.

**Config:** `disk.edge_erosion {enabled, strength=τ_KH, freq_u, freq_phi(int), freq_z,
octaves, soft_width(0⇒auto floor), seed}`. Base dials only — no CKS-13 resolver change.
````

- [ ] **Step 2: Update the file-map + version + revision history.** In the source-of-truth file map (~line 1926) add `CKS-22 KH edge erosion` against `taichi_renderer.py` (`_disk_density_cks`) and `noise.py` (`kh_field`). Bump the SKILL version header (current tail is v1.32 per the P3 work) to the next minor and add a revision-history row: `vX.YY | CKS-22 authored — KH outer-edge threshold erosion (VISUALIZATION), amends CKS-12 §3; clip H_soft(win_out − τ·N_KH), interior immunity via τ ≤ 1−w_soft, step-cap floor k_soft=1, shared envelope with CKS-19, default OFF bit-identical.`

- [ ] **Step 3: Commit.**

```bash
git add skills/kerr-physics/SKILL.md
git commit -m "docs(CKS-22): ratify KH edge-erosion formula (VISUALIZATION, amends CKS-12 §3)"
```

---

### Task P4.2: CPU twin — `NSEED_KH`, `kh_field`, `kh_erode_winout` (TDD)

**Files:**
- Modify: `src/renderer/noise.py` (add seed offset + two NumPy functions near the other NSEED defs ~line 663 and the dust modulator ~line 840).
- Test: `tests/test_noise.py`.

**Interfaces:**
- Consumes: `noise.sfbm3` (line 528, signature `sfbm3(x,y,z,period,octaves,lac,gain,seed)`), the `_NCYC_PHASE`/`_NCYC_CYCLE` reseed strides used by the dual-phase reset (reuse the exact constants the density twin `_advected_m` uses).
- Produces:
  - `NSEED_KH: int` (new decorrelated offset).
  - `kh_field(u, phi, zeta, t_disk, omega, shear_T, dynamism, freq_u, freq_phi, freq_z, octaves, seed) -> np.ndarray` → `N_KH ∈ [0,1]`, §2-advected.
  - `kh_erode_winout(win_out, n_kh, strength, w_soft) -> np.ndarray` → `smoothstep(0, w_soft, win_out − strength·n_kh)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_noise.py`:

```python
import numpy as np
from src.renderer import noise


def test_kh_field_in_unit_range():
    u = np.linspace(0.0, 1.0, 64)
    phi = np.linspace(-np.pi, np.pi, 64)
    n = noise.kh_field(u, phi, np.zeros_like(u), t_disk=3.0, omega=0.05,
                       shear_T=10.0, dynamism=1.0, freq_u=4.0, freq_phi=12,
                       freq_z=1.0, octaves=3, seed=1234)
    assert np.all(n >= 0.0) and np.all(n <= 1.0)


def test_kh_field_phi_seamless():
    # integer freq_phi ⇒ no seam at φ = 0 ≡ 2π
    a = noise.kh_field(np.array([0.5]), np.array([-np.pi + 1e-6]), np.array([0.0]),
                       3.0, 0.05, 10.0, 1.0, 4.0, 12, 1.0, 3, 1234)
    b = noise.kh_field(np.array([0.5]), np.array([np.pi - 1e-6]), np.array([0.0]),
                       3.0, 0.05, 10.0, 1.0, 4.0, 12, 1.0, 3, 1234)
    assert abs(float(a) - float(b)) < 1e-3


def test_kh_erode_interior_immune():
    # win_out == 1 (interior) stays 1 when strength <= 1 - w_soft
    win = np.ones(32)
    n = np.linspace(0.0, 1.0, 32)
    w_soft, strength = 0.15, 0.85  # strength == 1 - w_soft (the clamp bound)
    out = noise.kh_erode_winout(win, n, strength, w_soft)
    assert np.allclose(out, 1.0, atol=1e-6)


def test_kh_erode_tears_band():
    # win_out in the band (0.3) with high noise tears to 0
    win = np.full(8, 0.3)
    n = np.full(8, 1.0)
    out = noise.kh_erode_winout(win, n, strength=0.85, w_soft=0.15)
    assert np.all(out < 1e-6)  # 0.3 - 0.85 < 0 ⇒ clipped to 0
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `python -m pytest tests/test_noise.py -k kh_ -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'kh_field'`.

- [ ] **Step 3: Implement the CPU twins** in `src/renderer/noise.py`. Add the seed offset beside the others (after `NSEED_DUST = 911`, line 668):

```python
NSEED_KH = 1009  # CKS-22: KH edge-erosion high-freq simplex (own decorrelated stack)
```

Then add the two functions (place near the dust modulator, ~line 860). Mirror the §2 dual-phase reset of the density reference (`_advected_m`): two half-period-staggered phases, per-cycle reseed, triangle weights, sheared azimuth `φ − dynamism·Ω·(a_k·T)`:

```python
def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / np.where(e1 > e0, e1 - e0, 1.0), 0.0, 1.0)
    return np.where(e1 > e0, t * t * (3.0 - 2.0 * t), (x >= e0).astype(x.dtype))


def kh_field(u, phi, zeta, t_disk, omega, shear_T, dynamism,
             freq_u, freq_phi, freq_z, octaves, seed):
    """N_KH ∈ [0,1] — CKS-22 high-freq simplex, advected by the CKS-12 §2 dual-phase
    shear (material frame). Single sfbm3 layer per phase; convex triangle weights keep
    it in [0,1] (no variance-preserve divide — we want the [0,1] envelope domain).
    Twin of taichi_renderer._kh_field."""
    u = np.asarray(u, np.float32); phi = np.asarray(phi, np.float32)
    zeta = np.asarray(zeta, np.float32)
    fp = int(freq_phi)
    sd = seed + NSEED_KH

    def layer(ph, s):
        return sfbm3(u * freq_u, ph * (0.5 / np.pi) * freq_phi, zeta * freq_z,
                     fp, octaves, 2, 0.5, s).astype(np.float32)

    if shear_T <= 0.0:
        return layer(phi, sd)
    s = t_disk / shear_T
    c0 = np.floor(s); a0 = s - c0; w0 = 1.0 - abs(2.0 * a0 - 1.0)
    c1 = np.floor(s + 0.5); a1 = (s + 0.5) - c1; w1 = 1.0 - abs(2.0 * a1 - 1.0)
    sd0 = sd + int(c0) * _NCYC_CYCLE
    sd1 = sd + _NCYC_PHASE + int(c1) * _NCYC_CYCLE
    ph0 = phi - dynamism * omega * (a0 * shear_T)
    ph1 = phi - dynamism * omega * (a1 * shear_T)
    return (w0 * layer(ph0, sd0) + w1 * layer(ph1, sd1)).astype(np.float32)


def kh_erode_winout(win_out, n_kh, strength, w_soft):
    """Replace the smooth outer envelope with the soft-Heaviside clip (CKS-22).
    strength is assumed already clamped to [0, 1 - w_soft] by the caller/setup."""
    return _smoothstep(np.float32(0.0), np.float32(w_soft),
                       np.asarray(win_out, np.float32) - np.float32(strength) * np.asarray(n_kh, np.float32))
```

(If `_NCYC_PHASE` / `_NCYC_CYCLE` are not module-level in `noise.py`, reuse the exact constants the existing dual-phase reference defines — grep `_NCYC` in `noise.py` and import/reference them so the reseed strides match the GPU.)

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `python -m pytest tests/test_noise.py -k kh_ -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit.**

```bash
git add src/renderer/noise.py tests/test_noise.py
git commit -m "feat(CKS-22): CPU twin kh_field + kh_erode_winout + NSEED_KH (TDD)"
```

---

### Task P4.3: GPU twin — param buffer, `_kh_field`, clip in `_disk_density_cks` (TDD)

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (index block ~line 258, `_NSEED_*` aliases ~line 300, `_setup_disk_noise` ~line 529-650, add `_kh_field` `@ti.func`, apply clip in `_disk_density_cks` ~line 1429-1435).
- Test: `tests/test_noise_gpu.py`.

**Interfaces:**
- Consumes: `noise.NSEED_KH`, `noise.sfbm3_ti` (signature `sfbm3_ti(x,y,z,period,octaves,lac,gain,seed)`, line 1427), `_NCYC_PHASE`/`_NCYC_CYCLE`, `_NI_SHEAR_T`, `_NI_DYNAMISM`, `_smoothstep_ti` (line 1277), `_INV_TWO_PI`.
- Produces: `_kh_field(u, phi, zeta, t_disk, omega, seed)` `@ti.func` → GPU twin of `noise.kh_field`; the eroded `win` in `_disk_density_cks`.

- [ ] **Step 1: Write the failing parity test** in `tests/test_noise_gpu.py` (follow the existing dust/curl twin-parity pattern — `ti.init(arch=ti.cuda)`, a kernel that fills a field from `_kh_field`, compared to `noise.kh_field` within `_SATOL`):

```python
def test_kh_field_gpu_matches_cpu():
    import taichi as ti
    import numpy as np
    from src.renderer import taichi_renderer as tr, noise
    ti.init(arch=ti.cuda)
    N = 48
    out = ti.field(ti.f32, shape=N)
    us = np.linspace(0.1, 0.9, N).astype(np.float32)
    ph = np.linspace(-3.0, 3.0, N).astype(np.float32)
    tr._install_kh_test_params(freq_u=4.0, freq_phi=12, freq_z=1.0, octaves=3,
                               shear_T=10.0, dynamism=1.0, seed=1234)  # helper sets the buffer

    @ti.kernel
    def fill(us: ti.types.ndarray(), ph: ti.types.ndarray()):
        for i in range(N):
            out[i] = tr._kh_field(us[i], ph[i], 0.0, 3.0, 0.05, 1234)

    fill(us, ph)
    ref = noise.kh_field(us, ph, np.zeros(N, np.float32), 3.0, 0.05,
                         10.0, 1.0, 4.0, 12, 1.0, 3, 1234)
    assert np.allclose(out.to_numpy(), ref, atol=tr._SATOL)
```

(If a dedicated `_install_kh_test_params` helper is awkward, set the relevant `disk_noise_params[...]` slots directly in the test as the existing GPU-twin tests do. Match whatever the curl/dust twin tests already use.)

- [ ] **Step 2: Run the test to verify it fails.**

Run: `python -m pytest tests/test_noise_gpu.py -k kh_field_gpu -v`
Expected: FAIL — `_kh_field` undefined (and `_NI_EROS_*` undefined).

- [ ] **Step 3: Add the param-buffer indices.** In `taichi_renderer.py` after `_NI_MP_SIGFRAC = 56` (line 258), insert:

```python
# CKS-22 KH edge erosion. _NI_EROS_EN gates it: 0 ⇒ no clip (bit-identical, constraint 6).
_NI_EROS_EN = 57
_NI_EROS_STR = 58      # τ_KH (already clamped to [0, 1-w_soft] at setup)
_NI_EROS_FU = 59       # freq_u
_NI_EROS_FP = 60       # freq_phi (integer φ period, constraint 5)
_NI_EROS_FZ = 61       # freq_z
_NI_EROS_OCT = 62      # octaves
_NI_EROS_WSOFT = 63    # w_soft (resolved: max(soft_width, step-cap floor))
_NOISE_N = 64
```

Add the seed alias near the others (~line 300): `_NSEED_KH = noise.NSEED_KH`.

- [ ] **Step 4: Pack the dials in `_setup_disk_noise`** (~line 600-640, beside the multiphase/curl packing). Read `disk.edge_erosion`, resolve the `w_soft` floor, and clamp `strength`:

```python
eros = d.get("edge_erosion", {}) or {}
buf[_NI_EROS_EN] = 1.0 if eros.get("enabled", False) else 0.0
fu = float(eros.get("freq_u", 4.0)); fp = float(int(eros.get("freq_phi", 12)))
soft = float(mod.get("edge_softness", 0.4))          # CKS-12 §3 window width (geometric M)
sigma0 = float(d.get("sigma_theta0", d.get("scale_height", 0.1)))
r_outer = float(d["r_outer"])
vfrac = float(d.get("max_step_vfrac", 0.5))
floor = vfrac * (sigma0 * r_outer) / max(soft, 1e-6)  # CKS-22 step-cap floor, k_soft=1
w_soft = max(float(eros.get("soft_width", 0.0)) or 0.0, floor)
w_soft = min(max(w_soft, 0.02), 0.5)                  # keep H_soft sane
buf[_NI_EROS_WSOFT] = w_soft
buf[_NI_EROS_STR] = min(max(float(eros.get("strength", 0.0)), 0.0), 1.0 - w_soft)
buf[_NI_EROS_FU] = fu
buf[_NI_EROS_FP] = fp
buf[_NI_EROS_FZ] = float(eros.get("freq_z", 1.0))
buf[_NI_EROS_OCT] = float(eros.get("octaves", 3))
```

(Use the exact key for the base scale height — grep `sigma_theta0` / `scale_height` in the disk config block and match it; `sigma_theta0` is the symbol the kernel already uses.)

- [ ] **Step 5: Add the `_kh_field` `@ti.func`** (place near `_disk_blended_m`, ~line 1215). Mirror `_disk_blended_m`'s dual-phase reset exactly, but a single `sfbm3_ti` layer:

```python
@ti.func
def _kh_field(u, phi, zeta, t_disk, omega, seed):
    """N_KH ∈ [0,1] for CKS-22 KH edge erosion — GPU twin of noise.kh_field.
    Advected with the SAME §2 dual-phase shear as the density (_disk_blended_m)."""
    fu = disk_noise_params[_NI_EROS_FU]
    fpf = disk_noise_params[_NI_EROS_FP]
    fp = ti.cast(fpf, ti.i32)
    fz = disk_noise_params[_NI_EROS_FZ]
    oct_ = ti.cast(disk_noise_params[_NI_EROS_OCT], ti.i32)
    sd = seed + _NSEED_KH
    T = disk_noise_params[_NI_SHEAR_T]
    n = 0.5
    if T <= 0.0:
        n = noise.sfbm3_ti(u * fu, phi * _INV_TWO_PI * fpf, zeta * fz, fp, oct_, 2, 0.5, sd)
    else:
        s = t_disk / T
        g = disk_noise_params[_NI_DYNAMISM]
        c0 = ti.floor(s); a0 = s - c0; w0 = 1.0 - ti.abs(2.0 * a0 - 1.0)
        sd0 = sd + ti.cast(c0, ti.i32) * _NCYC_CYCLE
        ph0 = phi - g * omega * (a0 * T)
        ar1 = s + 0.5; c1 = ti.floor(ar1); a1 = ar1 - c1; w1 = 1.0 - ti.abs(2.0 * a1 - 1.0)
        sd1 = sd + _NCYC_PHASE + ti.cast(c1, ti.i32) * _NCYC_CYCLE
        ph1 = phi - g * omega * (a1 * T)
        n = (w0 * noise.sfbm3_ti(u * fu, ph0 * _INV_TWO_PI * fpf, zeta * fz, fp, oct_, 2, 0.5, sd0)
             + w1 * noise.sfbm3_ti(u * fu, ph1 * _INV_TWO_PI * fpf, zeta * fz, fp, oct_, 2, 0.5, sd1))
    return n
```

- [ ] **Step 6: Apply the clip in `_disk_density_cks`.** Replace the combined-`win` line (1429-1431) so the inner and outer factors are separate and the outer is clipped when erosion is on:

```python
            win_in = _smoothstep_ti(r_in_eff, r_in_eff + soft, r)
            win_out = 1.0 - _smoothstep_ti(r_out_eff - soft, r_out_eff, r)
            if disk_noise_params[_NI_EROS_EN] > 0.5:
                n_kh = _kh_field(u_n, phi_n, zeta_n, t_disk, omega, noise_seed)
                tau = disk_noise_params[_NI_EROS_STR]
                wsoft = disk_noise_params[_NI_EROS_WSOFT]
                win_out = _smoothstep_ti(0.0, wsoft, win_out - tau * n_kh)
            win = win_in * win_out
```

The downstream `density = gauss * dmult * win` (line 1435) and `density_cold = density` (1440) are unchanged ⇒ the clip shreds the SHARED envelope (both phases). `_NI_EROS_EN == 0` ⇒ this collapses to the original `win = win_in * win_out` ⇒ bit-identical.

- [ ] **Step 7: Run the parity test.**

Run: `python -m pytest tests/test_noise_gpu.py -k kh_field_gpu -v`
Expected: PASS.

- [ ] **Step 8: Commit.**

```bash
git add src/renderer/taichi_renderer.py tests/test_noise_gpu.py
git commit -m "feat(CKS-22): GPU twin _kh_field + win_out clip in _disk_density_cks (TDD)"
```

---

### Task P4.4: Config, bit-identity regression, tearing acceptance, docs-sync

**Files:**
- Modify: `configs/render.yaml` (add `disk.edge_erosion`).
- Test: `tests/test_disk_edge_erosion.py` (new), `tests/test_gpu_regression.py` (run unchanged).
- Modify: `PROJECT.md` (§6/§7).

**Interfaces:**
- Consumes: the full P4.2/P4.3 stack.
- Produces: the shipped, default-OFF `disk.edge_erosion` feature.

- [ ] **Step 1: Add the config block** to `configs/render.yaml` after the `multiphase`/`scatter` blocks (~line 254), default OFF:

```yaml
  edge_erosion:              # CKS-22 — Kelvin-Helmholtz threshold erosion of the OUTER edge.
                             # Replaces the CKS-12 §3 outer smoothstep with a soft-Heaviside
                             # clip H_soft(win_out − τ·N_KH): the rim tears into vacuum instead
                             # of a clean falloff. Shreds the SHARED envelope ⇒ with CKS-19 on,
                             # emission AND absorption fray together (silhouette-correct lanes).
                             # enabled:false ⇒ CKS-12 §3 ragged smoothstep, BIT-IDENTICAL.
    enabled: false           # master switch (golden frames intact when false).
    strength: 0.0            # τ_KH — erosion depth; clamped to [0, 1−w_soft] at load.
    freq_u: 4.0              # log-radial feature frequency of N_KH.
    freq_phi: 12             # azimuthal frequency (INTEGER — 2π periodicity, constraint 5).
    freq_z: 1.0              # vertical feature frequency.
    octaves: 3               # N_KH fBm octaves (high-freq fingers).
    soft_width: 0.0          # w_soft on the [0,1] envelope; 0 ⇒ auto = step-cap floor
                             #   max(soft_width, max_step_vfrac·σ0·r_outer/edge_softness) (k_soft=1).
    seed: 1234               # base seed (offset by NSEED_KH internally).
```

- [ ] **Step 2: Write the failing bit-identity + tearing tests** in `tests/test_disk_edge_erosion.py`. The bit-identity test renders a small disk frame with `edge_erosion.enabled:false` and asserts it equals the same render with the block ABSENT (byte-for-byte). The tearing test enables erosion and asserts the outer band gains disconnected ρ→0 holes interior to `r_out_eff` that are absent when off. Use the existing small-render harness from `tests/test_disk_noise.py` / `tests/test_disk_multiphase.py` as the template:

```python
import numpy as np
from tests._render_helpers import render_small_disk  # reuse the harness the disk tests use


def test_erosion_off_is_bit_identical(tmp_path):
    base = render_small_disk(overrides={})  # block absent
    off = render_small_disk(overrides={"disk": {"edge_erosion": {"enabled": False}}})
    assert np.array_equal(base, off)


def test_erosion_tears_outer_band():
    off = render_small_disk(overrides={"disk": {"noise": {"enabled": True}}})
    on = render_small_disk(overrides={"disk": {
        "noise": {"enabled": True},
        "edge_erosion": {"enabled": True, "strength": 0.8, "freq_phi": 12},
    }})
    # erosion removes flux in the outer band ⇒ strictly less total disk light there,
    # and introduces near-zero pixels the clean rim did not have.
    assert on.sum() < off.sum()
    assert (on < 1e-4).sum() > (off < 1e-4).sum()
```

(If there is no shared `render_small_disk` helper, lift the minimal setup+render from `tests/test_disk_multiphase.py::test_dust_carves_silhouette` — same `setup_renderer` + `render_beauty_*` call at low res, frame 0.)

- [ ] **Step 3: Run the new tests to verify they fail, then pass after wiring.**

Run: `python -m pytest tests/test_disk_edge_erosion.py -v`
Expected: initially FAIL (helper/feature path), PASS once the config + P4.3 kernel are in place.

- [ ] **Step 4: Run the regression guard — goldens must NOT move.**

Run: `python -m pytest tests/test_gpu_regression.py -v`
Expected: PASS, goldens bit-identical (the regression config keeps `edge_erosion` absent/false).

- [ ] **Step 5: Docs-sync.** Update `PROJECT.md` §6 (feature list / config reference) and §7 (kernel touch-map) with CKS-22: the `disk.edge_erosion` block, `_kh_field`, the `_disk_density_cks` clip, and the default-OFF bit-identity guarantee.

- [ ] **Step 6: Commit.**

```bash
git add configs/render.yaml tests/test_disk_edge_erosion.py PROJECT.md
git commit -m "feat(CKS-22): edge_erosion config + tearing/bit-identity tests + docs-sync (P4 complete)"
```

---

# PART P5 — Fractal LOD octave cascade (CKS-23)

**File structure (P5):**
- `skills/kerr-physics/SKILL.md` — promote CKS-23 to a full formula entry.
- `src/renderer/noise.py` — `lod_octave_weight`, `lod_noct`, and a gated `fbm2` reference.
- `src/renderer/taichi_renderer.py` — `_NI_LOD_*` indices; pack in `_setup_disk_noise`; a gated `fbm2_lod_ti` (in `noise.py`); thread `lod_noct` through `_disk_noise_m` → `_disk_blended_m` → `_disk_density_cks` → `_disk_emit_cks`; compute `n_oct` from camera distance in `render_beauty_physics`.
- `configs/render.yaml` — add `disk.lod` block.
- `tests/test_noise.py`, `tests/test_noise_gpu.py`, `tests/test_disk_lod.py` (new), `tests/test_gpu_regression.py`.
- `PROJECT.md` — §6/§7.

> **v1 scope (logged cap, per no-silent-caps):** LOD gates the **fBm** density layers (L0 base streaks, L2 patch, and the L1 coverage mask) — the dominant high-freq aliasing source. The L1 ridged-MF/Voronoi clump octaves and the §3 modulation/curl noise are NOT gated in v1 (they are lower-frequency structural fields). Anisotropic per-axis J and `dλ` step coarsening are also deferred (decisions C.6-1/2). This is noted in CKS-23 and PROJECT.md.

---

### Task P5.1: Ratify CKS-23 in SKILL.md

**Files:**
- Modify: `skills/kerr-physics/SKILL.md` (add `## Formula CKS-23`; update "When to use" line 22, the Formula-10 cross-reference, file map, version, revision history).

**Interfaces:**
- Produces: the authoritative CKS-23 math the later P5 tasks port verbatim.

- [ ] **Step 1: Add the full CKS-23 entry** (after the Formula-10 LOD section, since it extends it):

````markdown
## Formula CKS-23 — Fractal LOD octave cascade (SAMPLING)

**Class:** SAMPLING (sampling-rate only). Extends Formula 10 (screen-space LOD) to the
disk-noise octave loop. Never touches `p_μ/u^μ/g/g⁴/f_PT`/chroma — it only chooses how
many noise octaves a sample evaluates.

Per disk sample at camera distance `d` (isotropic scalar footprint, matching Formula 10's
`J = max(Jx,Jy)` philosophy):

```
ε        = fov_y / HEIGHT                         # pixel cone half-angle (rad)
J        = ε · d                                  # world-space footprint at the sample
n_oct    = clamp( N_max − log₂(J / J₀),  N_min,  N_max )    # fractional target octave count
g_o      = clamp( n_oct − o,  0,  1 )             # per-octave weight: 1 below cutoff,
                                                  #   fractional at the top (anti-pop), 0 above
fBm:  total = Σ_{o<N_max} g_o·gain^o·noise(coord·lac^o),   norm = Σ_{o<N_max} g_o·gain^o
```

Gating BOTH `total` and `norm` by `g_o` keeps normalization exact, so the disabled/full
path is bit-identical (constraint 6): with `n_oct ≥ N_max` every `g_o = 1` and the loop
reproduces the ungated fBm exactly (`x·1.0 == x` in IEEE float). The top partial octave
crossfades via the fractional `g_o ∈ (0,1)` ⇒ no integer popping as `n_oct` varies with
distance.

- `J₀` is a **base dial** (no CKS-13 resolver change): the footprint at which `n_oct = N_max`.
- `N_max ≥ disk.noise.layers.*.octaves`. `N_max == octaves` ⇒ no extra close-up octaves;
  `N_max > octaves` injects sub-octaves on close-ups.
- **Bit-identity:** `lod.enabled = false` ⇒ `n_oct` forced to `N_max` everywhere ⇒ ungated.
- **v1 scope:** gates the fBm density layers (L0/L2/L1-mask) only; octaves-only (no `dλ`);
  isotropic scalar `J`. Ridged/Voronoi/§3-modulation gating and anisotropic `J` deferred.
````

- [ ] **Step 2: Update file map + version + revision history** (same pattern as P4.1 Step 2): file map gets `CKS-23 fractal LOD` against `noise.py` (`fbm2_lod_ti`, `lod_noct`) and `taichi_renderer.py` (`render_beauty_physics`, `_disk_noise_m`). Bump version; add a revision row describing CKS-23 (extends Formula 10; `g_o = clamp(n_oct−o,0,1)`; gate total AND norm ⇒ bit-identical; default OFF; prerequisite for the V4 free camera).

- [ ] **Step 3: Commit.**

```bash
git add skills/kerr-physics/SKILL.md
git commit -m "docs(CKS-23): ratify fractal LOD octave-cascade formula (SAMPLING, extends Formula 10)"
```

---

### Task P5.2: CPU twin — `lod_noct`, `lod_octave_weight`, gated `fbm2` (TDD)

**Files:**
- Modify: `src/renderer/noise.py` (add `lod_noct`, `lod_octave_weight`, and a gated `fbm2` reference `fbm2_lod`).
- Test: `tests/test_noise.py`.

**Interfaces:**
- Consumes: the existing `fbm2` octave structure (`fbm2_ti` line 1136; its CPU reference).
- Produces:
  - `lod_noct(d, j0, n_max, n_min, eps_cone) -> float` = `clamp(n_max − log2(eps_cone·d/j0), n_min, n_max)`.
  - `lod_octave_weight(n_oct, o) -> float` = `clamp(n_oct − o, 0, 1)`.
  - `fbm2_lod(x, y, period, n_max, lac, gain, seed, n_oct) -> np.ndarray` = the gated fBm (gates total AND norm).

- [ ] **Step 1: Write the failing tests** in `tests/test_noise.py`:

```python
def test_lod_noct_monotonic_in_distance():
    near = noise.lod_noct(d=5.0, j0=1.0, n_max=7, n_min=1, eps_cone=1e-3)
    far = noise.lod_noct(d=500.0, j0=1.0, n_max=7, n_min=1, eps_cone=1e-3)
    assert far < near  # farther ⇒ fewer octaves
    assert 1 <= far <= 7 and 1 <= near <= 7


def test_lod_full_octaves_is_ungated():
    # n_oct >= n_max ⇒ gated fBm == ungated fBm bit-for-bit
    x = np.linspace(0, 4, 50); y = np.linspace(0, 4, 50)
    full = noise.fbm2_lod(x, y, 0, 5, 2, 0.5, 7, n_oct=99.0)
    base = noise.fbm2(x, y, 0, 5, 2, 0.5, 7)  # existing ungated reference
    assert np.allclose(full, base, atol=0.0)  # exact


def test_lod_octave_weight_crossfade():
    assert noise.lod_octave_weight(4.3, 4) == 0.3 or abs(noise.lod_octave_weight(4.3, 4) - 0.3) < 1e-6
    assert noise.lod_octave_weight(4.3, 3) == 1.0
    assert noise.lod_octave_weight(4.3, 5) == 0.0


def test_lod_gated_continuous_across_integer():
    # the summed fBm is continuous as n_oct crosses an integer (anti-pop)
    x = np.linspace(0, 4, 50); y = np.linspace(0, 4, 50)
    lo = noise.fbm2_lod(x, y, 0, 6, 2, 0.5, 7, n_oct=4.0 - 1e-3)
    hi = noise.fbm2_lod(x, y, 0, 6, 2, 0.5, 7, n_oct=4.0 + 1e-3)
    assert np.max(np.abs(hi - lo)) < 1e-3
```

(If the project's CPU `fbm2` reference has a different name than `noise.fbm2`, use the actual one — grep `def fbm2` in `noise.py`; there is a `fbm2_ti` GPU twin so a CPU `fbm2` or `_fbm2`/`fbm` exists. Match it.)

- [ ] **Step 2: Run to verify failure.**

Run: `python -m pytest tests/test_noise.py -k lod_ -v`
Expected: FAIL — `lod_noct` undefined.

- [ ] **Step 3: Implement the CPU twins** in `noise.py`:

```python
def lod_octave_weight(n_oct, o):
    return float(np.clip(n_oct - o, 0.0, 1.0))


def lod_noct(d, j0, n_max, n_min, eps_cone):
    j = max(eps_cone * d, 1e-12)
    return float(np.clip(n_max - np.log2(j / j0), n_min, n_max))


def fbm2_lod(x, y, period, n_max, lac, gain, seed, n_oct):
    """Gated fBm: identical to fbm2 when n_oct >= n_max; the top partial octave is
    crossfaded by frac(n_oct). Gates BOTH total and norm so normalization is exact."""
    total = np.zeros_like(np.asarray(x, np.float32))
    norm = np.float32(0.0)
    freq = 1; per = period; amp = np.float32(1.0)
    for o in range(int(n_max)):
        g = np.float32(lod_octave_weight(n_oct, o))
        total = total + g * snoise2(np.asarray(x) * freq, np.asarray(y) * freq, per, seed + o) * amp
        norm = norm + g * amp
        amp = amp * gain; freq = freq * lac; per = per * lac
    return (total / np.where(norm > 0, norm, 1.0)).astype(np.float32)
```

(Use the SAME per-octave primitive the existing `fbm2` reference uses — if `fbm2` is built on `snoise2`/`noise2`, mirror it so `n_oct≥n_max` is exact.)

- [ ] **Step 4: Run to verify pass.**

Run: `python -m pytest tests/test_noise.py -k lod_ -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit.**

```bash
git add src/renderer/noise.py tests/test_noise.py
git commit -m "feat(CKS-23): CPU twins lod_noct + lod_octave_weight + fbm2_lod (TDD)"
```

---

### Task P5.3: GPU twin — gated `fbm2_lod_ti`, thread `n_oct`, compute J in the march (TDD)

**Files:**
- Modify: `src/renderer/noise.py` (add `fbm2_lod_ti` `@ti.func`).
- Modify: `src/renderer/taichi_renderer.py` (`_NI_LOD_*` indices; `_setup_disk_noise`; thread `lod_noct` through `_disk_noise_m` → `_disk_blended_m` → `_disk_cold_mult_from_hot` → `_disk_density_cks` → `_disk_emit_cks`; compute `n_oct` in `render_beauty_physics`).
- Test: `tests/test_noise_gpu.py`.

**Interfaces:**
- Consumes: `fbm2_ti` call sites in `_disk_noise_m` (L0 line 1117, L1 mask 1151, L2 1168).
- Produces: `noise.fbm2_lod_ti(x, y, period, n_max, lac, gain, seed, n_oct)` `@ti.func`; an added trailing `lod_noct` arg on the density chain; `_NI_LOD_*` slots.

- [ ] **Step 1: Write the failing parity test** in `tests/test_noise_gpu.py` (GPU `fbm2_lod_ti` vs CPU `fbm2_lod` within `_SATOL`, plus an exactness check that `n_oct ≥ n_max` reproduces `fbm2_ti`):

```python
def test_fbm2_lod_gpu_matches_cpu():
    import taichi as ti, numpy as np
    from src.renderer import taichi_renderer as tr, noise
    ti.init(arch=ti.cuda)
    N = 40
    out = ti.field(ti.f32, shape=N)
    xs = np.linspace(0, 4, N).astype(np.float32)

    @ti.kernel
    def fill(xs: ti.types.ndarray(), noct: ti.f32):
        for i in range(N):
            out[i] = noise.fbm2_lod_ti(xs[i], xs[i], 0, 6, 2, 0.5, 7, noct)

    fill(xs, 4.3)
    ref = noise.fbm2_lod(xs, xs, 0, 6, 2, 0.5, 7, 4.3)
    assert np.allclose(out.to_numpy(), ref, atol=tr._SATOL)

    fill(xs, 99.0)  # ungated must equal plain fbm2_ti
    base = ti.field(ti.f32, shape=N)
    @ti.kernel
    def fill_base(xs: ti.types.ndarray()):
        for i in range(N):
            base[i] = noise.fbm2_ti(xs[i], xs[i], 0, 6, 2, 0.5, 7)
    fill_base(xs)
    assert np.array_equal(out.to_numpy(), base.to_numpy())
```

- [ ] **Step 2: Run to verify failure.**

Run: `python -m pytest tests/test_noise_gpu.py -k fbm2_lod -v`
Expected: FAIL — `fbm2_lod_ti` undefined.

- [ ] **Step 3: Add `fbm2_lod_ti`** in `noise.py` (beside `fbm2_ti`, ~line 1136), gating total AND norm; loop bound `n_max`:

```python
@ti.func
def fbm2_lod_ti(x, y, period, n_max, lac, gain, seed, n_oct):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    for o in range(n_max):
        g = ti.min(ti.max(n_oct - o, 0.0), 1.0)
        total += g * noise2_ti(x * freq, y * freq, per, seed + o) * amp  # same primitive fbm2_ti uses
        norm += g * amp
        amp *= gain
        freq *= lac
        per *= lac
    return total / norm if norm > 0.0 else 0.0
```

(Match the exact per-octave primitive in `fbm2_ti` — grep its body; if it calls `noise2_ti`/`pnoise2_ti`, use that. The `n_oct ≥ n_max` path must equal `fbm2_ti` bit-for-bit, which requires `n_max` ≥ the requested octaves and `g==1.0`.)

- [ ] **Step 4: Add the LOD param indices** after `_NI_EROS_WSOFT = 63` (from P4):

```python
# CKS-23 fractal LOD octave cascade. _NI_LOD_EN gates it: 0 ⇒ n_oct forced to N_max
# (ungated, bit-identical, constraint 6).
_NI_LOD_EN = 64
_NI_LOD_NMAX = 65      # N_max (>= max layer octaves)
_NI_LOD_NMIN = 66      # N_min
_NI_LOD_J0 = 67        # J0 reference footprint
_NI_LOD_EPS = 68       # ε = fov_y / HEIGHT (pixel cone half-angle), packed at setup
_NOISE_N = 69
```

Pack them in `_setup_disk_noise` from `disk.lod`, defaulting `n_max` to the max of the layer octaves so OFF == current:

```python
lod = d.get("lod", {}) or {}
layer_octs = [int(d["noise"]["layers"][k].get("octaves", 1)) for k in ("base", "clump", "patch")]
n_max = int(lod.get("n_max", 0)) or max(layer_octs)
buf[_NI_LOD_EN] = 1.0 if lod.get("enabled", False) else 0.0
buf[_NI_LOD_NMAX] = float(n_max)
buf[_NI_LOD_NMIN] = float(lod.get("n_min", 1))
buf[_NI_LOD_J0] = float(lod.get("j0", 1.0))
# ε is the pixel cone half-angle; fov_y/HEIGHT supplied by the caller (render cfg).
buf[_NI_LOD_EPS] = float(cfg_fov_y) / float(cfg_height)
```

(Wire `cfg_fov_y`/`cfg_height` from wherever `_setup_disk_noise` can see the camera/render config — if it only gets the `disk` dict today, pass the needed scalars in, or read `fov_deg`/`HEIGHT` from the full cfg as the rest of `setup_renderer` does.)

- [ ] **Step 5: Thread `lod_noct` through the density chain.** Add a trailing `lod_noct` parameter (default a large sentinel so non-LOD callers/tests stay ungated) to `_disk_noise_m`, `_disk_blended_m`, `_disk_cold_mult_from_hot`, `_disk_density_cks`, and `_disk_emit_cks`. In `_disk_noise_m`, swap the three fBm density calls to the gated form, e.g. L0 (line 1117):

```python
        n0 = noise.fbm2_lod_ti(
            u * disk_noise_params[_NI_L0_FU],
            phi01 * fpf,
            fp,
            ti.cast(disk_noise_params[_NI_LOD_NMAX], ti.i32),  # loop bound = N_max
            ti.cast(disk_noise_params[_NI_L0_LAC], ti.i32),
            disk_noise_params[_NI_L0_GAIN],
            seed + _NSEED_L0,
            lod_noct,
        )
```

Do the same for the L1 coverage mask (line 1151) and L2 (line 1168). **Bit-identity:** when `_NI_LOD_EN==0` the caller passes `lod_noct = N_max`, and `N_max` defaults to `max(layer octaves)`; for a layer whose own octaves `< N_max`, gating beyond its count would add octaves — so clamp the per-call loop bound to that layer's octaves when LOD is off. Simplest exact rule: pass `lod_noct = 1e9` AND use each layer's own octave count as the loop bound when off. Concretely, gate by `min(layer_oct, N_max)` and set `lod_noct = layer_oct` when off. Implement via:

```python
        l0_oct = ti.cast(disk_noise_params[_NI_L0_OCT], ti.i32)
        nm = l0_oct
        noct = ti.cast(l0_oct, ti.f32) * 1.0  # ungated default
        if disk_noise_params[_NI_LOD_EN] > 0.5:
            nm = ti.cast(disk_noise_params[_NI_LOD_NMAX], ti.i32)
            noct = lod_noct
        n0 = noise.fbm2_lod_ti(u * disk_noise_params[_NI_L0_FU], phi01 * fpf, fp,
                               nm, ti.cast(disk_noise_params[_NI_L0_LAC], ti.i32),
                               disk_noise_params[_NI_L0_GAIN], seed + _NSEED_L0, noct)
```

With LOD off: `nm = l0_oct`, `noct = l0_oct` ⇒ all `g_o=1` over exactly `l0_oct` octaves ⇒ `fbm2_lod_ti == fbm2_ti` bit-for-bit. (Apply the same off-guard to L1 mask + L2.)

- [ ] **Step 6: Compute `n_oct` in `render_beauty_physics`.** Inside the disk-emit branch (~line 2078, where `_disk_emit_cks` is called), compute the camera distance and `n_oct`, and pass it in. The sample world position is `(x,y,z)`; the camera origin is the ray origin available to the kernel (the same vector used to launch the primary ray — reuse it; if not in scope, add it as a kernel arg from the camera matrix). Then:

```python
                    lod_noct = disk_noise_params[_NI_LOD_NMAX]  # default: ungated (= N_max)
                    if disk_noise_params[_NI_LOD_EN] > 0.5:
                        d_cam = ti.sqrt((x - cam_x) ** 2 + (y - cam_y) ** 2 + (z - cam_z) ** 2)
                        j = disk_noise_params[_NI_LOD_EPS] * d_cam
                        nmax = disk_noise_params[_NI_LOD_NMAX]
                        nmin = disk_noise_params[_NI_LOD_NMIN]
                        lod_noct = ti.min(ti.max(nmax - ti.log(j / disk_noise_params[_NI_LOD_J0]) / ti.log(2.0), nmin), nmax)
```

Pass `lod_noct` as the new trailing arg to `_disk_emit_cks`. When `_NI_LOD_EN==0`, `lod_noct = N_max` and the per-layer off-guard (Step 5) makes every layer ungated ⇒ bit-identical.

- [ ] **Step 7: Run the parity test.**

Run: `python -m pytest tests/test_noise_gpu.py -k fbm2_lod -v`
Expected: PASS (parity + ungated-exactness).

- [ ] **Step 8: Commit.**

```bash
git add src/renderer/noise.py src/renderer/taichi_renderer.py tests/test_noise_gpu.py
git commit -m "feat(CKS-23): gated fbm2_lod_ti + n_oct threading + per-sample J in march (TDD)"
```

---

### Task P5.4: Config, bit-identity regression, two-distance acceptance, docs-sync

**Files:**
- Modify: `configs/render.yaml` (add `disk.lod`).
- Test: `tests/test_disk_lod.py` (new), `tests/test_gpu_regression.py` (run unchanged).
- Modify: `PROJECT.md` (§6/§7).

**Interfaces:**
- Consumes: the full P5.2/P5.3 stack.
- Produces: the shipped, default-OFF `disk.lod` feature.

- [ ] **Step 1: Add the config block** to `configs/render.yaml` after `edge_erosion`:

```yaml
  lod:                       # CKS-23 — fractal LOD octave cascade (SAMPLING; extends Formula 10).
                             # Per disk sample, n_oct = clamp(N_max − log2(ε·d/J0), N_min, N_max)
                             # gates the fBm density octaves (L0/L2/L1-mask) by g_o = clamp(n_oct−o,0,1).
                             # Holds on-screen detail density ~constant with camera distance: far views
                             # drop shimmering sub-pixel octaves, close-ups inject sub-octaves.
                             # Prerequisite for the V4 free camera. enabled:false ⇒ n_oct≡N_max ⇒
                             # current fixed-octave path, BIT-IDENTICAL (constraint 6).
    enabled: false           # master switch (golden frames intact when false).
    n_max: 0                 # N_max; 0 ⇒ auto = max(layer octaves) (the bit-identity anchor).
                             #   set > layer octaves to gain close-up sub-octaves.
    n_min: 1                 # floor octave count for far/macro views.
    j0: 1.0                  # reference footprint (cycles) at which n_oct = N_max. Look anchor.
```

- [ ] **Step 2: Write the failing acceptance tests** in `tests/test_disk_lod.py` — bit-identity OFF, distance-monotonic octave drop, and crossfade continuity at the render level. Reuse the small-render harness:

```python
import numpy as np
from tests._render_helpers import render_small_disk


def test_lod_off_is_bit_identical():
    base = render_small_disk(overrides={})
    off = render_small_disk(overrides={"disk": {"lod": {"enabled": False}}})
    assert np.array_equal(base, off)


def test_lod_far_has_less_high_freq_than_near():
    # render the same disk at two camera distances; the far frame's disk region should
    # carry less high-frequency variance (octaves culled) than a naive fixed-octave far frame.
    far_fixed = render_small_disk(overrides={"camera": {"radius": 200.0}})
    far_lod = render_small_disk(overrides={"camera": {"radius": 200.0},
                                           "disk": {"lod": {"enabled": True, "j0": 1.0}}})
    def hf_energy(img):  # Laplacian variance as a high-freq proxy
        g = img.mean(axis=-1) if img.ndim == 3 else img
        lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return float(np.var(lap))
    assert hf_energy(far_lod) <= hf_energy(far_fixed)
```

(If `camera.radius` is not the override key, use the actual camera-distance config path. Keep the render resolution small — these are GPU tests and slow.)

- [ ] **Step 3: Run the new tests.**

Run: `python -m pytest tests/test_disk_lod.py -v`
Expected: PASS once config + kernel are wired.

- [ ] **Step 4: Regression guard — goldens must NOT move.**

Run: `python -m pytest tests/test_gpu_regression.py -v`
Expected: PASS, goldens bit-identical (`lod` absent/false in the regression config).

- [ ] **Step 5: Docs-sync.** Update `PROJECT.md` §6 (the `disk.lod` config + the v1 scope cap) and §7 (touch-map: `fbm2_lod_ti`, the `n_oct` threading, the per-sample J in `render_beauty_physics`), and note CKS-23 is the V4-free-camera prerequisite.

- [ ] **Step 6: Commit.**

```bash
git add configs/render.yaml tests/test_disk_lod.py PROJECT.md
git commit -m "feat(CKS-23): disk.lod config + two-distance/bit-identity tests + docs-sync (P5 complete)"
```

---

## Self-Review

**Spec coverage:** P4 (B.1–B.7) → Tasks P4.1 (CKS-22 ratify) / P4.2 (CPU twin + NSEED_KH + erode) / P4.3 (GPU twin + clip at `_disk_density_cks`, shared `win`) / P4.4 (config + step-cap floor + tearing + bit-identity + docs). P5 (C.1–C.7) → Tasks P5.1 (CKS-23 ratify) / P5.2 (CPU `lod_noct`/`lod_octave_weight`/`fbm2_lod`) / P5.3 (gated `fbm2_lod_ti` + `n_oct` threading + per-sample J) / P5.4 (config + two-distance + bit-identity + docs). All 8 resolved decisions are encoded: §2-only advection (P4.2/P4.3 `_kh_field`), `k_soft=1` floor (P4.3 Step 4), shared-envelope erosion (P4.3 Step 6, before `density_cold`), `disk.edge_erosion` home (P4.4), isotropic scalar J (P5.3 Step 6), octaves-only/no-`dλ` (P5 scope note), `J₀` base dial (P5.3 Step 4), two-distance golden (P5.4 Step 2).

**Placeholder scan:** no "TBD"/"implement later"; every code step shows code; the parenthetical "grep the actual name" notes are guard-rails against name drift, not deferrals (each gives the concrete fallback).

**Type consistency:** `_NI_*` indices are sequential and non-overlapping (P4: 57-63, `_NOISE_N=64`; P5: 64-68, `_NOISE_N=69` — P5 builds on P4's renumber; if P5 lands first, shift its indices to start at 57). `kh_field`/`kh_erode_winout`/`_kh_field` signatures match between CPU and GPU twins and their tests. `lod_noct`/`lod_octave_weight`/`fbm2_lod`/`fbm2_lod_ti` signatures are consistent across P5.2/P5.3 and the threading in `_disk_noise_m`.

> **Ordering note for the executor:** P4 and P5 are independent and either may land first, BUT both renumber `_NOISE_N` and add `_NI_*` slots. Land them **sequentially** (finish one pillar's 4 tasks, then the other) and let the second pillar's P*.3 task start its indices at the first pillar's `_NOISE_N`. The plan as written assumes **P4 then P5** (P5 indices begin at 64). If you reverse the order, swap the index bases accordingly.

---

## Execution Handoff

Plan complete and saved to `docs/specs/2026-06-20-P4-P5-edge-erosion-lod-cascade-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints for review.

⚠️ Note: GPU test suites re-`ti.init` per test and the scatter/erosion mega-kernel cold-compiles slowly — expect long runs; use `render.advanced_optimization:false` / `cfg_optimization:false` (the `--fast-compile` knobs) during iteration, never for the final video.
