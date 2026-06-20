# P1 — Scale-dependent shear cascade (CKS-21) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CKS-12 §2 Keplerian shear **frequency-dependent** so coarse disk-noise octaves wind into filaments while high-frequency octaves are protected (Kolmogorov-like), instead of every octave laminarizing into the same spiral.

**Architecture:** A per-octave **de-shear correction** `Δφ_o = (1 − S(f_o))·shear_k` (`S(f)=1/(1+(f/f_c)^p)`) added **inside** the existing octave loops, *after* the unchanged shear→curl pipeline. It threads through the shared CPU `_octaves` generator (covers `fbm2`/`fbm2_lod`/`ridged3`) and the GPU `fbm2_lod_ti`/`ridged3_ti` twins. Scope: density octave stacks **L0 (`fbm2`), L2 (`fbm2`), L1's `ridged3`**; L1 Voronoi (no octaves) + L1 coverage mask + the §3 modulation envelopes keep uniform shear. A `shear_k = 0.0` sentinel default makes the correction exactly `0` ⇒ the disabled path is **bit-identical** and always-compiled (the CKS-23 precedent — no `ti.static` gate).

**Tech Stack:** Python 3, NumPy (CPU twin, source of truth), Taichi 1.7.4 on CUDA (GPU twin), pytest, YAML config.

**Spec:** `docs/specs/2026-06-20-P1-shear-cascade-design.md` (RATIFIED 2026-06-20). This plan implements it.

## Global Constraints

- **Governance class VISUALIZATION** — the cascade warps the noise **coordinate only**. It must NEVER touch `p_μ`, `u^μ` (CKS-8), `g` (CKS-9), the `g⁴` exponent (Formula 9), the `f_PT` radial shape (CKS-11), or the blackbody chroma form (CKS-12 §3 hard-constraint list, verbatim).
- **Constraint 6 (the key guard):** `disk.noise.shear_cascade.enabled: false` (or absent) ⇒ the kernel path is **bit-identical** to the current goldens. Backed by the `shear_k = 0.0` sentinel default ⇒ `(1−S)·0 = 0` in f32, `y + 0.0 = y` exact. **No `ti.static` `_SC_COMPILE` gate** (CKS-23 always-compiled precedent).
- **Constraint 5 (φ-seam):** the per-octave correction is **constant in φ** (`shear_k` depends on `r/u`, not φ), so integer-period lattice wrapping is preserved: `y_o(2π) − y_o(0) = period·lac^o ∈ ℤ`. Never break this.
- **CPU twin is the source of truth**; the GPU twin is ported **verbatim** (CLAUDE.md CRITICAL RULE — no re-derivation in code). Match float32 op-order exactly for bit-identity.
- **GPU backend locked:** `ti.init(arch=ti.cuda)` — never `ti.gpu`.
- **All params live in `configs/render.yaml`.** The cascade dials are **base look dials** ⇒ **no CKS-13 resolver change** (`kerr_params.resolve_config` is untouched).
- **Determinism:** integer-hashed seeds from config; **no `ti.random`** — same seed + same `t_disk` ⇒ identical frame.
- **Windows cp949 box:** always open text/YAML files with `encoding="utf-8"`.
- **Docs-sync (same task as the code):** SKILL.md (CKS-21) + PROJECT.md §6/§7 updated with the landed change.
- **Float32 everywhere** in the noise path (`np.float32` / Taichi `f32`).

---

## File Structure

- `skills/kerr-physics/SKILL.md` — promote **CKS-21** from reserved to a full formula (Task 1).
- `src/renderer/noise.py` — CPU twin (source of truth): `shear_transfer` helper + `SHEAR_FC_OFF`; per-octave correction in `_octaves`; forward params through `fbm2`/`fbm2_lod`/`_ridged`/`ridged3`; thread `shear_k` through `_noise_m_stack` + `_advected_m` (Tasks 2–3). GPU `@ti.func` twins `shear_transfer_ti` + `fbm2_lod_ti`/`ridged3_ti` (Task 5).
- `src/renderer/taichi_renderer.py` — `_NI_SC_*` buffer indices + `_NOISE_N` bump + `_setup_disk_noise` reads (Task 4); thread `shear_k` through `_disk_noise_m`/`_disk_blended_m` (Task 5).
- `configs/render.yaml` (+ any showcase configs) — `disk.noise.shear_cascade` block (Task 4).
- `tests/test_noise.py` — CPU transfer/collapse/displacement unit tests (Tasks 2–3).
- `tests/test_noise_gpu.py` — GPU twin parity (Task 5).
- `tests/test_disk_noise.py` — C0-at-reset (Task 5).
- `tests/test_disk_shear_cascade.py` (new) — render ON-re-textures acceptance (Task 6).
- `tests/test_gpu_regression.py` — bit-identity regression (Task 6, existing test, new OFF-config assertion).
- `PROJECT.md` — §6/§7 docs-sync (Task 6).

---

### Task 1: Ratify CKS-21 into SKILL.md (governance gate — owner review BEFORE kernel code)

**Files:**
- Modify: `skills/kerr-physics/SKILL.md` (promote the reserved CKS-21 line + add the full formula section + version/revision bumps)

**Interfaces:**
- Produces: the **math of record** every later task ports verbatim — `S(f)=1/(1+(f/f_c)^p)`; the de-shear-correction composition `φ′_{o,k} = curl_φ(φ−shear_k) + (1−S(f_o))·shear_k`; `f_o = f_base·lac^o`; the `disk.noise.shear_cascade` config block.

- [ ] **Step 1: Locate the reserved CKS-21 line and the current version/revision footer**

Open `skills/kerr-physics/SKILL.md` with `encoding="utf-8"`. Find line ~24:
```
- (RESERVED, design `docs/specs/2026-06-16-...-design.md`) scale-dependent shear cascade (CKS-21 — VISUALIZATION)
```
and the most recent revision (CKS-23 / v1.34, the P5 entry) to model the new section's format on.

- [ ] **Step 2: Promote the "When to use" line**

Replace the RESERVED line with the active form (mirror the CKS-18 / CKS-22 entries):
```
- Disk scale-dependent shear cascade / frequency-dependent shear transfer (Formula CKS-21 — VISUALIZATION)
```

- [ ] **Step 3: Add the full `## Formula CKS-21` section**

Insert after the CKS-18 section (the §2 advection family) or at the end of the CKS series, using this content (the design §B/§C are the source):

````markdown
## Formula CKS-21 — Scale-dependent shear cascade (owner-approved 2026-06-20; VISUALIZATION, NOT a metric)

> **Status:** same VISUALIZATION class as CKS-12 §2 / CKS-18 — it warps the disk-noise
> **coordinate** only (a per-octave azimuthal offset). It may not touch `p_μ`, `u^μ` (CKS-8),
> `g` (CKS-9), `g⁴` (Formula 9), `f_PT` (CKS-11), or the chroma form. Spec:
> `docs/specs/2026-06-20-P1-shear-cascade-design.md`.

CKS-12 §2 applies the Keplerian shear `φ′_k = φ − Ω(r)·a_k·T` **uniformly to the whole fBm**,
so every octave winds at the same rate and fine detail laminarizes into the same spiral as the
coarse structure. CKS-21 makes the shear **frequency-dependent**: per octave `o` of frequency
`f_o = f_base·lac^o`,

```
S(f)     = 1 / (1 + (f / f_c)^p)                       # transfer: low f → 1, high f → 0
shear_k  = dynamism · Ω(r) · (a_k · T)                 # the CKS-12 §2 shear amount, unchanged
φ′_{o,k} = φ − S(f_o) · shear_k                         # intuitive form (no curl)
```

**Composition with the CKS-18 curl warp (the implementation form).** The curl warp is nonlinear
and is applied to the already-sheared `φ_k` (CKS-18 §2). To keep that order (and bit-identity
when `S≡1`), the cascade is a per-octave **de-shear add-back** applied *after* curl:

```
φ_k       = φ − shear_k                                 # full §2 shear, before curl (UNCHANGED)
φ_c       = curl_φ(u, φ_k)                              # CKS-18 warp on φ_k (UNCHANGED order)
φ′_{o,k}  = φ_c + (1 − S(f_o)) · shear_k                # per-octave correction (the cascade)
```

`S(f_o) ≡ 1` (cascade off / `f_c → ∞`) ⇒ correction `0` ⇒ CKS-12 §2 uniform shear bit-for-bit.
C0-continuity at resets is preserved (the `w_k → 0` reset weight is independent of `S`).

**Scope:** the density octave stacks L0 (`fbm2`), L2 (`fbm2`), and L1's `ridged3`. The L1
Voronoi (single-frequency cellular — no octaves) and the L1 coverage mask + the §3 modulation
envelopes keep uniform shear. Density φ is linear-Perlin (`gnoise`, integer-period wrap) — the
constant-in-φ correction preserves the seam (constraint 5); no trig added.

Config `disk.noise.shear_cascade`: `enabled` (default false), `shear_cutoff` = `f_c` (0 ⇒ a
large sentinel ⇒ `S≡1`), `shear_falloff` = `p`. Base dials — no CKS-13 resolver change.
````

- [ ] **Step 4: Bump the version + revision history**

Update the SKILL.md version line to the next rev (e.g. `v1.35`) and add a revision-history row:
`CKS-21 — scale-dependent shear cascade (frequency-dependent shear transfer), VISUALIZATION`.

- [ ] **Step 5: Commit**

```bash
git add skills/kerr-physics/SKILL.md
git commit -m "feat(CKS-21): ratify scale-dependent shear cascade formula (SKILL.md, VISUALIZATION)"
```

- [ ] **Step 6: OWNER REVIEW GATE**

Per governance (design §B), pause for owner approval of the CKS-21 formula **before** any kernel
code. Do not proceed to Task 2 until approved.

---

### Task 2: CPU transfer helper + per-octave correction in `_octaves` (the primitive core)

**Files:**
- Modify: `src/renderer/noise.py` (add `SHEAR_FC_OFF`, `shear_transfer`; extend `_octaves`, `fbm2`, `fbm2_lod`, `_ridged`, `ridged3`)
- Test: `tests/test_noise.py`

**Interfaces:**
- Consumes: nothing (leaf primitives).
- Produces:
  - `SHEAR_FC_OFF: np.float32 = 1.0e9` — sentinel `f_c` meaning "no cutoff, `S≡1`".
  - `shear_transfer(f, f_c, p) -> np.ndarray` (float32) — `S(f)=1/(1+(f/f_c)^p)`; returns `1.0` when `f_c >= SHEAR_FC_OFF`.
  - Extended signatures (3 new trailing kwargs, all defaulting to the no-op sentinel):
    `fbm2(x, y, period, octaves=4, lacunarity=2, gain=0.5, seed=0, shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0)`
    `fbm2_lod(x, y, period, n_oct, octaves=4, lacunarity=2, gain=0.5, seed=0, shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0)`
    `ridged3(x, y, z, period, octaves=3, lacunarity=2, gain=0.5, offset=1.0, feedback=2.0, seed=0, shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0)`
  - The per-octave y-correction `Δy_o = (1 − S(f_o))·shear_k·(1/2π)·period`, applied to `y` before the `×freq` scale, with `f_o = period·freq`.

- [ ] **Step 1: Write the failing tests (transfer + collapse + per-octave displacement)**

Add to `tests/test_noise.py`:
```python
import numpy as np
from src.renderer import noise


def test_shear_transfer_monotone_and_sentinel():
    f = np.array([0.0, 1.0, 4.0, 16.0, 64.0], np.float32)
    s = noise.shear_transfer(f, np.float32(8.0), np.float32(2.0))
    # S(0) = 1, monotone decreasing, S(f≫f_c) → 0.
    assert np.isclose(s[0], 1.0, atol=1e-6)
    assert np.all(np.diff(s) <= 1e-7)
    assert s[-1] < 0.05
    # Sentinel f_c ⇒ S ≡ 1 exactly (the bit-identity hook).
    s_off = noise.shear_transfer(f, noise.SHEAR_FC_OFF, np.float32(2.0))
    assert np.all(s_off == np.float32(1.0))


def test_fbm2_shear_zero_is_bit_identical():
    # shear_k = 0 (default) ⇒ byte-for-byte the un-sheared fBm (constraint 6 hook).
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 64).astype(np.float32)
    y = rng.uniform(-3, 3, 64).astype(np.float32)
    a = noise.fbm2(x, y, 4, octaves=5, seed=7)
    b = noise.fbm2(x, y, 4, octaves=5, seed=7, shear_k=0.0)
    assert np.array_equal(a, b)


def test_fbm2_single_octave_displacement_matches_correction():
    # 1-octave fBm with shear_k = Δ equals the un-sheared fBm sampled at y displaced by
    # (1 − S(f_base))·Δ·(1/2π)·period  (the de-shear correction, octave 0).
    period = 4
    x = np.float32(0.3)
    y = np.float32(1.1)            # y is already (φ/2π)·period in the production caller
    delta = np.float32(0.7)        # shear_k (radians)
    f_c = np.float32(2.0)
    p = np.float32(2.0)
    f_base = np.float32(period * 1)               # octave 0: freq = 1
    s0 = noise.shear_transfer(f_base, f_c, p)
    y_corr = (np.float32(1.0) - s0) * delta * np.float32(noise._INV_TWO_PI) * np.float32(period)
    got = noise.fbm2(x, y, period, octaves=1, seed=3,
                     shear_k=delta, shear_fc=f_c, shear_p=p)
    ref = noise.fbm2(x, y + y_corr, period, octaves=1, seed=3)
    assert np.allclose(got, ref, atol=1e-6)


def test_fbm2_octave1_displacement_strictly_smaller_than_octave0():
    # The differential: octave 1 (higher f) is sheared LESS than octave 0.
    period = 4
    f_c, p = np.float32(3.0), np.float32(2.0)
    s0 = noise.shear_transfer(np.float32(period * 1), f_c, p)   # octave 0
    s1 = noise.shear_transfer(np.float32(period * 2), f_c, p)   # octave 1 (freq=2)
    # Displacement ∝ (1 − S); higher octave keeps MORE of its position (less displaced).
    assert (1.0 - s1) < (1.0 - s0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_noise.py -k "shear or displacement" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'shear_transfer'` / `SHEAR_FC_OFF`, and `fbm2() got an unexpected keyword argument 'shear_k'`.

- [ ] **Step 3: Add `SHEAR_FC_OFF` + `shear_transfer`**

Near the other module constants in `noise.py` (e.g. just above `_octaves`):
```python
# --------------------------------------------------------------------------- #
# CKS-21 — scale-dependent shear cascade: frequency transfer S(f).
# --------------------------------------------------------------------------- #
SHEAR_FC_OFF = np.float32(1.0e9)  # sentinel f_c ⇒ S ≡ 1 (no cutoff ⇒ uniform shear)


def shear_transfer(f, f_c, p):
    """CKS-21 frequency transfer ``S(f) = 1/(1 + (f/f_c)^p)`` ∈ (0, 1] (float32).

    Low frequencies → 1 (fully sheared into filaments); high frequencies → 0 (protected
    from the bulk shear). ``f_c >= SHEAR_FC_OFF`` short-circuits to exactly ``1.0`` (the
    bit-identity sentinel ⇒ uniform CKS-12 §2 shear). **CPU source of truth**; GPU twin
    ``shear_transfer_ti``."""
    f = np.asarray(f, np.float32)
    f_c = np.float32(f_c)
    if f_c >= SHEAR_FC_OFF:
        return np.ones_like(f, np.float32)
    ratio = (f / f_c).astype(np.float32)
    return (np.float32(1.0) / (np.float32(1.0) + np.power(ratio, np.float32(p)))).astype(np.float32)
```

- [ ] **Step 4: Extend `_octaves` with the per-octave φ-correction**

Replace the `_octaves` body (noise.py ~210-229) with the sheared form (the `shear_k == 0`
branch is the current code, untouched ⇒ bit-identical):
```python
def _octaves(base, x, y, z, period, octaves, lacunarity, gain, seed,
             shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0):
    """Yield ``(value, weight)`` per octave for the fBm-family loops.

    ``base`` is a callable ``(x*f, y*f[, z*f], period*f, seed+o) -> [0,1]``. ``lacunarity``
    is cast to ``int`` so ``period*f`` stays integral every octave (the φ-periodicity
    guarantee). ``z`` is ``None`` for the 2D primitives.

    **CKS-21 cascade.** When ``shear_k != 0`` the φ axis (``y``) gets a per-octave de-shear
    correction ``Δy_o = (1 − S(f_o))·shear_k·(1/2π)·period`` (``f_o = period·freq``), added
    BEFORE the ``×freq`` scale. The offset is constant in φ ⇒ the integer-period seam
    (constraint 5) survives. ``shear_k == 0`` ⇒ ``y`` unchanged ⇒ bit-identical.
    """
    lac = int(lacunarity)
    freq = 1
    per = int(period)
    amp = 1.0
    sk = np.float32(shear_k)
    for o in range(int(octaves)):
        yo = y
        if sk != np.float32(0.0):
            f_o = np.float32(period * freq)
            s_o = shear_transfer(f_o, shear_fc, shear_p)
            yo = (y + (np.float32(1.0) - s_o) * sk * np.float32(_INV_TWO_PI) * np.float32(period)).astype(np.float32)
        if z is None:
            n = base(x * freq, yo * freq, per, seed + o)
        else:
            n = base(x * freq, yo * freq, z * freq, per, seed + o)
        yield n, np.float32(amp)
        amp *= gain
        freq *= lac
        per *= lac
```
Confirm `_INV_TWO_PI` is module-level in `noise.py` (used by `_noise_m_stack`). If it is named differently, use that name.

- [ ] **Step 5: Forward the kwargs through `fbm2`, `fbm2_lod`, `_ridged`, `ridged3`**

`fbm2` (and `_fbm`): add the three kwargs and pass to `_octaves`. `_fbm`:
```python
def _fbm(base, x, y, z, period, octaves, lacunarity, gain, seed, transform,
         shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0):
    total = np.float32(0.0)
    norm = np.float32(0.0)
    for n, amp in _octaves(base, x, y, z, period, octaves, lacunarity, gain, seed,
                           shear_k, shear_fc, shear_p):
        total = total + transform(n) * amp
        norm = norm + amp
    return (total / norm).astype(np.float32)


def fbm2(x, y, period, octaves=4, lacunarity=2, gain=0.5, seed=0,
         shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0) -> np.ndarray:
    return _fbm(gnoise2, _f32(x), _f32(y), None, period, octaves, lacunarity, gain, seed,
                lambda n: n, shear_k, shear_fc, shear_p)
```
`fbm2_lod` (explicit loop): forward to `_octaves`:
```python
def fbm2_lod(x, y, period, n_oct, octaves=4, lacunarity=2, gain=0.5, seed=0,
             shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0) -> np.ndarray:
    total = np.float32(0.0)
    norm = np.float32(0.0)
    o = 0
    for n, amp in _octaves(gnoise2, _f32(x), _f32(y), None, period,
                           octaves, lacunarity, gain, seed, shear_k, shear_fc, shear_p):
        w = (amp * lod_octave_weight(n_oct, o)).astype(np.float32)
        total = total + n * w
        norm = norm + w
        o += 1
    return (total / norm).astype(np.float32)
```
`_ridged` + `ridged3`: add the three kwargs and forward to `_octaves`:
```python
def _ridged(base, x, y, z, period, octaves, lacunarity, gain, offset, feedback, seed,
            shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0):
    total = np.float32(0.0)
    norm = np.float32(0.0)
    prev = None
    for n, amp in _octaves(base, x, y, z, period, octaves, lacunarity, gain, seed,
                           shear_k, shear_fc, shear_p):
        if prev is None:
            w = np.float32(1.0)
        else:
            w = np.clip(prev * np.float32(feedback), 0.0, 1.0).astype(np.float32)
        d = np.float32(offset) - np.abs(np.float32(2.0) * n - np.float32(1.0))
        r = (d * d).astype(np.float32)
        prev = (w * r).astype(np.float32)
        total = total + prev * amp
        norm = norm + amp
    out = np.clip(total / norm, 0.0, 1.0)
    return out.astype(np.float32)


def ridged3(x, y, z, period, octaves=3, lacunarity=2, gain=0.5, offset=1.0, feedback=2.0,
            seed=0, shear_k=0.0, shear_fc=SHEAR_FC_OFF, shear_p=2.0) -> np.ndarray:
    return _ridged(gnoise3, _f32(x), _f32(y), _f32(z), period, octaves, lacunarity, gain,
                   offset, feedback, seed, shear_k, shear_fc, shear_p)
```

- [ ] **Step 6: Run the new tests + the full noise suite (no regressions)**

Run: `pytest tests/test_noise.py -v`
Expected: PASS — the new shear tests pass AND every pre-existing `test_noise.py` test still passes (the `shear_k == 0` default path is byte-for-byte unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/renderer/noise.py tests/test_noise.py
git commit -m "feat(CKS-21): CPU shear_transfer + per-octave de-shear correction in _octaves (default-off bit-identical)"
```

---

### Task 3: Thread `shear_k` through `_noise_m_stack` + `_advected_m` (route L0/L2/L1-ridged)

**Files:**
- Modify: `src/renderer/noise.py` (`_noise_m_stack`, `_advected_m`)
- Test: `tests/test_noise.py`

**Interfaces:**
- Consumes: `shear_transfer`, the extended `fbm2`/`ridged3` (Task 2).
- Produces:
  - `_noise_m_stack(u, phi, zeta, nz, seed, t_disk=0.0, shear_k=0.0)` — reads `nz["shear_cascade"]`; when enabled, applies `shear_k` + `f_c` + `p` to L0/L2 (`fbm2`) and L1 `ridged3`; the L1 Voronoi + mask are untouched (uniform shear).
  - `_advected_m` passes each phase's `shear_k = g·omega·(a_k·T)` into `_noise_m_stack` (static path ⇒ `shear_k = 0`).

- [ ] **Step 1: Write the failing test (cascade re-textures the modulator; off bit-identical)**

Add to `tests/test_noise.py`:
```python
def _nz_with_cascade(enabled, f_c=2.0, p=2.0):
    return {
        "m_max": 2.5, "variance_preserve": True, "dynamism": 1.0,
        "layers": {
            "base": {"enabled": True, "amp": 0.6, "octaves": 5, "lacunarity": 2,
                     "gain": 0.5, "freq_u": 6.0, "freq_phi": 4},
            "clump": {"enabled": False},
            "patch": {"enabled": False},
        },
        "shear_cascade": {"enabled": enabled, "shear_cutoff": f_c, "shear_falloff": p},
    }


def test_advected_m_cascade_off_is_static_reference():
    # Cascade OFF ⇒ identical to the pre-CKS-21 advected modulator (constraint 6).
    u = np.linspace(0.1, 0.9, 32, dtype=np.float32)
    phi = np.linspace(0.0, 6.0, 32, dtype=np.float32)
    zeta = np.zeros(32, np.float32)
    nz_off = _nz_with_cascade(False)
    a = noise._advected_m(u, phi, zeta, nz_off, seed=11, t_disk=40.0,
                          omega=np.float32(0.05), shear_period=10.0)
    # Same dict WITHOUT the shear_cascade key must give the byte-identical result.
    nz_nokey = {k: v for k, v in nz_off.items() if k != "shear_cascade"}
    b = noise._advected_m(u, phi, zeta, nz_nokey, seed=11, t_disk=40.0,
                          omega=np.float32(0.05), shear_period=10.0)
    assert np.array_equal(a, b)


def test_advected_m_cascade_on_changes_field_at_long_T():
    # Cascade ON re-textures the advected modulator vs OFF (the protected high octaves
    # no longer wind with the bulk). Different field, same shape.
    u = np.linspace(0.1, 0.9, 64, dtype=np.float32)
    phi = np.linspace(0.0, 6.0, 64, dtype=np.float32)
    zeta = np.zeros(64, np.float32)
    args = dict(seed=11, t_disk=80.0, omega=np.float32(0.08), shear_period=8.0)
    off = noise._advected_m(u, phi, zeta, _nz_with_cascade(False), **args)
    on = noise._advected_m(u, phi, zeta, _nz_with_cascade(True, f_c=2.0, p=2.0), **args)
    assert off.shape == on.shape
    assert not np.allclose(on, off, atol=1e-4)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_noise.py -k "advected_m_cascade" -v`
Expected: FAIL — the ON/OFF fields are currently identical (`not np.allclose` fails) because `_noise_m_stack` ignores the cascade.

- [ ] **Step 3: Thread `shear_k` + cascade dials into `_noise_m_stack`**

In `_noise_m_stack` add the `shear_k` param and resolve the cascade dials right after the curl
warp, then pass them to L0/L2/L1-ridged. Header + dial resolution:
```python
def _noise_m_stack(u, phi, zeta, nz, seed: int, t_disk: float = 0.0, shear_k: float = 0.0) -> np.ndarray:
    u, phi = _apply_curl(u, phi, nz.get("curl"), t_disk=t_disk)
    phi01 = phi * _INV_TWO_PI
    m = np.zeros(np.broadcast(u, phi, zeta).shape, dtype=np.float32)

    # CKS-21 cascade dials. Off / absent ⇒ sk = 0 ⇒ every layer is uniform-sheared
    # (bit-identical). Applied to the octave stacks L0/L2/L1-ridged only.
    sc = nz.get("shear_cascade", {}) or {}
    if sc.get("enabled", False):
        _sk = np.float32(shear_k)
        _fc = np.float32(sc.get("shear_cutoff", 0.0) or 0.0)
        _fc = _fc if _fc > np.float32(0.0) else SHEAR_FC_OFF
        _sp = np.float32(sc.get("shear_falloff", 2.0))
    else:
        _sk, _fc, _sp = np.float32(0.0), SHEAR_FC_OFF, np.float32(2.0)
    ...
```
Then add the kwargs to the L0 `fbm2`, the L1 `ridged3`, and the L2 `fbm2` calls:
```python
    # L0 — base streaks (fBm) + CKS-21 cascade.
    if base.get("enabled", False):
        fp = int(base["freq_phi"])
        n0 = fbm2(u * base["freq_u"], phi01 * fp, fp, octaves=int(base["octaves"]),
                  lacunarity=int(base["lacunarity"]), gain=base["gain"], seed=seed + NSEED_L0,
                  shear_k=_sk, shear_fc=_fc, shear_p=_sp)
        m = m + np.float32(base["amp"]) * (n0 - np.float32(0.5))
    ...
        ridge = ridged3(xu, yphi, zz, fp, octaves=int(clump["octaves"]),
                        lacunarity=int(clump["lacunarity"]), gain=clump["gain"],
                        offset=clump["ridge_offset"], feedback=RIDGE_FEEDBACK,
                        seed=seed + NSEED_L1_RIDGE,
                        shear_k=_sk, shear_fc=_fc, shear_p=_sp)
        voro = voronoi_billow3(xu, yphi, zz, fp, k=clump["voronoi_k"], seed=seed + NSEED_L1_VORO)
        # mask fbm2 keeps uniform shear (structural gate, NOT a turbulence cascade) — no kwargs.
    ...
    # L2 — patchiness (fBm) + CKS-21 cascade.
    if patch.get("enabled", False):
        fp = int(patch["freq_phi"])
        n2 = fbm2(u * patch["freq_u"], phi01 * fp, fp, octaves=int(patch["octaves"]),
                  lacunarity=int(patch["lacunarity"]), gain=patch["gain"], seed=seed + NSEED_L2,
                  shear_k=_sk, shear_fc=_fc, shear_p=_sp)
        m = m + np.float32(patch["amp"]) * (n2 - np.float32(0.5))
```

- [ ] **Step 4: Pass each phase's `shear_k` from `_advected_m`**

In `_advected_m`, forward `shear_k` (and `0.0` on the static path):
```python
    if T <= np.float32(0.0):
        return _noise_m_stack(u, phi, zeta, nz, int(seed), t_disk=t_disk, shear_k=0.0)
    ...
    for k in (0, 1):
        ...
        sk = g * omega * (ak * T)                         # the amount sheared off this phase
        phi_k = phi - sk                                  # CKS-12 §2: φ sheared (= old expression)
        mk = wk * _noise_m_stack(u, phi_k, zeta, nz, seed_k, t_disk=t_disk, shear_k=sk)
        m = mk if m is None else m + mk
        wsq = wsq + wk * wk
```
(`sk` is the same product `g*omega*(ak*T)` the old `phi_k` subtracted, now also handed down so
the octave loop can add the protected fraction back.)

- [ ] **Step 5: Run the cascade tests + full noise suite**

Run: `pytest tests/test_noise.py -v`
Expected: PASS — both new `_advected_m` tests pass; all pre-existing tests still pass (OFF byte-identical).

- [ ] **Step 6: Commit**

```bash
git add src/renderer/noise.py tests/test_noise.py
git commit -m "feat(CKS-21): route L0/L2/L1-ridged through the shear cascade in _noise_m_stack (off bit-identical)"
```

---

### Task 4: Config block + GPU param buffer (`_NI_SC_*`, `_NOISE_N`, `_setup_disk_noise`)

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (`_NI_SC_*` constants, `_NOISE_N`, `_SC_FC_OFF`, `_setup_disk_noise`)
- Modify: `configs/render.yaml` (add `disk.noise.shear_cascade`)
- Test: `tests/test_noise_gpu.py` (a setup smoke test)

**Interfaces:**
- Consumes: `noise.SHEAR_FC_OFF` (Task 2).
- Produces (read by Task 5):
  - `_NI_SC_EN = 69`, `_NI_SC_FC = 70`, `_NI_SC_P = 71`; `_NOISE_N = 72`.
  - `_SC_FC_OFF = float(noise.SHEAR_FC_OFF)` (a Python float for the `@ti.func` defaults).
  - `_setup_disk_noise` packs `buf[_NI_SC_EN/FC/P]` from `disk.noise.shear_cascade`.

- [ ] **Step 1: Write the failing setup smoke test**

Add to `tests/test_noise_gpu.py` (it already `ti.init`s CUDA in its fixtures — follow the file's
existing setup pattern; this test only checks the packed buffer, no kernel):
```python
def test_setup_packs_shear_cascade_params():
    import numpy as np
    from src.renderer import taichi_renderer as tr
    cfg = {
        "disk": {"noise": {"shear_cascade": {
            "enabled": True, "shear_cutoff": 5.0, "shear_falloff": 3.0}}},
    }
    tr._setup_disk_noise(cfg)
    buf = tr.disk_noise_params.to_numpy()
    assert buf.shape[0] == tr._NOISE_N == 72
    assert buf[tr._NI_SC_EN] == 1.0
    assert np.isclose(buf[tr._NI_SC_FC], 5.0)
    assert np.isclose(buf[tr._NI_SC_P], 3.0)
    # Disabled / absent ⇒ enabled flag 0 and f_c = the sentinel (S ≡ 1).
    tr._setup_disk_noise({"disk": {"noise": {}}})
    buf2 = tr.disk_noise_params.to_numpy()
    assert buf2[tr._NI_SC_EN] == 0.0
    assert buf2[tr._NI_SC_FC] >= tr._SC_FC_OFF
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_noise_gpu.py::test_setup_packs_shear_cascade_params -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_NI_SC_EN'` (and `_NOISE_N` is 69).

- [ ] **Step 3: Add the index constants, sentinel, and bump `_NOISE_N`**

In `taichi_renderer.py`, after the CKS-23 LOD block (`_NI_LOD_EPS = 68`):
```python
# CKS-21 scale-dependent shear cascade (frequency-dependent shear transfer). Base dials
# (no CKS-13 resolver change). enabled:false / absent ⇒ _NI_SC_EN=0 ⇒ shear_k=0 fed to the
# fBm twins ⇒ the de-shear correction is exactly 0 ⇒ bit-identical (constraint 6).
_NI_SC_EN = 69
_NI_SC_FC = 70    # f_c — shear cutoff frequency (sentinel _SC_FC_OFF ⇒ S≡1, no protection)
_NI_SC_P = 71     # p — transfer steepness
```
Bump:
```python
_NOISE_N = 72
```
And the sentinel (the `@ti.func` default + setup fallback), near `_LOD_OFF`:
```python
# CKS-21: sentinel f_c fed to the sheared fBm on the cascade-OFF path. f_c ≥ this ⇒ the
# transfer S(f) short-circuits to exactly 1.0 ⇒ no correction (constraint 6). Sourced from
# the CPU twin so the two cannot drift.
_SC_FC_OFF = float(noise.SHEAR_FC_OFF)
```

- [ ] **Step 4: Pack the params in `_setup_disk_noise`**

After the CKS-23 LOD packing block (`buf[_NI_LOD_EPS] = ...`), add:
```python
    # CKS-21 scale-dependent shear cascade. Absent block / enabled:false ⇒ _NI_SC_EN=0 ⇒
    # _disk_noise_m feeds shear_k=0 + f_c=_SC_FC_OFF to the fBm twins ⇒ (1−S)·0 = 0 ⇒
    # bit-identical (constraint 6). Base look dials — no CKS-13 resolver change.
    sc = nz.get("shear_cascade", {}) or {}
    buf[_NI_SC_EN] = 1.0 if sc.get("enabled", False) else 0.0
    fc = float(sc.get("shear_cutoff", 0.0) or 0.0)
    buf[_NI_SC_FC] = fc if fc > 0.0 else _SC_FC_OFF
    buf[_NI_SC_P] = float(sc.get("shear_falloff", 2.0))
```

- [ ] **Step 5: Add the config block to `configs/render.yaml`**

Under `disk: noise:` (sibling of `curl:`), add (open with `encoding="utf-8"`):
```yaml
    shear_cascade:           # CKS-21 — frequency-dependent shear transfer. Default OFF ⇒ bit-identical.
      enabled: false         # false (or shear_cutoff huge) ⇒ S≡1 ⇒ CKS-12 §2 uniform shear, bit-for-bit.
      shear_cutoff: 0.0      # f_c (cycles). 0 ⇒ auto = +inf sentinel ⇒ S≡1 (no protection). >0 ⇒ protect octaves above f_c.
      shear_falloff: 2.0     # p — transfer steepness; larger ⇒ sharper low/high split around f_c.
```

- [ ] **Step 6: Run the smoke test + full setup-touching suites**

Run: `pytest tests/test_noise_gpu.py::test_setup_packs_shear_cascade_params -v`
Expected: PASS.
Run: `pytest tests/test_disk_noise.py -k setup -v` (or the file's existing setup tests)
Expected: PASS — no `_NOISE_N`-shape regressions elsewhere.

- [ ] **Step 7: Commit**

```bash
git add src/renderer/taichi_renderer.py configs/render.yaml tests/test_noise_gpu.py
git commit -m "feat(CKS-21): disk.noise.shear_cascade config + _NI_SC_* param buffer (_NOISE_N 69->72)"
```

---

### Task 5: GPU twins — `shear_transfer_ti` + `fbm2_lod_ti`/`ridged3_ti` + thread through `_disk_noise_m`

**Files:**
- Modify: `src/renderer/noise.py` (`shear_transfer_ti`, `fbm2_lod_ti`, `ridged3_ti`)
- Modify: `src/renderer/taichi_renderer.py` (`_disk_noise_m`, `_disk_blended_m`)
- Test: `tests/test_noise_gpu.py`, `tests/test_disk_noise.py`

**Interfaces:**
- Consumes: `_NI_SC_*`, `_SC_FC_OFF` (Task 4); the CPU reference `_advected_m` (Task 3).
- Produces:
  - `shear_transfer_ti(f, f_c, p) -> f32` (`@ti.func`) — GPU twin of `shear_transfer`.
  - `fbm2_lod_ti(x, y, period, octaves, lac, gain, seed, n_oct, shear_k=0.0, shear_fc=_SC_FC_OFF, shear_p=2.0)`.
  - `ridged3_ti(x, y, z, period, octaves, lac, gain, offset, feedback, seed, shear_k=0.0, shear_fc=_SC_FC_OFF, shear_p=2.0)`.
  - `_disk_noise_m(u, phi, zeta, seed, t_disk, n_oct=_LOD_OFF, shear_k=0.0)` threading the dials to L0/L2/L1-ridged; `_disk_blended_m` passes each phase's `shear_k`.

- [ ] **Step 1: Write the failing GPU parity test**

Add to `tests/test_noise_gpu.py` (mirror an existing `_disk_blended_m` parity test; `_SATOL` is
the file's tolerance constant):
```python
def test_gpu_cascade_matches_cpu_advected_m():
    import numpy as np, taichi as ti
    from src.renderer import noise, taichi_renderer as tr
    # nz/config with L0 on + cascade on; mirror _nz_with_cascade from test_noise.py.
    cfg = {"disk": {"noise": {
        "m_max": 2.5, "variance_preserve": True, "dynamism": 1.0,
        "layers": {"base": {"enabled": True, "amp": 0.6, "octaves": 5, "lacunarity": 2,
                            "gain": 0.5, "freq_u": 6.0, "freq_phi": 4},
                   "clump": {"enabled": False}, "patch": {"enabled": False}},
        "shear_cascade": {"enabled": True, "shear_cutoff": 2.0, "shear_falloff": 2.0},
    }}}
    tr._setup_disk_noise(cfg)
    U = np.linspace(0.1, 0.9, 24, np.float32)
    PH = np.linspace(0.0, 6.0, 24, np.float32)
    out = ti.field(ti.f32, shape=24)

    @ti.kernel
    def run(seed: ti.i32, t_disk: ti.f32, omega: ti.f32):
        for i in range(24):
            out[i] = tr._disk_blended_m(U[i], PH[i], 0.0, t_disk, omega, seed)

    run(11, 80.0, 0.08)
    gpu = out.to_numpy()
    cpu = noise._advected_m(U, PH, np.zeros(24, np.float32), cfg["disk"]["noise"],
                            seed=11, t_disk=80.0, omega=np.float32(0.08), shear_period=8.0)
    # shear_period must match _NI_SHEAR_T; set disk.dynamics or buf directly in the test
    # setup so the GPU T equals 8.0 (see the file's existing pattern for forcing _NI_SHEAR_T).
    assert np.allclose(gpu, cpu, atol=tr_SATOL_OR_FILE_CONST)
```
(Use the test file's established way of forcing `_NI_SHEAR_T`/`omega` and its `_SATOL` constant —
match the existing `_disk_blended_m` parity test in the same file.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_noise_gpu.py::test_gpu_cascade_matches_cpu_advected_m -v`
Expected: FAIL — the GPU ignores the cascade (`_disk_noise_m` has no `shear_k`), so GPU≠CPU.

- [ ] **Step 3: Add `shear_transfer_ti` (GPU twin of `shear_transfer`)**

In `noise.py`, near the other `@ti.func` primitives:
```python
@ti.func
def shear_transfer_ti(f, f_c, p):
    """CKS-21 transfer S(f)=1/(1+(f/f_c)^p); twin of :func:`shear_transfer`. f_c at the
    sentinel ⇒ exactly 1.0 (the bit-identity short-circuit; no pow executed)."""
    s = 1.0
    if f_c < _SC_FC_OFF_TI:
        s = 1.0 / (1.0 + ti.pow(f / f_c, p))
    return s
```
Add the module constant in `noise.py`: `_SC_FC_OFF_TI = float(SHEAR_FC_OFF)` (a plain Python
float so it is usable as a `@ti.func` default and comparison literal).

- [ ] **Step 4: Extend `fbm2_lod_ti` with the per-octave correction**

Replace the `fbm2_lod_ti` loop body (the `shear_k == 0` path is bit-identical; the
`f_c == sentinel` short-circuit means no pow on the OFF path):
```python
@ti.func
def fbm2_lod_ti(x, y, period, octaves, lac, gain, seed, n_oct,
                shear_k=0.0, shear_fc=_SC_FC_OFF_TI, shear_p=2.0):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    for o in range(octaves):
        yo = y
        f_o = ti.cast(period * freq, ti.f32)
        s_o = shear_transfer_ti(f_o, shear_fc, shear_p)
        yo = y + (1.0 - s_o) * shear_k * _INV_TWO_PI_TI * ti.cast(period, ti.f32)
        n = gnoise2_ti(x * freq, yo * freq, per, seed + o)
        g = ti.min(ti.max(n_oct - ti.cast(o, ti.f32), 0.0), 1.0)
        w = amp * g
        total += n * w
        norm += w
        amp *= gain
        freq *= lac
        per *= lac
    return total / norm
```
Add `_INV_TWO_PI_TI = float(1.0 / (2.0 * math.pi))` in `noise.py` if not already present (the GPU
module uses `_INV_TWO_PI` in taichi_renderer; define the noise-module twin constant). With
`shear_k = 0` and `shear_fc = sentinel`: `s_o = 1.0` ⇒ `yo = y + 0 = y` ⇒ `gnoise2_ti(x*freq,
y*freq, ...)` exactly the current call ⇒ bit-identical.

- [ ] **Step 5: Extend `ridged3_ti` the same way**

In `ridged3_ti`'s loop (the 3D ridged twin), add the identical `yo` correction before
`gnoise3_ti(x*freq, yo*freq, z*freq, per, seed+o)`, with the same `(shear_k, shear_fc, shear_p)`
trailing defaults on the signature. Keep the ridged spectral-feedback `prev`/`w` logic unchanged.

- [ ] **Step 6: Thread `shear_k` + dials through `_disk_noise_m` and `_disk_blended_m`**

In `_disk_noise_m`, add the `shear_k` param and resolve the dials once after the curl warp:
```python
@ti.func
def _disk_noise_m(u, phi, zeta, seed, t_disk, n_oct=_LOD_OFF, shear_k=0.0):
    if disk_noise_params[_NI_CURL_EN] > 0.5:
        w = _disk_curl_warp(u, phi, t_disk)
        u = w[0]
        phi = w[1]
    # CKS-21 dials: off ⇒ sk=0 + f_c=sentinel ⇒ no correction (bit-identical).
    sk = 0.0
    fc = _SC_FC_OFF
    pp = 2.0
    if disk_noise_params[_NI_SC_EN] > 0.5:
        sk = shear_k
        fc = disk_noise_params[_NI_SC_FC]
        pp = disk_noise_params[_NI_SC_P]
    m = 0.0
    phi01 = phi * _INV_TWO_PI
    ...
```
Pass `(sk, fc, pp)` to the L0 `fbm2_lod_ti`, the L1 `ridged3_ti`, and the L2 `fbm2_lod_ti`
calls (append `sk, fc, pp` after their current last arg). Leave the L1 **mask** `fbm2_lod_ti`
call and the **Voronoi** call unchanged (uniform shear). E.g. L0:
```python
        n0 = noise.fbm2_lod_ti(
            u * disk_noise_params[_NI_L0_FU], phi01 * fpf, fp,
            ti.cast(disk_noise_params[_NI_L0_OCT], ti.i32),
            ti.cast(disk_noise_params[_NI_L0_LAC], ti.i32),
            disk_noise_params[_NI_L0_GAIN], seed + _NSEED_L0, n_oct,
            sk, fc, pp,
        )
```
In `_disk_blended_m`, pass each phase's `shear_k = g*omega*(a_k*T)` (the same product already
subtracted to form the sheared φ):
```python
        m += w0 * _disk_noise_m(u, phi - g * omega * (a0 * T), zeta, seed0, t_disk, n_oct,
                                g * omega * (a0 * T))
        ...
        m += w1 * _disk_noise_m(u, phi - g * omega * (a1 * T), zeta, seed1, t_disk, n_oct,
                                g * omega * (a1 * T))
```
The static branch (`T <= 0`) calls `_disk_noise_m(..., n_oct)` with the default `shear_k = 0.0`
(unchanged).

- [ ] **Step 7: Run the GPU parity test + the C0-at-reset guard**

Run: `pytest tests/test_noise_gpu.py::test_gpu_cascade_matches_cpu_advected_m -v`
Expected: PASS (GPU within `_SATOL` of the CPU twin).
Add/confirm a C0-at-reset case in `tests/test_disk_noise.py` (cascade ON; evaluate `_advected_m`
just before and after a reset boundary `t_disk = c·T ± ε` and assert continuity), then:
Run: `pytest tests/test_disk_noise.py -k "reset or cascade" -v`
Expected: PASS (continuous across the reset — `w_k → 0` independent of `S`).

- [ ] **Step 8: Commit**

```bash
git add src/renderer/noise.py src/renderer/taichi_renderer.py tests/test_noise_gpu.py tests/test_disk_noise.py
git commit -m "feat(CKS-21): GPU shear_transfer_ti + fbm2_lod_ti/ridged3_ti cascade threaded through _disk_noise_m (parity within _SATOL)"
```

---

### Task 6: Bit-identity regression, render acceptance, golden, docs-sync

> **DEVIATION (landed 2026-06-20, commit 3f1b620, final-review-approved):** Steps 1/5
> below were written against a stored-image-golden harness (`assert np.array_equal(frame,
> golden_frame)`) that does **not** exist in this repo — `tests/test_gpu_regression.py` is
> *metric-based* (Doppler ratio, disk-peak, NaN, spin-axis seam), and there are no `.npy`/
> image goldens anywhere (the CKS-22/23 precedent). Both OFF-bit-identity (compared against
> the block being *absent*, at a nonzero `t_disk` so the advected branch threads `shear_k`)
> and ON-re-texture were therefore placed in the feature's own `tests/test_disk_shear_cascade.py`,
> and no image golden was added. The final whole-branch review adjudicated this a faithful
> adaptation (coverage in fact stronger than planned), not a gap.

**Files:**
- Modify: `tests/test_gpu_regression.py` (assert OFF golden bit-identity with the new block present)
- Create: `tests/test_disk_shear_cascade.py` (render ON-re-textures acceptance)
- Modify: `PROJECT.md` (§6/§7 docs-sync)
- Add: one cascade-ON golden frame (under the repo's golden-image dir)

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: the constraint-6 guard + the acceptance guard + synced docs.

- [ ] **Step 1: Write the bit-identity regression assertion**

In `tests/test_gpu_regression.py`, add a case proving a config carrying
`disk.noise.shear_cascade.enabled: false` renders **bit-identical** to the existing golden (the
new block must not move any pixel). Follow the file's existing golden-compare harness:
```python
def test_shear_cascade_off_is_bit_identical(golden_frame, base_config):
    cfg = deepcopy(base_config)
    cfg["disk"].setdefault("noise", {})["shear_cascade"] = {
        "enabled": False, "shear_cutoff": 0.0, "shear_falloff": 2.0}
    frame = render_beauty_to_numpy(cfg)            # the file's existing render helper
    assert np.array_equal(frame, golden_frame)     # byte-for-byte
```

- [ ] **Step 2: Run it (expect PASS immediately — OFF must already be bit-identical)**

Run: `pytest tests/test_gpu_regression.py::test_shear_cascade_off_is_bit_identical -v`
Expected: PASS. If it FAILS, the disabled path is not bit-identical — STOP and fix the
sentinel/default plumbing (Tasks 2/4/5) before continuing.

- [ ] **Step 3: Write the render acceptance test (ON re-textures the disk; no emergent-spectrum assertion)**

Create `tests/test_disk_shear_cascade.py`. Render the **disk-only buffer at a face-on camera at a
long `T`** (the `project_p4p5_edge_lod` memory's calibration: `disk_buf.to_numpy()[:,:,0:3]`, a
face-on cam at distance < `render.r_max`), once with the cascade OFF and once ON, and assert the
ON field **differs** from OFF (re-textures). Do NOT assert any radial/φ power-spectrum property
(design §A.3 — laminarization is radial, a φ-spectrum can't see it):
```python
def test_cascade_on_retextures_disk_face_on_long_T(face_on_disk_render):
    off = face_on_disk_render(shear_cascade=False, t_disk=80.0)   # disk_buf rgb
    on = face_on_disk_render(shear_cascade=True, f_c=2.0, p=2.0, t_disk=80.0)
    # Same lit support, but the texture differs where the disk is bright.
    lit = off.sum(axis=2) > 1e-3
    assert lit.sum() > 100                                        # the disk is actually on screen
    diff = np.abs(on - off).sum(axis=2)
    assert diff[lit].mean() > 1e-3                                # ON re-textures the lit disk
```
Use the working face-on cam from the memory: `{pos:[0,0,45], fwd:[0,0,-1], up:[0,1,0], fov:1.1}`
at 480×270 (distance 45 < `render.r_max` 50), and render the **disk buffer**, not the composite.

- [ ] **Step 4: Run the acceptance test**

Run: `pytest tests/test_disk_shear_cascade.py -v`
Expected: PASS (ON differs from OFF on the lit disk; the suite is GPU — allow the usual
re-`ti.init` runtime).

- [ ] **Step 5: Generate + commit one cascade-ON golden frame**

Render a long-`T` beauty frame with `shear_cascade.enabled: true, shear_cutoff: ~2.0` via the
project's golden/showcase script, and add it under the golden-image directory the regression
suite reads. Keep it small (the project's thumb/showcase resolution).

- [ ] **Step 6: Docs-sync — update PROJECT.md §6/§7**

Add the landed CKS-21 cascade to PROJECT.md §6 (formula/feature list) and §7 (config/dials), per
the Docs Sync Policy: the `disk.noise.shear_cascade` dials, the VISUALIZATION class, the
density-octave-stack scope (L0/L2/L1-ridged), and the default-off bit-identity guarantee. (SKILL.md
was already synced in Task 1.)

- [ ] **Step 7: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS — the new tests plus every prior regression (including all existing goldens).

- [ ] **Step 8: Commit**

```bash
git add tests/test_gpu_regression.py tests/test_disk_shear_cascade.py PROJECT.md <golden-image-path>
git commit -m "feat(CKS-21): bit-identity regression + face-on re-texture acceptance + golden + PROJECT.md docs-sync (P1 complete)"
```

---

## Self-Review

**Spec coverage (design → task):**
- §B governance / VISUALIZATION guard → Global Constraints + Task 1 SKILL section.
- §C math of record + de-shear-correction composition → Task 1 (ratify) + Tasks 2/5 (code).
- §C.1 bit-identity hook (`shear_k=0` sentinel) → Tasks 2/4/5 defaults + Task 6 regression.
- §C.2 C0-at-reset → Task 5 Step 7.
- §D linear-Perlin, no trig, `_octaves`-centralized, CKS-23-precedent no-gate → Tasks 2/5.
- §D.1 scope L0/L2/L1-ridged, Voronoi/mask uniform → Task 3 + Task 5 Step 6.
- §E config block → Task 4.
- §F resolved decisions (transfer form, per-layer f_o, curl order unchanged, config home, perf) → Tasks 2–5.
- §G tests (transfer, collapse, per-octave displacement, GPU parity, C0, regression, ON-re-textures, golden) → Tasks 2/3/5/6.
- §H phasing + docs-sync → task order + Tasks 1/6.

**Placeholder scan:** the only deferred specifics are the test file's existing harness names
(`_SATOL`, the golden-compare fixture, the face-on render helper, the `_NI_SHEAR_T` forcing
pattern) — each step names the existing pattern to mirror rather than inventing one, because those
constants/fixtures already exist in the target files and must be reused verbatim. No `TBD`/`TODO`.

**Type consistency:** the three trailing kwargs `(shear_k=0.0, shear_fc=SHEAR_FC_OFF/_SC_FC_OFF,
shear_p=2.0)` are identical across `fbm2`/`fbm2_lod`/`ridged3` (CPU) and `fbm2_lod_ti`/`ridged3_ti`
(GPU); `_NI_SC_EN/FC/P = 69/70/71` and `_NOISE_N = 72` are defined in Task 4 and consumed in Task
5; `shear_transfer`/`shear_transfer_ti` share the `(f, f_c, p)` signature; `_INV_TWO_PI` (CPU) /
`_INV_TWO_PI_TI` (GPU noise module) are the same constant.
