# Pillar 2 — Multi-Phase Disk Media (CKS-19) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the disk's emission density from its absorption density so optically-thick, non-luminous dust lanes can carve high-contrast black silhouettes across the glowing plasma — without moving any existing golden frame.

**Architecture:** Today one mass density `ρ` (from `_disk_density_cks`) drives BOTH emission (`j ∝ ρ`) and absorption (`dτ = κ·ρ·ds`). CKS-19 splits it into `ρ_hot` (emission, the existing field renamed) and a NEW decoupled `ρ_cold` (absorption only). `ρ_cold`'s log-density modulator is correlated to `ρ_hot`'s by one coefficient `χ ∈ [−1,1]` via the variance-preserving Pearson construction `m_cold = χ·m_hot + √(1−χ²)·m_dust`. Absorption stays **grey** (one scalar `κ = absorption_coeff`) for this pillar; the per-channel/chromatic extension is isolated in a final optional task. Default OFF ⇒ `ρ_cold ≡ ρ_hot` ⇒ bit-identical.

**Tech Stack:** Python, NumPy (CPU twin `src/renderer/noise.py`), Taichi CUDA (`src/renderer/taichi_renderer.py`), pytest (CPU parity in `tests/test_noise.py`, GPU parity in `tests/test_disk_noise.py`/`test_noise_gpu.py`, regression in `tests/test_gpu_regression.py`). Physics source of truth: `skills/kerr-physics/SKILL.md` Formula CKS-19 (already authored).

---

## Key design decisions (read before starting)

1. **`m_dust` = the full layer-stack modulator re-evaluated with a new seed offset (`NSEED_DUST`), not a lone fBm.** The SKILL's variance-preservation claim (`Var(m_cold)` constant ⇒ Pearson `r = χ`) holds *only* when `Var(m_dust) = Var(m_hot)`. Re-running the identical generating process (same amps/layers) with a decorrelated seed guarantees equal variance and independence by construction. This is the single most important correctness point.

2. **Reuse the shear/curl advection for free.** Both `m_hot` and `m_dust` go through the *same* dual-phase reset + curl warp (CKS-12 §2 / CKS-18). We factor the existing pre-clamp blend into a helper (`_advected_m` CPU, `_disk_blended_m` GPU) and call it twice with different seeds. No advection logic is duplicated.

3. **Grey absorption now (scalar `κ`).** With `κ_R=κ_G=κ_B` the per-channel transmittance march is bit-identical to the scalar march, so we keep `dtau` scalar, `transm` scalar, and `disk_buf` 4-channel for Tasks 1–6. `ρ_cold` drives that scalar `dtau`. This fully delivers the acceptance test (dust carves a dark silhouette). The vec3-transmittance / chromatic-`κ⃗` plumbing (the "chromatic-ready" half of the owner decision) is **Task 7, optional/deferred** — it is structurally isolated and has its own golden risk.

4. **`σ_cold = σ_hot · dust_sigma_frac` is computed inside `_disk_density_cks`** from a ratio param — NO `kerr_params.resolve_config` (CKS-13) change. Keeps the resolver untouched (same discipline as the curl/flare blocks).

5. **The self-shadow bake reads `ρ_cold`** (CKS-19 constraint 2: the deep-shadow-map is an *absorption* optical depth). `_disk_density_cks` returns `vec3(ρ_hot, ρ_cold, temp_factor)`; the bake reads index `[1]`. When OFF, `[1] == [0]` ⇒ unchanged.

---

## File structure (what changes and why)

| File | Change | Responsibility |
|---|---|---|
| `skills/kerr-physics/SKILL.md` | **Already done** (CKS-19 authored, committed `88c9a59`). Promote status DESIGN→ACTIVE in the final task. | Physics source of truth. |
| `src/renderer/noise.py` | Add `NSEED_DUST`; factor `_advected_m` helper out of `noise_density_mult`; add `dust_density_mult`. | CPU source-of-truth twin for `ρ_cold`'s modulator. |
| `src/renderer/taichi_renderer.py` | Add `_NSEED_DUST` + `_NI_MP_*` indices (`_NOISE_N` 53→57); buffer fill; factor `_disk_blended_m`; widen `_disk_density_cks`→vec3; wire `_disk_emit_cks` (emission←ρ_hot, dτ←ρ_cold) + `bake_disk_shadow` (←ρ_cold) + step cap (thinner σ). | GPU twin. |
| `configs/render.yaml` | Add `disk.multiphase` block (default `enabled:false`). | Config (CKS-19 dials). |
| `tests/test_noise.py` | CPU Pearson-correlation + variance-preservation parity for `dust_density_mult`. | CPU correctness. |
| `tests/test_disk_noise.py` | GPU `ρ_cold` twin vs CPU within `_SATOL`; flag-off bit-identity. | CPU/GPU parity. |
| `tests/test_disk_multiphase.py` | **New.** `test_dust_carves_silhouette` (the B.1 acceptance test). | Acceptance. |
| `tests/test_gpu_regression.py` | Unchanged; re-run to prove `enabled:false` goldens bit-identical. | Constraint-6 guard. |

---

## Task 1: CPU twin — `NSEED_DUST` + factor `_advected_m` helper

**Files:**
- Modify: `src/renderer/noise.py` (seed constants near line 663; `noise_density_mult` ~line 759–823)

This task is a pure refactor — extract the pre-clamp blended `m` so Task 2 can reuse it. Behavior of `noise_density_mult` must not change.

- [ ] **Step 1: Write the failing test** (proves the helper exists and `noise_density_mult` still equals `exp(clamp(_advected_m))`)

Add to `tests/test_noise.py`:

```python
def test_advected_m_reconstructs_density_mult():
    """noise_density_mult == exp(clamp(_advected_m, ±m_max)) — refactor parity."""
    from renderer import noise as N
    nz = {
        "m_max": 2.5,
        "layers": {"base": {"enabled": True, "amp": 0.6, "octaves": 5,
                            "lacunarity": 2, "gain": 0.5, "freq_u": 6.0, "freq_phi": 24}},
    }
    u = np.linspace(0.0, 2.0, 64).astype(np.float32)
    phi = np.linspace(-np.pi, np.pi, 64).astype(np.float32)
    zeta = np.zeros(64, dtype=np.float32)
    m = N._advected_m(u, phi, zeta, nz, seed=7, t_disk=0.0, omega=0.0, shear_period=0.0)
    expect = np.exp(np.clip(m, -2.5, 2.5)).astype(np.float32)
    got = N.noise_density_mult(u, phi, zeta, nz, seed=7, t_disk=0.0,
                               omega=0.0, shear_period=0.0)
    np.testing.assert_allclose(got, expect, rtol=0, atol=0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_noise.py::test_advected_m_reconstructs_density_mult -v`
Expected: FAIL — `AttributeError: module 'renderer.noise' has no attribute '_advected_m'`.

- [ ] **Step 3: Implement the refactor**

In `src/renderer/noise.py`, add the seed constant next to the others (after `NSEED_L2 = 401`, ~line 667):

```python
NSEED_DUST = 911  # CKS-19: ρ_cold's independent modulator (own decorrelated stack)
```

Replace the body of `noise_density_mult` (the block from `u = _f32(u)` through `return np.exp(m).astype(np.float32)`, ~lines 791–823) by extracting the blend into a new helper placed immediately ABOVE `noise_density_mult`:

```python
def _advected_m(u, phi, zeta, nz, seed: int, t_disk: float = 0.0,
                omega=0.0, shear_period: float = 0.0) -> np.ndarray:
    """Pre-clamp blended log-density modulator m (CKS-12 §2 dual-phase shear
    advection + §4 layer stack), BEFORE the ±m_max clamp and exp. Factored out of
    :func:`noise_density_mult` so CKS-19 can evaluate it twice (hot seed + dust
    seed) for the ρ_cold correlation construction. Returns m; the density
    multiplier is exp(clamp(m, ±m_max))."""
    u = _f32(u)
    phi = _f32(phi)
    zeta = _f32(zeta)
    T = np.float32(shear_period)
    if T <= np.float32(0.0):
        return _noise_m_stack(u, phi, zeta, nz, int(seed), t_disk=t_disk)
    omega = _f32(omega)
    s = np.float32(t_disk) / T
    var_preserve = bool(nz.get("variance_preserve", True))
    g = np.float32(nz.get("dynamism", 1.0))
    m = None
    wsq = np.float32(0.0)
    for k in (0, 1):
        ar = s + np.float32(0.5 * k)
        ck = int(np.floor(ar))
        ak = ar - np.float32(ck)
        wk = np.float32(1.0) - np.abs(np.float32(2.0) * ak - np.float32(1.0))
        seed_k = int(seed) + k * NCYC_PHASE + ck * NCYC_CYCLE
        phi_k = phi - g * omega * (ak * T)
        mk = wk * _noise_m_stack(u, phi_k, zeta, nz, seed_k, t_disk=t_disk)
        m = mk if m is None else m + mk
        wsq = wsq + wk * wk
    if var_preserve and wsq > np.float32(0.0):
        m = m / np.sqrt(wsq)
    return m.astype(np.float32)
```

Then shrink `noise_density_mult`'s body to:

```python
    mmax = np.float32(nz.get("m_max", 2.5))
    m = _advected_m(u, phi, zeta, nz, seed, t_disk=t_disk,
                    omega=omega, shear_period=shear_period)
    m = np.clip(m, -mmax, mmax)
    return np.exp(m).astype(np.float32)
```

(Keep `noise_density_mult`'s signature and docstring; only its body changes.)

- [ ] **Step 4: Run the refactor-parity test + the existing noise suite**

Run: `pytest tests/test_noise.py -v`
Expected: PASS — including the new test and every pre-existing test (the refactor is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add src/renderer/noise.py tests/test_noise.py
git commit -m "refactor(CKS-19): factor _advected_m out of noise_density_mult"
```

---

## Task 2: CPU twin — `dust_density_mult` + correlation parity

**Files:**
- Modify: `src/renderer/noise.py` (add `dust_density_mult` after `noise_density_mult`)
- Test: `tests/test_noise.py`

- [ ] **Step 1: Write the failing tests** (Pearson `r ≈ χ`, variance preservation, and χ=+1 reduces to the hot modulator)

Add to `tests/test_noise.py`:

```python
def _dust_mp(chi):
    return {"enabled": True, "dust_correlation": chi, "dust_amp": 1.0, "dust_sigma_frac": 1.0}

_DUST_NZ = {
    "m_max": 8.0,  # loose clamp so the linear-correlation construction is visible
    "layers": {"base": {"enabled": True, "amp": 1.0, "octaves": 4,
                        "lacunarity": 2, "gain": 0.5, "freq_u": 6.0, "freq_phi": 24}},
}

def _grid():
    uu, pp = np.meshgrid(np.linspace(0.05, 3.0, 96, dtype=np.float32),
                         np.linspace(-np.pi, np.pi, 96, dtype=np.float32), indexing="ij")
    return uu.ravel(), pp.ravel(), np.zeros(uu.size, dtype=np.float32)

@pytest.mark.parametrize("chi", [-1.0, -0.6, 0.0, 0.6, 1.0])
def test_dust_correlation_matches_chi(chi):
    from renderer import noise as N
    u, phi, zeta = _grid()
    mmax = _DUST_NZ["m_max"]
    m_hot = N._advected_m(u, phi, zeta, _DUST_NZ, seed=7)
    rho_cold = N.dust_density_mult(u, phi, zeta, _DUST_NZ, _dust_mp(chi), seed=7)
    m_cold = np.log(rho_cold)            # a_cold=1 ⇒ log ρ_cold == clamp(m_cold)
    r = np.corrcoef(m_hot, m_cold)[0, 1]
    assert abs(r - chi) < 0.05, f"chi={chi}: Pearson r={r}"

def test_dust_variance_is_chi_invariant():
    from renderer import noise as N
    u, phi, zeta = _grid()
    var = [np.var(np.log(N.dust_density_mult(u, phi, zeta, _DUST_NZ, _dust_mp(c), seed=7)))
           for c in (-1.0, -0.6, 0.0, 0.6, 1.0)]
    assert max(var) / min(var) < 1.15, f"variance breathes across chi: {var}"

def test_dust_chi_plus_one_is_hot_modulator():
    from renderer import noise as N
    u, phi, zeta = _grid()
    rho_hot = N.noise_density_mult(u, phi, zeta, _DUST_NZ, seed=7)
    rho_cold = N.dust_density_mult(u, phi, zeta, _DUST_NZ, _dust_mp(1.0), seed=7)
    np.testing.assert_allclose(rho_cold, rho_hot, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_noise.py -k dust -v`
Expected: FAIL — `AttributeError: ... has no attribute 'dust_density_mult'`.

- [ ] **Step 3: Implement `dust_density_mult`**

Add to `src/renderer/noise.py` immediately after `noise_density_mult`:

```python
def dust_density_mult(u, phi, zeta, nz, mp, seed: int = 1234, t_disk: float = 0.0,
                      omega=0.0, shear_period: float = 0.0) -> np.ndarray:
    """CKS-19 cold (dust) density multiplier — **CPU source of truth** (GPU twin
    ``taichi_renderer._disk_density_cks`` index [1]). Returns
    ``exp(clamp(a_cold·m_cold, ±m_max))`` where the cold modulator is the
    variance-preserving Pearson mix of the hot modulator and an independent
    re-seeded copy of the SAME layer stack:

        m_hot  = _advected_m(..., seed)
        m_dust = _advected_m(..., seed + NSEED_DUST)   # equal variance, decorrelated
        m_cold = χ·m_hot + √(1−χ²)·m_dust

    χ = ``mp['dust_correlation'] ∈ [−1,1]``, a_cold = ``mp['dust_amp']``. Because the
    dust stack is the hot stack reseeded, Var(m_dust)=Var(m_hot), so the sampled
    Pearson correlation between m_hot and m_cold equals χ and Var(m_cold) is
    χ-invariant (CKS-19 variance preservation). This is ONLY the modulator; the
    caller multiplies by the cold Gaussian gauss(ζ;σ_cold) and the edge window.
    """
    mmax = np.float32(nz.get("m_max", 2.5))
    chi = np.float32(mp.get("dust_correlation", -0.6))
    a_cold = np.float32(mp.get("dust_amp", 1.0))
    m_hot = _advected_m(u, phi, zeta, nz, int(seed), t_disk=t_disk,
                        omega=omega, shear_period=shear_period)
    m_dust = _advected_m(u, phi, zeta, nz, int(seed) + NSEED_DUST, t_disk=t_disk,
                         omega=omega, shear_period=shear_period)
    s = np.sqrt(np.float32(1.0) - chi * chi)
    m_cold = chi * m_hot + s * m_dust
    m_cold = np.clip(a_cold * m_cold, -mmax, mmax)
    return np.exp(m_cold).astype(np.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_noise.py -k dust -v`
Expected: PASS — all 7 parametrizations (`r ≈ χ` within 0.05, variance ratio < 1.15, χ=+1 reduces to hot).

- [ ] **Step 5: Commit**

```bash
git add src/renderer/noise.py tests/test_noise.py
git commit -m "feat(CKS-19): CPU dust_density_mult — variance-preserving rho_cold modulator"
```

---

## Task 3: GPU param buffer — `_NI_MP_*` indices + buffer fill

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (index block ~line 226–231; buffer-fill ~line 504–591)

- [ ] **Step 1: Write the failing test** (the buffer carries the multiphase params from config)

Add to `tests/test_disk_noise.py`:

```python
def test_multiphase_params_uploaded():
    """disk.multiphase populates the _NI_MP_* slots; absent block ⇒ disabled defaults."""
    cfg_on = {"disk": {"multiphase": {"enabled": True, "dust_correlation": -0.6,
                                      "dust_amp": 1.0, "dust_sigma_frac": 0.8}}}
    tr.setup_disk_noise(cfg_on)
    buf = tr.disk_noise_params.to_numpy()
    assert buf[tr._NI_MP_EN] == 1.0
    assert abs(buf[tr._NI_MP_CHI] - (-0.6)) < 1e-6
    assert abs(buf[tr._NI_MP_SIGFRAC] - 0.8) < 1e-6
    tr.setup_disk_noise({"disk": {}})
    buf0 = tr.disk_noise_params.to_numpy()
    assert buf0[tr._NI_MP_EN] == 0.0
    assert buf0[tr._NI_MP_SIGFRAC] == 1.0  # ratio defaults to 1 (σ_cold=σ_hot)
```

> Replace `setup_disk_noise` with the actual name of the buffer-fill function if it differs (it is the function containing `buf = np.zeros(_NOISE_N, ...)` near line 512).

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_disk_noise.py::test_multiphase_params_uploaded -v`
Expected: FAIL — `AttributeError: module 'renderer.taichi_renderer' has no attribute '_NI_MP_EN'`.

- [ ] **Step 3: Add the indices and buffer fill**

In `src/renderer/taichi_renderer.py`, after `_NI_CURL_FLOWP = 52` and before `_NOISE_N` (~line 230):

```python
# CKS-19 multi-phase media (decoupled ρ_cold absorption). _NI_MP_EN gates it:
# 0 ⇒ ρ_cold ≡ ρ_hot, grey κ ⇒ single-phase march bit-identical (constraint 6).
_NI_MP_EN = 53
_NI_MP_CHI = 54       # χ ∈ [−1,1] dust↔plasma correlation
_NI_MP_AMP = 55       # a_cold — dust log-density gain
_NI_MP_SIGFRAC = 56   # σ_cold / σ_hot — dust slab thickness ratio
_NOISE_N = 57
```

Add `_NSEED_DUST` next to the other GPU seed constants (search for `_NSEED_L1_VORO` and place it adjacent):

```python
_NSEED_DUST = 911  # CKS-19: GPU twin of noise.NSEED_DUST
```

In the buffer-fill function, after the curl block (`buf[_NI_CURL_FLOWP] = ...`, ~line 591) and before the `ti.field(...)` upload:

```python
    # CKS-19 multi-phase media. Absent block ⇒ disabled, σ_cold=σ_hot, grey κ ⇒
    # ρ_cold≡ρ_hot ⇒ single-phase march bit-identical (constraint 6).
    mp = d.get("multiphase", {}) or {}
    buf[_NI_MP_EN] = 1.0 if mp.get("enabled", False) else 0.0
    buf[_NI_MP_CHI] = float(mp.get("dust_correlation", -0.6))
    buf[_NI_MP_AMP] = float(mp.get("dust_amp", 1.0))
    buf[_NI_MP_SIGFRAC] = float(mp.get("dust_sigma_frac", 1.0))
```

(`d` is already `cfg.get("disk", {})` at the top of the function — `multiphase` is a sibling of `noise` under `disk`.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_disk_noise.py::test_multiphase_params_uploaded -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/taichi_renderer.py tests/test_disk_noise.py
git commit -m "feat(CKS-19): GPU multiphase param-buffer slots (_NOISE_N 53->57)"
```

---

## Task 4: GPU twin — `_disk_blended_m` + `_disk_density_cks` returns `vec3`

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (`_disk_noise_density_mult` ~1114–1166; `_disk_density_cks` ~1245–1316)
- Test: `tests/test_disk_noise.py`

- [ ] **Step 1: Write the failing GPU parity test** (`ρ_cold` GPU vs CPU `dust_density_mult` within `_SATOL`)

Add to `tests/test_disk_noise.py` (CUDA-guarded like the others in this file):

```python
def test_rho_cold_gpu_matches_cpu():
    """GPU _disk_density_cks[1] (ρ_cold) == CPU dust_density_mult × gauss, within _SATOL."""
    _ensure_cuda()
    import taichi as ti
    from renderer import noise as N

    nz = {"m_max": 2.5,
          "layers": {"base": {"enabled": True, "amp": 0.6, "octaves": 5,
                              "lacunarity": 2, "gain": 0.5, "freq_u": 6.0, "freq_phi": 24}}}
    mp = {"enabled": True, "dust_correlation": -0.6, "dust_amp": 1.0, "dust_sigma_frac": 0.8}
    cfg = {"disk": {"noise": nz, "multiphase": mp}}
    tr.setup_disk_noise(cfg)

    # Sample grid in (u, φ, ζ); r = r_inner·e^u, dz_ang = ζ·σ0.
    r_inner, sigma0, beta = 4.0, 0.1, 0.0
    us = np.linspace(0.1, 2.0, 16).astype(np.float32)
    phis = np.linspace(-2.5, 2.5, 16).astype(np.float32)
    out = ti.field(ti.f32, shape=(us.size, phis.size, 3))

    @ti.kernel
    def probe(r_inner: ti.f32, sigma0: ti.f32, beta: ti.f32):
        for i, j in ti.ndrange(us.shape[0], phis.shape[0]):
            u = us[i]; phi = phis[j]
            r = r_inner * ti.exp(u)
            dz = 0.0 * sigma0          # ζ=0 midplane
            d = tr._disk_density_cks(ti.cos(phi)*1.0, ti.sin(phi)*1.0, r, dz,
                                     sigma0, beta, r_inner, 1e9, r_inner,
                                     1, 7, 0.0, 0.999)
            out[i, j, 0] = d[0]; out[i, j, 1] = d[1]; out[i, j, 2] = d[2]

    # (host arrays must be Taichi fields to index in-kernel — upload us/phis first)
    ...
```

> Implementation note for the worker: mirror the existing `test_noise_gpu.py` probe pattern (upload `us`/`phis` into `ti.field`s, run, compare). The CPU reference is `gauss(0;σ0)=1 × dust_density_mult(u,φ,0, nz, mp, seed=7)` for index `[1]` and `× noise_density_mult(...)` for index `[0]`. Tolerance: `_SATOL` (the same constant the other GPU twins use; import it from `taichi_renderer`).

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_disk_noise.py::test_rho_cold_gpu_matches_cpu -v`
Expected: FAIL — `_disk_density_cks` still returns a `vec2` (indexing `d[2]` errors, or `d[1]` equals `d[0]`).

- [ ] **Step 3: Factor `_disk_blended_m` and widen `_disk_density_cks`**

In `_disk_noise_density_mult` (~1114), extract the blend (lines 1136–1162) into a helper placed directly above it:

```python
@ti.func
def _disk_blended_m(u, phi, zeta, t_disk, omega, seed):
    """Pre-clamp blended modulator m (CKS-12 §2 dual-phase shear + §4 stack),
    BEFORE the ±m_max clamp and exp. GPU twin of noise._advected_m. CKS-19 calls
    it twice (hot seed, dust seed) for the ρ_cold correlation mix."""
    T = disk_noise_params[_NI_SHEAR_T]
    m = 0.0
    if T <= 0.0:
        m = _disk_noise_m(u, phi, zeta, seed, t_disk)
    else:
        s = t_disk / T
        g = disk_noise_params[_NI_DYNAMISM]
        wsq = 0.0
        c0 = ti.floor(s)
        a0 = s - c0
        w0 = 1.0 - ti.abs(2.0 * a0 - 1.0)
        seed0 = seed + ti.cast(c0, ti.i32) * _NCYC_CYCLE
        m += w0 * _disk_noise_m(u, phi - g * omega * (a0 * T), zeta, seed0, t_disk)
        wsq += w0 * w0
        ar1 = s + 0.5
        c1 = ti.floor(ar1)
        a1 = ar1 - c1
        w1 = 1.0 - ti.abs(2.0 * a1 - 1.0)
        seed1 = seed + _NCYC_PHASE + ti.cast(c1, ti.i32) * _NCYC_CYCLE
        m += w1 * _disk_noise_m(u, phi - g * omega * (a1 * T), zeta, seed1, t_disk)
        wsq += w1 * w1
        if disk_noise_params[_NI_VAR_PRESERVE] > 0.5 and wsq > 0.0:
            m = m / ti.sqrt(wsq)
    return m
```

Shrink `_disk_noise_density_mult`'s body to:

```python
    m = _disk_blended_m(u, phi, zeta, t_disk, omega, seed)
    mmax = disk_noise_params[_NI_M_MAX]
    m = ti.min(ti.max(m, -mmax), mmax)
    return ti.exp(m)
```

Add a cold-multiplier `@ti.func` next to it:

```python
@ti.func
def _disk_dust_density_mult(u, phi, zeta, t_disk, omega, seed):
    """CKS-19 cold modulator exp(clamp(a_cold·m_cold)) — GPU twin of
    noise.dust_density_mult. m_cold = χ·m_hot + √(1−χ²)·m_dust, m_dust the same
    stack reseeded by _NSEED_DUST (equal variance ⇒ Pearson r = χ)."""
    chi = disk_noise_params[_NI_MP_CHI]
    a_cold = disk_noise_params[_NI_MP_AMP]
    m_hot = _disk_blended_m(u, phi, zeta, t_disk, omega, seed)
    m_dust = _disk_blended_m(u, phi, zeta, t_disk, omega, seed + _NSEED_DUST)
    s = ti.sqrt(1.0 - chi * chi)
    m_cold = a_cold * (chi * m_hot + s * m_dust)
    mmax = disk_noise_params[_NI_M_MAX]
    m_cold = ti.min(ti.max(m_cold, -mmax), mmax)
    return ti.exp(m_cold)
```

In `_disk_density_cks` (~1245), change the return type to `vec3` and compute `ρ_cold`. Replace the final `density = gauss * dmult * win` / `return vec2(density, temp_factor)` (lines 1315–1316) with:

```python
        density = gauss * dmult * win
        # CKS-19: cold absorbing phase. OFF ⇒ ρ_cold ≡ ρ_hot (bit-identical).
        density_cold = density
        if disk_noise_params[_NI_MP_EN] > 0.5:
            dmult_cold = _disk_dust_density_mult(u_n, phi_n, zeta_n, t_disk, omega, noise_seed)
            sigma_cold = sigma_eff * disk_noise_params[_NI_MP_SIGFRAC]
            gauss_cold = ti.exp(-0.5 * (dz_ang / sigma_cold) ** 2)
            density_cold = gauss_cold * dmult_cold * win
        return vec3(density, density_cold, temp_factor)
    return vec3(density, density, temp_factor)
```

> Note: the early-return path (the `noise_enabled == 0` bare-Gaussian case, line 1275–1276 region) must also return `vec3(density, density, temp_factor)` — when noise is off there is no cold modulation, so `ρ_cold = ρ_hot` (the bare Gaussian). Update that `return vec2(...)` too.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_disk_noise.py::test_rho_cold_gpu_matches_cpu -v`
Expected: PASS — GPU `ρ_cold` matches CPU `dust_density_mult × gauss` within `_SATOL`.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/taichi_renderer.py tests/test_disk_noise.py
git commit -m "feat(CKS-19): GPU _disk_density_cks -> vec3(rho_hot, rho_cold, temp)"
```

---

## Task 5: Wire the march — emission←ρ_hot, absorption←ρ_cold, shadow & step-cap

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (`_disk_emit_cks` 1526–1589; `bake_disk_shadow` density read; step cap 1844–1849)

All three consumers of the density currently read index `[0]`. Emission keeps `[0]` (`ρ_hot`); absorption (`dtau`) and the shadow bake move to `[1]` (`ρ_cold`); the step cap resolves the thinner slab. OFF ⇒ `[1]==[0]` and `sigfrac=1` ⇒ unchanged.

- [ ] **Step 1: Write the failing parity test** (flag-off bit-identity of the full beauty frame)

Add to `tests/test_disk_noise.py`:

```python
def test_multiphase_off_bit_identical():
    """multiphase.enabled:false ⇒ beauty frame byte-identical to no multiphase block."""
    _ensure_cuda()
    base = _load_cfg()                       # the file's existing small-frame cfg helper
    cfg_a = copy.deepcopy(base)              # no multiphase block
    cfg_b = copy.deepcopy(base)
    cfg_b["disk"]["multiphase"] = {"enabled": False, "dust_correlation": -0.6,
                                   "dust_amp": 1.0, "dust_sigma_frac": 0.5}
    img_a = _render_small(cfg_a)
    img_b = _render_small(cfg_b)
    np.testing.assert_array_equal(img_a, img_b)
```

> Use the file's existing render helper (the one `test_disk_noise.py` already calls to produce a frame); if none is exposed, mirror `test_gpu_regression.py`'s frame call. `enabled:false` with a non-default `dust_sigma_frac` is the strict guard — it proves the disabled branch never reads the cold params.

- [ ] **Step 2: Run to verify it currently passes** (emission/absorption not yet rewired)

Run: `pytest tests/test_disk_noise.py::test_multiphase_off_bit_identical -v`
Expected: PASS already (nothing reads `[1]` yet) — this is the **guard we must keep green** through Step 3. (If you prefer a red-first step, temporarily set `enabled:true` in `cfg_b` and assert inequality after Step 3 instead.)

- [ ] **Step 3: Rewire the three consumers**

**(a) `_disk_emit_cks` absorption** — at the `dens_tf = _disk_density_cks(...)` call (~1526) the result is now `vec3`. Keep `density = dens_tf[0]` for emission; add `density_cold = dens_tf[1]`; and change the absorption term in the final `vec4(...)` (line 1588) from `absb_c * density * ds` to:

```python
                        absb_c * density_cold * ds,   # CKS-19: κ·ρ_cold (grey κ)
```

(Emission lines 1566 / 1570 keep `density` = `ρ_hot`. `temp_factor = dens_tf[2]` — note the index shifts from `[1]` to `[2]` because the middle slot is now `ρ_cold`.)

**(b) `bake_disk_shadow`** — the shadow bake reconstructs density via `_disk_density_cks` (per its docstring, ~1397). Wherever it reads the returned density for the τ accumulation, read index `[1]` (`ρ_cold`) instead of `[0]`:

```python
        rho = _disk_density_cks(ti.cos(phi), ti.sin(phi), r_j, dz_j, sigma_theta0,
                                flare_beta, r_inner, r_outer, r_isco,
                                noise_enabled, noise_seed, t_disk, a)[1]   # CKS-19: τ ≡ ∫κ·ρ_cold
```

> Find the existing `_disk_density_cks(...)[0]` (or `[0]` extraction) inside `bake_disk_shadow` and change the `[0]` to `[1]`. OFF ⇒ `[1]==[0]` ⇒ the deep-shadow-map is unchanged.

**(c) Vertical step cap** (~1844) — resolve the thinner of `σ_hot` and `σ_cold = σ_hot·sigfrac` when multiphase is on (CKS-19 constraint 3). After the existing `sigma_z = r * sigma_theta0` and the `_NI_MOD_EN` height-amp narrowing (line 1849), add:

```python
                    if disk_noise_params[_NI_MP_EN] > 0.5:
                        sf = disk_noise_params[_NI_MP_SIGFRAC]
                        if sf < 1.0:                      # cold slab thinner ⇒ tighter cap
                            sigma_z = sigma_z * sf
```

- [ ] **Step 4: Run the parity + a multiphase-on smoke**

Run: `pytest tests/test_disk_noise.py::test_multiphase_off_bit_identical -v`
Expected: PASS (off path still bit-identical).
Run: `pytest tests/test_disk_noise.py -k "multiphase or rho_cold" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/taichi_renderer.py tests/test_disk_noise.py
git commit -m "feat(CKS-19): march wires emission<-rho_hot, absorption+shadow<-rho_cold"
```

---

## Task 6: Config, acceptance test, regression, docs-sync

**Files:**
- Modify: `configs/render.yaml` (`disk:` block)
- Create: `tests/test_disk_multiphase.py`
- Modify: `skills/kerr-physics/SKILL.md` (CKS-19 status DESIGN→ACTIVE); PROJECT.md §6/§7 if present

- [ ] **Step 1: Write the acceptance test** (the B.1 silhouette criterion)

Create `tests/test_disk_multiphase.py`:

```python
"""CKS-19 acceptance: a cold-only slab carves a darker-than-background silhouette.

A region with ρ_cold>0 and ρ_hot≈0 placed in front of bright emission must read
DARKER than the bare background behind it — a true silhouette, not merely dimmer
emission. CUDA-mandatory (backend LOCKED to ti.cuda); skips cleanly without it.
"""
# pyright: reportInvalidTypeForm=false
import copy
import numpy as np
import pytest
from renderer import taichi_renderer as tr

pytestmark = pytest.mark.gpu

def test_dust_carves_silhouette():
    # Construct a config whose dust is anti-correlated (χ=-1) and strong, with a
    # bright emitting plasma behind. Compare the pixel luminance through the dust
    # lane against the same pixel with multiphase OFF (pure emission, no dust).
    base = _multiphase_scene()              # helper below: bright disk + bg
    cfg_off = copy.deepcopy(base); cfg_off["disk"]["multiphase"]["enabled"] = False
    cfg_on = copy.deepcopy(base);  cfg_on["disk"]["multiphase"]["enabled"] = True
    img_off = _render(cfg_off)
    img_on = _render(cfg_on)
    # In the densest dust column, ON must be strictly darker than OFF AND darker
    # than the background-only luminance there.
    lum_off = img_off.sum(axis=2)
    lum_on = img_on.sum(axis=2)
    darkened = (lum_on < lum_off - 1e-4)
    assert darkened.mean() > 0.02, "dust did not darken any appreciable region"
    assert lum_on.min() < lum_off.min(), "dust did not create a new darkest pixel"
```

> Implement `_multiphase_scene()` and `_render()` mirroring `test_disk_noise.py` (load the canonical camera, a small res, the production beauty path). The scene must enable `disk.noise` (so the dust modulator has structure) and set `multiphase: {enabled, dust_correlation: -1.0, dust_amp: 2.0, dust_sigma_frac: 1.0}` plus `absorption_coeff` high enough (e.g. 2.0) for visible obscuration.

- [ ] **Step 2: Run to verify it fails without config**

Run: `pytest tests/test_disk_multiphase.py -v`
Expected: FAIL (KeyError / no multiphase block) until the config exists.

- [ ] **Step 3: Add the config block**

In `configs/render.yaml` under `disk:` (sibling of `noise:`):

```yaml
  multiphase:                  # CKS-19 — decoupled hot/cold media. Default OFF ⇒ bit-identical.
    enabled: false             # false ⇒ ρ_cold≡ρ_hot, grey κ ⇒ legacy scalar march exactly.
    dust_correlation: -0.6     # χ ∈ [-1,1]: -1 anti-correlated lanes, 0 independent, +1 tracks plasma.
    dust_amp: 1.0              # a_cold — dust log-density gain (clamped by noise.m_max).
    dust_sigma_frac: 1.0       # σ_cold / σ_hot — dust slab thickness vs the emitting slab.
```

- [ ] **Step 4: Run acceptance + the full regression guard**

Run: `pytest tests/test_disk_multiphase.py -v`
Expected: PASS — dust darkens a region and creates a new darkest pixel.
Run: `pytest tests/test_gpu_regression.py -v`
Expected: PASS — every existing golden bit-identical (`multiphase` defaults OFF).

- [ ] **Step 5: Docs-sync + commit**

In `skills/kerr-physics/SKILL.md`, flip CKS-19's status line from `DESIGN ... NOT yet wired` to `ACTIVE (wired 2026-06-16)` and add a revision-history entry (v1.30: CKS-19 wired, grey scalar absorption; vec3 chromatic deferred). Update PROJECT.md §6/§7 if that file exists (per the docs-sync policy).

```bash
git add configs/render.yaml tests/test_disk_multiphase.py skills/kerr-physics/SKILL.md
git commit -m "feat(CKS-19): wire multiphase config + dust-silhouette acceptance; SKILL v1.30"
```

---

## Task 7 (OPTIONAL / DEFERRED): vec3 transmittance — chromatic-ready extinction

> This is the second half of the "grey now, **chromatic-ready**" owner decision. Tasks 1–6 deliver grey scalar absorption (κ_R=κ_G=κ_B), which fully satisfies the dust-lane acceptance test. This task makes the *march* carry per-channel transmittance so a future `extinction_rgb` (warm/brown dust) is a data-only change. It is isolated here because it widens `disk_buf` and touches the compositor — its own golden-risk surface. Implement only when chromatic dust is wanted.

**Files:**
- Modify: `src/renderer/taichi_renderer.py` (`disk_buf` alloc line 652; `_disk_emit_cks` dtau→vec3; march `transm`→vec3 ~1784–1920; composite ~2228–2233)

- [ ] **Step 1** — widen `disk_buf` shape `(height, width, 4)` → `(height, width, 6)` (line 652); channels 3,4,5 store `transm_rgb`.
- [ ] **Step 2** — `_disk_emit_cks` returns the absorption as a vec3 `dτ⃗ = κ⃗·ρ_cold·ds` (κ⃗ broadcast from `absb_c`, or a future `extinction_rgb`). Change the return from `vec4` to a struct/`vec3 emission + vec3 dtau` (or return `ρ_cold` and let the march form `dτ⃗`).
- [ ] **Step 3** — in `render_beauty_physics`, make `transm` a `vec3`; `w = 1 − exp(−dτ⃗)`, `disk_col += transm ⊙ (w·S)`, `transm *= exp(−dτ⃗)`, all componentwise. The depth/`total_emission` key uses `transm` luminance. Write `disk_buf[...,3:6] = transm`.
- [ ] **Step 4** — composite (line 2228–2230): `transm = vec3(disk_buf[...,3], disk_buf[...,4], disk_buf[...,5])`; `col = disk_col + transm ⊙ bg`.
- [ ] **Step 5** — regression: with grey κ (R=G=B) the result is bit-identical to Task 6; `test_gpu_regression.py` must stay green. Then add `extinction_rgb: [1.0, 0.7, 0.45]` to the config to demo warm dust. Commit.

---

## Self-review notes

- **Spec coverage:** B.1 acceptance → Task 6; B.2 physics (ρ_hot/ρ_cold, χ-mix, variance preservation) → Tasks 1–2 (CPU), 4 (GPU); B.3 architecture touch-list items 1–6 → Tasks 4 (density vec3), 5a (emit), 5b (shadow), 5c (step cap), 1–2 (CPU twin); B.4 config → Task 6; B.5 tests → Tasks 2/4/5/6; "grey now, chromatic-ready" → grey in Tasks 1–6, chromatic in Task 7.
- **Type consistency:** `_disk_density_cks` returns `vec3` from Task 4 onward; every consumer (emit, bake, the GPU probe test) reads `[0]=ρ_hot, [1]=ρ_cold, [2]=temp_factor` — the `temp_factor` index shifts `[1]→[2]` (called out in Task 5a). `_advected_m`/`_disk_blended_m` are the paired CPU/GPU helper names; `dust_density_mult`/`_disk_dust_density_mult` the paired cold modulators; `NSEED_DUST`/`_NSEED_DUST = 911` the paired seed offsets.
- **Bit-identity:** every task keeps `enabled:false` bit-identical (`ρ_cold≡ρ_hot`, `sigfrac=1`, grey κ); the dedicated guard is Task 5 Step 2 + Task 6 Step 4.
- **Open item the worker must resolve from the codebase, not invent:** the exact name of the buffer-fill function (Task 3) and the exact density-read line inside `bake_disk_shadow` (Task 5b) — both are pinpointed by the search hints given; do not guess signatures.
