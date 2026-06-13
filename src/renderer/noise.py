"""Procedural-noise primitive library — CPU (NumPy) source of truth (D2.1).

Disk turbulence (``docs/specs/2026-06-13-disk-noise-turbulence.md`` §3, SKILL.md
Formula **CKS-12**) is built from layered procedural noise. These primitives are
**texturing functions, not physics** — they live here, not in SKILL.md — but the
project's CPU-source-of-truth / Taichi-twin discipline still applies: every
function below has an ``@ti.func`` twin (same file, ``_ti`` suffix) that
``tests/test_noise_gpu.py`` holds to this module to ~1e-6 on a sample grid.

Design constraints baked in here so the GPU twins can match and the renderer
integration stays correct (CKS-12 §1–3, spec §3):

* **Deterministic integer hashing only** — a PCG-style ``u32`` hash, never
  ``ti.random`` (CKS-12 constraint 7). Same inputs + seed ⇒ identical output on
  CPU and GPU.
* **Exact φ-periodicity** — the φ axis is the *second* lattice coordinate
  (``y``). Its lattice index is wrapped to an integer ``period`` with the
  positive-modulo :func:`_wrap`, so the field is bit-identical at ``y`` and
  ``y + period`` — no seam at φ = 0 (CKS-12 constraint 5). The caller scales φ so
  that φ ∈ [0, 2π) maps to ``y ∈ [0, freq_phi)`` and passes ``period = freq_phi``
  (an integer); ``lacunarity`` must be integral for the fBm octaves to preserve
  this.
* **No transcendentals in the lattice path** — gradients use Ken Perlin's
  branch-selected integer gradient set (*Improving Noise*, SIGGRAPH 2002), so the
  only float ops are fade/lerp/dot and CPU↔GPU divergence stays sub-ULP.
* **float32 everywhere** — this reference is computed in float32 (not float64) so
  it is the literal target the f32 GPU kernel is held to.

Coordinates are **lattice-space** (already scaled by per-axis frequency); the
renderer-facing ``(u = ln r/r_inner, φ, ζ)`` → lattice mapping and the CKS-12
shear advection are applied by the caller in D2.2+, not here.

Primitive sources: Perlin, *Improving Noise* (2002); Worley, *A cellular texture
basis function* (1996); Musgrave, *Texturing & Modeling* (ridged construction).
See spec §3 and §10 for provenance.
"""
from __future__ import annotations

import numpy as np
import taichi as ti

# --------------------------------------------------------------------------- #
# Integer-hash mixing constants (all < 2^32). Knuth/Wang-style odd multipliers
# used to fold lattice coordinates + seed into one word before the PCG finalizer.
# --------------------------------------------------------------------------- #
_C_SEED = np.uint32(2654435769)  # 0x9E3779B9
_C_X = np.uint32(2246822507)  # 0x85EBCA6B
_C_Y = np.uint32(3266489917)  # 0xC2B2AE35
_C_Z = np.uint32(668265263)  # 0x27D4EB2F
_C_JY = np.uint32(0x9E3779B1)  # Worley jitter decorrelator (y vs x)
_C_JZ = np.uint32(0x85EBCA77)  # Worley jitter decorrelator (z)

_U32 = np.float64(2.0**32)  # PCG word → [0, 1) divisor


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


# --------------------------------------------------------------------------- #
# PCG hash — u32 -> u32 (the only entropy source; CKS-12 constraint 7)
# --------------------------------------------------------------------------- #
def pcg_hash(v) -> np.ndarray:
    """PCG-style integer hash ``u32 -> u32`` (Jarzynski & Olano, 2020).

    Vectorized; accepts any integer array, returns ``uint32``. The ``@ti.func``
    twin :func:`pcg_hash_ti` performs the identical ``u32`` arithmetic, so the
    integer path is bit-identical CPU↔GPU.
    """
    state = np.asarray(v, dtype=np.uint32) * np.uint32(747796405) + np.uint32(2891336453)
    word = ((state >> ((state >> np.uint32(28)) + np.uint32(4))) ^ state) * np.uint32(277803737)
    return (word >> np.uint32(22)) ^ word


def _u01(h: np.ndarray) -> np.ndarray:
    """Hash word -> float32 in [0, 1).

    Computed in **float32** (not float64) so the pure-f32 GPU twin rounds
    identically — the u32→f32 cast and the divide match bit-for-bit IEEE
    round-to-nearest. The lost low hash bits only set sub-texel jitter.
    """
    return (h.astype(np.float32) * np.float32(1.0 / _U32)).astype(np.float32)


def _hash2(ix, iy, seed) -> np.ndarray:
    h = (
        np.asarray(seed, np.uint32) * _C_SEED
        + np.asarray(ix, np.uint32) * _C_X
        + np.asarray(iy, np.uint32) * _C_Y
    )
    return pcg_hash(h)


def _hash3(ix, iy, iz, seed) -> np.ndarray:
    h = (
        np.asarray(seed, np.uint32) * _C_SEED
        + np.asarray(ix, np.uint32) * _C_X
        + np.asarray(iy, np.uint32) * _C_Y
        + np.asarray(iz, np.uint32) * _C_Z
    )
    return pcg_hash(h)


def _wrap(i: np.ndarray, period: int) -> np.ndarray:
    """Positive modulo ``((i % p) + p) % p`` (mirrors the GPU twin).

    Python/NumPy ``%`` is already floored, but Taichi integer ``%`` is truncated
    (C semantics), so the twin needs the double-mod; we use the same form here so
    the two cannot drift on negative indices.
    """
    return ((i % period) + period) % period


# --------------------------------------------------------------------------- #
# Gradients (Perlin 2002, branch-selected integer set) + quintic fade
# --------------------------------------------------------------------------- #
def _fade(t: np.ndarray) -> np.ndarray:
    """Quintic fade ``6t⁵ − 15t⁴ + 10t³`` (Perlin 2002)."""
    return t * t * t * (t * (t * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))


def _grad3(h: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Perlin ``grad(hash, x, y, z)`` — dot with one of 12 edge-midpoint gradients."""
    hh = h & np.uint32(15)
    u = np.where(hh < 8, x, y)
    v = np.where(hh < 4, y, np.where((hh == 12) | (hh == 14), x, z))
    gu = np.where((hh & np.uint32(1)) == 0, u, -u)
    gv = np.where((hh & np.uint32(2)) == 0, v, -v)
    return (gu + gv).astype(np.float32)


def _lerp(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _to01(raw: np.ndarray) -> np.ndarray:
    """Map raw Perlin (~[-1, 1]) to [0, 1] with a safety clamp."""
    return np.clip(np.float32(0.5) * (raw + np.float32(1.0)), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# §3.1 Lattice gradient noise, periodic in the φ (second) axis
# --------------------------------------------------------------------------- #
def gnoise2(x, y, period: int, seed: int = 0) -> np.ndarray:
    """2D gradient noise in ``[0, 1]``; exactly periodic in ``y`` with ``period``.

    ``x`` (log-radial ``u``) is non-periodic; ``y`` (azimuth φ) wraps. Built as a
    z = 0 slice of the 3D Perlin gradient so amplitude matches :func:`gnoise3`.
    """
    x = _f32(x)
    y = _f32(y)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    xf = (x - xi.astype(np.float32)).astype(np.float32)
    yf = (y - yi.astype(np.float32)).astype(np.float32)
    u = _fade(xf)
    v = _fade(yf)
    z0 = np.float32(0.0)

    acc = []
    for ix in (0, 1):
        col = []
        for iy in (0, 1):
            h = _hash3(xi + ix, _wrap(yi + iy, period), 0, seed)
            col.append(_grad3(h, xf - ix, yf - iy, z0))
        acc.append(_lerp(v, col[0], col[1]))
    raw = _lerp(u, acc[0], acc[1])
    return _to01(raw)


def gnoise3(x, y, z, period: int, seed: int = 0) -> np.ndarray:
    """3D gradient noise in ``[0, 1]``; exactly periodic in ``y`` with ``period``."""
    x = _f32(x)
    y = _f32(y)
    z = _f32(z)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    zi = np.floor(z).astype(np.int32)
    xf = (x - xi.astype(np.float32)).astype(np.float32)
    yf = (y - yi.astype(np.float32)).astype(np.float32)
    zf = (z - zi.astype(np.float32)).astype(np.float32)
    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    planes = []
    for ix in (0, 1):
        rows = []
        for iy in (0, 1):
            cells = []
            for iz in (0, 1):
                h = _hash3(xi + ix, _wrap(yi + iy, period), zi + iz, seed)
                cells.append(_grad3(h, xf - ix, yf - iy, zf - iz))
            rows.append(_lerp(w, cells[0], cells[1]))
        planes.append(_lerp(v, rows[0], rows[1]))
    raw = _lerp(u, planes[0], planes[1])
    return _to01(raw)


# --------------------------------------------------------------------------- #
# §3.2–3.4 Octave stacks: fBm, billow/turbulence, ridged multifractal
# --------------------------------------------------------------------------- #
def _octaves(base, x, y, z, period, octaves, lacunarity, gain, seed):
    """Yield ``(value, weight)`` per octave for the fBm-family loops.

    ``base`` is a callable ``(x*f, y*f[, z*f], period*f, seed+o) -> [0,1]``.
    ``lacunarity`` is cast to ``int`` so ``period*f`` stays integral every octave
    (the φ-periodicity guarantee). ``z`` is ``None`` for the 2D primitives.
    """
    lac = int(lacunarity)
    freq = 1
    per = int(period)
    amp = 1.0
    for o in range(int(octaves)):
        if z is None:
            n = base(x * freq, y * freq, per, seed + o)
        else:
            n = base(x * freq, y * freq, z * freq, per, seed + o)
        yield n, np.float32(amp)
        amp *= gain
        freq *= lac
        per *= lac


def _fbm(base, x, y, z, period, octaves, lacunarity, gain, seed, transform):
    total = np.float32(0.0)
    norm = np.float32(0.0)
    for n, amp in _octaves(base, x, y, z, period, octaves, lacunarity, gain, seed):
        total = total + transform(n) * amp
        norm = norm + amp
    return (total / norm).astype(np.float32)


def fbm2(x, y, period, octaves=4, lacunarity=2, gain=0.5, seed=0) -> np.ndarray:
    """fBm of :func:`gnoise2` — normalized to ``[0, 1]`` (spec §3.2)."""
    return _fbm(gnoise2, _f32(x), _f32(y), None, period, octaves, lacunarity, gain, seed,
                lambda n: n)


def fbm3(x, y, z, period, octaves=4, lacunarity=2, gain=0.5, seed=0) -> np.ndarray:
    return _fbm(gnoise3, _f32(x), _f32(y), _f32(z), period, octaves, lacunarity, gain, seed,
                lambda n: n)


def _billow(n: np.ndarray) -> np.ndarray:
    """Perlin turbulence octave value ``|2n − 1|`` (cusped, cloud-like; spec §3.3)."""
    return np.abs(np.float32(2.0) * n - np.float32(1.0)).astype(np.float32)


def billow2(x, y, period, octaves=4, lacunarity=2, gain=0.5, seed=0) -> np.ndarray:
    return _fbm(gnoise2, _f32(x), _f32(y), None, period, octaves, lacunarity, gain, seed,
                _billow)


def billow3(x, y, z, period, octaves=4, lacunarity=2, gain=0.5, seed=0) -> np.ndarray:
    return _fbm(gnoise3, _f32(x), _f32(y), _f32(z), period, octaves, lacunarity, gain, seed,
                _billow)


def _ridged(base, x, y, z, period, octaves, lacunarity, gain, offset, feedback, seed):
    """Musgrave-style ridged multifractal (spec §3.4).

    ``r_o = (offset − |2n − 1|)²``; spectral-weight feedback ``w_o = clamp(r_{o−1}
    · feedback, 0, 1)`` with ``w_0 = 1``; normalized by ``Σ gain^o`` and clamped
    to ``[0, 1]`` (exact for the default ``offset = 1``).
    """
    total = np.float32(0.0)
    norm = np.float32(0.0)
    prev = None
    for n, amp in _octaves(base, x, y, z, period, octaves, lacunarity, gain, seed):
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


def ridged2(x, y, period, octaves=3, lacunarity=2, gain=0.5, offset=1.0, feedback=2.0,
            seed=0) -> np.ndarray:
    return _ridged(gnoise2, _f32(x), _f32(y), None, period, octaves, lacunarity, gain,
                   offset, feedback, seed)


def ridged3(x, y, z, period, octaves=3, lacunarity=2, gain=0.5, offset=1.0, feedback=2.0,
            seed=0) -> np.ndarray:
    return _ridged(gnoise3, _f32(x), _f32(y), _f32(z), period, octaves, lacunarity, gain,
                   offset, feedback, seed)


# --------------------------------------------------------------------------- #
# §3.5 Worley / Voronoi cellular (jittered grid, F1 & F2), periodic in φ
# --------------------------------------------------------------------------- #
def worley2(x, y, period: int, seed: int = 0):
    """Jittered-grid Worley distances ``(F1, F2)``; periodic in ``y`` (spec §3.5).

    9-cell (3×3) search. Each cell's feature point is its corner plus a hashed
    ``[0, 1)²`` jitter; the jitter is keyed on the **wrapped** cell index so the
    field is exactly periodic in φ.
    """
    x = _f32(x)
    y = _f32(y)
    cx = np.floor(x).astype(np.int32)
    cy = np.floor(y).astype(np.int32)
    fx = (x - cx.astype(np.float32)).astype(np.float32)
    fy = (y - cy.astype(np.float32)).astype(np.float32)

    f1 = np.full(x.shape, np.float32(1e9), dtype=np.float32)
    f2 = np.full(x.shape, np.float32(1e9), dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            h = _hash2(cx + dx, _wrap(cy + dy, period), seed)
            jx = _u01(pcg_hash(h))
            jy = _u01(pcg_hash(h ^ _C_JY))
            ox = (np.float32(dx) + jx) - fx
            oy = (np.float32(dy) + jy) - fy
            d2 = (ox * ox + oy * oy).astype(np.float32)
            f2 = np.minimum(f2, np.maximum(f1, d2))
            f1 = np.minimum(f1, d2)
    return np.sqrt(f1).astype(np.float32), np.sqrt(f2).astype(np.float32)


def worley3(x, y, z, period: int, seed: int = 0):
    """27-cell (3×3×3) Worley distances ``(F1, F2)``; periodic in ``y``."""
    x = _f32(x)
    y = _f32(y)
    z = _f32(z)
    cx = np.floor(x).astype(np.int32)
    cy = np.floor(y).astype(np.int32)
    cz = np.floor(z).astype(np.int32)
    fx = (x - cx.astype(np.float32)).astype(np.float32)
    fy = (y - cy.astype(np.float32)).astype(np.float32)
    fz = (z - cz.astype(np.float32)).astype(np.float32)

    f1 = np.full(x.shape, np.float32(1e9), dtype=np.float32)
    f2 = np.full(x.shape, np.float32(1e9), dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                h = _hash3(cx + dx, _wrap(cy + dy, period), cz + dz, seed)
                jx = _u01(pcg_hash(h))
                jy = _u01(pcg_hash(h ^ _C_JY))
                jz = _u01(pcg_hash(h ^ _C_JZ))
                ox = (np.float32(dx) + jx) - fx
                oy = (np.float32(dy) + jy) - fy
                oz = (np.float32(dz) + jz) - fz
                d2 = (ox * ox + oy * oy + oz * oz).astype(np.float32)
                f2 = np.minimum(f2, np.maximum(f1, d2))
                f1 = np.minimum(f1, d2)
    return np.sqrt(f1).astype(np.float32), np.sqrt(f2).astype(np.float32)


def voronoi_billow2(x, y, period, k=4.0, seed=0) -> np.ndarray:
    """``exp(−k·F1)`` — bright clump cores (the "alligator" stand-in; spec §3.5)."""
    f1, _ = worley2(x, y, period, seed)
    return np.exp(-np.float32(k) * f1).astype(np.float32)


def voronoi_billow3(x, y, z, period, k=4.0, seed=0) -> np.ndarray:
    f1, _ = worley3(x, y, z, period, seed)
    return np.exp(-np.float32(k) * f1).astype(np.float32)


def cell_wall2(x, y, period, seed=0) -> np.ndarray:
    """``F2 − F1`` — membrane / tearing variant (spec §3.5)."""
    f1, f2 = worley2(x, y, period, seed)
    return (f2 - f1).astype(np.float32)


def cell_wall3(x, y, z, period, seed=0) -> np.ndarray:
    f1, f2 = worley3(x, y, z, period, seed)
    return (f2 - f1).astype(np.float32)


# --------------------------------------------------------------------------- #
# §4 layer stack — combined density multiplier (CPU source of truth).
#
# Constants shared with the GPU twin
# (``taichi_renderer._disk_noise_density_mult``): keep the two in lockstep — the
# GPU side imports these so a change here moves both.
# --------------------------------------------------------------------------- #
NSEED_L0 = 0  # per-layer/sublayer seed offsets from disk.noise.seed (decorrelate)
NSEED_L1_RIDGE = 101
NSEED_L1_VORO = 211
NSEED_L1_MASK = 307
NSEED_L2 = 401
RIDGE_FEEDBACK = 2.0  # ridged-MF spectral feedback (spec §3.4; not a §5 look dial)
MASK_SOFT = 0.15  # coverage-mask smoothstep half-width around the 1−coverage threshold
_INV_TWO_PI = np.float32(1.0 / (2.0 * np.pi))

# Dual-phase shear-advection reseed offsets (CKS-12 §2, D2.3). The base
# ``disk.noise.seed`` is offset per advection phase (k∈{0,1}) and per reset cycle
# (c_k = floor(s + k/2)) so each crossfade cycle draws a decorrelated pattern —
# without this the whole animation repeats with period T (CKS-12 §2 "per-cycle
# reseed is mandatory"). Plain integer strides (not a hash): the PCG lattice hash
# avalanches any seed delta, so distinct (k, c_k) ⇒ fully independent fields. Kept
# small so ``seed + k·PHASE + c_k·CYCLE`` stays a valid u32 for any realistic
# animation length (c_k = footage_seconds·time_scale / shear_period_M ≪ 10⁴).
NCYC_PHASE = 50021   # k-phase offset
NCYC_CYCLE = 100003  # per-reset-cycle stride


def _noise_m_stack(u, phi, zeta, nz, seed: int) -> np.ndarray:
    """Unclamped log-density sum ``m = Σ amp·(layer − bias)`` at the (already-f32)
    disk-natural coords, BEFORE the ``m_max`` clamp and ``exp`` (CKS-12 §3, spec §4).

    Factored out of :func:`noise_density_mult` so the D2.3 shear-advection blend can
    evaluate the whole layer stack at each of its two reset phases (a different
    ``phi`` and ``seed`` per phase) and crossfade the two ``m`` values before the
    single shared clamp+exp. φ enters each lattice as ``y = phi/(2π)·freq_phi`` with
    integer period ``freq_phi`` (exact 2π-periodicity, constraint 5). Disabled layers
    contribute nothing (``m`` stays 0 ⇒ multiplier 1.0, the kernel's skipped branch).
    """
    phi01 = phi * _INV_TWO_PI  # φ/(2π); ×freq_phi → lattice y
    m = np.zeros(np.broadcast(u, phi, zeta).shape, dtype=np.float32)

    layers = nz.get("layers", {}) or {}
    base = layers.get("base", {}) or {}
    clump = layers.get("clump", {}) or {}
    patch = layers.get("patch", {}) or {}

    # L0 — base streaks (fBm).
    if base.get("enabled", False):
        fp = int(base["freq_phi"])
        n0 = fbm2(u * base["freq_u"], phi01 * fp, fp, octaves=int(base["octaves"]),
                  lacunarity=int(base["lacunarity"]), gain=base["gain"], seed=seed + NSEED_L0)
        m = m + np.float32(base["amp"]) * (n0 - np.float32(0.5))

    # L1 — clump/tear (ridged MF × Voronoi billow, coverage-masked).
    if clump.get("enabled", False):
        fp = int(clump["freq_phi"])
        xu = u * clump["freq_u"]
        yphi = phi01 * fp
        zz = zeta * clump["freq_z"]
        ridge = ridged3(xu, yphi, zz, fp, octaves=int(clump["octaves"]),
                        lacunarity=int(clump["lacunarity"]), gain=clump["gain"],
                        offset=clump["ridge_offset"], feedback=RIDGE_FEEDBACK,
                        seed=seed + NSEED_L1_RIDGE)
        voro = voronoi_billow3(xu, yphi, zz, fp, k=clump["voronoi_k"], seed=seed + NSEED_L1_VORO)
        cl = ridge * voro
        mfp = int(clump["mask_freq_phi"])
        mask_raw = fbm2(u * clump["mask_freq_u"], phi01 * mfp, mfp, octaves=2,
                        lacunarity=2, gain=0.5, seed=seed + NSEED_L1_MASK)
        thr = np.float32(1.0 - clump["coverage"])
        t = (mask_raw - (thr - np.float32(MASK_SOFT))) / np.float32(2.0 * MASK_SOFT)
        t = np.clip(t, 0.0, 1.0)
        mask = t * t * (np.float32(3.0) - np.float32(2.0) * t)  # smoothstep
        m = m + np.float32(clump["amp"]) * mask * (cl - np.float32(clump["bias"]))

    # L2 — patchiness (fBm).
    if patch.get("enabled", False):
        fp = int(patch["freq_phi"])
        n2 = fbm2(u * patch["freq_u"], phi01 * fp, fp, octaves=int(patch["octaves"]),
                  lacunarity=int(patch["lacunarity"]), gain=patch["gain"], seed=seed + NSEED_L2)
        m = m + np.float32(patch["amp"]) * (n2 - np.float32(0.5))

    return m


def noise_density_mult(u, phi, zeta, nz, seed: int = 1234,
                       t_disk: float = 0.0, omega=0.0, shear_period: float = 0.0) -> np.ndarray:
    """Combined L0/L1/L2 procedural-density multiplier (SKILL.md CKS-12 §2–3, spec §4).

    **CPU source of truth** for the disk-turbulence layer stack; the GPU twin is
    ``taichi_renderer._disk_noise_density_mult`` (held to this by
    ``tests/test_disk_noise.py``). Evaluates the stack at the disk-natural coords
    (``u = ln r/r_inner``, ``phi`` = azimuth in radians, ``zeta`` = vertical scale
    heights) and returns ``exp(clamp(Σ amp·(layer − bias), ±m_max)) > 0`` — the
    multiplier on the Gaussian vertical density (feeds BOTH emission and absorption).

    **D2.3 — Keplerian shear advection (CKS-12 §2).** When ``shear_period`` (= the
    CKS-13-derived ``disk.dynamics.shear_period_M``) is > 0 the stack is advected by
    the gas flow: the whole layer stack is evaluated at two staggered reset phases
    ``φ′_k = φ − Ω(r)·a_k·T`` (``a_k = frac(s + k/2)``, ``s = t_disk/T``) and
    crossfaded with triangle weights ``w_k = 1 − |2a_k − 1|`` (``w_0 + w_1 ≡ 1``).
    Each phase draws a per-cycle reseed (``NCYC_PHASE``/``NCYC_CYCLE``) so the loop
    does not repeat with period ``T``. ``variance_preserve`` (``nz`` key, default
    True) divides the blended ``m`` by ``√(w_0² + w_1²)`` to remove the mid-crossfade
    contrast "breathing". ``omega`` = Ω(r) (Formula 3) is supplied per sample by the
    caller (it has ``r`` already); it may be a scalar or broadcastable array.

    ``dynamism`` (``nz`` key, default 1.0) is a **non-physical viz gain** on the shear
    amount: ``φ′_k = φ − dynamism·Ω·a_k·T``. At 1.0 it reproduces the CKS-12 §2 formula
    bit-for-bit; >1 exaggerates the per-frame differential winding (the swirl) without
    touching the reset cadence. Same dial spirit as ``disk.doppler_strength`` —
    artistic emphasis, not a metric change.

    With ``shear_period ≤ 0`` (the default, and any config without a ``disk.dynamics``
    block) the field is **static** — sampled directly at ``phi`` — i.e. exactly the
    D2.2 path, so existing static callers/tests are unchanged. ``nz`` disabled layers
    (or an ``enabled: false`` block) give an identically-1.0 multiplier either way.
    """
    u = _f32(u)
    phi = _f32(phi)
    zeta = _f32(zeta)
    mmax = np.float32(nz.get("m_max", 2.5))
    T = np.float32(shear_period)

    if T <= np.float32(0.0):
        # Static (D2.2 / backward-compatible default): no advection, sample at φ.
        m = _noise_m_stack(u, phi, zeta, nz, int(seed))
    else:
        omega = _f32(omega)
        s = np.float32(t_disk) / T
        var_preserve = bool(nz.get("variance_preserve", True))
        g = np.float32(nz.get("dynamism", 1.0))  # viz gain on the shear amount
        m = None
        wsq = np.float32(0.0)
        # Two staggered reset phases, crossfaded (Neyret-style advected texture).
        for k in (0, 1):
            ar = s + np.float32(0.5 * k)
            ck = int(np.floor(ar))           # reset-cycle index (≥ 0 for t_disk ≥ 0)
            ak = ar - np.float32(ck)         # age fraction ∈ [0, 1)
            wk = np.float32(1.0) - np.abs(np.float32(2.0) * ak - np.float32(1.0))
            seed_k = int(seed) + k * NCYC_PHASE + ck * NCYC_CYCLE
            phi_k = phi - g * omega * (ak * T)   # CKS-12 §2: φ sheared for ≤ T (×gain)
            mk = wk * _noise_m_stack(u, phi_k, zeta, nz, seed_k)
            m = mk if m is None else m + mk
            wsq = wsq + wk * wk
        if var_preserve and wsq > np.float32(0.0):
            m = m / np.sqrt(wsq)

    m = np.clip(m, -mmax, mmax)
    return np.exp(m).astype(np.float32)


# =========================================================================== #
# Taichi twins (@ti.func) — GPU-side mirror of every primitive above.
#
# These are held to the NumPy reference by ``tests/test_noise_gpu.py`` (~1e-6 on
# a shared grid; the integer-hash path is bit-exact). The integer ops below use
# explicit ``ti.u32`` so they wrap exactly like the NumPy ``uint32`` arithmetic,
# and every float op is f32 so it rounds to the same f32 the reference targets.
# Defined at import (compiled lazily on first kernel call, after ``ti.init``);
# the LOCKED backend is ``ti.init(arch=ti.cuda)`` per CLAUDE.md.
# =========================================================================== #
_INV_U32 = float(np.float32(1.0 / _U32))  # exact 2⁻³², the PCG word → [0,1) scale


@ti.func
def pcg_hash_ti(v):
    """``u32 -> u32`` PCG twin of :func:`pcg_hash` (identical wrapping arithmetic)."""
    state = v * ti.u32(747796405) + ti.u32(2891336453)
    shift = (state >> ti.u32(28)) + ti.u32(4)
    word = ((state >> shift) ^ state) * ti.u32(277803737)
    return (word >> ti.u32(22)) ^ word


@ti.func
def _u01_ti(h):
    """Hash word -> f32 in [0, 1); twin of :func:`_u01` (cast × exact 2⁻³²)."""
    return ti.cast(h, ti.f32) * ti.f32(_INV_U32)


@ti.func
def _hash2_ti(ix, iy, seed):
    h = (
        ti.cast(seed, ti.u32) * ti.u32(2654435769)
        + ti.cast(ix, ti.u32) * ti.u32(2246822507)
        + ti.cast(iy, ti.u32) * ti.u32(3266489917)
    )
    return pcg_hash_ti(h)


@ti.func
def _hash3_ti(ix, iy, iz, seed):
    h = (
        ti.cast(seed, ti.u32) * ti.u32(2654435769)
        + ti.cast(ix, ti.u32) * ti.u32(2246822507)
        + ti.cast(iy, ti.u32) * ti.u32(3266489917)
        + ti.cast(iz, ti.u32) * ti.u32(668265263)
    )
    return pcg_hash_ti(h)


@ti.func
def _wrap_ti(i, period):
    """Positive modulo; twin of :func:`_wrap`. Taichi ``%`` is truncated (C
    semantics), so the double-mod is what makes negative indices floor like NumPy."""
    return ((i % period) + period) % period


@ti.func
def _fade_ti(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@ti.func
def _grad3_ti(h, x, y, z):
    """Perlin branch-selected gradient dot; twin of :func:`_grad3`."""
    hh = h & ti.u32(15)
    u = x if hh < 8 else y
    v = y if hh < 4 else (x if (hh == 12 or hh == 14) else z)
    gu = u if (hh & ti.u32(1)) == 0 else -u
    gv = v if (hh & ti.u32(2)) == 0 else -v
    return gu + gv


@ti.func
def _lerp_ti(t, a, b):
    return a + t * (b - a)


@ti.func
def _to01_ti(raw):
    return ti.max(0.0, ti.min(1.0, 0.5 * (raw + 1.0)))


@ti.func
def gnoise2_ti(x, y, period, seed):
    """Twin of :func:`gnoise2` (z = 0 slice of the 3D gradient lattice)."""
    xi = ti.cast(ti.floor(x), ti.i32)
    yi = ti.cast(ti.floor(y), ti.i32)
    xf = x - ti.cast(xi, ti.f32)
    yf = y - ti.cast(yi, ti.f32)
    u = _fade_ti(xf)
    v = _fade_ti(yf)

    gy0 = _wrap_ti(yi, period)
    gy1 = _wrap_ti(yi + 1, period)
    c00 = _grad3_ti(_hash3_ti(xi, gy0, 0, seed), xf, yf, 0.0)
    c01 = _grad3_ti(_hash3_ti(xi, gy1, 0, seed), xf, yf - 1.0, 0.0)
    c10 = _grad3_ti(_hash3_ti(xi + 1, gy0, 0, seed), xf - 1.0, yf, 0.0)
    c11 = _grad3_ti(_hash3_ti(xi + 1, gy1, 0, seed), xf - 1.0, yf - 1.0, 0.0)
    a = _lerp_ti(v, c00, c01)
    b = _lerp_ti(v, c10, c11)
    return _to01_ti(_lerp_ti(u, a, b))


@ti.func
def gnoise3_ti(x, y, z, period, seed):
    """Twin of :func:`gnoise3` (8-corner trilinear gradient lattice)."""
    xi = ti.cast(ti.floor(x), ti.i32)
    yi = ti.cast(ti.floor(y), ti.i32)
    zi = ti.cast(ti.floor(z), ti.i32)
    xf = x - ti.cast(xi, ti.f32)
    yf = y - ti.cast(yi, ti.f32)
    zf = z - ti.cast(zi, ti.f32)
    u = _fade_ti(xf)
    v = _fade_ti(yf)
    w = _fade_ti(zf)

    gy0 = _wrap_ti(yi, period)
    gy1 = _wrap_ti(yi + 1, period)
    x00 = _lerp_ti(
        w,
        _grad3_ti(_hash3_ti(xi, gy0, zi, seed), xf, yf, zf),
        _grad3_ti(_hash3_ti(xi, gy0, zi + 1, seed), xf, yf, zf - 1.0),
    )
    x01 = _lerp_ti(
        w,
        _grad3_ti(_hash3_ti(xi, gy1, zi, seed), xf, yf - 1.0, zf),
        _grad3_ti(_hash3_ti(xi, gy1, zi + 1, seed), xf, yf - 1.0, zf - 1.0),
    )
    x10 = _lerp_ti(
        w,
        _grad3_ti(_hash3_ti(xi + 1, gy0, zi, seed), xf - 1.0, yf, zf),
        _grad3_ti(_hash3_ti(xi + 1, gy0, zi + 1, seed), xf - 1.0, yf, zf - 1.0),
    )
    x11 = _lerp_ti(
        w,
        _grad3_ti(_hash3_ti(xi + 1, gy1, zi, seed), xf - 1.0, yf - 1.0, zf),
        _grad3_ti(_hash3_ti(xi + 1, gy1, zi + 1, seed), xf - 1.0, yf - 1.0, zf - 1.0),
    )
    y0 = _lerp_ti(v, x00, x01)
    y1 = _lerp_ti(v, x10, x11)
    return _to01_ti(_lerp_ti(u, y0, y1))


# --- fBm / billow octave stacks (mode 0 = identity fBm, 1 = billow |2n−1|) --- #
@ti.func
def _stack2_ti(x, y, period, octaves, lac, gain, seed, mode):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    for o in range(octaves):
        n = gnoise2_ti(x * freq, y * freq, per, seed + o)
        t = ti.abs(2.0 * n - 1.0) if mode == 1 else n
        total += t * amp
        norm += amp
        amp *= gain
        freq *= lac
        per *= lac
    return total / norm


@ti.func
def _stack3_ti(x, y, z, period, octaves, lac, gain, seed, mode):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    for o in range(octaves):
        n = gnoise3_ti(x * freq, y * freq, z * freq, per, seed + o)
        t = ti.abs(2.0 * n - 1.0) if mode == 1 else n
        total += t * amp
        norm += amp
        amp *= gain
        freq *= lac
        per *= lac
    return total / norm


@ti.func
def fbm2_ti(x, y, period, octaves, lac, gain, seed):
    return _stack2_ti(x, y, period, octaves, lac, gain, seed, 0)


@ti.func
def fbm3_ti(x, y, z, period, octaves, lac, gain, seed):
    return _stack3_ti(x, y, z, period, octaves, lac, gain, seed, 0)


@ti.func
def billow2_ti(x, y, period, octaves, lac, gain, seed):
    return _stack2_ti(x, y, period, octaves, lac, gain, seed, 1)


@ti.func
def billow3_ti(x, y, z, period, octaves, lac, gain, seed):
    return _stack3_ti(x, y, z, period, octaves, lac, gain, seed, 1)


# --- Musgrave ridged multifractal (twin of :func:`_ridged`) ----------------- #
@ti.func
def ridged2_ti(x, y, period, octaves, lac, gain, offset, feedback, seed):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    prev = 0.0
    for o in range(octaves):
        n = gnoise2_ti(x * freq, y * freq, per, seed + o)
        w = 1.0 if o == 0 else ti.max(0.0, ti.min(1.0, prev * feedback))
        d = offset - ti.abs(2.0 * n - 1.0)
        r = d * d
        prev = w * r
        total += prev * amp
        norm += amp
        amp *= gain
        freq *= lac
        per *= lac
    return ti.max(0.0, ti.min(1.0, total / norm))


@ti.func
def ridged3_ti(x, y, z, period, octaves, lac, gain, offset, feedback, seed):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    prev = 0.0
    for o in range(octaves):
        n = gnoise3_ti(x * freq, y * freq, z * freq, per, seed + o)
        w = 1.0 if o == 0 else ti.max(0.0, ti.min(1.0, prev * feedback))
        d = offset - ti.abs(2.0 * n - 1.0)
        r = d * d
        prev = w * r
        total += prev * amp
        norm += amp
        amp *= gain
        freq *= lac
        per *= lac
    return ti.max(0.0, ti.min(1.0, total / norm))


# --- Worley / Voronoi cellular (twin of :func:`worley2` / :func:`worley3`) --- #
@ti.func
def worley2_ti(x, y, period, seed):
    """Returns ``ti.Vector([F1, F2])``. Cell iteration order matches the NumPy
    reference (the F2 update depends on the running F1)."""
    cx = ti.cast(ti.floor(x), ti.i32)
    cy = ti.cast(ti.floor(y), ti.i32)
    fx = x - ti.cast(cx, ti.f32)
    fy = y - ti.cast(cy, ti.f32)
    f1 = 1e9
    f2 = 1e9
    for dx in ti.static((-1, 0, 1)):
        for dy in ti.static((-1, 0, 1)):
            h = _hash2_ti(cx + dx, _wrap_ti(cy + dy, period), seed)
            jx = _u01_ti(pcg_hash_ti(h))
            jy = _u01_ti(pcg_hash_ti(h ^ ti.u32(0x9E3779B1)))
            ox = (float(dx) + jx) - fx
            oy = (float(dy) + jy) - fy
            d2 = ox * ox + oy * oy
            f2 = ti.min(f2, ti.max(f1, d2))
            f1 = ti.min(f1, d2)
    return ti.Vector([ti.sqrt(f1), ti.sqrt(f2)])


@ti.func
def worley3_ti(x, y, z, period, seed):
    """Returns ``ti.Vector([F1, F2])`` (27-cell); twin of :func:`worley3`."""
    cx = ti.cast(ti.floor(x), ti.i32)
    cy = ti.cast(ti.floor(y), ti.i32)
    cz = ti.cast(ti.floor(z), ti.i32)
    fx = x - ti.cast(cx, ti.f32)
    fy = y - ti.cast(cy, ti.f32)
    fz = z - ti.cast(cz, ti.f32)
    f1 = 1e9
    f2 = 1e9
    for dx in ti.static((-1, 0, 1)):
        for dy in ti.static((-1, 0, 1)):
            for dz in ti.static((-1, 0, 1)):
                h = _hash3_ti(cx + dx, _wrap_ti(cy + dy, period), cz + dz, seed)
                jx = _u01_ti(pcg_hash_ti(h))
                jy = _u01_ti(pcg_hash_ti(h ^ ti.u32(0x9E3779B1)))
                jz = _u01_ti(pcg_hash_ti(h ^ ti.u32(0x85EBCA77)))
                ox = (float(dx) + jx) - fx
                oy = (float(dy) + jy) - fy
                oz = (float(dz) + jz) - fz
                d2 = ox * ox + oy * oy + oz * oz
                f2 = ti.min(f2, ti.max(f1, d2))
                f1 = ti.min(f1, d2)
    return ti.Vector([ti.sqrt(f1), ti.sqrt(f2)])


@ti.func
def voronoi_billow2_ti(x, y, period, k, seed):
    d = worley2_ti(x, y, period, seed)
    return ti.exp(-k * d[0])


@ti.func
def voronoi_billow3_ti(x, y, z, period, k, seed):
    d = worley3_ti(x, y, z, period, seed)
    return ti.exp(-k * d[0])


@ti.func
def cell_wall2_ti(x, y, period, seed):
    d = worley2_ti(x, y, period, seed)
    return d[1] - d[0]


@ti.func
def cell_wall3_ti(x, y, z, period, seed):
    d = worley3_ti(x, y, z, period, seed)
    return d[1] - d[0]
