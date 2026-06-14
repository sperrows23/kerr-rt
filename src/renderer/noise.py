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

Primitive sources: Perlin, *Improving Noise* (2002); Perlin, *Simplex noise*
(2001) / Gustavson, *Simplex noise demystified* (2005) — the isotropic ``snoise*``
basis (§3.6, V1.5); Worley, *A cellular texture basis function* (1996); Musgrave,
*Texturing & Modeling* (ridged construction). See spec §3 and §10 for provenance.

Note on periodicity: the cubic-lattice family (``gnoise*`` and everything built on
it) is **exactly φ-periodic** via wrapped lattice indices (CKS-12 constraint 5);
the skewed simplex family (``snoise*``) is **not** lattice-periodic and is a
library basis for the V3 curl-flow potential, not wired into the φ-periodic disk
density stack — see §3.6.
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
# §3.6 Simplex gradient noise (Perlin/Gustavson skewed-simplex) — V1.5
#
# An *isotropic* gradient basis: the lattice is the skewed simplicial grid
# (triangles in 2D, tetrahedra in 3D), not the axis-aligned cubic grid of
# :func:`gnoise2`/:func:`gnoise3`. Its whole reason to exist here is that the
# square-lattice Perlin basis leaks faint **axis-aligned directional artifacts**
# (visible as a grid bias in a curl/flow field); the hexagonal simplex lattice
# has no such preferred direction (pinned by ``test_noise.py``'s isotropy guard).
#
# Scope (volumetric spec §1a / V3 step 7, decision D-V4 → "add Simplex, V1.5"):
# this is the basis for the **curl-flow potential** built in V3, NOT a drop-in for
# the φ-periodic disk density stack. Classic simplex is **not exactly φ-periodic**
# — the input skew couples the axes, so a 2π φ-period is not a lattice period the
# way it is for the wrapped cubic lattice (CKS-12 constraint 5). Exact φ-seamlessness
# for the disk is obtained at the V3 integration point (cylinder embedding of the
# periodic axis), not here. Accordingly nothing in V1.5 wires simplex into the
# render path, so every golden frame is bit-identical (it is pure library addition,
# exactly as the D2.1 primitives were added before D2.2 consumed them).
#
# Construction follows Perlin (*Simplex noise*, 2001) / Gustavson (*Simplex noise
# demystified*, 2005) verbatim — the project does not re-derive it. Reuses this
# file's PCG corner hash and the Perlin-2002 12-gradient set (:func:`_grad3`, which
# already returns grad·(x,y,z)); the radial kernel is ``(r0² − |d|²)₊⁴ · grad·d``.
# float32 throughout, no transcendentals on the lattice path — same discipline as
# the rest of the library, so the @ti.func twins match to ~1e-6.
# --------------------------------------------------------------------------- #
# Skew/unskew constants, written as f32-exact decimal literals so np.float32(lit)
# and the GPU ti.f32(lit) round to identical bits.
_F2 = np.float32(0.3660254037844386)   # (√3 − 1)/2
_G2 = np.float32(0.21132486540518713)  # (3 − √3)/6
_F3 = np.float32(0.3333333333333333)   # 1/3
_G3 = np.float32(0.16666666666666666)  # 1/6
_SR2 = np.float32(0.5)   # 2D corner support radius² (r0² = 0.5)
_SR3 = np.float32(0.6)   # 3D corner support radius² (r0² = 0.6)
_SSCALE2 = np.float32(70.0)  # raw → ~[−1, 1] normalizer (Gustavson)
_SSCALE3 = np.float32(32.0)


def _scorner2(ix, iy, x, y, seed) -> np.ndarray:
    """One 2D simplex corner: ``(0.5 − x² − y²)₊⁴ · grad·(x, y)`` (Gustavson §2D)."""
    t = np.maximum(_SR2 - x * x - y * y, np.float32(0.0)).astype(np.float32)
    t2 = (t * t).astype(np.float32)
    g = _grad3(_hash3(ix, iy, 0, seed), x, y, np.float32(0.0))
    return (t2 * t2 * g).astype(np.float32)


def snoise2(x, y, period: int = 0, seed: int = 0) -> np.ndarray:
    """2D simplex noise in ``[0, 1]`` (Perlin/Gustavson). ``period`` is accepted and
    **ignored** (simplex is not lattice-periodic) so it slots into the shared
    :func:`_octaves` fBm machinery exactly like :func:`gnoise2`."""
    x = _f32(x)
    y = _f32(y)
    s = ((x + y) * _F2).astype(np.float32)            # skew to the simplex grid
    i = np.floor(x + s).astype(np.int32)
    j = np.floor(y + s).astype(np.int32)
    t = ((i + j).astype(np.float32) * _G2).astype(np.float32)
    x0 = (x - (i.astype(np.float32) - t)).astype(np.float32)  # unskewed offset from cell origin
    y0 = (y - (j.astype(np.float32) - t)).astype(np.float32)

    upper = x0 > y0                                   # which of the two triangles
    i1 = upper.astype(np.int32)
    j1 = (~upper).astype(np.int32)
    x1 = (x0 - i1.astype(np.float32) + _G2).astype(np.float32)
    y1 = (y0 - j1.astype(np.float32) + _G2).astype(np.float32)
    x2 = (x0 - np.float32(1.0) + np.float32(2.0) * _G2).astype(np.float32)
    y2 = (y0 - np.float32(1.0) + np.float32(2.0) * _G2).astype(np.float32)

    n = (_scorner2(i, j, x0, y0, seed)
         + _scorner2(i + i1, j + j1, x1, y1, seed)
         + _scorner2(i + 1, j + 1, x2, y2, seed))
    return _to01((_SSCALE2 * n).astype(np.float32))


def _scorner3(ix, iy, iz, x, y, z, seed) -> np.ndarray:
    """One 3D simplex corner: ``(0.6 − x² − y² − z²)₊⁴ · grad·(x, y, z)``."""
    t = np.maximum(_SR3 - x * x - y * y - z * z, np.float32(0.0)).astype(np.float32)
    t2 = (t * t).astype(np.float32)
    g = _grad3(_hash3(ix, iy, iz, seed), x, y, z)
    return (t2 * t2 * g).astype(np.float32)


def snoise3(x, y, z, period: int = 0, seed: int = 0) -> np.ndarray:
    """3D simplex noise in ``[0, 1]`` (Perlin/Gustavson). ``period`` accepted/ignored
    (see :func:`snoise2`)."""
    x = _f32(x)
    y = _f32(y)
    z = _f32(z)
    s = ((x + y + z) * _F3).astype(np.float32)
    i = np.floor(x + s).astype(np.int32)
    j = np.floor(y + s).astype(np.int32)
    k = np.floor(z + s).astype(np.int32)
    t = ((i + j + k).astype(np.float32) * _G3).astype(np.float32)
    x0 = (x - (i.astype(np.float32) - t)).astype(np.float32)
    y0 = (y - (j.astype(np.float32) - t)).astype(np.float32)
    z0 = (z - (k.astype(np.float32) - t)).astype(np.float32)

    # Rank the unskewed offsets to pick the tetrahedron (Gustavson's 6 cases). The
    # comparison operators (≥) match the GPU twin exactly so the two cannot split a tie.
    c1 = x0 >= y0
    c2 = y0 >= z0
    c3 = x0 >= z0
    A = c1 & c2                # X Y Z
    B = c1 & ~c2 & c3          # X Z Y
    C = c1 & ~c2 & ~c3         # Z X Y
    D = ~c1 & ~c2              # Z Y X
    E = ~c1 & c2 & ~c3         # Y Z X
    F = ~c1 & c2 & c3          # Y X Z
    sel = [A, B, C, D, E, F]
    i1 = np.select(sel, [1, 1, 0, 0, 0, 0]).astype(np.int32)
    j1 = np.select(sel, [0, 0, 0, 0, 1, 1]).astype(np.int32)
    k1 = np.select(sel, [0, 0, 1, 1, 0, 0]).astype(np.int32)
    i2 = np.select(sel, [1, 1, 1, 0, 0, 1]).astype(np.int32)
    j2 = np.select(sel, [1, 0, 0, 1, 1, 1]).astype(np.int32)
    k2 = np.select(sel, [0, 1, 1, 1, 1, 0]).astype(np.int32)

    g3 = _G3
    x1 = (x0 - i1.astype(np.float32) + g3).astype(np.float32)
    y1 = (y0 - j1.astype(np.float32) + g3).astype(np.float32)
    z1 = (z0 - k1.astype(np.float32) + g3).astype(np.float32)
    x2 = (x0 - i2.astype(np.float32) + np.float32(2.0) * g3).astype(np.float32)
    y2 = (y0 - j2.astype(np.float32) + np.float32(2.0) * g3).astype(np.float32)
    z2 = (z0 - k2.astype(np.float32) + np.float32(2.0) * g3).astype(np.float32)
    x3 = (x0 - np.float32(1.0) + np.float32(3.0) * g3).astype(np.float32)
    y3 = (y0 - np.float32(1.0) + np.float32(3.0) * g3).astype(np.float32)
    z3 = (z0 - np.float32(1.0) + np.float32(3.0) * g3).astype(np.float32)

    n = (_scorner3(i, j, k, x0, y0, z0, seed)
         + _scorner3(i + i1, j + j1, k + k1, x1, y1, z1, seed)
         + _scorner3(i + i2, j + j2, k + k2, x2, y2, z2, seed)
         + _scorner3(i + 1, j + 1, k + 1, x3, y3, z3, seed))
    return _to01((_SSCALE3 * n).astype(np.float32))


def sfbm2(x, y, period: int = 0, octaves=4, lacunarity=2, gain=0.5, seed=0) -> np.ndarray:
    """fBm of :func:`snoise2`, normalized to ``[0, 1]`` (isotropic counterpart of
    :func:`fbm2`). ``period`` is carried for signature symmetry but unused."""
    return _fbm(snoise2, _f32(x), _f32(y), None, period, octaves, lacunarity, gain, seed,
                lambda n: n)


def sfbm3(x, y, z, period: int = 0, octaves=4, lacunarity=2, gain=0.5, seed=0) -> np.ndarray:
    """fBm of :func:`snoise3` (isotropic counterpart of :func:`fbm3`)."""
    return _fbm(snoise3, _f32(x), _f32(y), _f32(z), period, octaves, lacunarity, gain, seed,
                lambda n: n)


# --------------------------------------------------------------------------- #
# §3.7 Curl-flow domain warp (V3.0; SKILL.md Formula CKS-18).
#
# An in-plane, divergence-free distortion of the disk-noise coordinate (u, φ),
# built as the 2-D curl of a scalar potential on the V1.5 simplex basis. The
# potential is sampled on the (cosφ, sinφ, u) CYLINDER EMBEDDING of the φ-axis, so
# the displacement is exactly 2π-periodic in φ — seamless across φ=0 (CKS-12
# constraint 5) even though classic simplex is not lattice-periodic — and ρ_c /
# k_u (freq_phi / freq_u) may be any real. The curl of a scalar is divergence-free
# by construction (the incompressible-flow look). Texturing only: it relocates the
# noise sample coordinate, never a metric/transport quantity (CKS-18 governance).
# --------------------------------------------------------------------------- #
CURL_FD_EPS = np.float32(1e-3)  # default central-difference step in the (u,φ) chart


def curl_warp(u, phi, amp, freq_phi=3.0, freq_u=1.0, octaves=4, lacunarity=2,
              gain=0.5, seed=0, fd_eps=CURL_FD_EPS):
    """In-plane divergence-free curl warp of the disk-noise coords ``(u, φ)`` (CKS-18).

    **CPU source of truth**; the GPU twin is :func:`curl_warp_ti` (held to this by
    ``tests/test_disk_noise.py``). Returns ``(u', φ')`` displaced by the 2-D curl of
    the scalar potential ``ψ(u,φ) = sfbm3(cosφ·ρ_c, sinφ·ρ_c, u·k_u)``::

        δu = +∂ψ/∂φ      δφ = −∂ψ/∂u      (∇·(δu,δφ) ≡ 0)
        u' = u + amp·δu   φ' = φ + amp·δφ

    The gradient is a CENTRAL finite difference with step ``ε = fd_eps`` (simplex has
    no analytic gradient on this path). Because ψ is built on ``cos φ`` / ``sin φ``,
    ``δu`` and ``δφ`` are exactly 2π-periodic in φ ⇒ ``φ'`` is continuous across the
    seam. ``amp == 0`` ⇒ identity (the disabled path). ``u`` / ``phi`` may be scalars
    or broadcastable arrays.
    """
    u = _f32(u)
    phi = _f32(phi)
    rho = np.float32(freq_phi)
    ku = np.float32(freq_u)
    eps = np.float32(fd_eps)
    oct_ = int(octaves)
    lac = int(lacunarity)
    g = np.float32(gain)
    sd = int(seed)

    def _psi(uu, pp):
        return sfbm3(np.cos(pp) * rho, np.sin(pp) * rho, uu * ku,
                     0, octaves=oct_, lacunarity=lac, gain=g, seed=sd)

    inv2e = np.float32(1.0) / (np.float32(2.0) * eps)
    dpsi_du = ((_psi(u + eps, phi) - _psi(u - eps, phi)) * inv2e).astype(np.float32)
    dpsi_dphi = ((_psi(u, phi + eps) - _psi(u, phi - eps)) * inv2e).astype(np.float32)
    a = np.float32(amp)
    u_w = (u + a * dpsi_dphi).astype(np.float32)   # δu = +∂ψ/∂φ
    phi_w = (phi - a * dpsi_du).astype(np.float32)  # δφ = −∂ψ/∂u
    return u_w, phi_w


def _apply_curl(u, phi, curl):
    """Apply :func:`curl_warp` if the ``curl`` config block is enabled, else return
    ``(u, phi)`` unchanged. Shared entry for the density layer stack AND the §3
    modulation stack so the two warp identically (one warp, coherent swirl). An
    absent block / ``enabled: false`` / ``amp == 0`` is the bit-identical disabled
    path (CKS-12 constraint 6)."""
    if not curl or not curl.get("enabled", False):
        return u, phi
    amp = float(curl.get("amp", 0.0))
    if amp == 0.0:
        return u, phi
    return curl_warp(
        u, phi, amp,
        freq_phi=curl.get("freq_phi", 3.0),
        freq_u=curl.get("freq_u", 1.0),
        octaves=int(curl.get("octaves", 4)),
        lacunarity=int(curl.get("lacunarity", 2)),
        gain=curl.get("gain", 0.5),
        seed=int(curl.get("seed", 0)),
        fd_eps=float(curl.get("fd_eps", float(CURL_FD_EPS))),
    )


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

# D2.4 modulation-field seed offsets (CKS-12 §3). The four §3 envelopes
# (emitted temperature, inner/outer edge, scale height) are decorrelated from
# each other AND from the density stack's NSEED_* family by these strides — the
# PCG lattice hash avalanches any seed delta, so the four envelopes are visually
# independent fBm fields sharing the same advection/reseed bookkeeping.
NSEED_MOD_T = 503     # n_T  — emitted-temperature lumps
NSEED_MOD_EIN = 601   # n_e  — inner-edge raggedness
NSEED_MOD_EOUT = 701  # n_e' — outer-edge raggedness
NSEED_MOD_H = 809     # n_h  — scale-height lumpiness


def _noise_m_stack(u, phi, zeta, nz, seed: int) -> np.ndarray:
    """Unclamped log-density sum ``m = Σ amp·(layer − bias)`` at the (already-f32)
    disk-natural coords, BEFORE the ``m_max`` clamp and ``exp`` (CKS-12 §3, spec §4).

    Factored out of :func:`noise_density_mult` so the D2.3 shear-advection blend can
    evaluate the whole layer stack at each of its two reset phases (a different
    ``phi`` and ``seed`` per phase) and crossfade the two ``m`` values before the
    single shared clamp+exp. φ enters each lattice as ``y = phi/(2π)·freq_phi`` with
    integer period ``freq_phi`` (exact 2π-periodicity, constraint 5). Disabled layers
    contribute nothing (``m`` stays 0 ⇒ multiplier 1.0, the kernel's skipped branch).

    The CKS-18 curl warp (if ``nz["curl"]`` is enabled) distorts ``(u, φ)`` HERE, at
    the stack entry — applied to the already-sheared per-phase ``φ`` so the eddies are
    frozen into the gas's material frame and the §2 shear winds them into filaments.
    """
    u, phi = _apply_curl(u, phi, nz.get("curl"))
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


def _mod_fbm_stack(u, phi, mod, seed_offsets, seed_base, curl=None):
    """Evaluate the four §3 modulation envelopes at one (already-advected) phase.

    Each is a single fBm of :func:`fbm2` in ``[0, 1]`` over the disk-natural
    lattice ``(u·freq_u, φ/(2π)·freq_phi)`` with integer φ-period ``freq_phi``
    (exact 2π periodicity, constraint 5). Returns a 4-tuple of f32 arrays
    ``(n_T, n_e_in, n_e_out, n_h)`` keyed by ``seed_offsets`` so the envelopes are
    mutually decorrelated. Shared by the static and advected paths of
    :func:`noise_modulation_fields`.

    The CKS-18 curl warp (``curl`` block, if enabled) distorts ``(u, φ)`` at entry —
    the SAME warp the density stack applies, so the §3 envelopes swirl coherently with
    the density.
    """
    u, phi = _apply_curl(u, phi, curl)
    fp = int(mod["freq_phi"])
    x = u * np.float32(mod["freq_u"])
    y = phi * _INV_TWO_PI * np.float32(fp)
    oct_ = int(mod.get("octaves", 3))
    lac = int(mod.get("lacunarity", 2))
    gain = mod.get("gain", 0.5)
    return tuple(
        fbm2(x, y, fp, octaves=oct_, lacunarity=lac, gain=gain, seed=int(seed_base) + off)
        for off in seed_offsets
    )


def noise_modulation_fields(u, phi, zeta, nz, seed: int = 1234,
                            t_disk: float = 0.0, omega=0.0, shear_period: float = 0.0):
    """Advected ``[0, 1]`` envelopes ``(n_T, n_e_in, n_e_out, n_h)`` for the CKS-12
    §3 temperature / edge / scale-height modulation. **CPU source of truth**; the
    GPU twin is ``taichi_renderer._disk_noise_mod_fields`` (held to this by
    ``tests/test_disk_noise.py``). These are NOT the density field — they are the
    four *amplitude* envelopes the disk kernel uses to modulate the **emitted**
    temperature (before the g shift), the inner/outer edge windows, and the local
    Gaussian scale height (SKILL.md CKS-12 §3, hard constraints 1–4).

    Advected with the SAME dual-phase reset + ``dynamism`` gain as the density
    field (:func:`noise_density_mult`) so the envelopes co-move with the gas:
    ``φ′_k = φ − dynamism·Ω(r)·(a_k·T)``, triangle weights ``w_k = 1 − |2a_k − 1|``
    (``w_0 + w_1 ≡ 1``), per-cycle reseed ``seed + k·NCYC_PHASE + c_k·NCYC_CYCLE``.
    Unlike the density blend there is **no** ``variance_preserve`` divide — a
    ``√(Σw²)`` rescale could push a ``[0, 1]`` fBm outside ``[0, 1]``; the convex
    triangle weights keep each blended envelope in ``[0, 1]`` (so ``n − ½`` stays a
    bounded ``±½`` modulation). ``zeta`` is accepted for signature symmetry with
    the density sampler (and so the caller can pass the same coords) but the
    envelopes are sampled in ``(u, φ)`` only — they are slowly-varying along the
    slab.

    With ``shear_period ≤ 0`` (no ``disk.dynamics`` block) the field is static
    (sampled at ``φ``). When the ``disk.noise.modulation`` block is absent or
    ``enabled: false`` every envelope is the **no-op midpoint 0.5** (so ``n − ½ = 0``
    ⇒ identity modulation), matching the kernel's skipped branch.
    """
    mod = nz.get("modulation", {}) or {}
    shape = np.broadcast(u, phi, zeta).shape
    half = np.full(shape, np.float32(0.5), dtype=np.float32)
    if not mod.get("enabled", False):
        return half, half, half, half

    u = _f32(u)
    phi = _f32(phi)
    offs = (NSEED_MOD_T, NSEED_MOD_EIN, NSEED_MOD_EOUT, NSEED_MOD_H)
    T = np.float32(shear_period)

    curl = nz.get("curl")
    if T <= np.float32(0.0):
        vals = _mod_fbm_stack(u, phi, mod, offs, int(seed), curl=curl)
        return tuple(np.broadcast_to(v, shape).astype(np.float32) for v in vals)

    omega = _f32(omega)
    s = np.float32(t_disk) / T
    g = np.float32(nz.get("dynamism", 1.0))  # same viz gain as the density shear
    acc = [np.zeros(shape, dtype=np.float32) for _ in offs]
    for k in (0, 1):
        ar = s + np.float32(0.5 * k)
        ck = int(np.floor(ar))
        ak = ar - np.float32(ck)
        wk = np.float32(1.0) - np.abs(np.float32(2.0) * ak - np.float32(1.0))
        seed_k = int(seed) + k * NCYC_PHASE + ck * NCYC_CYCLE
        phi_k = phi - g * omega * (ak * T)
        vk = _mod_fbm_stack(u, phi_k, mod, offs, seed_k, curl=curl)
        acc = [a + wk * v for a, v in zip(acc, vk)]
    return tuple(np.broadcast_to(a, shape).astype(np.float32) for a in acc)


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


# --- Simplex (Perlin/Gustavson) — twin of :func:`snoise2` / :func:`snoise3` --- #
# The skew/unskew constants are written as the same decimal literals as the CPU
# module constants so the f32 rounding is identical; ``period`` is accepted for
# signature parity with the cubic-lattice twins but ignored (simplex is not
# lattice-periodic). The 3D tetrahedron selection mirrors the CPU ``np.select``
# branch case-for-case (≥ comparisons identical ⇒ no tie split).
@ti.func
def _scorner2_ti(ix, iy, x, y, seed):
    """One 2D simplex corner; twin of :func:`_scorner2`."""
    t = 0.5 - x * x - y * y
    res = 0.0
    if t > 0.0:
        t2 = t * t
        res = t2 * t2 * _grad3_ti(_hash3_ti(ix, iy, 0, seed), x, y, 0.0)
    return res


@ti.func
def snoise2_ti(x, y, period, seed):
    """Twin of :func:`snoise2` (2D simplex in [0, 1])."""
    s = (x + y) * 0.3660254037844386
    i = ti.cast(ti.floor(x + s), ti.i32)
    j = ti.cast(ti.floor(y + s), ti.i32)
    t = ti.cast(i + j, ti.f32) * 0.21132486540518713
    x0 = x - (ti.cast(i, ti.f32) - t)
    y0 = y - (ti.cast(j, ti.f32) - t)
    i1 = 1 if x0 > y0 else 0
    j1 = 0 if x0 > y0 else 1
    g2 = 0.21132486540518713
    x1 = x0 - ti.cast(i1, ti.f32) + g2
    y1 = y0 - ti.cast(j1, ti.f32) + g2
    x2 = x0 - 1.0 + 2.0 * g2
    y2 = y0 - 1.0 + 2.0 * g2
    n = (_scorner2_ti(i, j, x0, y0, seed)
         + _scorner2_ti(i + i1, j + j1, x1, y1, seed)
         + _scorner2_ti(i + 1, j + 1, x2, y2, seed))
    return _to01_ti(70.0 * n)


@ti.func
def _scorner3_ti(ix, iy, iz, x, y, z, seed):
    """One 3D simplex corner; twin of :func:`_scorner3`."""
    t = 0.6 - x * x - y * y - z * z
    res = 0.0
    if t > 0.0:
        t2 = t * t
        res = t2 * t2 * _grad3_ti(_hash3_ti(ix, iy, iz, seed), x, y, z)
    return res


@ti.func
def snoise3_ti(x, y, z, period, seed):
    """Twin of :func:`snoise3` (3D simplex in [0, 1])."""
    g3 = 0.16666666666666666
    s = (x + y + z) * 0.3333333333333333
    i = ti.cast(ti.floor(x + s), ti.i32)
    j = ti.cast(ti.floor(y + s), ti.i32)
    k = ti.cast(ti.floor(z + s), ti.i32)
    t = ti.cast(i + j + k, ti.f32) * g3
    x0 = x - (ti.cast(i, ti.f32) - t)
    y0 = y - (ti.cast(j, ti.f32) - t)
    z0 = z - (ti.cast(k, ti.f32) - t)

    i1 = 0
    j1 = 0
    k1 = 0
    i2 = 0
    j2 = 0
    k2 = 0
    if x0 >= y0:
        if y0 >= z0:        # A: X Y Z
            i1 = 1
            i2 = 1
            j2 = 1
        elif x0 >= z0:      # B: X Z Y
            i1 = 1
            i2 = 1
            k2 = 1
        else:               # C: Z X Y
            k1 = 1
            i2 = 1
            k2 = 1
    else:
        if y0 < z0:         # D: Z Y X
            k1 = 1
            j2 = 1
            k2 = 1
        elif x0 < z0:       # E: Y Z X
            j1 = 1
            j2 = 1
            k2 = 1
        else:               # F: Y X Z
            j1 = 1
            i2 = 1
            j2 = 1

    x1 = x0 - ti.cast(i1, ti.f32) + g3
    y1 = y0 - ti.cast(j1, ti.f32) + g3
    z1 = z0 - ti.cast(k1, ti.f32) + g3
    x2 = x0 - ti.cast(i2, ti.f32) + 2.0 * g3
    y2 = y0 - ti.cast(j2, ti.f32) + 2.0 * g3
    z2 = z0 - ti.cast(k2, ti.f32) + 2.0 * g3
    x3 = x0 - 1.0 + 3.0 * g3
    y3 = y0 - 1.0 + 3.0 * g3
    z3 = z0 - 1.0 + 3.0 * g3

    n = (_scorner3_ti(i, j, k, x0, y0, z0, seed)
         + _scorner3_ti(i + i1, j + j1, k + k1, x1, y1, z1, seed)
         + _scorner3_ti(i + i2, j + j2, k + k2, x2, y2, z2, seed)
         + _scorner3_ti(i + 1, j + 1, k + 1, x3, y3, z3, seed))
    return _to01_ti(32.0 * n)


# Simplex fBm octave stacks (identity transform only; twin of :func:`sfbm2/3`).
@ti.func
def _sstack2_ti(x, y, period, octaves, lac, gain, seed):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    for o in range(octaves):
        total += snoise2_ti(x * freq, y * freq, per, seed + o) * amp
        norm += amp
        amp *= gain
        freq *= lac
        per *= lac
    return total / norm


@ti.func
def _sstack3_ti(x, y, z, period, octaves, lac, gain, seed):
    total = 0.0
    norm = 0.0
    freq = 1
    per = period
    amp = 1.0
    for o in range(octaves):
        total += snoise3_ti(x * freq, y * freq, z * freq, per, seed + o) * amp
        norm += amp
        amp *= gain
        freq *= lac
        per *= lac
    return total / norm


@ti.func
def sfbm2_ti(x, y, period, octaves, lac, gain, seed):
    return _sstack2_ti(x, y, period, octaves, lac, gain, seed)


@ti.func
def sfbm3_ti(x, y, z, period, octaves, lac, gain, seed):
    return _sstack3_ti(x, y, z, period, octaves, lac, gain, seed)


@ti.func
def curl_warp_ti(u, phi, amp, freq_phi, freq_u, octaves, lac, gain, seed, eps):
    """Twin of :func:`curl_warp` (CKS-18 in-plane divergence-free curl warp).

    Returns ``ti.Vector([u', φ'])`` displaced by the 2-D curl of the scalar potential
    ``ψ = sfbm3(cosφ·ρ_c, sinφ·ρ_c, u·k_u)``, with a central finite-difference
    gradient (step ``eps``) — the SAME 4-point stencil as the CPU reference, so the
    two agree to ~1e-5. Built on ``cos φ`` / ``sin φ`` ⇒ seamless across φ=0."""
    inv2e = 1.0 / (2.0 * eps)
    cphi = ti.cos(phi)
    sphi = ti.sin(phi)
    cphi_p = ti.cos(phi + eps)
    sphi_p = ti.sin(phi + eps)
    cphi_m = ti.cos(phi - eps)
    sphi_m = ti.sin(phi - eps)
    psi_up = sfbm3_ti(cphi * freq_phi, sphi * freq_phi, (u + eps) * freq_u, 0, octaves, lac, gain, seed)
    psi_um = sfbm3_ti(cphi * freq_phi, sphi * freq_phi, (u - eps) * freq_u, 0, octaves, lac, gain, seed)
    psi_pp = sfbm3_ti(cphi_p * freq_phi, sphi_p * freq_phi, u * freq_u, 0, octaves, lac, gain, seed)
    psi_pm = sfbm3_ti(cphi_m * freq_phi, sphi_m * freq_phi, u * freq_u, 0, octaves, lac, gain, seed)
    dpsi_du = (psi_up - psi_um) * inv2e
    dpsi_dphi = (psi_pp - psi_pm) * inv2e
    return ti.Vector([u + amp * dpsi_dphi, phi - amp * dpsi_du])
