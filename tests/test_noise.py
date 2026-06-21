"""D2.1 — procedural-noise primitive library guard (``renderer.noise``).

The disk turbulence design (``docs/specs/2026-06-13-disk-noise-turbulence.md`` §3,
SKILL.md Formula CKS-12) makes ``src/renderer/noise.py`` the **NumPy source of
truth** for the noise primitives and holds the ``@ti.func`` GPU twins to it by
test. This module is the CPU-side guard (no CUDA): it pins the public API and the
hard invariants that the renderer integration (D2.2+) and the GPU twins rely on:

  * **Exact φ-periodicity** — every primitive is bit-identical at ``y`` and
    ``y + period`` (CKS-12 constraint 5: integer φ-period ⇒ no seam at φ = 0).
  * **Range** — base gradient noise and the fBm family stay in ``[0, 1]``.
  * **Determinism** — pure function of inputs + seed; no ``ti.random`` /
    global RNG (CKS-12 constraint 7).
  * **Construction identities** — quintic-fade endpoints, fBm normalization,
    Worley ``F1 ≤ F2``, ridged ridges peak where the lattice noise is mid-band.

The CPU↔GPU agreement / GPU φ-periodicity checks live in
``tests/test_noise_gpu.py`` (gpu-marked, skips without CUDA).
"""
from __future__ import annotations

import numpy as np
import pytest

from renderer import noise


# --------------------------------------------------------------------------- #
# Sample grids. φ-axis (the SECOND lattice axis, ``y``) is the periodic one.
# --------------------------------------------------------------------------- #
def _grid(period: int, n: int = 23):
    """A non-degenerate (u, φ, ζ) lattice-space sample grid.

    ``x`` (log-radial) and ``z`` (vertical) are non-periodic; ``y`` (azimuth φ)
    spans ``[0, period)`` so wrap tests exercise the seam. ``y`` is sampled on a
    **dyadic** grid (multiples of 1/16): the noise is exactly 2π-periodic as a
    function of real ``y``, but ``y`` and ``y + period`` only carry *bit-identical*
    fractional parts (hence bit-identical output) when both are float-exact — true
    for dyadic samples, not for arbitrary floats where ``(y+period) − period``
    rounds. Renderer φ inputs are arbitrary, so the real-world guarantee is "no
    visible seam" (continuity to ~1e-7); this grid pins the stronger exact form.
    """
    x = np.linspace(-3.3, 4.7, n)
    y = (np.arange(n) * ((period * 16) // n)) / 16.0  # dyadic, spans [0, period)
    z = np.linspace(-1.9, 2.1, n)
    return x, y, z


_PERIOD = 6  # a representative integer freq_phi


# --------------------------------------------------------------------------- #
# pcg hash — deterministic integer mixing (CKS-12 constraint 7)
# --------------------------------------------------------------------------- #
def test_pcg_hash_is_deterministic_and_u32():
    v = np.arange(0, 1000, dtype=np.uint32)
    h1 = noise.pcg_hash(v)
    h2 = noise.pcg_hash(v.copy())
    assert h1.dtype == np.uint32
    assert np.array_equal(h1, h2)


def test_pcg_hash_avalanches():
    """Adjacent inputs must not produce correlated outputs (no obvious banding)."""
    v = np.arange(0, 4096, dtype=np.uint32)
    h = noise.pcg_hash(v).astype(np.float64) / 2**32
    # Mean ~0.5, and consecutive values are essentially uncorrelated.
    assert abs(h.mean() - 0.5) < 0.02
    corr = np.corrcoef(h[:-1], h[1:])[0, 1]
    assert abs(corr) < 0.1


# --------------------------------------------------------------------------- #
# Gradient lattice noise — range + exact φ-periodicity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn,dim", [(noise.gnoise2, 2), (noise.gnoise3, 3)])
def test_gnoise_in_unit_range(fn, dim):
    x, y, z = _grid(_PERIOD)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    n = fn(X, Y, _PERIOD) if dim == 2 else fn(X, Y, Z, _PERIOD)
    assert n.min() >= 0.0
    assert n.max() <= 1.0
    # not a constant field
    assert n.std() > 0.05


@pytest.mark.parametrize("fn,dim", [(noise.gnoise2, 2), (noise.gnoise3, 3)])
def test_gnoise_exact_phi_periodicity(fn, dim):
    x, y, z = _grid(_PERIOD)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    if dim == 2:
        a = fn(X, Y, _PERIOD)
        b = fn(X, Y + _PERIOD, _PERIOD)
    else:
        a = fn(X, Y, Z, _PERIOD)
        b = fn(X, Y + _PERIOD, Z, _PERIOD)
    # Integer-hash lattice ⇒ bit-identical across the 2π seam, not merely close.
    assert np.array_equal(a, b)


def test_gnoise_seed_changes_field():
    x, y, _ = _grid(_PERIOD)
    X, Y = np.meshgrid(x, y, indexing="ij")
    a = noise.gnoise2(X, Y, _PERIOD, seed=1)
    b = noise.gnoise2(X, Y, _PERIOD, seed=2)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# fBm / billow / ridged — normalization, range, periodicity
# --------------------------------------------------------------------------- #
def test_fbm_stays_in_unit_range_and_periodic():
    x, y, z = _grid(_PERIOD)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    n = noise.fbm3(X, Y, Z, _PERIOD, octaves=5, lacunarity=2, gain=0.5)
    assert n.min() >= 0.0 and n.max() <= 1.0
    # lacunarity is integer ⇒ every octave's φ-period is an integer multiple ⇒
    # the sum is still exactly 2π-periodic.
    n2 = noise.fbm3(X, Y + _PERIOD, Z, _PERIOD, octaves=5, lacunarity=2, gain=0.5)
    assert np.array_equal(n, n2)


def test_fbm_single_octave_equals_base_noise():
    x, y, _ = _grid(_PERIOD)
    X, Y = np.meshgrid(x, y, indexing="ij")
    one = noise.fbm2(X, Y, _PERIOD, octaves=1, gain=0.5)
    base = noise.gnoise2(X, Y, _PERIOD)
    assert np.allclose(one, base, atol=1e-6)


def test_billow_is_cusped_nonnegative_and_periodic():
    x, y, _ = _grid(_PERIOD)
    X, Y = np.meshgrid(x, y, indexing="ij")
    b = noise.billow2(X, Y, _PERIOD, octaves=4)
    assert b.min() >= 0.0 and b.max() <= 1.0
    assert np.array_equal(b, noise.billow2(X, Y + _PERIOD, _PERIOD, octaves=4))


def test_ridged_in_range_and_periodic():
    x, y, z = _grid(_PERIOD)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = noise.ridged3(X, Y, Z, _PERIOD, octaves=3)
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert r.std() > 0.05
    assert np.array_equal(r, noise.ridged3(X, Y + _PERIOD, Z, _PERIOD, octaves=3))


# --------------------------------------------------------------------------- #
# Worley / Voronoi cellular — F1 ≤ F2, periodicity, billow brightness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn,dim", [(noise.worley2, 2), (noise.worley3, 3)])
def test_worley_f1_le_f2_and_nonnegative(fn, dim):
    x, y, z = _grid(_PERIOD)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    f1, f2 = fn(X, Y, _PERIOD) if dim == 2 else fn(X, Y, Z, _PERIOD)
    assert f1.min() >= 0.0
    assert np.all(f1 <= f2 + 1e-6)


def test_worley_exact_phi_periodicity():
    x, y, z = _grid(_PERIOD)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    a1, a2 = noise.worley3(X, Y, Z, _PERIOD)
    b1, b2 = noise.worley3(X, Y + _PERIOD, Z, _PERIOD)
    assert np.array_equal(a1, b1)
    assert np.array_equal(a2, b2)


def test_voronoi_billow_bright_cores_in_unit_range():
    x, y, _ = _grid(_PERIOD)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vb = noise.voronoi_billow2(X, Y, _PERIOD, k=4.0)
    # exp(-k F1) ∈ (0, 1]; equals 1 only exactly at a feature point (unlikely here)
    assert vb.min() > 0.0 and vb.max() <= 1.0
    assert np.array_equal(vb, noise.voronoi_billow2(X, Y + _PERIOD, _PERIOD, k=4.0))


def test_cell_wall_is_f2_minus_f1():
    x, y, _ = _grid(_PERIOD)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f1, f2 = noise.worley2(X, Y, _PERIOD)
    assert np.allclose(noise.cell_wall2(X, Y, _PERIOD), f2 - f1, atol=1e-6)


# --------------------------------------------------------------------------- #
# §3.6 Simplex (Perlin/Gustavson) — the isotropic basis for the V3 curl potential.
#
# Unlike the cubic-lattice family these are NOT lattice-periodic (the input skew
# couples the axes), so they carry no φ-periodicity guard — that is precisely why
# they are a library basis, not wired into the φ-periodic disk stack (noise.py
# §3.6). Their reason to exist is the isotropy guard at the bottom of this section:
# simplex has no axis-aligned grid bias the square-lattice Perlin basis leaks.
# --------------------------------------------------------------------------- #
def _simplex_grid(n: int = 24):
    """A non-periodic (x, y, z) sample grid (simplex needs no integer φ-period)."""
    return (np.linspace(-4.1, 5.3, n), np.linspace(-3.7, 6.1, n),
            np.linspace(-2.9, 3.3, n))


@pytest.mark.parametrize("fn,dim", [(noise.snoise2, 2), (noise.snoise3, 3)])
def test_snoise_in_unit_range(fn, dim):
    """Simplex maps to [0, 1] (the Gustavson 70/32 normalizers + _to01 clamp) and is
    not a constant field."""
    x, y, z = _simplex_grid()
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    n = fn(X, Y, 0) if dim == 2 else fn(X, Y, Z, 0)
    assert n.dtype == np.float32
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert n.std() > 0.05


def test_snoise_deterministic():
    """Pure function of inputs + seed (PCG hash, no global RNG; CKS-12 constraint 7)."""
    x, y, z = _simplex_grid()
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    assert np.array_equal(noise.snoise3(X, Y, Z, 0, seed=4),
                          noise.snoise3(X, Y, Z, 0, seed=4))


def test_snoise_seed_changes_field():
    x, y, _ = _simplex_grid()
    X, Y = np.meshgrid(x, y, indexing="ij")
    assert not np.array_equal(noise.snoise2(X, Y, 0, seed=1),
                              noise.snoise2(X, Y, 0, seed=2))


def test_sfbm_single_octave_equals_base():
    """One octave of the simplex fBm is exactly the base simplex noise (the octave
    machinery is shared with the cubic-lattice fBm)."""
    x, y, _ = _simplex_grid()
    X, Y = np.meshgrid(x, y, indexing="ij")
    one = noise.sfbm2(X, Y, 0, octaves=1, gain=0.5)
    base = noise.snoise2(X, Y, 0)
    assert np.allclose(one, base, atol=1e-6)


def test_sfbm_in_unit_range_3d():
    x, y, z = _simplex_grid()
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    n = noise.sfbm3(X, Y, Z, 0, octaves=5, lacunarity=2, gain=0.5)
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert n.std() > 0.03


def test_snoise_is_continuous():
    """Simplex noise is C2 — the radial corner kernel (r₀²−|d|²)₊⁴ vanishes smoothly
    at the support boundary, so there is no lattice-edge discontinuity. Sampled on a
    fine line (step 0.01), the largest adjacent jump must stay far below the field's
    own [0, 1] span (a discontinuity would spike this)."""
    t = np.arange(0.0, 30.0, 0.01, dtype=np.float32)
    line = noise.snoise2(t, 0.37 * t + 1.1, 0, seed=3)
    assert np.abs(np.diff(line)).max() < 0.05


def _m4_anisotropy(field: np.ndarray) -> float:
    """Normalized amplitude of the cos(4θ)/sin(4θ) angular harmonic of the noise
    power spectrum within a mid radial band. A **square lattice** (Perlin) is
    strongly 4-fold symmetric ⇒ large; a hexagonal **simplex** lattice is not ⇒
    near zero. This is the quantitative signature of the axis-aligned grid bias."""
    f = field - field.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    n = P.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    ky = yy - c
    kx = xx - c
    rad = np.hypot(kx, ky)
    th = np.arctan2(ky, kx)
    band = (rad >= 0.06 * n) & (rad <= 0.30 * n)  # skip DC + extreme highs
    w = P[band]
    w = w / w.sum()
    t = th[band]
    c4 = float((w * np.cos(4 * t)).sum())
    s4 = float((w * np.sin(4 * t)).sum())
    return np.hypot(c4, s4)


def test_simplex_more_isotropic_than_perlin():
    """THE acceptance for V1.5: simplex removes the axis-aligned grid artifact of the
    square-lattice Perlin basis. At a matched feature scale (10 cells across a 256²
    grid) the 4-fold angular anisotropy of the power spectrum is ~12× smaller for
    simplex than for Perlin gradient noise (measured: Perlin ≈ 0.52, simplex ≈ 0.04,
    stable across seeds). The margins below are conservative against that gap."""
    n = 256
    cells = 10
    g = np.linspace(0.0, cells, n, endpoint=False).astype(np.float32)
    X, Y = np.meshgrid(g, g, indexing="ij")
    perlin = np.mean([_m4_anisotropy(noise.gnoise2(X, Y, cells, seed=s)) for s in (1, 2, 3)])
    simplex = np.mean([_m4_anisotropy(noise.snoise2(X, Y, 0, seed=s)) for s in (1, 2, 3)])
    assert perlin > 0.30, perlin           # the square lattice IS strongly 4-fold
    assert simplex < 0.15, simplex         # simplex is near-isotropic at m=4
    assert simplex < 0.4 * perlin, (simplex, perlin)  # ≥2.5× more isotropic


# --------------------------------------------------------------------------- #
# §4 combined layer stack — noise_density_mult (CPU source of truth, D2.2)
# --------------------------------------------------------------------------- #
_NZ = {
    "m_max": 2.5,
    "layers": {
        "base": {"enabled": True, "amp": 0.6, "octaves": 5, "lacunarity": 2,
                 "gain": 0.5, "freq_u": 6.0, "freq_phi": 24},
        "clump": {"enabled": True, "amp": 1.2, "bias": 0.35, "octaves": 3,
                  "lacunarity": 2, "gain": 0.5, "freq_u": 3.0, "freq_phi": 12,
                  "freq_z": 1.0, "coverage": 0.45, "mask_freq_u": 1.0,
                  "mask_freq_phi": 3, "ridge_offset": 1.0, "voronoi_k": 4.0},
        "patch": {"enabled": True, "amp": 0.35, "octaves": 2, "lacunarity": 2,
                  "gain": 0.5, "freq_u": 1.5, "freq_phi": 4},
    },
}


def _stack_grid(n: int = 20):
    return (np.linspace(0.0, 3.0, n), np.linspace(-np.pi, np.pi, n),
            np.linspace(-2.5, 2.5, n))


def test_density_mult_positive_and_bounded():
    """ρ-multiplier = exp(clamp(m, ±m_max)) — strictly > 0 (CKS-12 §3, ρ stays > 0)
    and bounded by the m_max clamp."""
    u, phi, zeta = _stack_grid()
    d = noise.noise_density_mult(u, phi, zeta, _NZ, seed=7)
    assert d.dtype == np.float32
    assert np.all(d > 0.0)
    assert np.all(d <= np.exp(2.5) + 1e-3)
    assert np.all(d >= np.exp(-2.5) - 1e-6)


def test_density_mult_disabled_is_identity():
    """All layers off ⇒ multiplier ≡ 1.0 exactly — matches the kernel's skipped
    branch (CKS-12 constraint 6, bit-identical legacy disk)."""
    u, phi, zeta = _stack_grid()
    d = noise.noise_density_mult(u, phi, zeta, {"m_max": 2.5, "layers": {}}, seed=7)
    assert np.array_equal(d, np.ones_like(d))


def test_density_mult_deterministic():
    """Pure function of inputs + seed (CKS-12 constraint 7, no global RNG)."""
    u, phi, zeta = _stack_grid()
    a = noise.noise_density_mult(u, phi, zeta, _NZ, seed=3)
    b = noise.noise_density_mult(u, phi, zeta, _NZ, seed=3)
    assert np.array_equal(a, b)


def test_density_mult_no_phi_seam():
    """No seam at φ = 0: stack(φ) ≈ stack(φ + 2π). Each lattice is exactly
    2π-periodic (pinned per-primitive above); composing + exp leaves only float
    rounding of the (φ + 2π) coordinate, so the visible-seam guarantee is continuity
    (~1e-3 after the exp), matching the renderer's arbitrary-φ inputs."""
    u, phi, zeta = _stack_grid()
    a = noise.noise_density_mult(u, phi, zeta, _NZ, seed=5)
    b = noise.noise_density_mult(u, phi + 2.0 * np.pi, zeta, _NZ, seed=5)
    assert np.allclose(a, b, atol=1e-3)


# --------------------------------------------------------------------------- #
# §2 Keplerian shear advection — dual-phase reset blend (D2.3, CKS-12 §2)
# --------------------------------------------------------------------------- #
def _adv_grid(n: int = 2000):
    """Random (u, φ, ζ, Ω) sample — Ω(r) is per-point as in the kernel."""
    rng = np.random.default_rng(0)
    return (
        rng.uniform(0.0, 3.0, n).astype(np.float32),
        rng.uniform(-np.pi, np.pi, n).astype(np.float32),
        rng.uniform(-2.5, 2.5, n).astype(np.float32),
        rng.uniform(0.05, 0.4, n).astype(np.float32),
    )


def test_advection_static_when_shear_period_zero():
    """``shear_period ≤ 0`` (no disk.dynamics block) ⇒ the static D2.2 path, exactly
    — t_disk/Ω are ignored, so every legacy static caller stays bit-identical."""
    u, phi, zeta, omega = _adv_grid()
    static = noise.noise_density_mult(u, phi, zeta, _NZ, seed=9)
    # Same call, now passing t_disk/omega but shear_period still 0 → must not change.
    same = noise.noise_density_mult(u, phi, zeta, _NZ, seed=9,
                                    t_disk=12.3, omega=omega, shear_period=0.0)
    assert np.array_equal(static, same)


def test_advection_engages_and_evolves():
    """With shear_period > 0 the field advects: it differs from the static field
    (the dual-phase blend is engaged at t=0) and evolves in time."""
    u, phi, zeta, omega = _adv_grid()
    T = 4.0
    static = noise.noise_density_mult(u, phi, zeta, _NZ, seed=9)
    d0 = noise.noise_density_mult(u, phi, zeta, _NZ, seed=9,
                                  t_disk=0.0, omega=omega, shear_period=T)
    d1 = noise.noise_density_mult(u, phi, zeta, _NZ, seed=9,
                                  t_disk=0.7 * T, omega=omega, shear_period=T)
    assert not np.allclose(d0, static)  # advection path actually taken
    assert not np.allclose(d0, d1)      # pattern moves with t_disk


def test_advection_deterministic():
    """Same (seed, t_disk, Ω, T) ⇒ identical (constraint 7, no global RNG)."""
    u, phi, zeta, omega = _adv_grid()
    kw = dict(seed=7, t_disk=1.3 * 4.0, omega=omega, shear_period=4.0)
    a = noise.noise_density_mult(u, phi, zeta, _NZ, **kw)
    b = noise.noise_density_mult(u, phi, zeta, _NZ, **kw)
    assert np.array_equal(a, b)


def test_advection_continuous_across_resets():
    """No pop at a phase reset. Each phase's triangle weight is exactly 0 at its own
    reset (c_k increment), so the per-cycle reseed swaps in at zero weight and the
    blended field is C0-continuous — the whole point of the staggered reset (CKS-12
    §2). Sampling finely across two periods (which contains every reset of both
    phases), the largest single-step change must not be an outlier vs the median."""
    u, phi, zeta, omega = _adv_grid(512)
    T = 5.0
    ts = np.linspace(0.0, 2.0 * T, 401)  # spans resets at t = T (phase 0) and T/2, 3T/2 (phase 1)
    fields = np.stack([
        noise.noise_density_mult(u, phi, zeta, _NZ, seed=11,
                                 t_disk=float(t), omega=omega, shear_period=T)
        for t in ts
    ])
    step = np.abs(np.diff(fields, axis=0)).max(axis=1)
    assert step.max() < 10.0 * np.median(step)  # a reset pop would spike this


def test_variance_preserve_restores_contrast():
    """``variance_preserve`` divides the blend by √(w₀²+w₁²) so the mid-crossfade
    contrast does not sag (CKS-12 §2). At s = 0.25 the weights are equal (0.5/0.5),
    so the preserved field has the larger spatial spread of log-density."""
    u, phi, zeta, omega = _adv_grid()
    T = 4.0
    t_mid = 0.25 * T  # w0 = w1 = 0.5 (maximal crossfade dilution)
    nz_on = {**_NZ, "variance_preserve": True}
    nz_off = {**_NZ, "variance_preserve": False}
    on = noise.noise_density_mult(u, phi, zeta, nz_on, seed=5,
                                  t_disk=t_mid, omega=omega, shear_period=T)
    off = noise.noise_density_mult(u, phi, zeta, nz_off, seed=5,
                                   t_disk=t_mid, omega=omega, shear_period=T)
    assert np.log(on).std() > np.log(off).std()


def test_dynamism_unit_gain_is_bit_identical():
    """``dynamism: 1.0`` (and an omitted key) reproduce the CKS-12 §2 formula exactly —
    the viz gain is opt-in and the default path is unchanged bit-for-bit."""
    u, phi, zeta, omega = _adv_grid()
    T = 4.0
    kw = dict(seed=7, t_disk=1.3 * T, omega=omega, shear_period=T)
    base = noise.noise_density_mult(u, phi, zeta, _NZ, **kw)
    unit = noise.noise_density_mult(u, phi, zeta, {**_NZ, "dynamism": 1.0}, **kw)
    assert np.array_equal(base, unit)


def test_dynamism_gain_emphasises_winding():
    """A gain > 1 scales the shear (φ′ = φ − dynamism·Ω·a·T), so the field is pushed
    further from the unsheared (static) field than gain = 1 is — i.e. the dial makes
    the swirl read stronger for a given frame."""
    u, phi, zeta, omega = _adv_grid()
    T = 4.0
    t = 0.3 * T
    static = noise.noise_density_mult(u, phi, zeta, _NZ, seed=5)
    g1 = noise.noise_density_mult(u, phi, zeta, {**_NZ, "dynamism": 1.0}, seed=5,
                                  t_disk=t, omega=omega, shear_period=T)
    g4 = noise.noise_density_mult(u, phi, zeta, {**_NZ, "dynamism": 4.0}, seed=5,
                                  t_disk=t, omega=omega, shear_period=T)
    dev1 = np.abs(np.log(g1) - np.log(static)).mean()
    dev4 = np.abs(np.log(g4) - np.log(static)).mean()
    assert dev4 > dev1


# --------------------------------------------------------------------------- #
# §3 modulation envelopes — noise_modulation_fields (D2.4, CKS-12 §3)
# --------------------------------------------------------------------------- #
_MOD = {
    **_NZ,
    "dynamism": 1.0,
    "modulation": {
        "enabled": True, "octaves": 3, "lacunarity": 2, "gain": 0.5,
        "freq_u": 4.0, "freq_phi": 16, "temp_amp": 0.6, "edge_in_amp": 0.3,
        "edge_out_amp": 0.2, "edge_softness": 0.4, "height_amp": 0.5,
    },
}


def test_modulation_disabled_is_half():
    """No ``modulation`` block (or ``enabled: false``) ⇒ every envelope is the no-op
    midpoint 0.5 (n − ½ = 0 ⇒ identity), matching the kernel's skipped §3 branch."""
    u, phi, zeta = _stack_grid()
    for nz in (_NZ, {**_MOD, "modulation": {**_MOD["modulation"], "enabled": False}}):
        fields = noise.noise_modulation_fields(u, phi, zeta, nz, seed=7)
        assert len(fields) == 4
        for f in fields:
            assert f.dtype == np.float32
            assert np.array_equal(f, np.full_like(f, 0.5))


def test_modulation_fields_in_unit_range():
    """The four envelopes are fBm in [0,1]; the advected dual-phase blend (convex
    triangle weights, no variance_preserve) stays in [0,1] so n − ½ is a bounded ±½."""
    u, phi, zeta, omega = _adv_grid()
    for kw in ({}, {"t_disk": 1.7 * 4.0, "omega": omega, "shear_period": 4.0}):
        fields = noise.noise_modulation_fields(u, phi, zeta, _MOD, seed=5, **kw)
        for f in fields:
            assert f.min() >= -1e-5 and f.max() <= 1.0 + 1e-5


def test_modulation_envelopes_decorrelated():
    """The four envelopes use distinct seed offsets ⇒ they are different fields
    (a shared field would make T/edge/height move in lockstep)."""
    u, phi, zeta = _stack_grid()
    nT, nein, neout, nh = noise.noise_modulation_fields(u, phi, zeta, _MOD, seed=5)
    assert not np.array_equal(nT, nein)
    assert not np.array_equal(nT, nh)
    assert not np.array_equal(nein, neout)


def test_modulation_advects_and_is_deterministic():
    """Advection moves the envelopes with t_disk, and the result is a pure function
    of (inputs, seed, t_disk) — no global RNG (constraint 7)."""
    u, phi, zeta, omega = _adv_grid()
    T = 4.0
    kw = dict(seed=9, omega=omega, shear_period=T)
    a = noise.noise_modulation_fields(u, phi, zeta, _MOD, t_disk=0.2 * T, **kw)
    b = noise.noise_modulation_fields(u, phi, zeta, _MOD, t_disk=0.2 * T, **kw)
    c = noise.noise_modulation_fields(u, phi, zeta, _MOD, t_disk=0.7 * T, **kw)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))  # deterministic
    assert not np.array_equal(a[0], c[0])                   # evolves with t_disk


# --------------------------------------------------------------------------- #
# §3.7 / CKS-18 — curl-flow domain warp (V3.0)
#
# An in-plane, divergence-free distortion of (u, φ) = the 2-D curl of an sfbm3
# scalar potential on the (cosφ, sinφ, u) cylinder embedding. These pin the three
# defining properties: divergence-free (∇·displacement ≈ 0), seamless across φ=0,
# and deterministic; plus the zero-amp identity (the disabled bit-identical path).
# --------------------------------------------------------------------------- #
def _warp_grid(n=40):
    """A dense (u, φ) grid; φ spans a full turn [−π, π) so the seam is exercised."""
    u = np.linspace(-1.0, 3.0, n).astype(np.float32)
    phi = np.linspace(-np.pi, np.pi, n, endpoint=False).astype(np.float32)
    return u, phi


_CURL_KW = dict(amp=0.15, freq_phi=3.0, freq_u=1.3, octaves=4,
                lacunarity=2, gain=0.5, seed=1337)


def test_curl_warp_zero_amp_is_identity():
    """amp = 0 ⇒ the warp returns the coords unchanged (the disabled / bit-identical
    path; CKS-12 constraint 6)."""
    u, phi = _warp_grid()
    U, P = np.meshgrid(u, phi, indexing="ij")
    u_w, phi_w = noise.curl_warp(U, P, amp=0.0, freq_phi=3.0, freq_u=1.3, seed=1337)
    assert np.array_equal(u_w, U.astype(np.float32))
    assert np.array_equal(phi_w, P.astype(np.float32))


def test_curl_warp_is_divergence_free():
    """The displacement ``(δu, δφ) = (+∂ψ/∂φ, −∂ψ/∂u)`` is the 2-D curl of a scalar
    potential, so ``∇·(δu,δφ) ≡ ψ_uφ − ψ_uφ = 0`` (mixed partials commute) — the
    defining property of the curl construction.

    Measured HONESTLY with one consistent operator (``np.gradient``) on the same
    potential ``ψ`` curl_warp uses: build the curl field and a generic GRADIENT field
    ``(∂ψ/∂u, ∂ψ/∂φ)`` (NOT divergence-free — its divergence is the Laplacian), then
    show the curl field's discrete divergence is orders of magnitude smaller. (A bare
    FD on curl_warp's f32 output mixes the warp's internal ε-step with the grid step
    and would conflate truncation with the property — so we test the construction.)"""
    rho, ku, seed, oct_, lac, gain = 3.0, 1.3, 1337, 4, 2, 0.5
    n = 96
    u = np.linspace(0.0, 2.0, n)
    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    U, P = np.meshgrid(u, phi, indexing="ij")
    psi = noise.sfbm3(np.cos(P) * rho, np.sin(P) * rho, U * ku, 0,
                      octaves=oct_, lacunarity=lac, gain=gain, seed=seed).astype(np.float64)
    hu, hp = u[1] - u[0], phi[1] - phi[0]
    dpsi_du, dpsi_dphi = np.gradient(psi, hu, hp, edge_order=2)

    def _divergence(fu, fphi):
        d_u = np.gradient(fu, hu, axis=0, edge_order=2)
        d_phi = np.gradient(fphi, hp, axis=1, edge_order=2)
        return (d_u + d_phi)[2:-2, 2:-2]  # trim edges (one-sided stencils)

    div_curl = _divergence(dpsi_dphi, -dpsi_du)   # ∇·curl  → ~0
    div_grad = _divergence(dpsi_du, dpsi_dphi)    # ∇·grad = ∇²ψ → O(1)
    assert np.abs(div_curl).max() < 1e-3 * np.abs(div_grad).max(), (
        np.abs(div_curl).max(), np.abs(div_grad).max())


def test_curl_warp_is_seamless_in_phi():
    """The displacement is built on cos φ / sin φ ⇒ it is a globally-continuous,
    exactly 2π-periodic function of φ — so the warped coordinate has no jump across
    the seam (CKS-12 constraint 5; periodicity of a continuous field IS seamlessness).
    Tested as ``displacement(φ) ≈ displacement(φ + 2π)`` over a generic φ range.

    A coarser ``fd_eps`` is used here so the FD curl's ``1/2ε`` factor does not amplify
    the ~6e-7 ``cos(φ+2π)`` argument-reduction roundoff (at the shipped ε=1e-3 that
    inflates the residual to a sub-pixel ~1e-4 — an f32 artifact of comparing trig a
    full turn apart, not a real seam). The exact ±π atan2-wrap point is excluded for
    the same reason (f32 ``sin(±π)`` ≈ ±9e-8)."""
    u = np.linspace(-1.0, 3.0, 24).astype(np.float32)
    phi = np.linspace(-2.0, 2.0, 24).astype(np.float32)
    U, P = np.meshgrid(u, phi, indexing="ij")
    kw = {**_CURL_KW, "fd_eps": 0.05}
    u_a, phi_a = noise.curl_warp(U, P, **kw)
    u_b, phi_b = noise.curl_warp(U, (P + 2.0 * np.pi).astype(np.float32), **kw)
    assert np.allclose(u_a - U, u_b - U, atol=3e-5)
    assert np.allclose(phi_a - P, phi_b - (P + 2.0 * np.pi), atol=3e-5)


def test_curl_warp_is_deterministic_and_seed_sensitive():
    """Pure function of (coords, seed) — no global RNG (constraint 7) — and a
    different seed gives a genuinely different warp."""
    u, phi = _warp_grid()
    U, P = np.meshgrid(u, phi, indexing="ij")
    a = noise.curl_warp(U, P, **_CURL_KW)
    b = noise.curl_warp(U, P, **_CURL_KW)
    c = noise.curl_warp(U, P, **{**_CURL_KW, "seed": 4242})
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_curl_warp_actually_moves_the_density_stack():
    """Wired through the layer stack: an enabled curl block changes the density
    multiplier vs the same config with curl off — and curl-off is the bit-identical
    no-warp path."""
    u, phi, zeta = _stack_grid()
    nz_off = _NZ
    nz_on = {**_NZ, "curl": {"enabled": True, "amp": 0.2, "freq_phi": 3.0,
                             "freq_u": 1.3, "seed": 1337}}
    nz_disabled = {**_NZ, "curl": {"enabled": False, "amp": 0.2}}
    m_off = noise.noise_density_mult(u, phi, zeta, nz_off, seed=7)
    m_on = noise.noise_density_mult(u, phi, zeta, nz_on, seed=7)
    m_dis = noise.noise_density_mult(u, phi, zeta, nz_disabled, seed=7)
    assert not np.array_equal(m_off, m_on)            # the warp does something
    assert np.array_equal(m_off, m_dis)              # enabled:false ⇒ bit-identical


# --------------------------------------------------------------------------- #
# §3.7 / CKS-18 §2 — curl-flow advection (V3.1): time-dependent ψ.
#
# The static V3.0 potential ψ becomes time-dependent via the SAME CKS-12 §2
# dual-phase reset blend (ω_k triangle weights, per-cycle reseed). These pin: the
# static-fallback bit-identity (flow_period ≤ 0 ⇒ V3.0 exactly — the regression
# hook), divergence-free AT EACH t (curl is linear so the blend preserves it),
# seamless at each t, C0-continuity through reseeds (ω_k → 0 at each reset), and
# evolution + determinism. Same honest np.gradient methodology as the V3.0 tests.
# --------------------------------------------------------------------------- #
_CURL_FLOW_KW = dict(amp=0.15, freq_phi=3.0, freq_u=1.3, octaves=4,
                     lacunarity=2, gain=0.5, seed=1337)


def _blended_psi(U, P, t_disk, flow_period, rho=3.0, ku=1.3, seed=1337,
                 oct_=4, lac=2, gain=0.5):
    """Reconstruct the time-blended potential ``ψ = Σ ω_k ψ_k`` exactly as the
    ``flow_period > 0`` branch of :func:`noise.curl_warp` builds it — so a test can
    apply a consistent ``np.gradient`` operator to the SAME potential the warp curls
    (the honest divergence methodology, lifted to the time axis)."""
    sc = np.float32(t_disk) / np.float32(flow_period)
    out = np.zeros(np.broadcast(U, P).shape, dtype=np.float64)
    for k in (0, 1):
        ar = sc + np.float32(0.5 * k)
        ck = int(np.floor(ar))
        ak = ar - np.float32(ck)
        wk = np.float32(1.0) - np.abs(np.float32(2.0) * ak - np.float32(1.0))
        seed_k = int(seed) + k * noise.NCYC_PHASE + ck * noise.NCYC_CYCLE
        out += float(wk) * noise.sfbm3(np.cos(P) * rho, np.sin(P) * rho, U * ku, 0,
                                       octaves=oct_, lacunarity=lac, gain=gain,
                                       seed=seed_k).astype(np.float64)
    return out


def test_curl_flow_static_fallback_is_bit_identical():
    """``flow_period ≤ 0`` (and an absent clock) ⇒ the V3.0 static warp BIT-FOR-BIT,
    for any ``t_disk`` — the regression hook (mirror of CKS-12 §2's static fallback).
    This is what keeps the V3.0 / default-off goldens valid."""
    u, phi = _warp_grid()
    U, P = np.meshgrid(u, phi, indexing="ij")
    static = noise.curl_warp(U, P, **_CURL_FLOW_KW)                       # no clock
    for t in (0.0, 3.7, 100.0):
        zero = noise.curl_warp(U, P, t_disk=t, flow_period=0.0, **_CURL_FLOW_KW)
        neg = noise.curl_warp(U, P, t_disk=t, flow_period=-5.0, **_CURL_FLOW_KW)
        assert np.array_equal(static[0], zero[0]) and np.array_equal(static[1], zero[1])
        assert np.array_equal(static[0], neg[0]) and np.array_equal(static[1], neg[1])


def test_curl_flow_is_divergence_free_at_each_time():
    """The blend ``ψ = ω_0ψ_0 + ω_1ψ_1`` is still a scalar potential, so its curl
    ``(+∂ψ/∂φ, −∂ψ/∂u)`` is divergence-free at every ``t_disk`` (curl is linear ⇒ a
    convex combination of div-free fields is div-free). Same honest operator as the
    V3.0 test: build the curl field and a generic gradient field from the SAME blended
    ψ, show the curl divergence is orders of magnitude smaller."""
    n, rho, ku = 96, 3.0, 1.3
    u = np.linspace(0.0, 2.0, n)
    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    U, P = np.meshgrid(u, phi, indexing="ij")
    hu, hp = u[1] - u[0], phi[1] - phi[0]
    for t in (0.0, 1.3, 4.1):
        psi = _blended_psi(U, P, t, flow_period=6.0, rho=rho, ku=ku)
        dpsi_du, dpsi_dphi = np.gradient(psi, hu, hp, edge_order=2)

        def _divergence(fu, fphi):
            d_u = np.gradient(fu, hu, axis=0, edge_order=2)
            d_phi = np.gradient(fphi, hp, axis=1, edge_order=2)
            return (d_u + d_phi)[2:-2, 2:-2]

        div_curl = _divergence(dpsi_dphi, -dpsi_du)
        div_grad = _divergence(dpsi_du, dpsi_dphi)
        assert np.abs(div_curl).max() < 1e-3 * np.abs(div_grad).max(), (
            t, np.abs(div_curl).max(), np.abs(div_grad).max())


def test_curl_flow_is_seamless_at_each_time():
    """The blended displacement is still built on ``cos φ`` / ``sin φ`` ⇒ exactly
    2π-periodic in φ at every ``t_disk`` (seamlessness survives the time blend). Coarse
    ``fd_eps`` as in the V3.0 seam test so the ``1/2ε`` factor doesn't amplify the f32
    ``cos(φ+2π)`` argument-reduction roundoff."""
    u = np.linspace(-1.0, 3.0, 24).astype(np.float32)
    phi = np.linspace(-2.0, 2.0, 24).astype(np.float32)
    U, P = np.meshgrid(u, phi, indexing="ij")
    kw = {**_CURL_FLOW_KW, "fd_eps": 0.05, "flow_period": 6.0}
    for t in (0.0, 1.7, 3.3):
        u_a, phi_a = noise.curl_warp(U, P, t_disk=t, **kw)
        u_b, phi_b = noise.curl_warp(U, (P + 2.0 * np.pi).astype(np.float32), t_disk=t, **kw)
        assert np.allclose(u_a - U, u_b - U, atol=3e-5), t
        assert np.allclose(phi_a - P, phi_b - (P + 2.0 * np.pi), atol=3e-5), t


def test_curl_flow_is_c0_continuous_through_resets():
    """``ω_k → 0`` exactly at each reset (``α_k = 0``), so the warp is continuous as the
    potential reseeds (``γ_k`` steps) — the proven §2 property, now on the time axis.
    With ``flow_period = 1`` the phases reset every ``Δs = 0.5`` (k=0 at integer t, k=1
    at half-integer t); a fine ``t`` sweep across those boundaries must show NO jump —
    the max adjacent step stays the same order at a reset as away from it (a seed swap
    with O(1) weight would spike by O(amp))."""
    u = np.linspace(-0.5, 2.5, 20).astype(np.float32)
    phi = np.linspace(-np.pi, np.pi, 20, endpoint=False).astype(np.float32)
    U, P = np.meshgrid(u, phi, indexing="ij")
    ts = np.linspace(0.0, 2.0, 401)  # Δt = 0.005; crosses resets at 0.5,1.0,1.5,2.0
    warps = np.stack([noise.curl_warp(U, P, t_disk=float(t), flow_period=1.0,
                                      **_CURL_FLOW_KW)[0] for t in ts])
    steps = np.abs(np.diff(warps, axis=0)).reshape(len(ts) - 1, -1).max(axis=1)
    # Continuity is scale-free: NO adjacent-in-time step is an outlier vs the typical
    # boil step. A reseed discontinuity (ω≠0 seed swap) would spike ONE step to O(amp)
    # ≈ 0.15 ≈ 6× the median; a continuous reseed (ω→0 at the reset) keeps every step
    # within a small factor. The observed max lands AWAY from the resets (the uniform
    # boil rate), not on them.
    assert steps.max() < 3.0 * np.median(steps), (
        steps.max(), float(np.median(steps)), float(ts[steps.argmax()]))


def test_curl_flow_evolves_and_is_deterministic():
    """The warp boils: it changes between distinct ``t_disk`` (stays finite), and is a
    pure function of ``(coords, seed, t_disk)`` — same ``t`` ⇒ identical (no global
    RNG, constraint 7)."""
    u, phi = _warp_grid()
    U, P = np.meshgrid(u, phi, indexing="ij")
    a = noise.curl_warp(U, P, t_disk=0.0, flow_period=5.0, **_CURL_FLOW_KW)
    b = noise.curl_warp(U, P, t_disk=2.5, flow_period=5.0, **_CURL_FLOW_KW)
    b2 = noise.curl_warp(U, P, t_disk=2.5, flow_period=5.0, **_CURL_FLOW_KW)
    assert not np.array_equal(a[0], b[0])                  # evolves over time
    assert np.isfinite(b[0]).all() and np.isfinite(b[1]).all()
    assert np.array_equal(b[0], b2[0]) and np.array_equal(b[1], b2[1])  # deterministic


# --------------------------------------------------------------------------- #
# CKS-19 multi-phase media — _advected_m refactor parity + dust modulator.
# --------------------------------------------------------------------------- #
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


def _dust_mp(chi):
    return {"enabled": True, "dust_correlation": chi, "dust_amp": 1.0, "dust_sigma_frac": 1.0}

_DUST_NZ = {
    "m_max": 8.0,  # loose clamp so the linear-correlation construction is visible
    "layers": {"base": {"enabled": True, "amp": 1.0, "octaves": 4,
                        "lacunarity": 2, "gain": 0.5, "freq_u": 6.0, "freq_phi": 24}},
}

def _dust_grid():
    uu, pp = np.meshgrid(np.linspace(0.05, 3.0, 96, dtype=np.float32),
                         np.linspace(-np.pi, np.pi, 96, dtype=np.float32), indexing="ij")
    return uu.ravel(), pp.ravel(), np.zeros(uu.size, dtype=np.float32)

@pytest.mark.parametrize("chi", [-1.0, -0.6, 0.0, 0.6, 1.0])
def test_dust_correlation_matches_chi(chi):
    from renderer import noise as N
    u, phi, zeta = _dust_grid()
    m_hot = N._advected_m(u, phi, zeta, _DUST_NZ, seed=7)
    rho_cold = N.dust_density_mult(u, phi, zeta, _DUST_NZ, _dust_mp(chi), seed=7)
    m_cold = np.log(rho_cold)            # a_cold=1 ⇒ log ρ_cold == clamp(m_cold)
    r = np.corrcoef(m_hot, m_cold)[0, 1]
    assert abs(r - chi) < 0.05, f"chi={chi}: Pearson r={r}"

def test_dust_variance_is_chi_invariant():
    from renderer import noise as N
    u, phi, zeta = _dust_grid()
    var = [np.var(np.log(N.dust_density_mult(u, phi, zeta, _DUST_NZ, _dust_mp(c), seed=7)))
           for c in (-1.0, -0.6, 0.0, 0.6, 1.0)]
    assert max(var) / min(var) < 1.15, f"variance breathes across chi: {var}"

def test_dust_chi_plus_one_is_hot_modulator():
    from renderer import noise as N
    u, phi, zeta = _dust_grid()
    rho_hot = N.noise_density_mult(u, phi, zeta, _DUST_NZ, seed=7)
    rho_cold = N.dust_density_mult(u, phi, zeta, _DUST_NZ, _dust_mp(1.0), seed=7)
    np.testing.assert_allclose(rho_cold, rho_hot, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# CKS-22 — Kelvin-Helmholtz edge erosion (kh_field N_KH + kh_erode_winout clip)
# --------------------------------------------------------------------------- #
def test_kh_field_in_unit_range():
    # N_KH is a convex (triangle-weight) blend of sfbm3 layers ⇒ stays in [0, 1].
    u = np.linspace(0.0, 1.0, 64)
    phi = np.linspace(-np.pi, np.pi, 64)
    n = noise.kh_field(u, phi, np.zeros_like(u), t_disk=3.0, omega=0.05,
                       shear_T=10.0, dynamism=1.0, freq_u=4.0, freq_phi=12,
                       freq_z=1.0, octaves=3, seed=1234)
    assert np.all(n >= 0.0) and np.all(n <= 1.0)


def test_kh_field_phi_seamless():
    # Seamless across φ = ±π via the CKS-18 cylinder embedding (cos/sin φ), NOT a
    # lattice period (classic simplex is non-periodic — SKILL v1.23 / constraint 5).
    a = noise.kh_field(np.array([0.5]), np.array([-np.pi + 1e-6]), np.array([0.0]),
                       3.0, 0.05, 10.0, 1.0, 4.0, 12, 1.0, 3, 1234)
    b = noise.kh_field(np.array([0.5]), np.array([np.pi - 1e-6]), np.array([0.0]),
                       3.0, 0.05, 10.0, 1.0, 4.0, 12, 1.0, 3, 1234)
    assert abs(float(a.reshape(-1)[0]) - float(b.reshape(-1)[0])) < 1e-3


def test_kh_field_static_path_in_range():
    # shear_T <= 0 ⇒ single static layer (no advection); still a valid [0,1] envelope.
    u = np.linspace(0.0, 1.0, 32)
    phi = np.linspace(-np.pi, np.pi, 32)
    n = noise.kh_field(u, phi, np.zeros_like(u), t_disk=0.0, omega=0.0,
                       shear_T=0.0, dynamism=1.0, freq_u=4.0, freq_phi=12,
                       freq_z=1.0, octaves=3, seed=1234)
    assert np.all(n >= 0.0) and np.all(n <= 1.0)


def test_kh_erode_interior_immune():
    # win_out == 1 (disk interior) stays 1 when strength <= 1 - w_soft (the clamp bound).
    win = np.ones(32)
    n = np.linspace(0.0, 1.0, 32)
    w_soft, strength = 0.15, 0.85  # strength == 1 - w_soft
    out = noise.kh_erode_winout(win, n, strength, w_soft)
    assert np.allclose(out, 1.0, atol=1e-6)


def test_kh_erode_tears_band():
    # win_out in the transition band (0.3) with high noise tears to 0 (vacuum).
    win = np.full(8, 0.3)
    n = np.full(8, 1.0)
    out = noise.kh_erode_winout(win, n, strength=0.85, w_soft=0.15)
    assert np.all(out < 1e-6)  # 0.3 - 0.85 < 0 ⇒ clipped to 0


# --------------------------------------------------------------------------- #
# CKS-23 — fractal LOD octave cascade (gates the CKS-12 fBm density octaves)
# --------------------------------------------------------------------------- #
def test_fbm2_lod_full_matches_fbm2():
    # n_oct >= octaves ⇒ every gate g_o = 1 ⇒ fbm2_lod is fbm2 byte-for-byte
    # (the LOD-off / shadow-bake sentinel path leaves the goldens unshifted).
    x = np.linspace(0.0, 7.0, 96)
    y = np.linspace(0.0, 5.0, 96)
    full = noise.fbm2(x, y, period=4, octaves=4, seed=77)
    lod = noise.fbm2_lod(x, y, period=4, n_oct=1.0e9, octaves=4, seed=77)
    assert np.array_equal(full, lod)


def test_fbm2_lod_integer_equals_truncated():
    # At integer n_oct = k the gates are 1 for o < k and 0 for o >= k, so the
    # gated stack (numerator AND denominator) equals fbm2 truncated to k octaves.
    x = np.linspace(0.0, 7.0, 96)
    y = np.linspace(0.0, 5.0, 96)
    trunc = noise.fbm2(x, y, period=4, octaves=2, seed=77)
    lod = noise.fbm2_lod(x, y, period=4, n_oct=2.0, octaves=5, seed=77)
    assert np.array_equal(trunc, lod)


def test_lod_octave_weight_crossfade():
    # g_o = clamp(n_oct - o, 0, 1): the top octave fades linearly (no integer pop).
    assert float(noise.lod_octave_weight(1.5, 0)) == 1.0   # fully resolved
    assert abs(float(noise.lod_octave_weight(1.5, 1)) - 0.5) < 1e-7  # crossfading
    assert float(noise.lod_octave_weight(1.5, 2)) == 0.0   # culled
    # Monotone non-increasing in o at fixed n_oct.
    g = [float(noise.lod_octave_weight(2.3, o)) for o in range(5)]
    assert all(g[i] >= g[i + 1] for i in range(len(g) - 1))


def test_lod_noct_monotone_and_clamped():
    # n_oct = clamp(N_max - log2(eps*d/J0), N_min, N_max): drops one octave per
    # footprint doubling, clamped to [N_min, N_max].
    eps, j0, n_max, n_min = 1.0e-3, 0.5, 6.0, 2.0
    d = np.array([1.0, 10.0, 100.0, 1.0e3, 1.0e5], np.float32)
    n = noise.lod_noct(d, j0, n_max, n_min, eps)
    assert np.all(np.diff(n) <= 1e-6)              # non-increasing with distance
    assert np.all(n >= n_min - 1e-6) and np.all(n <= n_max + 1e-6)
    # At J = J0 (eps*d == j0) the cascade is exactly Nyquist ⇒ n_oct == N_max.
    d0 = j0 / eps
    assert abs(float(noise.lod_noct(d0, j0, n_max, n_min, eps)) - n_max) < 1e-5
    # One footprint doubling beyond J0 drops exactly one octave.
    assert abs(float(noise.lod_noct(2.0 * d0, j0, n_max, n_min, eps)) - (n_max - 1.0)) < 1e-5


# --------------------------------------------------------------------------- #
# CKS-21 — scale-dependent shear cascade (frequency-dependent shear transfer)
# --------------------------------------------------------------------------- #
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


def test_fbm2_octave1_net_shear_strictly_smaller_than_octave0():
    # The differential / cascade: octave 1 (higher f) is sheared LESS than octave 0.
    period = 4
    f_c, p = np.float32(3.0), np.float32(2.0)
    s0 = noise.shear_transfer(np.float32(period * 1), f_c, p)   # octave 0 (freq=1)
    s1 = noise.shear_transfer(np.float32(period * 2), f_c, p)   # octave 1 (freq=2)
    # Net shear applied to octave o is S(f_o)·shear_k (intuitive form φ′ = φ − S·shear_k),
    # so the higher-frequency octave keeps MORE of its position — less net shear.
    assert s1 < s0


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
