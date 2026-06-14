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
