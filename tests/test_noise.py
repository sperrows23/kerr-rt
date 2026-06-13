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
