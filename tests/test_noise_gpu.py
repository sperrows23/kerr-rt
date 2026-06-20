"""D2.1 — CPU↔GPU agreement guard for the noise primitives (gpu-marked).

``src/renderer/noise.py`` is the NumPy source of truth; this module holds its
``@ti.func`` twins to it on CUDA, exactly as ``test_gpu_regression`` does for the
physics kernels. The backend is LOCKED to ``ti.init(arch=ti.cuda)`` (CLAUDE.md);
on a host without a working CUDA backend the whole module skips cleanly.

Asserts (spec §8):
  * every twin matches the NumPy primitive to ~1e-6 on a shared sample grid,
  * the lattice path is exactly φ-periodic on the GPU too,
  * the twins are deterministic (no ``ti.random``): same args ⇒ identical output.

NOTE: this module must NOT ``from __future__ import annotations`` — Taichi resolves
the ``f32 = ti.types.ndarray(...)`` kernel-arg annotations as live type objects, and
PEP 563 would stringify them into unresolvable names (``TaichiSyntaxError``).
"""
# ti.types.ndarray() kernel-arg annotations read as variables by pyright:
# pyright: reportInvalidTypeForm=false
import numpy as np
import pytest

from renderer import noise

pytestmark = pytest.mark.gpu

_ATOL = 2e-6  # f32 twin vs f32 NumPy reference; integer hash path is exact


@pytest.fixture(scope="module")
def ti_cuda():
    """Init the LOCKED CUDA backend or skip (mirrors test_gpu_regression)."""
    import taichi as ti

    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without a CUDA driver
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")
    return ti


# --------------------------------------------------------------------------- #
# ndarray-arg kernels: evaluate each twin over a flattened grid into ``out``.
# Defined at import (compiled lazily on first call, after ti.init).
# --------------------------------------------------------------------------- #
def _kernels(ti):
    f32 = ti.types.ndarray(dtype=ti.f32, ndim=1)

    @ti.kernel
    def k_gnoise2(x: f32, y: f32, out: f32, period: ti.i32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.gnoise2_ti(x[i], y[i], period, seed)

    @ti.kernel
    def k_gnoise3(x: f32, y: f32, z: f32, out: f32, period: ti.i32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.gnoise3_ti(x[i], y[i], z[i], period, seed)

    @ti.kernel
    def k_fbm2(x: f32, y: f32, out: f32, period: ti.i32, oct: ti.i32, lac: ti.i32,
               gain: ti.f32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.fbm2_ti(x[i], y[i], period, oct, lac, gain, seed)

    @ti.kernel
    def k_fbm3(x: f32, y: f32, z: f32, out: f32, period: ti.i32, oct: ti.i32, lac: ti.i32,
               gain: ti.f32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.fbm3_ti(x[i], y[i], z[i], period, oct, lac, gain, seed)

    @ti.kernel
    def k_fbm2_lod(x: f32, y: f32, out: f32, period: ti.i32, oct: ti.i32, lac: ti.i32,
                   gain: ti.f32, seed: ti.i32, n_oct: ti.f32):
        for i in range(x.shape[0]):
            out[i] = noise.fbm2_lod_ti(x[i], y[i], period, oct, lac, gain, seed, n_oct)

    @ti.kernel
    def k_billow3(x: f32, y: f32, z: f32, out: f32, period: ti.i32, oct: ti.i32,
                  lac: ti.i32, gain: ti.f32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.billow3_ti(x[i], y[i], z[i], period, oct, lac, gain, seed)

    @ti.kernel
    def k_ridged3(x: f32, y: f32, z: f32, out: f32, period: ti.i32, oct: ti.i32,
                  lac: ti.i32, gain: ti.f32, offset: ti.f32, feedback: ti.f32,
                  seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.ridged3_ti(x[i], y[i], z[i], period, oct, lac, gain,
                                      offset, feedback, seed)

    @ti.kernel
    def k_worley3(x: f32, y: f32, z: f32, f1: f32, f2: f32, period: ti.i32, seed: ti.i32):
        for i in range(x.shape[0]):
            d = noise.worley3_ti(x[i], y[i], z[i], period, seed)
            f1[i] = d[0]
            f2[i] = d[1]

    @ti.kernel
    def k_voronoi2(x: f32, y: f32, out: f32, period: ti.i32, k: ti.f32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.voronoi_billow2_ti(x[i], y[i], period, k, seed)

    @ti.kernel
    def k_snoise2(x: f32, y: f32, out: f32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.snoise2_ti(x[i], y[i], 0, seed)

    @ti.kernel
    def k_snoise3(x: f32, y: f32, z: f32, out: f32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.snoise3_ti(x[i], y[i], z[i], 0, seed)

    @ti.kernel
    def k_sfbm3(x: f32, y: f32, z: f32, out: f32, oct: ti.i32, lac: ti.i32,
                gain: ti.f32, seed: ti.i32):
        for i in range(x.shape[0]):
            out[i] = noise.sfbm3_ti(x[i], y[i], z[i], 0, oct, lac, gain, seed)

    @ti.kernel
    def k_curl(u: f32, phi: f32, u_out: f32, phi_out: f32, amp: ti.f32, fp: ti.f32,
               fu: ti.f32, oct: ti.i32, lac: ti.i32, gain: ti.f32, seed: ti.i32,
               eps: ti.f32, t_disk: ti.f32, flow_period: ti.f32):
        for i in range(u.shape[0]):
            w = noise.curl_warp_ti(u[i], phi[i], amp, fp, fu, oct, lac, gain, seed, eps,
                                   t_disk, flow_period)
            u_out[i] = w[0]
            phi_out[i] = w[1]

    return locals()


# --------------------------------------------------------------------------- #
# Shared grid (matches tests/test_noise.py dyadic-φ grid)
# --------------------------------------------------------------------------- #
_PERIOD = 6


def _grid(n=17):
    x = np.linspace(-3.3, 4.7, n)
    y = (np.arange(n) * ((_PERIOD * 16) // n)) / 16.0
    z = np.linspace(-1.9, 2.1, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return (X.ravel().astype(np.float32), Y.ravel().astype(np.float32),
            Z.ravel().astype(np.float32))


def _out(n):
    return np.empty(n, dtype=np.float32)


def test_gnoise2_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    out = _out(x.size)
    K["k_gnoise2"](x, y, out, _PERIOD, 7)
    ref = noise.gnoise2(x, y, _PERIOD, seed=7)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


def test_gnoise3_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    out = _out(x.size)
    K["k_gnoise3"](x, y, z, out, _PERIOD, 7)
    ref = noise.gnoise3(x, y, z, _PERIOD, seed=7)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


def test_gnoise3_twin_is_phi_periodic(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    a, b = _out(x.size), _out(x.size)
    K["k_gnoise3"](x, y, z, a, _PERIOD, 3)
    K["k_gnoise3"](x, (y + _PERIOD).astype(np.float32), z, b, _PERIOD, 3)
    assert np.array_equal(a, b)


def test_gnoise3_twin_is_deterministic(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    a, b = _out(x.size), _out(x.size)
    K["k_gnoise3"](x, y, z, a, _PERIOD, 5)
    K["k_gnoise3"](x, y, z, b, _PERIOD, 5)
    assert np.array_equal(a, b)


def test_fbm2_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    out = _out(x.size)
    K["k_fbm2"](x, y, out, _PERIOD, 5, 2, 0.5, 1)
    ref = noise.fbm2(x, y, _PERIOD, octaves=5, lacunarity=2, gain=0.5, seed=1)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


def test_fbm3_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    out = _out(x.size)
    K["k_fbm3"](x, y, z, out, _PERIOD, 5, 2, 0.5, 1)
    ref = noise.fbm3(x, y, z, _PERIOD, octaves=5, lacunarity=2, gain=0.5, seed=1)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


def test_fbm2_lod_twin_matches_reference(ti_cuda):
    # CKS-23: gated fBm GPU twin == CPU noise.fbm2_lod at a partial (crossfading) n_oct.
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    out = _out(x.size)
    K["k_fbm2_lod"](x, y, out, _PERIOD, 5, 2, 0.5, 1, 2.6)
    ref = noise.fbm2_lod(x, y, _PERIOD, n_oct=2.6, octaves=5, lacunarity=2, gain=0.5, seed=1)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


def test_fbm2_lod_twin_off_is_fbm2(ti_cuda):
    # n_oct ≥ octaves ⇒ every gate=1 ⇒ the gated twin is fbm2_ti byte-for-byte (the
    # LOD-off / shadow-bake sentinel path leaves the golden frames unshifted, constraint 6).
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    lod_off = _out(x.size)
    plain = _out(x.size)
    K["k_fbm2_lod"](x, y, lod_off, _PERIOD, 5, 2, 0.5, 1, 1.0e9)
    K["k_fbm2"](x, y, plain, _PERIOD, 5, 2, 0.5, 1)
    assert np.array_equal(lod_off, plain)


def test_billow3_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    out = _out(x.size)
    K["k_billow3"](x, y, z, out, _PERIOD, 4, 2, 0.5, 2)
    ref = noise.billow3(x, y, z, _PERIOD, octaves=4, lacunarity=2, gain=0.5, seed=2)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


def test_ridged3_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    out = _out(x.size)
    K["k_ridged3"](x, y, z, out, _PERIOD, 3, 2, 0.5, 1.0, 2.0, 4)
    ref = noise.ridged3(x, y, z, _PERIOD, octaves=3, lacunarity=2, gain=0.5,
                        offset=1.0, feedback=2.0, seed=4)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


def test_worley3_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    f1, f2 = _out(x.size), _out(x.size)
    K["k_worley3"](x, y, z, f1, f2, _PERIOD, 9)
    r1, r2 = noise.worley3(x, y, z, _PERIOD, seed=9)
    assert np.allclose(f1, r1, atol=_ATOL), np.abs(f1 - r1).max()
    assert np.allclose(f2, r2, atol=_ATOL), np.abs(f2 - r2).max()


def test_voronoi_billow2_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    out = _out(x.size)
    K["k_voronoi2"](x, y, out, _PERIOD, 4.0, 9)
    ref = noise.voronoi_billow2(x, y, _PERIOD, k=4.0, seed=9)
    assert np.allclose(out, ref, atol=_ATOL), np.abs(out - ref).max()


# --------------------------------------------------------------------------- #
# §3.6 Simplex twins (V1.5). Non-periodic, so no φ-periodicity guard — the twin
# fidelity (skew/floor/branch all on identical f32 inputs) and determinism are the
# invariants. A touch looser than _ATOL: the per-corner radial kernel piles a few
# more f32 ops on than the cubic lattice, so 1-ULP spread can reach ~1e-5.
# --------------------------------------------------------------------------- #
_SATOL = 1e-5


def test_snoise2_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    out = _out(x.size)
    K["k_snoise2"](x, y, out, 7)
    ref = noise.snoise2(x, y, 0, seed=7)
    assert np.allclose(out, ref, atol=_SATOL), np.abs(out - ref).max()


def test_snoise3_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    out = _out(x.size)
    K["k_snoise3"](x, y, z, out, 7)
    ref = noise.snoise3(x, y, z, 0, seed=7)
    assert np.allclose(out, ref, atol=_SATOL), np.abs(out - ref).max()


def test_snoise3_twin_is_deterministic(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    a, b = _out(x.size), _out(x.size)
    K["k_snoise3"](x, y, z, a, 5)
    K["k_snoise3"](x, y, z, b, 5)
    assert np.array_equal(a, b)


def test_sfbm3_twin_matches_reference(ti_cuda):
    K = _kernels(ti_cuda)
    x, y, z = _grid()
    out = _out(x.size)
    K["k_sfbm3"](x, y, z, out, 5, 2, 0.5, 1)
    ref = noise.sfbm3(x, y, z, 0, octaves=5, lacunarity=2, gain=0.5, seed=1)
    assert np.allclose(out, ref, atol=_SATOL), np.abs(out - ref).max()


def test_curl_warp_twin_matches_reference(ti_cuda):
    """CKS-18 curl warp GPU twin (``curl_warp_ti``) matches the NumPy source of truth
    (``curl_warp``) — same 4-point central-difference stencil.

    Tolerance is NOT ``_SATOL``: the warp is a *derivative* of the noise, dividing
    ``(ψ₊ − ψ₋)`` by ``2·fd_eps``. The GPU/CPU ``sfbm3`` twins agree only to ``_SATOL``
    (FMA / transcendental ordering), so the displacement ``amp·(ψ₊−ψ₋)/(2ε)`` inherits a
    worst-case twin error of ``amp·_SATOL/ε``. Asserting ``_SATOL`` on a ``1/(2ε)``-amplified
    quantity is the wrong expectation (observed ~6.5e-5); the bound below is derived from
    the call's own amp/ε so it cannot desync."""
    amp, fd_eps = 0.15, 1e-3
    curl_atol = amp * _SATOL / fd_eps  # = 1.5e-3 worst-case; observed ~6.5e-5
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    u_out, phi_out = _out(x.size), _out(x.size)
    # flow_period = 0 ⇒ the static V3.0 warp (curl-flow off); twin of the default path.
    K["k_curl"](x, y, u_out, phi_out, amp, 3.0, 1.3, 4, 2, 0.5, 1337, fd_eps, 0.0, 0.0)
    ru, rp = noise.curl_warp(x, y, amp=amp, freq_phi=3.0, freq_u=1.3, octaves=4,
                             lacunarity=2, gain=0.5, seed=1337, fd_eps=fd_eps)
    assert np.allclose(u_out, ru, atol=curl_atol), np.abs(u_out - ru).max()
    assert np.allclose(phi_out, rp, atol=curl_atol), np.abs(phi_out - rp).max()


def test_curl_flow_twin_matches_reference(ti_cuda):
    """CKS-18 §2 curl-flow advection GPU twin: with ``flow_period > 0`` the
    time-dependent dual-phase-blended ψ warp (``curl_warp_ti``) still matches the NumPy
    source of truth (``curl_warp``) at several ``t_disk`` — verifies the §2 reseed
    strides (``NCYC_PHASE``/``NCYC_CYCLE``) and triangle weights agree across the twin.

    Same derived ``amp·_SATOL/ε`` bound as the static test — the blend is a convex
    combination of the same per-phase derivatives, so the FD-amplification analysis is
    unchanged (the ``ω_k`` weights are exact f32 on both sides)."""
    amp, fd_eps, T_c = 0.15, 1e-3, 5.0
    curl_atol = amp * _SATOL / fd_eps
    K = _kernels(ti_cuda)
    x, y, _z = _grid()
    u_out, phi_out = _out(x.size), _out(x.size)
    for t in (0.0, 1.7, 6.3):  # spans within-cycle and a reseed (s_c crosses 1)
        K["k_curl"](x, y, u_out, phi_out, amp, 3.0, 1.3, 4, 2, 0.5, 1337, fd_eps, float(t), T_c)
        ru, rp = noise.curl_warp(x, y, amp=amp, freq_phi=3.0, freq_u=1.3, octaves=4,
                                 lacunarity=2, gain=0.5, seed=1337, fd_eps=fd_eps,
                                 t_disk=float(t), flow_period=T_c)
        assert np.allclose(u_out, ru, atol=curl_atol), (t, np.abs(u_out - ru).max())
        assert np.allclose(phi_out, rp, atol=curl_atol), (t, np.abs(phi_out - rp).max())


# --------------------------------------------------------------------------- #
# CKS-22 — KH edge-erosion field _kh_field GPU twin (reads disk_noise_params, so
# the param buffer is populated via _setup_disk_noise rather than kernel args).
# --------------------------------------------------------------------------- #
def _kh_cfg(enabled=True, strength=0.3, shear_T=10.0, dynamism=1.0):
    return {"disk": {
        "theta_half_width": 0.15, "vertical_sigma_frac": 0.2, "r_outer": 25.0,
        "max_step_vfrac": 0.5,
        "dynamics": {"shear_period_M": shear_T},
        "noise": {"dynamism": dynamism, "modulation": {"edge_softness": 0.4}},
        "edge_erosion": {"enabled": enabled, "strength": strength,
                         "freq_u": 4.0, "freq_phi": 12, "freq_z": 1.0, "octaves": 3},
    }}


def test_kh_field_gpu_matches_cpu(ti_cuda):
    """CKS-22 _kh_field GPU twin matches noise.kh_field (the §2-advected, cylinder-
    embedded simplex N_KH). Tolerance above _SATOL: the φ embedding multiplies the
    cos/sin twin error by freq_phi before sfbm3, on top of the base sfbm3 twin gap."""
    ti = ti_cuda
    from renderer import taichi_renderer as tr

    tr._setup_disk_noise(_kh_cfg())  # populates disk_noise_params (EROS + SHEAR_T + DYNAMISM)

    N = 48
    out = ti.field(ti.f32, shape=N)
    us = np.linspace(0.1, 0.9, N).astype(np.float32)
    ph = np.linspace(-3.0, 3.0, N).astype(np.float32)

    @ti.kernel
    def fill(us: ti.types.ndarray(dtype=ti.f32, ndim=1),
             ph: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(N):
            out[i] = tr._kh_field(us[i], ph[i], 0.0, 3.0, 0.05, 1234)

    fill(us, ph)
    ref = noise.kh_field(us, ph, np.zeros(N, np.float32), t_disk=3.0, omega=0.05,
                         shear_T=10.0, dynamism=1.0, freq_u=4.0, freq_phi=12,
                         freq_z=1.0, octaves=3, seed=1234)
    got = out.to_numpy()
    assert np.all(got >= -1e-6) and np.all(got <= 1.0 + 1e-6), (got.min(), got.max())
    assert np.allclose(got, ref, atol=1e-4), np.abs(got - ref).max()


def test_kh_field_gpu_static_path(ti_cuda):
    """shear_T <= 0 ⇒ the GPU static branch (single sfbm3 layer at φ) matches the CPU."""
    ti = ti_cuda
    from renderer import taichi_renderer as tr

    tr._setup_disk_noise(_kh_cfg(shear_T=0.0))
    N = 40
    out = ti.field(ti.f32, shape=N)
    us = np.linspace(0.1, 0.9, N).astype(np.float32)
    ph = np.linspace(-3.0, 3.0, N).astype(np.float32)

    @ti.kernel
    def fill(us: ti.types.ndarray(dtype=ti.f32, ndim=1),
             ph: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(N):
            out[i] = tr._kh_field(us[i], ph[i], 0.0, 0.0, 0.0, 1234)

    fill(us, ph)
    ref = noise.kh_field(us, ph, np.zeros(N, np.float32), t_disk=0.0, omega=0.0,
                         shear_T=0.0, dynamism=1.0, freq_u=4.0, freq_phi=12,
                         freq_z=1.0, octaves=3, seed=1234)
    assert np.allclose(out.to_numpy(), ref, atol=1e-4), np.abs(out.to_numpy() - ref).max()


# --------------------------------------------------------------------------- #
# CKS-21 — shear cascade param buffer packing (no kernel; setup smoke test)
# --------------------------------------------------------------------------- #
def test_setup_packs_shear_cascade_params(ti_cuda):
    from renderer import taichi_renderer as tr
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
