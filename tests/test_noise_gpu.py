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
