"""CKS-20 single-scattering + Henyey-Greenstein tests.

CPU tests run anywhere; GPU tests are CUDA-mandatory (backend LOCKED to ti.cuda)
and skip cleanly without it.
"""
# pyright: reportInvalidTypeForm=false
import copy
import math

import numpy as np
import pytest


def test_hg_phase_normalized():
    """∫_{4π} P(cosθ) dΩ = 1 for representative g (HG is a normalized phase function)."""
    from renderer.disk import hg_phase
    th = np.linspace(0.0, math.pi, 4001)
    cos_th = np.cos(th)
    for g in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
        # ∫ P · 2π sinθ dθ over θ∈[0,π]; trapezoid on a fine grid.
        integ = np.trapz(hg_phase(cos_th, g) * 2.0 * math.pi * np.sin(th), th)
        assert abs(integ - 1.0) < 1e-3, f"g={g}: ∮P dΩ={integ}"


def test_hg_phase_forward_dominant():
    """g=0.6 ⇒ forward (cosθ=+1) > isotropic (cosθ=0) > back (cosθ=−1)."""
    from renderer.disk import hg_phase
    fwd = float(hg_phase(1.0, 0.6))
    iso = float(hg_phase(0.0, 0.6))
    bak = float(hg_phase(-1.0, 0.6))
    assert fwd > iso > bak
    assert abs(float(hg_phase(0.0, 0.0)) - 1.0 / (4.0 * math.pi)) < 1e-6  # g=0 ⇒ 1/4π


pytestmark_gpu = pytest.mark.gpu


def _cuda_or_skip():
    import taichi as ti
    from renderer import taichi_renderer as tr  # noqa: F401
    try:
        ti.init(arch=ti.cuda)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA unavailable: {e}")


@pytest.mark.gpu
def test_hg_phase_gpu_matches_cpu():
    """GPU _hg_phase == CPU hg_phase across cosθ∈[−1,1], g∈{−0.6,0,0.6}."""
    _cuda_or_skip()
    import taichi as ti
    from renderer import taichi_renderer as tr
    from renderer.disk import hg_phase

    cos_vals = np.linspace(-1.0, 1.0, 64).astype(np.float32)
    g_vals = np.array([-0.6, 0.0, 0.6], dtype=np.float32)
    cf = ti.field(ti.f32, shape=cos_vals.size)
    gf = ti.field(ti.f32, shape=g_vals.size)
    out = ti.field(ti.f32, shape=(g_vals.size, cos_vals.size))
    cf.from_numpy(cos_vals)
    gf.from_numpy(g_vals)

    @ti.kernel
    def probe():
        for gi, ci in ti.ndrange(g_vals.size, cos_vals.size):
            out[gi, ci] = tr._hg_phase(cf[ci], gf[gi])

    probe()
    got = out.to_numpy()
    for gi, g in enumerate(g_vals):
        ref = hg_phase(cos_vals, float(g)).astype(np.float32)
        np.testing.assert_allclose(got[gi], ref, rtol=1e-4, atol=1e-5)
