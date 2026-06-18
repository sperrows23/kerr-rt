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


@pytest.mark.gpu
def test_disk_scatter_cks_analytic():
    """With noise OFF (ρ_cold = bare Gaussian) and self_shadow OFF (e^{−τ_src}=1),
    _disk_scatter_cks returns σ_dτ = albedo·κ·ρ·ds and J = σ_dτ·P(cosθ_s)·src_rgb
    for a hand-placed sample — pins the σ_s=ϖ·κ assembly and the HG/geometry."""
    _cuda_or_skip()
    import taichi as ti
    from renderer import taichi_renderer as tr
    from renderer.disk import hg_phase

    a = 0.999
    r_inner, r_outer = 4.0, 25.0

    # Single-phase scene, noise + shadow OFF ⇒ _disk_density_cks midplane density is
    # the bare Gaussian gauss(dz_ang/σ)=1 at ζ=0, edge-window≈1 well inside the band.
    tr._setup_disk_noise({"disk": {}})          # noise off ⇒ ρ_cold = bare Gaussian
    # The scatter func references _sample_shadow_tau (under self_shadow==1); the field
    # must exist for the @ti.func to compile even though the probe runs with shadow OFF.
    tr._setup_disk_shadow({"disk": {"r_inner": r_inner, "r_outer": r_outer}})
    sigma0, beta = 0.15, 0.0
    theta_half = 0.3
    absb_c, albedo, hg_g, ds = 0.8, 0.5, 0.6, 0.1
    src = (1.0, 0.7, 0.4)

    # Sample on the midplane (z=0) at r=8, φ=0.6; camera far on +x so ŝ_view is known.
    r = 8.0
    phi = 0.6
    x, y, z = r * math.cos(phi), r * math.sin(phi), 0.0
    cx, cy, cz = 60.0, 0.0, 0.0

    out = ti.field(ti.f32, shape=4)

    @ti.kernel
    def probe():
        sc = tr._disk_scatter_cks(
            x, y, z, cx, cy, cz, a, r_inner, r_outer, 4.0, theta_half,
            sigma0, beta, 0, 1234, 0.0, 0, 0.0, absb_c, ds, albedo, hg_g,
            tr.vec3(src[0], src[1], src[2]),
        )
        for c in ti.static(range(4)):
            out[c] = sc[c]

    probe()
    got = out.to_numpy()

    # ρ_cold at midplane, noise off ≈ 1.0 (bare Gaussian peak). σ_dτ = albedo·κ·ρ·ds.
    rho = 1.0
    sigma_dtau = albedo * absb_c * rho * ds
    assert abs(got[3] - sigma_dtau) < 1e-4, f"sigma_dtau {got[3]} vs {sigma_dtau}"

    # cosθ_s = ŝ_src·ŝ_view, both straight CKS rays.
    import numpy as _np
    s_src = _np.array([x - r_inner * math.cos(phi), y - r_inner * math.sin(phi), 0.0])
    s_src /= _np.linalg.norm(s_src)
    s_view = _np.array([cx - x, cy - y, cz - z])
    s_view /= _np.linalg.norm(s_view)
    cos_s = float(s_src @ s_view)
    P = float(hg_phase(cos_s, hg_g))
    j_r = sigma_dtau * P * src[0]          # e^{−τ_src}=1 (shadow off)
    assert abs(got[0] - j_r) < 1e-4, f"J_r {got[0]} vs {j_r}"
