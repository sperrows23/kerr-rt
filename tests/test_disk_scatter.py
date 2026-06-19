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

    # The scatter func references _sample_shadow_tau (under self_shadow==1); the field
    # must exist for the @ti.func to compile even though the probe runs with shadow OFF.
    tr._setup_disk_shadow({"disk": {"r_inner": r_inner, "r_outer": r_outer}})
    sigma0, beta = 0.15, 0.0
    absb_c, albedo, hg_g, ds = 0.8, 0.5, 0.6, 0.1
    src = (1.0, 0.7, 0.4)

    # Sample on the midplane (z=0) at r=8, φ=0.6; camera far on +x so ŝ_view is known.
    r = 8.0
    phi = 0.6
    x, y, z = r * math.cos(phi), r * math.sin(phi), 0.0
    cx, cy, cz = 60.0, 0.0, 0.0

    # grey_dtau = absb_c·ρ_cold·ds is what the emission march hands in; ρ_cold = 1.0 here
    # (bare Gaussian peak at the midplane, noise off) ⇒ the scatter func re-derives
    # σ_dτ = albedo·grey_dtau = albedo·κ·ρ·ds WITHOUT re-evaluating the density stack.
    rho = 1.0
    grey_dtau = absb_c * rho * ds

    out = ti.field(ti.f32, shape=4)

    @ti.kernel
    def probe():
        sc = tr._disk_scatter_cks(
            x, y, z, cx, cy, cz, a, r_inner,
            sigma0, beta, 0, 0.0,
            grey_dtau, albedo, hg_g,
            tr.vec3(src[0], src[1], src[2]),
        )
        for c in ti.static(range(4)):
            out[c] = sc[c]

    probe()
    got = out.to_numpy()

    # σ_dτ = albedo·grey_dtau = albedo·κ·ρ·ds.
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


# --------------------------------------------------------------------------- #
# End-to-end beauty-frame scenes (mirror tests/test_disk_noise.py's render path:
# the canonical frame-0 edge-on camera + the production render_beauty_frame).
# --------------------------------------------------------------------------- #
import json  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_CAMERA_PATH = _ROOT / "camera_matrix.json"
_SC_W, _SC_H = 480, 270


def _cam():
    if not _CAMERA_PATH.exists():
        pytest.skip(f"camera_matrix.json not found at {_CAMERA_PATH}")
    with open(_CAMERA_PATH, encoding="utf-8-sig") as fh:
        return json.load(fh)[0]


def _scatter_scene():
    """Edge-on, backlit cloudy disk: the canonical frame-0 camera looks across a
    ρ_cold dust field lit from behind by the hot inner edge, so the camera-side
    cloud edges face forward-scatter (cosθ_s→+1). Noise ON (structured dust),
    multiphase ON (a decoupled ρ_cold absorber), self_shadow ON (τ_src defined)."""
    from renderer import taichi_renderer as tr
    cam = _cam()                                   # also gates the skip on the camera file
    cfg = copy.deepcopy(tr.load_config())
    # The scatter-ON beauty mega-kernel is enormous; with Taichi's IR/CFG opt
    # passes ON a cold compile exceeds 80 min / tens of GB. These passes only buy
    # kernel RUNTIME speed (irrelevant for a one-shot test) so disable them here —
    # both render paths (OFF vs ON) use the same setting, keeping the comparison
    # valid. Production keeps the YAML defaults (true).
    cfg["render"]["advanced_optimization"] = False
    cfg["render"]["cfg_optimization"] = False
    d = cfg["disk"]
    d["temperature_model"] = "simple"
    d.setdefault("noise", {})["enabled"] = True
    d["absorption_coeff"] = 2.0
    d["multiphase"] = {"enabled": True, "dust_correlation": -0.8, "dust_amp": 1.5,
                       "dust_sigma_frac": 1.0}
    vol = d.setdefault("volumetric", {})
    ss = vol.setdefault("self_shadow", {})
    ss["enabled"] = True
    ss["lod_max_camera_radius"] = 0.0              # keep self_shadow active at this camera
    return cfg, cam


def _render_scatter(cfg, cam):
    from renderer import taichi_renderer as tr
    tr.setup_renderer(cfg)                          # re-inits ti + sets _SCATTER_COMPILE
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    return tr.render_beauty_frame(cfg, cam, _SC_W, _SC_H, with_disk=True, lod_enabled=False)


@pytest.mark.gpu
def test_scatter_albedo_zero_identical():
    """scatter.enabled:true but albedo:0 ⇒ σ_s=0 ⇒ frame identical to scatter OFF."""
    _cuda_or_skip()
    base, cam = _scatter_scene()
    cfg_off = copy.deepcopy(base)
    cfg_off["disk"].pop("scatter", None)
    cfg_on0 = copy.deepcopy(base)
    cfg_on0["disk"]["scatter"] = {"enabled": True, "albedo": 0.0, "hg_g": 0.6, "inner_glow": 1.0}
    img_off = _render_scatter(cfg_off, cam)
    img_on0 = _render_scatter(cfg_on0, cam)
    np.testing.assert_allclose(img_on0, img_off, rtol=0, atol=1e-6)


@pytest.mark.gpu
def test_scatter_rim_light():
    """CKS-20 single-scatter end-to-end on the canonical edge-on camera.

    Asserts the physically-true ratified behaviour (NOT a net-brighten — for this scene
    σ_s ADDS opacity and single-scatter only re-injects the inner-glow bounce):
      (A) scatter ON injects in-scattered light into an appreciable region;
      (B) that brightening lands on DIM cloud edges (rim light), not the bright core;
      (C) the HG phase is DIRECTIONAL — forward (g=+0.6) vs backward (g=−0.6) give
          markedly different rim patterns;
      (D) σ_s is carried in the extinction (SKILL.md CKS-20 constraint 2) — adding
          scattering REMOVES forward light, so the frame net-darkens for this config.

    NOTE: on this canonical edge-on camera most visible dust is FRONT-lit (cosθ_s<0), so
    the broad back-scatter haze dominates the aggregate and the forward 'silver-lining'
    is a smaller localized limb (empirically g=−0.6 brightens ~25× more area than g=+0.6;
    see the SKILL.md CKS-20 'Aggregate vs limb' scene note). The cosθ_s/HG kernel matches
    the spec exactly — test_disk_scatter_cks_analytic pins it — so this is a property of
    the SCENE geometry, not the code."""
    _cuda_or_skip()
    base, cam = _scatter_scene()
    cfg_off = copy.deepcopy(base); cfg_off["disk"].pop("scatter", None)
    cfg_fwd = copy.deepcopy(base)
    cfg_fwd["disk"]["scatter"] = {"enabled": True, "albedo": 0.6, "hg_g": 0.6, "inner_glow": 2.0}
    cfg_back = copy.deepcopy(base)
    cfg_back["disk"]["scatter"] = {"enabled": True, "albedo": 0.6, "hg_g": -0.6, "inner_glow": 2.0}

    lum_off = _render_scatter(cfg_off, cam).sum(axis=2)
    lum_fwd = _render_scatter(cfg_fwd, cam).sum(axis=2)     # forward HG lobe
    lum_back = _render_scatter(cfg_back, cam).sum(axis=2)   # backward HG lobe (same cached kernel)

    # (A) in-scatter adds light to an appreciable region.
    brighter = lum_fwd > lum_off + 1e-4
    assert brighter.mean() > 0.01, "scatter injected no appreciable in-scattered light"

    # (B) the brightening is rim-light on DIM edges, not the bright emission core.
    db = (lum_fwd - lum_off).ravel()
    top = np.argsort(db)[-100:]                             # the 100 most-brightened pixels
    assert np.median(lum_off.ravel()[top]) < 0.1 * lum_off.max(), \
        "rim brightening should land on dim edges, not the bright core"

    # (C) the HG phase is directional: forward vs backward differ markedly.
    pos_fwd = float(np.clip(lum_fwd - lum_off, 0.0, None).sum())
    pos_back = float(np.clip(lum_back - lum_off, 0.0, None).sum())
    assert abs(pos_fwd - pos_back) > 0.5 * max(pos_fwd, pos_back), \
        "HG phase produced no directional difference between forward and backward"

    # (D) σ_s is in the extinction (constraint 2): scattering removes forward light.
    assert lum_fwd.sum() < lum_off.sum(), \
        "σ_s extinction missing — scattering must remove forward light (SKILL.md CKS-20 constraint 2)"
