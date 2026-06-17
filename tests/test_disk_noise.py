"""D2.2 — disk procedural-noise wiring guard (gpu-marked; SKILL.md CKS-12).

Drives the production ``render_beauty_frame`` (the same path as
``test_gpu_regression``) at a modest resolution and checks the static
density-modulation wiring:

  * ``disk.noise.enabled: false`` is **bit-identical** to a config with no noise
    block at all — the disabled branch never touches density (CKS-12 constraint 6,
    golden frames untouched);
  * ``enabled: true`` actually changes the disk emission, stays NaN-free, still
    emits, and keeps the g⁴ Doppler beaming (the noise is amplitude-only — it must
    not break the redshift chain);
  * the field is deterministic: same seed + same (static) frame ⇒ identical render,
    and a different seed gives a different pattern (CKS-12 constraint 7, no
    ``ti.random``).

CUDA is mandatory (backend LOCKED to ``ti.init(arch=ti.cuda)`` per CLAUDE.md); the
module skips cleanly without it. Resolution is kept small — correctness of the
branch/determinism does not need FHD, and the disk is still well inside the frame
at the canonical edge-on frame 0.

NOTE: this module must NOT ``from __future__ import annotations`` — the in-test
``@ti.kernel`` (stack-agreement guard) carries a live ``ti.i32`` arg annotation that
PEP 563 would stringify into an unresolvable name (TaichiSyntaxError; the D2.1
gotcha, see tests/test_noise_gpu.py).
"""
# ti.i32 kernel-arg annotation reads as a variable to pyright:
# pyright: reportInvalidTypeForm=false
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from renderer import taichi_renderer as tr

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).resolve().parents[1]
_CAMERA_PATH = _ROOT / "camera_matrix.json"
_FRAME_INDEX = 0
_WIDTH = 480
_HEIGHT = 270


def _ensure_cuda():
    if not _CAMERA_PATH.exists():
        pytest.skip(f"camera_matrix.json not found at {_CAMERA_PATH}")
    import taichi as ti

    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without a CUDA driver
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")


def _cam():
    with open(_CAMERA_PATH, encoding="utf-8-sig") as fh:
        return json.load(fh)[_FRAME_INDEX]


def _render(noise_cfg, *, seed=None, temperature_model="simple", doppler_strength=1.0):
    """Render frame 0 with ``disk.noise`` replaced by ``noise_cfg`` (a dict, or the
    sentinel ``"DELETE"`` to drop the block entirely). Returns (hdr, disk_rgb)."""
    _ensure_cuda()
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = temperature_model
    cfg["disk"]["doppler_strength"] = doppler_strength
    if noise_cfg == "DELETE":
        cfg["disk"].pop("noise", None)
    else:
        cfg["disk"]["noise"] = copy.deepcopy(noise_cfg)
        if seed is not None:
            cfg["disk"]["noise"]["seed"] = seed

    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    hdr = tr.render_beauty_frame(cfg, _cam(), _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)
    disk_rgb = np.nan_to_num(tr.disk_buf.to_numpy()[:, :, :3]).copy()
    return hdr, disk_rgb


# A fully-on noise config (all three layers) for the "changes the disk" checks.
_NOISE_ON = {
    "enabled": True,
    "seed": 1234,
    "m_max": 2.5,
    "layers": {
        "base": {"enabled": True, "amp": 0.6, "octaves": 5, "lacunarity": 2.0,
                 "gain": 0.5, "freq_u": 6.0, "freq_phi": 24},
        "clump": {"enabled": True, "amp": 1.2, "bias": 0.35, "octaves": 3,
                  "lacunarity": 2.0, "gain": 0.5, "freq_u": 3.0, "freq_phi": 12,
                  "freq_z": 1.0, "coverage": 0.45, "mask_freq_u": 1.0,
                  "mask_freq_phi": 3, "ridge_offset": 1.0, "voronoi_k": 4.0},
        "patch": {"enabled": True, "amp": 0.35, "octaves": 2, "lacunarity": 2.0,
                  "gain": 0.5, "freq_u": 1.5, "freq_phi": 4},
    },
}


def _disabled(noise_on):
    off = copy.deepcopy(noise_on)
    off["enabled"] = False
    return off


def test_noise_disabled_is_bit_identical_to_no_block():
    """``enabled: false`` must render byte-for-byte the same as no noise block —
    the disabled branch is skipped, density is untouched (CKS-12 constraint 6)."""
    hdr_off, _ = _render(_disabled(_NOISE_ON))
    hdr_none, _ = _render("DELETE")
    assert np.array_equal(hdr_off, hdr_none)


def test_noise_enabled_changes_the_disk():
    """``enabled: true`` must perturb the disk emission, stay NaN-free, still emit,
    and keep the g⁴ Doppler beaming (amplitude-only modulation)."""
    _, disk_off = _render(_disabled(_NOISE_ON))
    hdr_on, disk_on = _render(_NOISE_ON)

    assert int(np.isnan(hdr_on).sum()) == 0
    assert float(disk_on.max()) > 0.0  # still emits
    # The pattern actually changes the disk (not a no-op branch).
    assert not np.array_equal(disk_on, disk_off)
    # Beaming preserved: approaching (right) half brighter than receding (left).
    lum = disk_on.sum(axis=2)
    w = lum.shape[1]
    left = float(lum[:, : w // 2].mean())
    right = float(lum[:, w // 2 :].mean())
    assert right > left


def test_noise_is_deterministic():
    """Same seed + same static frame ⇒ identical render (no ``ti.random``)."""
    hdr_a, _ = _render(_NOISE_ON, seed=777)
    hdr_b, _ = _render(_NOISE_ON, seed=777)
    assert np.array_equal(hdr_a, hdr_b)


def test_noise_seed_changes_pattern():
    """Different seeds ⇒ different texture (the seed actually reaches the hash)."""
    _, disk_a = _render(_NOISE_ON, seed=1)
    _, disk_b = _render(_NOISE_ON, seed=2)
    assert not np.array_equal(disk_a, disk_b)


def test_noise_stack_matches_cpu_reference():
    """The renderer's GPU layer stack must match the CPU source of truth.

    ``noise.noise_density_mult`` is canonical (spec §4); the kernel's
    ``_disk_noise_density_mult`` is its twin. The per-primitive twins are pinned in
    ``test_noise_gpu.py``; this pins their *composition* plus the ``_setup_disk_noise``
    param-buffer packing (index map ↔ config keys). ti.field I/O is used (not an
    ``ndarray`` kernel arg) so the module's ``from __future__ import annotations``
    cannot stringify a Taichi type annotation (the D2.1 gotcha).
    """
    import taichi as ti

    from renderer import noise

    _ensure_cuda()
    tr._setup_disk_noise({"disk": {"noise": copy.deepcopy(_NOISE_ON)}})
    seed = int(_NOISE_ON["seed"])

    rng = np.random.default_rng(0)
    n = 4096
    u = rng.uniform(0.0, 3.0, n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    zeta = rng.uniform(-2.5, 2.5, n).astype(np.float32)

    uf = ti.field(ti.f32, n)
    pf = ti.field(ti.f32, n)
    zf = ti.field(ti.f32, n)
    of = ti.field(ti.f32, n)
    uf.from_numpy(u)
    pf.from_numpy(phi)
    zf.from_numpy(zeta)

    @ti.kernel
    def k_density(s: ti.i32):
        for i in range(n):
            # t_disk = 0, omega = 0 ⇒ the static path (param buffer has no dynamics
            # block ⇒ _NI_SHEAR_T = 0), matching the static CPU reference below.
            of[i] = tr._disk_noise_density_mult(uf[i], pf[i], zf[i], 0.0, 0.0, s)

    k_density(seed)
    gpu = of.to_numpy()
    cpu = noise.noise_density_mult(u, phi, zeta, _NOISE_ON, seed=seed)
    assert np.allclose(gpu, cpu, rtol=1e-4, atol=5e-4), float(np.abs(gpu - cpu).max())


def test_advected_stack_matches_cpu_reference():
    """D2.3 shear-advection (CKS-12 §2): the kernel's dual-phase reset blend must
    match the CPU reference. Uploading a ``disk.dynamics.shear_period_M`` turns on
    advection (``_NI_SHEAR_T > 0``); the kernel is driven with a per-sample Ω and a
    ``t_disk`` past one reset cycle (``s = t_disk/T = 1.3`` ⇒ ``c_k = 1``), so the
    per-cycle reseed stride and the triangle-weight crossfade are both exercised.
    """
    import taichi as ti

    from renderer import noise

    _ensure_cuda()
    T = 10.0
    t_disk = 1.3 * T  # s = 1.3 → both phases active, reset-cycle index c_k = 1
    tr._setup_disk_noise({"disk": {"noise": copy.deepcopy(_NOISE_ON),
                                   "dynamics": {"shear_period_M": T}}})
    seed = int(_NOISE_ON["seed"])

    rng = np.random.default_rng(1)
    n = 4096
    u = rng.uniform(0.0, 3.0, n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    zeta = rng.uniform(-2.5, 2.5, n).astype(np.float32)
    omega = rng.uniform(0.05, 0.4, n).astype(np.float32)  # Ω(r) per sample

    uf = ti.field(ti.f32, n)
    pf = ti.field(ti.f32, n)
    zf = ti.field(ti.f32, n)
    omf = ti.field(ti.f32, n)
    of = ti.field(ti.f32, n)
    uf.from_numpy(u)
    pf.from_numpy(phi)
    zf.from_numpy(zeta)
    omf.from_numpy(omega)

    @ti.kernel
    def k_adv(s: ti.i32, td: ti.f32):
        for i in range(n):
            of[i] = tr._disk_noise_density_mult(uf[i], pf[i], zf[i], td, omf[i], s)

    k_adv(seed, t_disk)
    gpu = of.to_numpy()
    cpu = noise.noise_density_mult(u, phi, zeta, _NOISE_ON, seed=seed,
                                   t_disk=t_disk, omega=omega, shear_period=T)
    # Looser than the static guard: s/a_k/w_k/φ′ are f32 on both sides but the φ shift
    # (Ω·a·T ~ 1 rad) feeds a high-freq_phi lattice, so f32 rounding spreads a touch.
    assert np.allclose(gpu, cpu, rtol=1e-3, atol=2e-3), float(np.abs(gpu - cpu).max())


def test_curl_flow_advection_matches_cpu_and_animates():
    """CKS-18 §2 curl-flow advection WIRED through the kernel. With a ``disk.noise.curl``
    block carrying ``flow_period_M > 0`` and the STATIC shear path (no ``disk.dynamics``,
    ``_NI_SHEAR_T = 0``), the time evolution comes purely from the curl flow — so this
    isolates the new ``t_disk → _disk_curl_warp`` threading. Checks (a) the kernel's
    ``_disk_noise_density_mult`` matches the CPU source of truth at a given ``t_disk``
    (the §2 reset blend + reseed strides agree end-to-end through ``_setup_disk_noise``),
    and (b) the field actually changes between two ``t_disk`` (the warp boils ⇒ ``t_disk``
    really reaches the GPU warp; a dropped thread would render identical frames)."""
    import taichi as ti

    from renderer import noise

    _ensure_cuda()
    nz = copy.deepcopy(_NOISE_ON)
    nz["curl"] = {"enabled": True, "amp": 0.25, "freq_phi": 3.0, "freq_u": 1.3,
                  "octaves": 4, "lacunarity": 2, "gain": 0.5, "seed": 1337,
                  "flow_period_M": 5.0}
    # No dynamics block ⇒ _NI_SHEAR_T = 0 ⇒ static shear; only the curl flow animates.
    tr._setup_disk_noise({"disk": {"noise": nz}})
    seed = int(nz["seed"])

    rng = np.random.default_rng(2)
    n = 4096
    u = rng.uniform(0.0, 3.0, n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    zeta = rng.uniform(-2.5, 2.5, n).astype(np.float32)

    uf = ti.field(ti.f32, n); pf = ti.field(ti.f32, n); zf = ti.field(ti.f32, n)
    of = ti.field(ti.f32, n)
    uf.from_numpy(u); pf.from_numpy(phi); zf.from_numpy(zeta)

    @ti.kernel
    def k_flow(s: ti.i32, td: ti.f32):
        for i in range(n):
            # omega = 0 / static shear; td drives the CKS-18 §2 curl flow only.
            of[i] = tr._disk_noise_density_mult(uf[i], pf[i], zf[i], td, 0.0, s)

    t_a, t_b = 1.7, 3.9
    k_flow(seed, t_a)
    gpu_a = of.to_numpy()
    cpu_a = noise.noise_density_mult(u, phi, zeta, nz, seed=seed,
                                     t_disk=t_a, omega=0.0, shear_period=0.0)
    # Same FD-amplified curl tolerance family as the advected guard (warp perturbs the
    # lattice coords; GPU/CPU sfbm3 twins agree to ~1e-5, amplified by 1/2ε then fed fBm).
    assert np.allclose(gpu_a, cpu_a, rtol=1e-3, atol=2e-3), float(np.abs(gpu_a - cpu_a).max())

    k_flow(seed, t_b)
    gpu_b = of.to_numpy()
    assert not np.allclose(gpu_a, gpu_b, rtol=1e-3, atol=2e-3)  # the flow boils over t


def test_dynamism_gain_matches_cpu_and_changes_shear():
    """The non-physical ``disk.noise.dynamism`` viz gain (φ′ = φ − dynamism·Ω·a·T)
    must (a) still agree GPU↔CPU at a gain ≠ 1, and (b) actually move the field versus
    gain = 1 (so the dial does something). Same advection setup as the test above.
    """
    import taichi as ti

    from renderer import noise

    _ensure_cuda()
    T = 10.0
    t_disk = 1.3 * T
    nz_gain = copy.deepcopy(_NOISE_ON)
    nz_gain["dynamism"] = 4.0
    tr._setup_disk_noise({"disk": {"noise": nz_gain,
                                   "dynamics": {"shear_period_M": T}}})
    seed = int(nz_gain["seed"])

    rng = np.random.default_rng(2)
    n = 4096
    u = rng.uniform(0.0, 3.0, n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    zeta = rng.uniform(-2.5, 2.5, n).astype(np.float32)
    omega = rng.uniform(0.05, 0.4, n).astype(np.float32)

    uf = ti.field(ti.f32, n)
    pf = ti.field(ti.f32, n)
    zf = ti.field(ti.f32, n)
    omf = ti.field(ti.f32, n)
    of = ti.field(ti.f32, n)
    uf.from_numpy(u); pf.from_numpy(phi); zf.from_numpy(zeta); omf.from_numpy(omega)

    @ti.kernel
    def k_adv(s: ti.i32, td: ti.f32):
        for i in range(n):
            of[i] = tr._disk_noise_density_mult(uf[i], pf[i], zf[i], td, omf[i], s)

    k_adv(seed, t_disk)
    gpu = of.to_numpy()
    cpu_gain = noise.noise_density_mult(u, phi, zeta, nz_gain, seed=seed,
                                        t_disk=t_disk, omega=omega, shear_period=T)
    assert np.allclose(gpu, cpu_gain, rtol=1e-3, atol=2e-3), float(np.abs(gpu - cpu_gain).max())

    # gain = 1 is a different field: the dial demonstrably emphasises the winding.
    cpu_unit = noise.noise_density_mult(u, phi, zeta, _NOISE_ON, seed=seed,
                                        t_disk=t_disk, omega=omega, shear_period=T)
    assert not np.allclose(cpu_gain, cpu_unit, rtol=1e-2, atol=1e-2)


# A modulation block (D2.4 §3) for the twin-agreement test below.
_MOD_BLOCK = {
    "enabled": True, "octaves": 3, "lacunarity": 2.0, "gain": 0.5,
    "freq_u": 4.0, "freq_phi": 16, "temp_amp": 0.6, "edge_in_amp": 0.3,
    "edge_out_amp": 0.2, "edge_softness": 0.4, "height_amp": 0.5,
}


def test_mod_fields_match_cpu_reference():
    """D2.4 (CKS-12 §3): the kernel's modulation-envelope twin
    ``_disk_noise_mod_fields`` (n_T, n_e_in, n_e_out, n_h) must match the CPU source
    of truth ``noise.noise_modulation_fields`` under the same dual-phase shear
    advection as the density field. Driven past one reset cycle (s = 1.3) with a
    per-sample Ω and the ``dynamism`` gain so the reseed + crossfade + gain all bite.
    """
    import taichi as ti

    from renderer import noise

    _ensure_cuda()
    T = 10.0
    t_disk = 1.3 * T
    nz = copy.deepcopy(_NOISE_ON)
    nz["dynamism"] = 4.0
    nz["modulation"] = copy.deepcopy(_MOD_BLOCK)
    tr._setup_disk_noise({"disk": {"noise": nz, "dynamics": {"shear_period_M": T}}})
    seed = int(nz["seed"])

    rng = np.random.default_rng(3)
    n = 4096
    u = rng.uniform(0.0, 3.0, n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    zeta = rng.uniform(-2.5, 2.5, n).astype(np.float32)
    omega = rng.uniform(0.05, 0.4, n).astype(np.float32)

    uf = ti.field(ti.f32, n)
    pf = ti.field(ti.f32, n)
    omf = ti.field(ti.f32, n)
    out = ti.field(ti.f32, shape=(n, 4))
    uf.from_numpy(u); pf.from_numpy(phi); omf.from_numpy(omega)

    @ti.kernel
    def k_mod(s: ti.i32, td: ti.f32):
        for i in range(n):
            v = tr._disk_noise_mod_fields(uf[i], pf[i], td, omf[i], s)
            for j in ti.static(range(4)):
                out[i, j] = v[j]

    k_mod(seed, t_disk)
    gpu = out.to_numpy()  # (n, 4)
    cpu = noise.noise_modulation_fields(u, phi, zeta, nz, seed=seed,
                                        t_disk=t_disk, omega=omega, shear_period=T)
    for j in range(4):
        assert np.allclose(gpu[:, j], cpu[j], rtol=1e-3, atol=2e-3), (
            j, float(np.abs(gpu[:, j] - cpu[j]).max())
        )
    # Sanity: the advected envelopes stay in [0,1] (convex triangle-weight blend).
    assert gpu.min() >= -1e-4 and gpu.max() <= 1.0 + 1e-4


# --------------------------------------------------------------------------- #
# CKS-19 multi-phase media — GPU param buffer + ρ_cold twin.
# --------------------------------------------------------------------------- #
def test_multiphase_params_uploaded():
    """disk.multiphase populates the _NI_MP_* slots; absent block ⇒ disabled defaults."""
    _ensure_cuda()
    cfg_on = {"disk": {"multiphase": {"enabled": True, "dust_correlation": -0.6,
                                      "dust_amp": 1.0, "dust_sigma_frac": 0.8}}}
    tr._setup_disk_noise(cfg_on)
    buf = tr.disk_noise_params.to_numpy()
    assert buf[tr._NI_MP_EN] == 1.0
    assert abs(buf[tr._NI_MP_CHI] - (-0.6)) < 1e-6
    assert abs(buf[tr._NI_MP_SIGFRAC] - 0.8) < 1e-6
    tr._setup_disk_noise({"disk": {}})
    buf0 = tr.disk_noise_params.to_numpy()
    assert buf0[tr._NI_MP_EN] == 0.0
    assert buf0[tr._NI_MP_SIGFRAC] == 1.0  # ratio defaults to 1 (σ_cold=σ_hot)


_SATOL = 1e-5  # same scalar-twin tolerance as test_noise_gpu.py


def test_rho_cold_gpu_matches_cpu():
    """GPU _disk_density_cks[1] (ρ_cold) == CPU dust_density_mult × gauss, within _SATOL.

    Sampled at the midplane (ζ=0 ⇒ gauss=gauss_cold=1) so the index [1] reference is
    exactly ``dust_density_mult`` and index [0] is ``noise_density_mult`` (the hot
    multiplier). t_disk=0 with no dynamics block ⇒ the static stack (omega unused).
    """
    _ensure_cuda()
    import taichi as ti
    from renderer import noise as N

    nz = {"m_max": 2.5,
          "layers": {"base": {"enabled": True, "amp": 0.6, "octaves": 5,
                              "lacunarity": 2, "gain": 0.5, "freq_u": 6.0, "freq_phi": 24}}}
    mp = {"enabled": True, "dust_correlation": -0.6, "dust_amp": 1.0, "dust_sigma_frac": 0.8}
    cfg = {"disk": {"noise": nz, "multiphase": mp}}
    tr._setup_disk_noise(cfg)

    r_inner, sigma0, beta, seed = 4.0, 0.1, 0.0, 7
    us = np.linspace(0.1, 2.0, 16).astype(np.float32)
    phis = np.linspace(-2.5, 2.5, 16).astype(np.float32)
    uf = ti.field(ti.f32, us.size); uf.from_numpy(us)
    pf = ti.field(ti.f32, phis.size); pf.from_numpy(phis)
    out = ti.field(ti.f32, shape=(us.size, phis.size, 3))

    @ti.kernel
    def probe(r_inner: ti.f32, sigma0: ti.f32, beta: ti.f32, s: ti.i32):
        for i, j in ti.ndrange(us.shape[0], phis.shape[0]):
            u = uf[i]
            phi = pf[j]
            r = r_inner * ti.exp(u)
            dz = 0.0 * sigma0          # ζ=0 midplane ⇒ gauss=1
            d = tr._disk_density_cks(ti.cos(phi) * 1.0, ti.sin(phi) * 1.0, r, dz,
                                     sigma0, beta, r_inner, 1e9, r_inner,
                                     1, s, 0.0, 0.999)
            out[i, j, 0] = d[0]
            out[i, j, 1] = d[1]
            out[i, j, 2] = d[2]

    probe(r_inner, sigma0, beta, seed)
    gpu = out.to_numpy()

    # CPU reference on the same (u, φ) grid, ζ=0.
    uu, pp = np.meshgrid(us, phis, indexing="ij")
    u_flat = uu.ravel(); p_flat = pp.ravel()
    z_flat = np.zeros_like(u_flat)
    rho_hot = N.noise_density_mult(u_flat, p_flat, z_flat, nz, seed=seed).reshape(uu.shape)
    rho_cold = N.dust_density_mult(u_flat, p_flat, z_flat, nz, mp, seed=seed).reshape(uu.shape)

    assert np.allclose(gpu[:, :, 0], rho_hot, atol=_SATOL), float(np.abs(gpu[:, :, 0] - rho_hot).max())
    assert np.allclose(gpu[:, :, 1], rho_cold, atol=_SATOL), float(np.abs(gpu[:, :, 1] - rho_cold).max())
