"""V1.2 — CKS-15 radial deep-shadow-map self-shadow guard (gpu-marked).

Drives the production ``render_beauty_frame`` (same path as ``test_gpu_regression``)
and checks the radial self-shadow gated by ``disk.volumetric.self_shadow.enabled``
(SKILL.md Formula CKS-15):

  * **flag off ⇒ bit-identical.** ``enabled: false`` renders byte-for-byte the same
    as a config with no ``self_shadow`` block at all — no bake, no lookup, so the
    legacy emission march is untouched (golden frames intact).
  * **bake vs analytic column.** With noise OFF the density is the pure Gaussian
    ``ρ(ζ) = exp(−½ζ²)`` (independent of u, φ), so the baked deep-shadow-map
    ``τ_shadow[i_u, ·, i_z] = Σ_{j<i_u} absb_c · ρ(ζ) · r_j·du`` has a closed form.
    The GPU bake must match it to f32 tolerance — pins the radial-scan quadrature,
    the inner-edge accumulation (a cell stores τ from gas STRICTLY inward of it),
    the ``dr = r·du`` element, and the ``max_tau`` clamp.
  * **self-shadow dims the disk, more outward.** Turning the lookup on only ever
    multiplies emissivity by ``e^{−strength·τ_s} ≤ 1`` and τ_s GROWS outward, so the
    rendered disk gets dimmer overall AND the outer disk is dimmed more than the
    inner — the radial brightness profile steepens (the dark-lane mechanism).
  * **voids: self-shadow raises disk contrast.** With the procedural noise on, the
    shadow carves dark wakes behind dense clumps, so the disk's luminance
    coefficient-of-variation RISES versus the unshadowed frame — gas, not haze.

CUDA is mandatory (backend LOCKED to ``ti.init(arch=ti.cuda)`` per CLAUDE.md); the
module skips cleanly without it.
"""
from __future__ import annotations

import copy
import json
import math
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

_RENDER_CACHE: dict = {}


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


def _render(enabled, *, strength=1.0, noise=False, drop_block=False,
            source_function=False, lod_max=None):
    """Render frame 0 disk-only RGB at a forced self_shadow setting.

    ``drop_block`` removes the whole ``disk.volumetric.self_shadow`` block (loader
    sees no flag) — used to prove ``enabled: false`` equals the legacy code. Simple
    model + fixed doppler so the comparison is clean; ``noise`` toggles the CKS-12
    turbulence (off for the analytic/dimming checks, on for the contrast/void check).
    ``source_function`` pairs CKS-14 with the shadow (the full V1 look); ``lod_max``
    sets ``self_shadow.lod_max_camera_radius`` to exercise the V1.3 distance gate.
    """
    key = (bool(enabled), float(strength), bool(noise), bool(drop_block),
           bool(source_function), None if lod_max is None else float(lod_max))
    if key in _RENDER_CACHE:
        return _RENDER_CACHE[key]

    _ensure_cuda()
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = "simple"
    cfg["disk"]["doppler_strength"] = 1.0
    cfg["disk"].setdefault("noise", {})["enabled"] = bool(noise)
    vol = cfg["disk"].setdefault("volumetric", {})
    vol["source_function"] = bool(source_function)
    if drop_block:
        vol.pop("self_shadow", None)
    else:
        ss = vol.setdefault("self_shadow", {})
        ss["enabled"] = bool(enabled)
        ss["strength"] = float(strength)
        if lod_max is not None:
            ss["lod_max_camera_radius"] = float(lod_max)

    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    hdr = tr.render_beauty_frame(cfg, _cam(), _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)
    disk_rgb = np.nan_to_num(tr.disk_buf.to_numpy()[:, :, :3]).copy()
    out = {"hdr": np.nan_to_num(hdr), "disk": disk_rgb}
    _RENDER_CACHE[key] = out
    return out


def _disk_mask(*disks):
    """Pixels where any of the supplied disk frames actually emit (union)."""
    m = np.zeros(disks[0].shape[:2], dtype=bool)
    for d in disks:
        m |= d.sum(axis=2) > 1e-6
    return m


def test_self_shadow_off_is_bit_identical_to_no_block():
    """``self_shadow.enabled: false`` ⇒ byte-for-byte the legacy march (dead code).

    Off-flag must equal the no-``self_shadow``-block render exactly — proves the
    entire CKS-15 path (bake + lookup) is inert when not opted in, so the pinned
    goldens (test_gpu_regression) are unaffected.
    """
    off = _render(False)["disk"]
    none = _render(False, drop_block=True)["disk"]
    assert np.array_equal(off, none)


def test_bake_matches_analytic_gaussian_column():
    """The GPU bake equals the closed-form Gaussian column when noise is off.

    ρ(ζ)=exp(−½ζ²) is independent of (u, φ), so every φ column of the deep-shadow-
    map is the same radial cumulative ``τ[i_u]=Σ_{j<i_u} absb_c·ρ·r_j·du`` (clamped
    to ``max_tau``). Bake directly (the same call ``render_beauty_frame`` makes) and
    compare ``tr.disk_shadow_tau`` to the analytic field to f32 tolerance.
    """
    _ensure_cuda()
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"].setdefault("noise", {})["enabled"] = False
    d = cfg["disk"]
    ss = d.setdefault("volumetric", {}).setdefault("self_shadow", {})
    ss["enabled"] = True

    tr.setup_renderer(cfg)

    r_inner = float(d["r_inner"])
    r_outer = float(d["r_outer"])
    r_isco = float(cfg.get("black_hole", {}).get("r_isco", r_inner))
    theta_half = float(d["theta_half_width"])
    sigma_frac = float(d["vertical_sigma_frac"])
    zeta_max = float(ss.get("zeta_max", 3.0))
    max_tau = float(ss.get("max_tau", 8.0))
    absb_c = float(d["absorption_coeff"])

    tr.bake_disk_shadow(
        r_inner, r_outer, r_isco, theta_half, sigma_frac,
        zeta_max, max_tau, absb_c, 0, 1234, 0.0, float(cfg["black_hole"]["spin"]),
    )
    got = tr.disk_shadow_tau.to_numpy()  # (NU, NPHI, NZ)
    nu, nphi, nz = got.shape

    # ζ is already the σ_θ-normalized vertical coord, so the analytic density is just
    # exp(−½ζ²) — σ_θ does not enter the column quadrature (only u and ζ do).
    u_max = math.log(r_outer / r_inner)
    du = u_max / nu
    expected = np.zeros((nu, nz), dtype=np.float64)
    for iz in range(nz):
        zeta = -zeta_max + (iz + 0.5) * (2.0 * zeta_max / nz)
        g = math.exp(-0.5 * zeta * zeta)
        tau = 0.0
        for iu in range(nu):
            expected[iu, iz] = min(tau, max_tau)         # τ from strictly inner gas
            r = r_inner * math.exp((iu + 0.5) * du)
            tau += absb_c * g * (r * du)                 # dr = r·du

    # τ is nonzero somewhere (sanity the bake actually integrated something).
    assert expected.max() > 0.0
    # Every φ column matches the analytic field (noise off ⇒ axisymmetric).
    for iphi in (0, nphi // 2, nphi - 1):
        np.testing.assert_allclose(got[:, iphi, :], expected, rtol=2e-4, atol=1e-5)


def test_self_shadow_dims_disk_more_outward():
    """The lookup only dims (e^{−τ_s} ≤ 1) and dims the OUTER disk more than inner.

    τ_s accumulates outward from r_inner, so the bright inner edge is barely
    shadowed while the outer gas sits behind the whole inner column. The rendered
    disk must (a) be globally dimmer with shadow on, (b) still emit + be NaN-free,
    (c) NOT be bit-identical (genuinely active), and (d) steepen radially — the
    far-half mean dims by a larger fraction than the near (brightest) region.
    """
    off = _render(False)["disk"]
    on = _render(True, strength=1.0)["disk"]
    assert np.isnan(on).sum() == 0

    mask = _disk_mask(off, on)
    assert mask.any(), "no disk pixels emitted — framing/absorption sanity"
    off_l = off.sum(axis=2)
    on_l = on.sum(axis=2)

    off_mean = float(off_l[mask].mean())
    on_mean = float(on_l[mask].mean())
    assert on_mean > 0.0                       # still emits
    assert on_mean < off_mean                  # only ever dims
    assert not np.array_equal(off, on)         # genuinely active
    rel = (off_mean - on_mean) / max(off_mean, 1e-12)
    assert rel > 0.02, f"self-shadow barely dimmed the disk ({rel:.3%})"

    # Radial steepening: split the emitting pixels into the brightest quartile (the
    # lit inner edge, least shadowed) vs the dimmer outer gas, and show the outer
    # region is attenuated by a LARGER fraction. Use the OFF luminance to classify
    # so the split is shadow-independent.
    vals = off_l[mask]
    thresh = np.quantile(vals, 0.75)
    inner = mask & (off_l >= thresh)
    outer = mask & (off_l < thresh) & (off_l > 0.0)
    assert inner.any() and outer.any()
    inner_ratio = float(on_l[inner].mean()) / max(float(off_l[inner].mean()), 1e-12)
    outer_ratio = float(on_l[outer].mean()) / max(float(off_l[outer].mean()), 1e-12)
    assert outer_ratio < inner_ratio, (inner_ratio, outer_ratio)


def test_self_shadow_raises_disk_contrast_with_noise():
    """The void payoff: with turbulence on, the shadow carves dark wakes.

    Self-shadow dims dense-clump-occluded regions far more than the gaps, so the
    disk luminance coefficient-of-variation (std/mean over emitting pixels) RISES
    versus the unshadowed frame — the disk reads as gas with voids, not flat haze.
    """
    off = _render(False, noise=True)["disk"]
    on = _render(True, strength=1.5, noise=True)["disk"]
    assert np.isnan(on).sum() == 0

    mask = _disk_mask(off, on)
    assert mask.any()
    off_l = off.sum(axis=2)[mask]
    on_l = on.sum(axis=2)[mask]

    cov_off = float(off_l.std()) / max(float(off_l.mean()), 1e-12)
    cov_on = float(on_l.std()) / max(float(on_l.mean()), 1e-12)
    assert cov_on > cov_off, f"contrast did not rise (off={cov_off:.3f}, on={cov_on:.3f})"


def test_lod_gate_drops_self_shadow_for_distant_camera():
    """V1.3 LOD gate: ``self_shadow.lod_max_camera_radius`` skips the bake when far.

    A tiny ``lod_max`` (below any camera radius) must make a ``self_shadow: true``
    render byte-for-byte equal to the flag-off frame — the gate skipped the bake +
    lookup, so the cheap legacy march carried the frame. A huge ``lod_max`` (above
    the camera radius) leaves the shadow fully active (equal to the un-gated on
    frame). Together: the gate is a pure on/off on distance, no partial state.
    """
    off = _render(False)["disk"]
    on = _render(True)["disk"]
    gated = _render(True, lod_max=1.0)["disk"]          # camera is far beyond 1 M
    ungated = _render(True, lod_max=1.0e9)["disk"]      # camera well within 1e9 M

    assert np.array_equal(gated, off)        # gate off ⇒ legacy march, bit-identical
    assert np.array_equal(ungated, on)       # gate inactive ⇒ identical to un-gated on
    assert not np.array_equal(off, on)       # (sanity: the shadow does something here)


def test_combined_source_function_and_self_shadow_golden():
    """V1 golden: CKS-14 + CKS-15 compose into the glowing-gas-with-voids path.

    The defining V1 deliverable is the PAIR — CKS-14 materialises the source function
    ``S``; CKS-15 dims it by ``e^{−τ_s}`` so shadowed thick gas reads dark. This
    relational golden (in the repo's metric-pinned style, not a brittle absolute
    constant) asserts the composition is genuinely active and physically ordered:
      * NaN-free and still emitting,
      * differs from the source-function-only frame (the shadow term fires),
      * differs from the self-shadow-only frame (the source-function term fires),
      * is dimmer than the source-function-only frame (the shadow only attenuates
        ``S``, never brightens it).
    Noise OFF for determinism (the void *texture* is exercised by the contrast test
    above; this golden guards the two-term composition itself).
    """
    both = _render(True, source_function=True)
    sf_only = _render(False, source_function=True)
    ss_only = _render(True, source_function=False)

    assert np.isnan(both["disk"]).sum() == 0
    mask = _disk_mask(both["disk"], sf_only["disk"], ss_only["disk"])
    assert mask.any()

    both_l = both["disk"].sum(axis=2)
    sf_l = sf_only["disk"].sum(axis=2)

    assert float(both_l[mask].mean()) > 0.0                 # still emits
    assert not np.array_equal(both["disk"], sf_only["disk"])  # shadow term active
    assert not np.array_equal(both["disk"], ss_only["disk"])  # source-fn term active
    # The self-shadow only ever dims the materialised S (e^{−τ_s} ≤ 1).
    assert float(both_l[mask].mean()) < float(sf_l[mask].mean())
