"""CKS-21 acceptance: scale-dependent shear cascade on the disk turbulence.

The product claim of Pillar 1 (VISUALIZATION): the CKS-12 §2 uniform Keplerian
shear ``shear_k = dynamism·Ω(r)·a_k·T`` is replaced, per density octave ``o``, by a
*frequency-dependent* net shear ``S(f_o)·shear_k`` with ``S(f) = 1/(1+(f/f_c)^p)``
(Butterworth-like; low spatial frequencies wind into the spiral, high-frequency
micro-vortices are protected — Kolmogorov-like). It is implemented as a per-octave
**de-shear add-back** ``Δφ_o = (1 − S(f_o))·shear_k`` applied AFTER the CKS-18 curl
warp inside the shared ``_octaves`` loop, so the curl order is unchanged and
``S ≡ 1`` collapses to the uniform-shear stack bit-for-bit.

Two guards (mirroring the CKS-23 LOD precedent in ``tests/test_disk_lod.py``):
  * **OFF bit-identity** (constraint 6): ``shear_cascade.enabled:false`` renders
    byte-for-byte the same frame as the block being ABSENT (both ⇒ ``_NI_SC_EN``=0 ⇒
    the noise stack feeds the ``_SC_FC_OFF`` f_c sentinel ⇒ ``S ≡ 1`` ⇒ the de-shear
    add-back is exactly 0). Like CKS-23 (and unlike the multiphase/scatter/erosion
    gates) there is NO ``ti.static`` recompile — the sentinel makes the sheared
    fBm bit-exact, so it is always compiled and the goldens are unshifted. Rendered
    at a NONZERO ``t_disk`` so the dynamic advected branch (where ``shear_k`` is
    actually computed and threaded into the octave loop) is exercised.
  * **ON re-textures** (the render-level acceptance): with ``shear_cascade.enabled:true``
    and ``f_c=2.0`` at a long ``t_disk``, the per-octave net shear differs from the
    uniform shear (corrections ``(1−S(f_o))·shear_k`` grow with octave frequency), so
    the disk-only render DIFFERS from the cascade-off render over the lit disk — while
    pixels that carry no disk light in either render stay exactly 0 (a coordinate
    warp re-textures the disk, it never moves the silhouette).

WHY NOT AN EMERGENT-SPECTRUM RENDER METRIC: at a fixed radius the shear is a pure
azimuthal (φ) translation — spectrum-invariant — and the laminarization the cascade
produces is RADIAL (coarse filaments wind, fine structure does not), which a φ power
spectrum cannot see (design §A.3). The cascade MATH (monotone net shear ``S·shear_k``,
per-octave displacement, ``S≡1`` collapse, C0-at-reset) is proven directly by the CPU
unit tests in ``tests/test_noise.py`` (``test_shear_transfer_monotone_and_sentinel``,
``test_fbm2_single_octave_displacement_matches_correction``,
``test_fbm2_octave1_net_shear_strictly_smaller_than_octave0``) and the GPU twin parity
in ``tests/test_disk_noise.py`` (``test_shear_cascade_stack_matches_cpu_reference``).
The render guard here asserts only the sound facts (ON re-textures, OFF is bit-identical).

Drives the production ``render_beauty_frame`` (same path as test_gpu_regression /
test_disk_lod). CUDA is mandatory (backend LOCKED to ``ti.init(arch=ti.cuda)`` per
CLAUDE.md); the module skips cleanly without it. Resolution kept small.
"""
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

# A long disk-animation time so the dual-phase reset blend is mid-advection (some
# a_k meaningfully nonzero ⇒ shear_k ≠ 0), making the cascade's per-octave de-shear
# add-back a visible re-texture rather than a no-op.
_T_DISK = 80.0


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


def _faceon_cam():
    """Face-on view down the +z spin axis at d=45 < render.r_max (50): a clean disk
    annulus so the cascade's re-texturing is read on a fully-lit face, not tangled
    with the edge-on multi-radius integration (same cam as the CKS-23 LOD test)."""
    return {"pos": [0.0, 0.0, 45.0], "fwd": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0], "fov": 1.1}


def _render(cfg, cam, t_disk):
    """Render frame 0 (full composite) → HDR RGB float array (for OFF bit-identity)."""
    _ensure_cuda()
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    hdr = tr.render_beauty_frame(cfg, cam, _WIDTH, _HEIGHT, with_disk=True,
                                 lod_enabled=True, t_disk=t_disk)
    return np.nan_to_num(np.asarray(hdr))


def _render_disk_lum(cfg, cam, t_disk):
    """Render frame 0 and return the DISK-ONLY luminance (Σ RGB of tr.disk_buf, no
    lensed starfield) so the on/off comparison isolates the disk texture."""
    _ensure_cuda()
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    tr.render_beauty_frame(cfg, cam, _WIDTH, _HEIGHT, with_disk=True,
                           lod_enabled=True, t_disk=t_disk)
    disk = tr.disk_buf.to_numpy()[:, :, 0:3]
    return np.nan_to_num(disk).sum(axis=2)


def _noisy_scene():
    """A turbulent, advecting disk (L0/L2 fBm + L1 ridged on, shear-advected). The
    caller flips ``disk.noise.shear_cascade.enabled`` (and tunes ``shear_cutoff``).
    The resolved config carries ``disk.dynamics`` (CKS-13), so T = shear_period_M > 0
    and a nonzero ``t_disk`` puts the stack on the dynamic advected branch."""
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = "simple"
    cfg["disk"]["doppler_strength"] = 1.0
    nz = cfg["disk"].setdefault("noise", {})
    nz["enabled"] = True
    # shear_cutoff/enabled are set per-test; the block presence is what the OFF test
    # toggles against absence.
    nz["shear_cascade"] = {"enabled": False, "shear_cutoff": 2.0, "shear_falloff": 2.0}
    return cfg


def test_shear_cascade_off_is_bit_identical():
    """shear_cascade.enabled:false renders byte-for-byte the same as the block absent
    (both ⇒ _NI_SC_EN=0 ⇒ the _SC_FC_OFF sentinel ⇒ S≡1 ⇒ the de-shear add-back is
    exactly 0, no recompile). Rendered at a nonzero t_disk so the advected branch that
    computes and threads shear_k is exercised — the sentinel must zero the correction
    even while shear_k flows."""
    base = _noisy_scene()
    cfg_absent = copy.deepcopy(base)
    cfg_absent["disk"]["noise"].pop("shear_cascade", None)
    cfg_off = copy.deepcopy(base)
    cfg_off["disk"]["noise"]["shear_cascade"]["enabled"] = False

    img_absent = _render(cfg_absent, _cam(), _T_DISK)
    img_off = _render(cfg_off, _cam(), _T_DISK)
    assert np.array_equal(img_absent, img_off)


def test_shear_cascade_on_retextures_disk_face_on_long_T():
    """Enabling the cascade with f_c=2.0 at a long t_disk applies a per-octave de-shear
    add-back that differs from the uniform shear (the correction (1−S(f_o))·shear_k grows
    with octave frequency), visibly re-texturing the disk: the disk-only render DIFFERS
    from the otherwise-identical cascade-off render over a substantial number of disk
    pixels — proving the wiring — while pixels with no disk light in either render stay
    exactly 0 (a coordinate warp; it never invents disk where the silhouette has none).

    The cascade MATH is proven precisely by the CPU twins in tests/test_noise.py and the
    GPU twin parity in tests/test_disk_noise.py — see the module docstring for why an
    emergent φ-spectrum is not a sound render-level observable for it."""
    cam = _faceon_cam()
    cfg_off = _noisy_scene()  # enabled:false ⇒ uniform CKS-12 §2 shear on every octave
    cfg_on = _noisy_scene()
    cfg_on["disk"]["noise"]["shear_cascade"]["enabled"] = True  # f_c=2.0 ⇒ per-octave net shear

    lum_off = _render_disk_lum(cfg_off, cam, _T_DISK)
    lum_on = _render_disk_lum(cfg_on, cam, _T_DISK)

    thr = 1e-3
    lit_off = lum_off > thr
    assert lit_off.sum() > 100, "disk did not render enough lit pixels to measure"

    # The cascade must actually change the disk texture (its sole job).
    changed = int((np.abs(lum_off - lum_on) > 1e-3).sum())
    assert changed > 100, f"cascade did not re-texture the disk (only {changed} pixels changed)"
    assert not np.array_equal(lum_on, lum_off)

    # Containment: where NEITHER render shows disk light, the cascade created none
    # (a coordinate warp re-shears octaves, it never moves the disk silhouette).
    vacuum_both = (lum_off <= thr) & (lum_on <= thr)
    assert float(np.abs(lum_on[vacuum_both]).max(initial=0.0)) <= thr

    # The disk still exists (the cascade re-textured it, it did not erase it).
    assert (lum_on > thr).sum() > 0.5 * lit_off.sum(), "cascade erased the disk"
