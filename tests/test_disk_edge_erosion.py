"""CKS-22 acceptance: Kelvin-Helmholtz threshold erosion of the OUTER disk edge.

The product claim of Pillar 4 (VISUALIZATION): the clean CKS-12 §3 outer smoothstep
rim is replaced by a noise-thresholded soft-Heaviside clip
``win_out' = H_soft(win_out − τ_KH·N_KH)`` so the rim TEARS into vacuum — disconnected
holes/fingers appear interior to ``r_out_eff`` — rather than fading smoothly. Because
the clip is a *remap* (not a pure multiply), where ``N_KH`` is low the surviving edge can
sharpen; the robust observable is therefore "rim pixels newly torn to ~0" (asymmetric:
torn ≫ filled), NOT net flux (the redistribution is flux-neutral: ratio ≈ 1.0).

Two guards:
  * **OFF bit-identity** (constraint 6): ``edge_erosion.enabled:false`` renders byte-for-byte
    the same frame as the block being ABSENT (the ``_EROS_COMPILE`` ti.static gate is False
    in both ⇒ identical kernel; the §3 ``win_out`` is the unmodified smoothstep). This guard
    drives the full production composite at the canonical camera (the strongest OFF claim).
  * **Tearing** (the acceptance): with erosion ON (and the required §3 modulation on) the
    outer rim gains near-zero holes the clean rim did not have.

GEOMETRY: the tearing guard renders the **disk-only buffer** (``tr.disk_buf[:,:,0:3]`` —
no lensed starfield) at a **face-on** camera looking down the +z spin axis. Both are
necessary to OBSERVE the tear: at the canonical edge-on view every ray through the outer
rim also integrates near/far disk material (and, in the composite, the rim sits in front
of stars), so a torn finger is always back-filled and never reaches vacuum. Face-on, the
disk projects to a clean annulus whose outer rim maps to identifiable pixels with vacuum
(0) behind them; the camera sits at d=45 < ``render.r_max`` (50) so rays are not culled.

Drives the production ``render_beauty_frame`` (same path as test_gpu_regression /
test_disk_multiphase). CUDA is mandatory (backend LOCKED to ``ti.init(arch=ti.cuda)`` per
CLAUDE.md); the module skips cleanly without it. Resolution kept small.

NOTE: erosion REQUIRES ``disk.noise.modulation.enabled`` (the only producer of a soft
``win_out`` to clip). Enabling erosion flips the ``ti.static`` ``_EROS_COMPILE`` gate, so the
ON render pays a one-time kernel recompile (OFF default keeps the original fast JIT).
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
    """Face-on view down the +z spin axis at d=45 < render.r_max (50): the disk projects
    to a clean annulus so the outer rim is identifiable pixels with vacuum (0) behind."""
    return {"pos": [0.0, 0.0, 45.0], "fwd": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0], "fov": 1.1}


def _render(cfg):
    """Render frame 0 (full composite, canonical camera) → HDR RGB float array."""
    _ensure_cuda()
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    hdr = tr.render_beauty_frame(cfg, _cam(), _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)
    return np.nan_to_num(np.asarray(hdr))


def _render_disk_lum(cfg, cam):
    """Render frame 0 and return the DISK-ONLY luminance (Σ RGB of tr.disk_buf, no
    lensed starfield) — the rim tear is only observable without the back-fill."""
    _ensure_cuda()
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    tr.render_beauty_frame(cfg, cam, _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)
    disk = tr.disk_buf.to_numpy()[:, :, 0:3]
    return np.nan_to_num(disk).sum(axis=2)


def _eroding_scene():
    """A turbulent disk with §3 ragged-edge modulation ON (so a soft win_out exists to
    erode). The caller flips ``edge_erosion.enabled``."""
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = "simple"
    cfg["disk"]["doppler_strength"] = 1.0
    nz = cfg["disk"].setdefault("noise", {})
    nz["enabled"] = True
    nz.setdefault("modulation", {})["enabled"] = True   # CKS-22 requires the §3 window
    cfg["disk"]["edge_erosion"] = {
        "enabled": True,
        "strength": 0.8,    # clamped to [0, 1-w_soft] at load — deep tears
        "freq_u": 5.0,
        "freq_phi": 16,     # fine azimuthal fingers
        "freq_z": 1.0,
        "octaves": 4,
    }
    return cfg


def test_erosion_off_is_bit_identical():
    """edge_erosion.enabled:false renders byte-for-byte the same as the block absent
    (both ⇒ _EROS_COMPILE=False ⇒ identical kernel; win_out is the §3 smoothstep)."""
    base = _eroding_scene()
    cfg_absent = copy.deepcopy(base)
    cfg_absent["disk"].pop("edge_erosion", None)
    cfg_off = copy.deepcopy(base)
    cfg_off["disk"]["edge_erosion"]["enabled"] = False

    img_absent = _render(cfg_absent)
    img_off = _render(cfg_off)
    assert np.array_equal(img_absent, img_off)


def test_erosion_tears_outer_band():
    """Erosion ON tears the face-on outer rim into vacuum: disk-only pixels that carried
    light with erosion OFF drop to ~0 with it ON (disconnected holes the clean rim did not
    have), and the tear is ASYMMETRIC — far more pixels are newly torn than newly filled
    (the clip removes fingers; the redistribution leaves total flux ≈ unchanged)."""
    cam = _faceon_cam()
    cfg_off = _eroding_scene(); cfg_off["disk"]["edge_erosion"]["enabled"] = False
    cfg_on = _eroding_scene();  cfg_on["disk"]["edge_erosion"]["enabled"] = True

    lum_off = _render_disk_lum(cfg_off, cam)
    lum_on = _render_disk_lum(cfg_on, cam)

    thr = 1e-3
    newly_torn = int(((lum_off > thr) & (lum_on < thr)).sum())   # lit → vacuum (the tear)
    newly_filled = int(((lum_off < thr) & (lum_on > thr)).sum())  # vacuum → lit (sharpening)
    # The rim must develop vacuum holes (diagnostic: ~120 torn at d=45, 480×270).
    assert newly_torn > 30, (
        f"erosion did not tear the rim into vacuum (only {newly_torn} pixels newly torn)"
    )
    # Tearing dominates filling — the rim FRAYS, it does not merely shift outward.
    assert newly_torn > 5 * max(newly_filled, 1), (
        f"tear not asymmetric (torn={newly_torn}, filled={newly_filled})"
    )
    # The disk must still exist (we tore the rim, not the whole disk).
    assert (lum_on > thr).sum() > 0.5 * (lum_off > thr).sum(), "erosion erased the disk"
