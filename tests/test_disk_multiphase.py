"""CKS-19 acceptance: a cold dust slab carves a darker-than-background silhouette.

The single product claim of Pillar 2 (multi-phase media): cold dust must
**absorb** light from the hot plasma behind it WITHOUT emitting, so an
anti-correlated (χ=−1) dust field reads DARKER than the same scene with
multiphase OFF — a true silhouette, not merely dimmer emission. With χ=−1 the
dust log-density is ``m_cold = −m_hot`` (the √(1−χ²) term vanishes), so dust
piles into the hot voids and obscures the background/disk seen through them.

Drives the production ``render_beauty_frame`` (same path as test_gpu_regression /
test_disk_noise). CUDA is mandatory (backend LOCKED to ``ti.init(arch=ti.cuda)``
per CLAUDE.md); the module skips cleanly without it. Resolution is kept small.

NOTE: enabling multiphase flips the ``ti.static`` ``_MP_COMPILE`` gate, so the ON
render pays a one-time kernel recompile (the dust branch is only emitted when the
config enables it — the OFF default keeps the original fast JIT).
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


def _multiphase_scene():
    """A bright, turbulent disk with a strong, maximally anti-correlated dust phase.

    Noise is on (so the dust modulator has structure to carve), absorption is
    raised for visible obscuration, and the dust is χ=−1 / high-amp so it fills the
    hot voids. Returns a full render config; the caller flips ``multiphase.enabled``.
    """
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = "simple"
    cfg["disk"]["doppler_strength"] = 1.0
    # Visible obscuration: grey κ high enough that ρ_cold meaningfully attenuates.
    cfg["disk"]["absorption_coeff"] = 2.0
    # Structured turbulence is required for the dust field to have lanes to carve.
    cfg["disk"].setdefault("noise", {})["enabled"] = True
    cfg["disk"]["multiphase"] = {
        "enabled": True,
        "dust_correlation": -1.0,   # χ=−1 ⇒ m_cold = −m_hot ⇒ dust fills the hot voids
        "dust_amp": 2.0,            # strong dust log-density (clamped by noise.m_max)
        "dust_sigma_frac": 1.0,     # same slab thickness ⇒ isolates the χ/absorption effect
    }
    return cfg


def _render(cfg):
    """Render frame 0 from a full config dict → HDR RGB float array."""
    _ensure_cuda()
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    hdr = tr.render_beauty_frame(cfg, _cam(), _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)
    return np.nan_to_num(np.asarray(hdr))


def test_dust_carves_silhouette():
    """ON (χ=−1 dust) must darken an appreciable region vs OFF and create a new
    darkest pixel — absorption removes light, it does not merely fail to add it."""
    base = _multiphase_scene()
    cfg_off = copy.deepcopy(base); cfg_off["disk"]["multiphase"]["enabled"] = False
    cfg_on = copy.deepcopy(base);  cfg_on["disk"]["multiphase"]["enabled"] = True

    img_off = _render(cfg_off)
    img_on = _render(cfg_on)

    lum_off = img_off.sum(axis=2)
    lum_on = img_on.sum(axis=2)

    darkened = lum_on < lum_off - 1e-4
    assert darkened.mean() > 0.02, "dust did not darken any appreciable region"
    assert lum_on.min() < lum_off.min(), "dust did not create a new darkest pixel"


def _absorbing_scene():
    """A bright, turbulent, optically-THICK single-phase disk (multiphase OFF).

    Task 7's chromatic extinction κ⃗ applies in the march regardless of the
    emission/absorption density split, so it is exercised here on the single-phase
    (MP-off) path — the same kernel as ``test_gpu_regression`` — to avoid the
    separate MP-on cold compile. High absorption gives the disk real optical depth
    so per-channel extinction visibly reddens the light transmitted through it."""
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = "simple"
    cfg["disk"]["doppler_strength"] = 1.0
    cfg["disk"]["absorption_coeff"] = 3.0   # thick enough that κ⃗ reddening is visible
    cfg["disk"].setdefault("noise", {})["enabled"] = True
    cfg["disk"].setdefault("multiphase", {})["enabled"] = False
    return cfg


def test_chromatic_extinction_reddens():
    """CKS-19 Task 7: per-channel κ⃗ = absb_c·extinction_rgb. With κ_B > κ_R the cold
    medium absorbs blue more than red, so light surviving through it is WARMER
    (reddened) than under grey extinction — astrophysical dust reddening.

    Compared to a grey [1,1,1] reference (same scene), a strongly blue-weighted
    extinction [0.3, 1.0, 3.0] must let proportionally MORE red than blue survive:
    red retention dr = ΣR_red/ΣR_grey > blue retention db = ΣB_red/ΣB_grey.
    On the pre-Task-7 march (scalar transmittance, extinction_rgb ignored) the two
    renders are identical ⇒ dr == db ⇒ this fails, which is the missing feature."""
    base = _absorbing_scene()
    cfg_grey = copy.deepcopy(base); cfg_grey["disk"]["extinction_rgb"] = [1.0, 1.0, 1.0]
    cfg_red = copy.deepcopy(base);  cfg_red["disk"]["extinction_rgb"] = [0.3, 1.0, 3.0]

    img_grey = _render(cfg_grey)
    img_red = _render(cfg_red)

    dr = img_red[..., 0].sum() / max(img_grey[..., 0].sum(), 1e-6)
    db = img_red[..., 2].sum() / max(img_grey[..., 2].sum(), 1e-6)
    assert dr > db * 1.01, (
        f"chromatic extinction did not redden the disk: red retention {dr:.4f} "
        f"not > blue retention {db:.4f}"
    )
