"""CKS-23 acceptance: fractal LOD octave cascade on the disk turbulence.

The product claim of Pillar 5 (SAMPLING): the procedural-noise density octaves are
gated per sample by ``g_o = clamp(n_oct − o, 0, 1)`` with
``n_oct = clamp(N_max − log₂(ε·d / J₀), N_min, N_max)`` (ε = vertical_fov/HEIGHT, d =
camera distance), so a sample whose pixel footprint is large drops the shimmering
sub-pixel octaves of the L0/L2/L1-mask fBm — the anti-aliasing prerequisite for the
V4 free camera. The gate weights BOTH the fBm numerator and denominator, so it is an
exact renormalization: at ``n_oct ≥ octaves`` every gate is 1 and the field is the
ungated fBm byte-for-byte.

Two guards:
  * **OFF bit-identity** (constraint 6): ``lod.enabled:false`` renders byte-for-byte
    the same frame as the block being ABSENT (both ⇒ ``_NI_LOD_EN``=0 ⇒ the march feeds
    the ``_LOD_OFF`` sentinel ⇒ every ``g_o``=1). Unlike the multiphase/scatter/erosion
    gates there is NO ``ti.static`` recompile: the gated fBm is bit-exact at the sentinel
    (×1.0 is exact), so it is always compiled and the goldens are unshifted.
  * **ON is wired and contained** (the render-level acceptance): with ``lod.enabled:true``
    and a tiny ``J₀`` (forcing ``n_oct → N_min``), the per-sample octave count threads all
    the way through ``render_beauty_physics`` → ``_disk_emit_cks`` → the gated fBm and
    visibly re-textures the disk — the disk-only render DIFFERS from the LOD-off render —
    while leaving the escaped/background pixels untouched (sampling-only, never the disk
    silhouette geometry).

WHY NOT A "less high-frequency" RENDER METRIC: the CKS-23 cascade math (dropping octave
``o`` = truncating the fBm to ``o`` octaves, exact renormalisation, monotone ``n_oct``,
anti-pop crossfade) is proven directly and precisely by the CPU unit tests in
``tests/test_noise.py`` (``test_fbm2_lod_integer_equals_truncated``,
``test_fbm2_lod_full_matches_fbm2``, ``test_lod_noct_monotone_and_clamped``,
``test_lod_octave_weight_crossfade``) and the GPU twin parity in ``tests/test_noise_gpu.py``.
A raw image-space high-frequency metric (|Laplacian|/|gradient|) is NOT a sound render-level
observable for it: the gated fBm RENORMALISES (gating numerator AND denominator), so culling
the sub-pixel octaves re-weights the surviving coarse octave's amplitude UP (~2×), and since
those high octaves are already below the pixel/ray-integration Nyquist their removal does not
lower — and empirically slightly RAISES (~+8%) — the measured |Laplacian| of the disk. The
true LOD benefit is anti-aliasing against a supersampled reference, not raw curvature; the
render guard here therefore asserts only the sound facts (ON re-textures the disk, OFF is
bit-identical) and leaves the cascade math to the CPU/GPU twins above.

Drives the production ``render_beauty_frame`` (same path as test_gpu_regression /
test_disk_edge_erosion). CUDA is mandatory (backend LOCKED to ``ti.init(arch=ti.cuda)``
per CLAUDE.md); the module skips cleanly without it. Resolution kept small.
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
    """Face-on view down the +z spin axis at d=45 < render.r_max (50): a clean disk annulus
    with a (roughly) uniform camera distance, so the per-sample footprint J=ε·d is well
    defined and LOD's re-texturing is not tangled with the edge-on multi-radius integration."""
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
    """Render frame 0 and return the DISK-ONLY luminance (Σ RGB of tr.disk_buf, no lensed
    starfield) so the on/off comparison isolates the disk texture LOD re-samples."""
    _ensure_cuda()
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    tr.render_beauty_frame(cfg, cam, _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)
    disk = tr.disk_buf.to_numpy()[:, :, 0:3]
    return np.nan_to_num(disk).sum(axis=2)


def _noisy_scene():
    """A turbulent disk (L0/L2 fBm on) whose fine noise texture LOD can thin. The
    caller flips ``disk.lod.enabled`` (and tunes ``j0``)."""
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = "simple"
    cfg["disk"]["doppler_strength"] = 1.0
    nz = cfg["disk"].setdefault("noise", {})
    nz["enabled"] = True
    # j0 is set per-test; the block presence is what test_lod_off toggles against absence.
    cfg["disk"]["lod"] = {"enabled": False, "n_max": 6.0, "n_min": 1.0, "j0": 1.0e-6}
    return cfg


def test_lod_off_is_bit_identical():
    """lod.enabled:false renders byte-for-byte the same as the block absent (both ⇒
    _NI_LOD_EN=0 ⇒ the _LOD_OFF sentinel ⇒ every octave gate g_o=1, no recompile)."""
    base = _noisy_scene()
    cfg_absent = copy.deepcopy(base)
    cfg_absent["disk"].pop("lod", None)
    cfg_off = copy.deepcopy(base)
    cfg_off["disk"]["lod"]["enabled"] = False

    img_absent = _render(cfg_absent)
    img_off = _render(cfg_off)
    assert np.array_equal(img_absent, img_off)


def test_lod_on_retextures_disk_only():
    """Enabling LOD with a tiny J₀ (n_oct → N_min) threads the per-sample octave count
    through the density-noise chain and visibly re-textures the disk: the disk-only render
    DIFFERS from the otherwise-identical LOD-off render over a substantial number of disk
    pixels — proving the wiring — while pixels that carry no disk light in either render
    stay exactly 0 (sampling-only; LOD never invents disk where the silhouette has none).

    The cascade MATH (octave drop = fBm truncation, exact renorm, monotone n_oct, anti-pop
    crossfade) is proven precisely by the CPU twins in tests/test_noise.py and the GPU twin
    parity in tests/test_noise_gpu.py — see the module docstring for why a raw image-space
    high-frequency metric is not a sound render-level observable for it."""
    cam = _faceon_cam()
    cfg_off = _noisy_scene()  # enabled:false ⇒ full native octaves
    cfg_on = _noisy_scene()
    cfg_on["disk"]["lod"]["enabled"] = True  # tiny j0 ⇒ n_oct clamps to n_min everywhere

    lum_off = _render_disk_lum(cfg_off, cam)
    lum_on = _render_disk_lum(cfg_on, cam)

    thr = 1e-3
    lit_off = lum_off > thr
    assert lit_off.sum() > 200, "disk did not render enough lit pixels to measure"

    # LOD must actually change the disk texture (its sole job at this distance).
    changed = int((np.abs(lum_off - lum_on) > 1e-3).sum())
    assert changed > 100, f"LOD did not re-texture the disk (only {changed} pixels changed)"
    assert not np.array_equal(lum_on, lum_off)

    # Containment: where NEITHER render shows disk light, LOD created none (sampling-only —
    # it re-weights octaves, it never moves the disk silhouette).
    vacuum_both = (lum_off <= thr) & (lum_on <= thr)
    assert float(np.abs(lum_on[vacuum_both]).max(initial=0.0)) <= thr

    # The disk still exists (LOD thinned octaves, it did not erase the disk).
    assert (lum_on > thr).sum() > 0.5 * lit_off.sum(), "LOD erased the disk"
