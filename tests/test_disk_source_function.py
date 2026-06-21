"""V1.1 — CKS-14 volumetric RTE source-function march guard (gpu-marked).

Drives the production ``render_beauty_frame`` (same path as ``test_gpu_regression``)
and checks the radiative-transfer source-function reinterpretation gated by
``disk.volumetric.source_function`` (SKILL.md Formula CKS-14):

  * **flag off ⇒ bit-identical.** ``source_function: false`` renders byte-for-byte
    the same as a config with no ``volumetric`` block at all — the legacy
    ``disk_col += transm·emission`` branch is untouched (golden frames intact).
  * **optically-thin equivalence.** With a tiny ``absorption_coeff`` the per-step
    ``dτ`` is small, ``w = 1−e^{−dτ} → dτ``, and ``w·S → emission`` — so the RTE
    frame matches the legacy emission frame to a tight tolerance (the CKS-14
    back-compatibility proof, §1.2). It is NOT bit-identical (the paths differ at
    O(dτ²)), so the tolerance is small-but-nonzero.
  * **thick-regime quadrature divergence.** Legacy and CKS-14 evaluate the SAME
    continuum integral ``I = ∫ S e^{−τ} dτ`` (since ``transm·j·ds = transm·S·dτ``);
    they differ only in discretization. With a large ``absorption_coeff`` (big
    per-step ``dτ``) the legacy left-endpoint rectangle rule **over-counts** each
    opaque step by ``dτ/(1−e^{−dτ})``, while CKS-14 is the exact per-step solution
    for piecewise-constant ``S``. So the RTE disk must differ measurably from the
    legacy disk AND be the *dimmer, non-over-counting* one — this is CKS-14 acting
    as a better quadrature (its standalone value; the void LOOK needs CKS-15's
    self-shadow on top, which dims the materialised ``S`` by ``e^{−τ_shadow}``).

Noise is forced OFF and ``temperature_model: simple`` so the check is a clean
physics comparison (both frames share the same density field; only the march
accumulation differs). CUDA is mandatory (backend LOCKED to
``ti.init(arch=ti.cuda)`` per CLAUDE.md); the module skips cleanly without it.
"""
from __future__ import annotations

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

# Absorption settings (visualization knob, not physics) bracketing the two limits.
_THIN_ABSORPTION = 0.02    # per-step dτ ≪ 1 ⇒ w→dτ ⇒ RTE ≈ legacy emission sum
_THICK_ABSORPTION = 8.0    # transm collapses ⇒ legacy reads ~black, RTE → surface

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


def _render(source_function, absorption, *, drop_volumetric=False):
    """Render frame 0 disk-only RGB at a forced (source_function, absorption).

    ``drop_volumetric`` removes the whole ``disk.volumetric`` block (so the loader
    sees no flag at all) — used to prove the flag-off path equals the legacy code.
    Noise OFF, simple model, doppler_strength fixed so the comparison is clean.
    """
    key = (bool(source_function), float(absorption), bool(drop_volumetric))
    if key in _RENDER_CACHE:
        return _RENDER_CACHE[key]

    _ensure_cuda()
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = "simple"
    cfg["disk"]["doppler_strength"] = 1.0
    cfg["disk"]["absorption_coeff"] = float(absorption)
    cfg["disk"].setdefault("noise", {})["enabled"] = False
    if drop_volumetric:
        cfg["disk"].pop("volumetric", None)
    else:
        cfg["disk"].setdefault("volumetric", {})["source_function"] = bool(source_function)

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


def test_flag_off_is_bit_identical_to_no_volumetric_block():
    """``source_function: false`` ⇒ byte-for-byte the legacy march (dead code off).

    The default disk absorption is used here (the realistic frame); off-flag must
    equal the no-block render exactly — proves the entire CKS-14 path is inert when
    not opted in, so the pinned goldens (test_gpu_regression) are unaffected.
    """
    default_absb = float(tr.load_config()["disk"]["absorption_coeff"])
    off = _render(False, default_absb)["disk"]
    none = _render(False, default_absb, drop_volumetric=True)["disk"]
    assert np.array_equal(off, none)


def test_thin_limit_matches_legacy_emission():
    """CKS-14 back-compat: at tiny dτ the RTE march ≈ the legacy emission sum.

    ``w·S = (1−e^{−dτ})/dτ · emission → emission`` as dτ→0. With a small absorption
    the per-pixel disk radiance from source_function:true must match the legacy
    (false) frame to a tight relative tolerance — but NOT be bit-identical (the
    reduction is only first-order in dτ; an exact match would mean the flag did
    nothing).
    """
    off = _render(False, _THIN_ABSORPTION)["disk"]
    on = _render(True, _THIN_ABSORPTION)["disk"]
    assert np.isnan(on).sum() == 0

    mask = _disk_mask(off, on)
    assert mask.any(), "no disk pixels emitted — framing/absorption sanity"
    off_l = off[mask].sum(axis=1)
    on_l = on[mask].sum(axis=1)
    denom = float(off_l.sum())
    rel = float(np.abs(on_l - off_l).sum()) / max(denom, 1e-12)
    assert rel < 0.02, f"thin-limit RTE deviates from legacy by {rel:.3%} (expected <2%)"
    # ...but the flag is genuinely active (paths differ at O(dτ²)).
    assert not np.array_equal(off, on)


def test_source_function_changes_thick_disk():
    """CKS-14 thick limit: exact per-step quadrature ≠ the legacy rectangle rule.

    Legacy and RTE integrate the same ``∫ S e^{−τ} dτ``; at high absorption the big
    per-step ``dτ`` makes the legacy left-endpoint rectangle rule over-count each
    opaque step by ``dτ/(1−e^{−dτ}) > 1``. So the RTE disk must (a) differ from the
    legacy disk by a clear margin (the flag is genuinely active in the thick
    regime), (b) be the *dimmer* one (CKS-14 removes the over-count), and (c) stay
    NaN-free and still emit. This is CKS-14 as a better quadrature — NOT a brightness
    boost (the glowing-gas-with-voids look needs CKS-15 self-shadow on top).
    """
    off = _render(False, _THICK_ABSORPTION)
    on = _render(True, _THICK_ABSORPTION)
    assert on["hdr"] is not None and off["hdr"] is not None
    assert np.isnan(on["disk"]).sum() == 0

    mask = _disk_mask(off["disk"], on["disk"])
    assert mask.any()
    off_mean = float(off["disk"][mask].sum(axis=1).mean())
    on_mean = float(on["disk"][mask].sum(axis=1).mean())
    assert on_mean > 0.0                       # still emits
    rel = abs(on_mean - off_mean) / max(off_mean, 1e-12)
    assert rel > 0.03, f"thick-regime RTE barely differs from legacy ({rel:.3%})"
    assert on_mean < off_mean, (off_mean, on_mean)  # exact quadrature ≤ over-counting rectangle
