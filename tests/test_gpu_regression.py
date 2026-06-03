"""GPU beauty-render regression test — automated Doppler / NaN / disk-peak guard.

Physics policy (see CLAUDE.md): every GR formula exercised here lives in
``skills/kerr-physics/SKILL.md`` and the production renderer
(``renderer.taichi_renderer``). This test does NOT re-derive or re-implement any
physics — it drives the *production* entrypoint ``render_beauty_frame`` (the same
code path ``scripts/gpu_test.py`` and ``scripts/export_exr.py`` use) and asserts on
the rendered HDR buffer.

It automates the manual smoke check that IMPLEMENTATION_PLAN.md calls the "real
physics regression guard" (``scripts/gpu_test.py``'s left/right Doppler-asymmetry
test), turning it into a pytest that CI can run unattended.

What it guarantees, per frame (frame 0 of ``camera_matrix.json``, disk on):

  1. Doppler asymmetry — the g⁴-beamed approaching (right) disk edge is far
     brighter than the receding (left) edge. The left/right luminance ratio must
     stay in a physical band (Formulas 8/9; baseline ≈ 7.77×). A sign flip or a
     broken g-factor collapses this ratio.
  2. NaN-free — no pixel is NaN. RK4 overshoot near the inner disk edge or a
     polar 1/sin²θ blow-up (guarded by Formula 12's Θ_u) would poison pixels.
  3. Disk emission peak — the brightest pixel (the beamed disk edge) must match a
     pinned reference within tolerance. Drift flags a change in the disk emission
     / redshift chain (Formulas 3/4/5/8/9).

CUDA is mandatory (the backend is LOCKED to ``ti.init(arch=ti.cuda)`` per
CLAUDE.md). On a host without a working CUDA backend the whole module skips
cleanly — collection stays cheap because the Taichi init is deferred into the
fixture, not done at import time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# `renderer` is importable via pyproject's pythonpath=["src"] (same as the other
# tests); no sys.path manipulation needed.
from renderer import taichi_renderer as tr

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).resolve().parents[1]
_CAMERA_PATH = _ROOT / "camera_matrix.json"

# --------------------------------------------------------------------------- #
# Test-only knobs (NOT physics): a single, modest, deterministic frame. The
# render itself is sub-second at this resolution; the fixed cost is the one-time
# CUDA JIT + 16K starmap upload in setup_renderer (shared across all asserts via
# the module-scoped fixture). Frame 0 is the canonical regression framing used by
# scripts/gpu_test.py — edge-on, so the left/right Doppler split is meaningful.
# --------------------------------------------------------------------------- #
_FRAME_INDEX = 0
_WIDTH = 1920
_HEIGHT = 1080

# --------------------------------------------------------------------------- #
# Golden references — MEASURED on the production code path (RTX 5060, sm_120,
# Taichi 1.7.4 CUDA), not copied from docs. Tolerances are wide enough to absorb
# cross-GPU last-bit FP / driver variance but tight enough that a real physics
# regression (e.g. the ~10% disk-peak shift between Phase-1 and the c45d24b fix)
# trips them.
# --------------------------------------------------------------------------- #
_DOPPLER_RATIO_MIN = 7.0      # baseline 7.77× (matches gpu_test "≈7–8×" guard)
_DOPPLER_RATIO_MAX = 8.5
_DISK_MAX_REF = 12.7707       # peak HDR (disk emission edge); rel tol below
_DISK_MAX_RTOL = 0.05


@pytest.fixture(scope="module")
def beauty_frame() -> dict:
    """Render one production beauty frame on CUDA, or skip if CUDA is unavailable.

    Returns a dict of derived metrics so the individual assertions stay cheap and
    read independently:
      nan_count     : number of NaN pixels in the HDR buffer
      doppler_ratio : max(left, right) / min(left, right) half-frame mean luminance
      disk_max      : peak HDR value (the g⁴-beamed disk edge, Pipe B)
    """
    if not _CAMERA_PATH.exists():
        pytest.skip(f"camera_matrix.json not found at {_CAMERA_PATH}")

    import taichi as ti

    # Probe the LOCKED backend first (cheap, no starmap upload). Taichi falls back
    # to CPU when CUDA is unavailable instead of raising, so check the active arch.
    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without a CUDA driver
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")

    cfg = tr.load_config()

    # camera_matrix.json is UTF-8-with-BOM (Blender export), per scripts/gpu_test.py.
    with open(_CAMERA_PATH, "r", encoding="utf-8-sig") as fh:
        frames = json.load(fh)
    if not 0 <= _FRAME_INDEX < len(frames):
        pytest.skip(f"frame {_FRAME_INDEX} out of range (have {len(frames)})")
    cam = frames[_FRAME_INDEX]

    # Real production setup + render (re-inits Taichi on CUDA with the configured
    # device memory, then uploads the starmap mip pyramid).
    tr.setup_renderer(cfg)
    hdr = tr.render_beauty_frame(
        cfg, cam, _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True
    )

    nan_count = int(np.isnan(hdr).sum())

    # Doppler asymmetry: per-pixel luminance = channel sum, split into left/right
    # halves (same metric as scripts/gpu_test.py).
    lum = hdr.sum(axis=2)
    w = hdr.shape[1]
    left = float(lum[:, : w // 2].mean())
    right = float(lum[:, w // 2:].mean())
    doppler_ratio = max(left, right) / max(min(left, right), 1e-9)

    # With the disk on, the global peak pixel IS the beamed disk edge (frame max
    # == disk_buf max), so hdr.max() guards the disk emission peak via public API.
    disk_max = float(np.nan_to_num(hdr).max())

    return {
        "nan_count": nan_count,
        "doppler_ratio": doppler_ratio,
        "disk_max": disk_max,
        "left": left,
        "right": right,
    }


def test_no_nan_pixels(beauty_frame):
    assert beauty_frame["nan_count"] == 0


def test_doppler_asymmetry_in_band(beauty_frame):
    ratio = beauty_frame["doppler_ratio"]
    # The approaching (right) edge must be the bright one (g⁴ beaming).
    assert beauty_frame["right"] > beauty_frame["left"]
    assert _DOPPLER_RATIO_MIN <= ratio <= _DOPPLER_RATIO_MAX


def test_disk_peak_matches_reference(beauty_frame):
    assert beauty_frame["disk_max"] == pytest.approx(
        _DISK_MAX_REF, rel=_DISK_MAX_RTOL
    )
