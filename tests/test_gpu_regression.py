"""GPU beauty-render regression test — automated Doppler / NaN / disk-peak guard.

Physics policy (see CLAUDE.md): every GR formula exercised here lives in
``skills/kerr-physics/SKILL.md`` and the production renderer
(``renderer.taichi_renderer``). This test does NOT re-derive or re-implement any
physics — it drives the *production* entrypoint ``render_beauty_frame`` (the same
code path ``scripts/gpu_test.py`` and ``scripts/export_exr.py`` use) and asserts on
the rendered HDR buffer.

It automates the manual "real physics regression guard" (``scripts/gpu_test.py``'s
left/right Doppler-asymmetry smoke check; see PROJECT.md §6), turning it into a
pytest that CI can run unattended.

What it guarantees, per frame (frame 0 of ``camera_matrix.json``, disk on):

  1. Doppler asymmetry — the g⁴-beamed approaching (right) disk edge is far
     brighter than the receding (left) edge. The left/right luminance ratio must
     stay in a physical band (Formulas CKS-9/9; CKS baseline ≈ 4.32×). A sign flip
     or a broken g-factor collapses this ratio.
  2. NaN-free — no pixel is NaN. RK4 overshoot near the inner disk edge would
     poison pixels. (The BL polar 1/sin²θ blow-up is *gone at the source* under
     CKS — the chart is regular on the spin axis and across the horizon.)
  3. Disk emission peak — the brightest pixel (the beamed disk edge) must match a
     pinned reference within tolerance. Drift flags a change in the disk emission
     / redshift chain (Formulas 3 / CKS-8 / CKS-9 / 9).
  4. Spin-axis seam (A4) — no sharp vertical luminance discontinuity at the center
     column. Under CKS the BH spin-axis seam is **eliminated at the source**
     (Formula CKS-10: the escaped-ray celestial direction is a genuine Cartesian
     unit vector — no meridian caustic, no φ-accumulation blow-up), so this is now
     a cheap sanity guard that should sit near the off-seam background baseline
     (~2×) rather than the pre-CKS BL/dngr seam spikes. A reintroduced vertical
     band (e.g. a regression back to a coordinate-singular path) still trips it.

CUDA is mandatory (the backend is LOCKED to ``ti.init(arch=ti.cuda)`` per
CLAUDE.md). On a host without a working CUDA backend the whole module skips
cleanly — collection stays cheap because the Taichi init is deferred into the
fixture, not done at import time.
"""

from __future__ import annotations

import copy
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
# CKS baseline (2026-06 migration, measured on RTX 5060): Doppler 4.32×, disk peak
# 6.17. The CKS affine emission measure reweights the g⁴ integral relative to the
# retired BL Mino path (which read 7.77× / 12.77), so these are NOT the old BL
# numbers — see PROJECT.md §6 "CKS migration" and the render.yaml emission/
# absorption recalibration. The band stays the same fractional width as the old
# guard, recentred on the CKS value.
# Re-anchored goldens — re-MEASURED on the current calibration (simple model,
# doppler_strength = 1.0, frame 0, RTX 5060). The CKS-era 4.32× / 6.1667 references
# held through the doppler_strength knob landing (PROJECT.md §7 measured 4.317× /
# 6.1665), but the D3 kerr_params resolver (commit 30f8511) now DERIVES the disk
# amplitude T_0 from disk.target_peak_temperature (CKS-13). That rescaled the disk
# emission peak (~2.3×) and — through the g-dependent blackbody chroma feeding the
# channel-sum luminance — the half-frame Doppler ratio. Rather than re-pin a single
# brittle s = 1.0 band, these guards now verify the beaming RESPONSE to
# doppler_strength (g_eff = g^s): monotone in s, symmetric at s = 0, anchored at
# the re-measured s = 1.0 value. Disk-only metrics (disk_buf RGB) keep the
# asymmetric DNGR sky from contaminating the ratio.
_DOPPLER_RATIO_REF = 5.15   # disk-only right/left at s = 1.0 (simple)
_DOPPLER_RATIO_RTOL = 0.10
_DISK_MAX_REF = 14.45       # disk emission peak at s = 1.0 (simple)
_DISK_MAX_RTOL = 0.08
# s = 0 ⇒ g_eff ≡ 1 (beaming off): the only residual left/right split is pure
# lensing geometry (frame-dragging), which the knob deliberately does not touch.
_SYMMETRIC_RATIO_MAX = 1.5
# doppler_strength samples the scaling guards sweep (monotone non-decreasing).
_STRENGTH_SWEEP = (0.0, 0.5, 1.0)

# Spin-axis seam guard (A4): max tolerated ratio of the center-column luminance
# jump to the median background jump. Measured ≈1.9× on the legacy texture path
# (the fold-saturation fix) and ≈2.06× on the dngr default after the R2 splat-
# placement fix (SKILL.md F13 guard (b′)); a reintroduced static band is a dominant
# vertical discontinuity (many× median — the pre-R2 dngr pileup measured ≈15×).
# 6.0 sits in the empty gap above both fixed modes and well below a real seam.
_SEAM_JUMP_OVER_MEDIAN_MAX = 6.0


@pytest.fixture(scope="module")
def beauty_frame() -> dict:
    """Render one production beauty frame on CUDA, or skip if CUDA is unavailable.

    Returns a dict of derived metrics so the individual assertions stay cheap and
    read independently:
      nan_count     : number of NaN pixels in the HDR buffer
      doppler_ratio : max(left, right) / min(left, right) half-frame mean luminance
      disk_max      : peak disk emission (g⁴-beamed disk edge, Pipe B) — read from
                      disk_buf RGB, so it is independent of background star brightness
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
    # D2.4: the production config ships disk.noise.enabled: true. This regression
    # guards the GR / redshift / calibration chain (a pure-physics check), so force
    # the procedural turbulence OFF — its whole job is to break disk symmetry, which
    # would make the pinned Doppler ratio / disk-peak goldens fragile. Noise has its
    # own twin/agreement guards (test_disk_noise.py).
    cfg.setdefault("disk", {}).setdefault("noise", {})["enabled"] = False

    # camera_matrix.json is UTF-8-with-BOM (Blender export), per scripts/gpu_test.py.
    with open(_CAMERA_PATH, encoding="utf-8-sig") as fh:
        frames = json.load(fh)
    if not 0 <= _FRAME_INDEX < len(frames):
        pytest.skip(f"frame {_FRAME_INDEX} out of range (have {len(frames)})")
    cam = frames[_FRAME_INDEX]

    # Real production setup + render (re-inits Taichi on CUDA with the configured
    # device memory, then uploads the starmap mip pyramid).
    tr.setup_renderer(cfg)
    hdr = tr.render_beauty_frame(cfg, cam, _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)

    nan_count = int(np.isnan(hdr).sum())

    # Doppler asymmetry: per-pixel luminance = channel sum, split into left/right
    # halves (same metric as scripts/gpu_test.py).
    lum = hdr.sum(axis=2)
    w = hdr.shape[1]
    left = float(lum[:, : w // 2].mean())
    right = float(lum[:, w // 2 :].mean())
    doppler_ratio = max(left, right) / max(min(left, right), 1e-9)

    # Disk emission peak (Pipe B) read straight from the disk buffer's RGB, so the
    # guard is robust to background brightness: with luminous Layer-A stars the
    # global frame max can be a lensed star, not the disk. disk_buf = (disk_rgb,
    # transmittance) and the final pixel is disk_rgb + transm*bg, so the max over
    # disk_buf[..., :3] is the beamed disk edge independent of the sky behind it
    # (at the peak transm≈0, so it matches the historical frame-max reference).
    disk_max = float(np.nan_to_num(tr.disk_buf.to_numpy()[:, :, :3]).max())

    # Spin-axis meridian seam guard (A4). The committed center-seam fix collapses
    # escaped pixels straddling the spin-axis meridian caustic to the coarsest mip,
    # killing the former center "static" band. Measure column-to-column smoothness
    # at the center column in the TOP sky band (above the edge-on disk), averaging
    # over rows so isolated point stars wash out while a persistent vertical seam
    # survives. Reference: the center jump is ~1.9× the median background jump on
    # the fixed code path; a reintroduced static band spikes far above that.
    safe_lum = np.nan_to_num(lum)
    top = safe_lum[: safe_lum.shape[0] // 5, :]
    col_mean = top.mean(axis=0)
    col_diff = np.abs(np.diff(col_mean))
    c = w // 2
    seam_center_jump = float(col_diff[c - 4 : c + 4].max())
    # Baseline background variation: interior columns, excluding the center
    # neighborhood so the seam under test can't inflate its own reference.
    interior = np.concatenate([col_diff[5 : c - 4], col_diff[c + 4 : -5]])
    seam_bg_median = float(np.median(interior))

    return {
        "nan_count": nan_count,
        "doppler_ratio": doppler_ratio,
        "disk_max": disk_max,
        "left": left,
        "right": right,
        "seam_center_jump": seam_center_jump,
        "seam_bg_median": seam_bg_median,
    }


# --------------------------------------------------------------------------- #
# Forced-strength disk renders for the doppler_strength scaling guards.
#
# Unlike ``beauty_frame`` (which renders the YAML-configured frame, whatever the
# user's current doppler_strength / temperature_model), these render frame 0 at an
# EXPLICIT (doppler_strength, model) so the guards control the knob instead of
# inheriting it. Results are cached per (model, s) so the multi-strength sweep
# shares renders across tests. Metrics are read from disk_buf RGB (disk emission
# only), making the left/right ratio independent of the asymmetric DNGR sky.
# --------------------------------------------------------------------------- #
_DISK_METRICS_CACHE: dict = {}


def _ensure_cuda():
    """Probe the LOCKED CUDA backend or skip (mirrors the beauty_frame fixture)."""
    if not _CAMERA_PATH.exists():
        pytest.skip(f"camera_matrix.json not found at {_CAMERA_PATH}")
    import taichi as ti

    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without a CUDA driver
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")


def _disk_metrics(doppler_strength: float, model: str = "simple") -> dict:
    """Render frame 0 disk-only metrics at a forced (doppler_strength, model)."""
    key = (model, float(doppler_strength))
    if key in _DISK_METRICS_CACHE:
        return _DISK_METRICS_CACHE[key]

    _ensure_cuda()
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"]["temperature_model"] = model
    cfg["disk"]["doppler_strength"] = float(doppler_strength)
    # Force the D2.4 procedural turbulence off — these guards pin the GR/beaming
    # response, not the art (see the beauty_frame fixture note).
    cfg["disk"].setdefault("noise", {})["enabled"] = False

    with open(_CAMERA_PATH, encoding="utf-8-sig") as fh:
        cam = json.load(fh)[_FRAME_INDEX]

    # setup_renderer re-runs ti.init (destroying all fields); reset the cached frame
    # buffers so _alloc_frame rebuilds them against the fresh runtime (same stale-
    # handle workaround the page_thorne / s=0 guards already use).
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    hdr = tr.render_beauty_frame(cfg, cam, _WIDTH, _HEIGHT, with_disk=True, lod_enabled=True)

    disk = np.nan_to_num(tr.disk_buf.to_numpy()[:, :, :3])
    dlum = disk.sum(axis=2)
    w = dlum.shape[1]
    left = float(dlum[:, : w // 2].mean())
    right = float(dlum[:, w // 2 :].mean())
    m = {
        "nan_count": int(np.isnan(hdr).sum()),
        "left": left,
        "right": right,
        "ratio": max(left, right) / max(min(left, right), 1e-9),
        "disk_max": float(disk.max()),
    }
    _DISK_METRICS_CACHE[key] = m
    return m


def test_no_nan_pixels(beauty_frame):
    assert beauty_frame["nan_count"] == 0


def test_doppler_asymmetry_scales_with_strength():
    """g⁴ beaming must grow monotonically with disk.doppler_strength (g_eff = g^s).

    Replaces the old fixed [3.8, 4.9] band, which silently assumed s = 1.0 AND the
    pre-D3 disk calibration. Renders the disk-only (sky-independent) left/right
    luminance split at several strengths and asserts the physically required shape:
      * s = 0  ⇒ near-symmetric (only residual frame-dragging lensing, < 1.5),
      * the approaching (right) edge is the bright one whenever s > 0,
      * the ratio is monotone non-decreasing in s (more shift ⇒ more beaming),
      * at full physics (s = 1) it matches the re-anchored golden.
    A sign flip or a broken g-factor collapses or inverts this response.
    """
    ms = [_disk_metrics(s, "simple") for s in _STRENGTH_SWEEP]
    for s, m in zip(_STRENGTH_SWEEP, ms, strict=True):
        assert m["nan_count"] == 0, (s, m)
        if s > 0.0:
            assert m["right"] > m["left"], (s, m)  # approaching edge is brighter

    ratios = [m["ratio"] for m in ms]
    assert ratios[0] <= _SYMMETRIC_RATIO_MAX, ratios  # s = 0 symmetric floor
    assert ratios == sorted(ratios), ratios          # monotone non-decreasing in s
    assert ratios[-1] > ratios[0]                     # genuinely responds to the knob
    assert ratios[-1] == pytest.approx(_DOPPLER_RATIO_REF, rel=_DOPPLER_RATIO_RTOL)


def test_disk_peak_scales_with_strength():
    """Disk emission peak must brighten monotonically with doppler_strength and
    match the re-anchored s = 1.0 golden.

    The g⁴ beaming raises the approaching-edge peak as s grows (g_eff = g^s). The
    old _DISK_MAX_REF = 6.1667 predated the D3 T_0 derivation (commit 30f8511); it
    is re-anchored to the current simple-model s = 1.0 peak. Read from disk_buf so a
    lensed background star can never stand in for the disk edge.
    """
    ms = [_disk_metrics(s, "simple") for s in _STRENGTH_SWEEP]
    peaks = [m["disk_max"] for m in ms]
    for s, m in zip(_STRENGTH_SWEEP, ms, strict=True):
        assert m["nan_count"] == 0, (s, m)
        assert m["disk_max"] > 0.0, (s, m)
    assert peaks == sorted(peaks), peaks  # beaming brightens the peak with s
    assert peaks[-1] > peaks[0]
    assert peaks[-1] == pytest.approx(_DISK_MAX_REF, rel=_DISK_MAX_RTOL)


def test_no_spin_axis_seam(beauty_frame):
    # Under CKS the BH spin-axis seam is eliminated at the source (Formula CKS-10),
    # so the center-column jump sits at the off-seam background baseline (~2×), well
    # under the 6.0 limit. A regression back to a coordinate-singular path (BL
    # meridian caustic / φ-accumulation) would reintroduce a sharp vertical
    # discontinuity here and trip the guard.
    jump = beauty_frame["seam_center_jump"]
    baseline = max(beauty_frame["seam_bg_median"], 1e-9)
    assert jump / baseline <= _SEAM_JUMP_OVER_MEDIAN_MAX


def test_page_thorne_disk_model_renders():
    """D1 guard: the page_thorne flux LUT path renders a physical, beamed frame.

    Forces ``doppler_strength = 1.0`` (full physics) so the beaming guard is
    independent of the YAML's visualization knob — at the configured s = 0.1 the
    asymmetry is deliberately suppressed (g_eff = g^0.1) and must not be mistaken for
    a regression. Renders frame 0 with ``disk.temperature_model = page_thorne`` on a
    deepcopied cfg (the simple-model goldens are untouched). Asserts the Page-Thorne
    branch is NaN-free, still g⁴-Doppler-beamed (right > left by > 2×, disk-only),
    and actually emits (peak disk luminance > 0).
    """
    m = _disk_metrics(1.0, "page_thorne")
    assert m["nan_count"] == 0
    assert m["right"] > m["left"]  # approaching edge is the bright one (g⁴ beaming)
    assert m["ratio"] > 2.0
    assert m["disk_max"] > 0.0


def test_doppler_strength_zero_symmetrizes_disk():
    """``disk.doppler_strength`` guard: s=0 ⇒ g_eff≡1 ⇒ the beamed asymmetry dies.

    Visualization knob, not physics (g_eff = g^s feeding both g⁴ and the blackbody
    chroma; default 1.0 leaves the kernel path bit-identical). With s=0 the disk's
    left/right split must collapse to near-symmetric — measured on ``disk_buf``
    (disk emission only), since the DNGR sky background is itself asymmetric.
    Residual asymmetry from frame-dragged photon paths (lensing geometry, which the
    knob deliberately does NOT touch) is allowed, hence a loose < 1.5 bound vs the
    full-physics beamed ratio. Forces ``temperature_model: simple`` so the bound
    does not depend on the user's current YAML state; shares the s=0 render with the
    scaling sweep above via the metrics cache.
    """
    m = _disk_metrics(0.0, "simple")
    assert m["nan_count"] == 0
    assert m["disk_max"] > 0.0  # still emits with the shift disabled
    assert m["ratio"] < _SYMMETRIC_RATIO_MAX, (m["left"], m["right"])
