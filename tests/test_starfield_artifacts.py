"""Visual-artifact regression for the escaped-ray background (disk OFF).

WHY THIS FILE EXISTS
--------------------
``tests/test_gpu_regression.py`` pins coarse scalars (disk peak, Doppler ratio, a
center-column seam ratio). Those passed green while the rendered starfield still
showed two plainly-visible artifacts (confirmed 2026-06-06 by inspecting the
isolated, disk-off starfield):

  A. **Radial star-smear** — in ``mode=texture`` the Formula-10 *isotropic* scalar
     mip-LOD cannot represent the anisotropic lensing footprint near the photon
     ring, so background stars smear into tangential streaks. **RESOLVED 2026-06-06**
     by promoting ``mode=dngr`` (Formula-13 anisotropic EWA) to the config default;
     ``test_background_has_no_radial_smear`` is now a *live* regression guard (no
     longer xfail) — it asserts the smear stays absent under the shipped default.
  B. **Spin-axis seam** — a thin vertical stripe on the meridian: a blocky
     coarse-mip column (``j_fold`` collapse) in ``texture``, a piled-up rope of
     point stars (Formula-13 Layer-A ``detJ`` blow-up) in ``dngr``. The dngr
     pileup was **FIXED 2026-06-06 by the R2 splat-placement rule** (SKILL.md
     Formula 13 guard (b′): on the seam, place Layer-A splats by the undeflected
     proper-separation footprint instead of the degenerate ``J⁻¹``). With R2 the
     *field is seam-free* — masking the single brightest lensed star drops this
     detector to z≈14 (the clean-field range), and the coarse center-column metric
     in ``test_gpu_regression`` is back to ~2.06×. This test nonetheless **remains
     ``xfail(strict=True)``**: the cinematic frame-0 framing leaves only an ~80-row
     sky band, in which a single bright lensed star (row 12, col ~1075) dominates a
     column's amplitude and the location-agnostic stripe metric reads z≈28 — a
     bright-point *confound*, not a residual seam (verified: the pre-R2 ``J⁻¹``
     render peaks at the same star/column at z≈27, so the score is unchanged by the
     fix). Making the detector bright-point-robust (clip the top ~1% before the
     column autocorrelation) is a deferred recalibration; until then the marker
     records this known confound. The ``texture`` blocky-stripe (a separate
     Formula-10 LOD issue) is also still unfixed — see spec §7.2.

The coarse seam ratio missed both: a low-frequency blocky stripe has small local
jumps, and the smear is field-wide rather than column-local. These tests encode
the *visual* correctness properties the coarse metrics lack.

POLICY
------
Pure image statistics — no GR/lensing formula is evaluated or re-derived here, so
nothing in this file is governed by ``skills/kerr-physics/SKILL.md``.

CALIBRATION + VALIDATION (see scripts log 2026-06-06, then deleted)
Each detector was validated for sensitivity AND specificity, not just fit to the
two real renders. A synthetic artifact-free field (smooth band + isotropic point
stars + shadow, 5 seeds) was used as ground truth; the named artifact was then
injected and an equal-energy off-axis perturbation used as the negative control:

    metric             clean(synth)  +control(neg)  +inject   dngr   texture  thresh
    smear coherence      0.01-0.03    iso 0.07-0.10  dir 0.75  0.257   0.50    < 0.36
    vertical-stripe z    9.3 - 14     hband 10-14    vstr 68+  27.8    138     < 20

  - smear: rises ONLY for a *directional* blur (isotropic Gaussian leaves it
    ~0.08), so it measures anisotropy, not blur. 0.36 sits in the 0.10->0.50 gap.
  - seam: rises ONLY for a *vertical* stripe that varies along its length
    (a uniform column is removed by the per-column demean — accepted: real seams
    are textured). An equal-energy horizontal band stays at clean levels.
    The threshold is 20 (NOT 12): a clean field scores 9-14, so 12 sat INSIDE the
    clean distribution and would false-positive. 20 is in the empty 14->27.8 gap,
    keeping margin above clean and below today's real seam.

As of the 2026-06 CKS migration both artifacts are live PASS guards. Artifact A
(smear) is fixed by the dngr anisotropic EWA (smear coherence 0.257 < 0.36).
Artifact B (seam) is now eliminated **at the source** by Cartesian Kerr-Schild:
the escaped-ray celestial direction is a genuine Cartesian unit vector (Formula
CKS-10), so there is no spin-axis meridian caustic, no φ-accumulation blow-up, and
no Layer-A detJ pileup — the residual BL/dngr seam reading (which had previously
kept ``test_background_has_no_vertical_seam_stripe`` at ``xfail(strict=True)``,
confounded by a bright lensed star in frame-0's thin sky band) drops below the
z=20 threshold, so the marker was removed and the test is a live guard. Calibrated
for 1280x720; keep that render size.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from renderer import taichi_renderer as tr

_ROOT = Path(__file__).resolve().parents[1]
_CAMERA_PATH = _ROOT / "camera_matrix.json"

_RENDER_W, _RENDER_H = 1280, 720
_SMEAR_COHERENCE_MAX = 0.36   # bright-feature structure-tensor coherence (streaky => high)
_SEAM_STRIPE_Z_MAX = 20.0     # strongest thin vertical stripe, MAD-z over columns
                              # (clean fields score 9-14; real seam 27.8+ -> 20 has margin)


# --------------------------------------------------------------------------- #
# Pure-numpy image-statistics helpers (no scipy)
# --------------------------------------------------------------------------- #
def _boxblur(a: np.ndarray, k: int) -> np.ndarray:
    """Separable k x k uniform blur, reflect-padded, same shape."""
    out = a.astype(np.float64)
    for axis in (0, 1):
        pad = k // 2
        ap = np.pad(out, [(pad, pad) if ax == axis else (0, 0) for ax in (0, 1)],
                    mode="reflect")
        cs = np.cumsum(ap, axis=axis)
        csz = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs], axis=axis)
        hi = [slice(None), slice(None)]; lo = [slice(None), slice(None)]
        hi[axis] = slice(k, None); lo[axis] = slice(0, -k)
        out = (csz[tuple(hi)] - csz[tuple(lo)]) / k
    return out


def _boxblur1d_h(a: np.ndarray, k: int) -> np.ndarray:
    """Horizontal-only k-wide moving average, reflect-padded, same shape."""
    pad = k // 2
    ap = np.pad(a, ((0, 0), (pad, pad)), mode="reflect")
    cs = np.cumsum(ap, axis=1)
    csz = np.concatenate([np.zeros((a.shape[0], 1)), cs], axis=1)
    return (csz[:, k:] - csz[:, :-k]) / k


def _shadow_geom(lum: np.ndarray) -> tuple[int, int]:
    """Centroid-x and top row of the *solid* shadow (dense-black region), as
    opposed to the scattered black-sky texels that also satisfy lum==0."""
    dark = (lum < 1e-6).astype(np.float64)
    solid = _boxblur(dark, 15) > 0.8
    ys, xs = np.nonzero(solid)
    if ys.size == 0:
        return lum.shape[1] // 2, lum.shape[0] // 2
    return int(round(xs.mean())), int(ys.min())


def smear_coherence(lum: np.ndarray) -> float:
    """Gradient-energy-weighted structure-tensor coherence over the brightest fine
    features. Streaks (one dominant gradient direction) -> ~1; isolated point
    stars (radial gradients) -> ~0. The smooth lensed band is removed by the
    high-pass so it cannot dominate."""
    hp = lum - _boxblur(lum, 9)
    pos = hp[hp > 0]
    if pos.size == 0:
        return 0.0
    bright = hp > np.percentile(pos, 99.5)
    gy, gx = np.gradient(lum)
    Jxx = _boxblur(gx * gx, 7); Jyy = _boxblur(gy * gy, 7); Jxy = _boxblur(gx * gy, 7)
    tr_ = Jxx + Jyy
    disc = np.sqrt(np.maximum(tr_ * tr_ / 4 - (Jxx * Jyy - Jxy * Jxy), 0.0))
    l1 = tr_ / 2 + disc; l2 = tr_ / 2 - disc
    coh = np.where(l1 + l2 > 1e-20, (l1 - l2) / (l1 + l2 + 1e-20), 0.0)
    w = gx * gx + gy * gy
    denom = w[bright].sum()
    return float((coh[bright] * w[bright]).sum() / denom) if denom > 1e-20 else 0.0


def seam_stripe_z(lum: np.ndarray) -> float:
    """Strongest thin vertical stripe in the background above the shadow, as a
    MAD-z over columns. Horizontal-only high pass turns a vertical stripe into a
    persistent bar; score = vertical lag-1 autocorrelation x amplitude. A curved
    lensed band (horizontal) and random point stars score low; smear streaks and
    the seam pileup score high. Location-agnostic on purpose."""
    _, shadow_top = _shadow_geom(lum)
    if shadow_top < 24:
        return 0.0
    bg = lum[:shadow_top, :]
    hh = bg - _boxblur1d_h(bg, 31)
    a = hh - hh.mean(axis=0, keepdims=True)
    vcoh = (a[1:] * a[:-1]).sum(axis=0) / ((a * a).sum(axis=0) + 1e-20)
    amp = np.sqrt((a * a).mean(axis=0))
    score = vcoh * amp
    m = score.size // 20                      # drop outer 5% (reflect-pad boundary)
    core = score[m:-m]
    med = np.median(core)
    mad = np.median(np.abs(core - med)) * 1.4826 + 1e-20
    return float((core.max() - med) / mad)


# --------------------------------------------------------------------------- #
# GPU fixture: render the CONFIG-DEFAULT starfield with the disk off
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def background_lum() -> np.ndarray:
    if not _CAMERA_PATH.exists():
        pytest.skip(f"camera_matrix.json not found at {_CAMERA_PATH}")

    import taichi as ti
    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without CUDA
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")

    cfg = tr.load_config()
    mode = cfg.get("starfield", {}).get("mode", "texture")
    # Skip if the configured default mode's required assets are absent.
    if mode == "dngr":
        if not (_ROOT / cfg["starfield"]["catalog_path"]).exists():
            pytest.skip("dngr default but ingested catalog missing")
        if not (_ROOT / cfg["starfield"]["diffuse_map"]).exists():
            pytest.skip("dngr default but diffuse Milky-Way map missing")
    else:
        if not (_ROOT / cfg["starmap"]["path"]).exists():
            pytest.skip("texture default but baked starmap missing")

    with open(_CAMERA_PATH, "r", encoding="utf-8-sig") as fh:
        cam = json.load(fh)[0]

    tr.setup_renderer(cfg)
    hdr = tr.render_beauty_frame(cfg, cam, _RENDER_W, _RENDER_H,
                                 with_disk=False, lod_enabled=True)
    return np.nan_to_num(hdr).sum(axis=2)


def test_background_has_no_radial_smear(background_lum):
    # Live regression guard since 2026-06-06: the dngr default's anisotropic EWA
    # keeps escaped stars sharp (coherence ~0.257). If this fails, the smear has
    # regressed (e.g. default reverted to texture, or EWA path broke).
    coh = smear_coherence(background_lum)
    assert coh < _SMEAR_COHERENCE_MAX, (
        f"bright-feature coherence {coh:.3f} >= {_SMEAR_COHERENCE_MAX} "
        "=> stars are smeared into directional streaks (anisotropic-LOD aliasing)")


def test_background_has_no_vertical_seam_stripe(background_lum):
    # Live regression guard since the 2026-06 CKS migration: the BH spin-axis seam
    # is eliminated at the source (Formula CKS-10 — the escaped-ray celestial
    # direction is a genuine Cartesian unit vector, so there is no meridian caustic,
    # no φ-accumulation blow-up, and no Layer-A detJ pileup). Under CKS this metric
    # drops below threshold even in frame-0's thin sky band, so it now PASSES as a
    # live guard (was xfail(strict) under the BL/dngr path, where a residual seam +
    # bright-star confound held it high). A regression back to a coordinate-singular
    # path would reintroduce the stripe and trip this.
    z = seam_stripe_z(background_lum)
    assert z < _SEAM_STRIPE_Z_MAX, (
        f"strongest vertical stripe z={z:.1f} >= {_SEAM_STRIPE_Z_MAX} "
        "=> a thin tall vertical seam/streak exists in the background")
