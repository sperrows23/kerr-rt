"""DNGR background (Formula 13) tests — Layer A point stars + Layer B diffuse.

Physics policy (CLAUDE.md): every GR/lensing formula exercised here lives in
``skills/kerr-physics/SKILL.md`` (Formula 13) and the production renderer
(``renderer.taichi_renderer``). Nothing is re-derived. Two host-only checks pin
the parts that do not need a GPU — the magnification normalization (guard (a):
μ→1 in flat space) and the catalog cell-grid binning — and a CUDA-gated smoke
render confirms the two-layer path produces sharp, finite point stars.

The host μ check mirrors, in NumPy, the *exact* arithmetic of the kernel's
``_dngr_shade`` (FD beam Jacobian ``det J`` + analytic per-pixel solid angle
``dΩ``): a flat-space pinhole camera, where there is no lensing, must yield
μ = dΩ / |det J · sinθ′| = 1 everywhere. This is the regression guard for
Formula-13 guard (a) — owner-approved 2026-06-05.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from renderer import taichi_renderer as tr

_ROOT = Path(__file__).resolve().parents[1]
_CAMERA_PATH = _ROOT / "camera_matrix.json"
_CATALOG_PATH = _ROOT / "assets" / "stars.npy"


# --------------------------------------------------------------------------- #
# Host: magnification normalization (Formula 13 guard (a)) — μ → 1 in flat space
# --------------------------------------------------------------------------- #
def _flat_space_mu(width: int, height: int, fov_deg: float):
    """μ over a flat-space pinhole camera, mirroring ``_dngr_shade`` arithmetic.

    Exit direction == camera ray direction (no lensing); for an orthonormal,
    off-pole camera basis the source footprint |det J · sinθ′| equals the
    analytic pixel solid angle dΩ, so μ must be 1 to FD discretization error.
    """
    tan_y = math.tan(math.radians(fov_deg) / 2.0)
    tan_x = tan_y * (width / height)
    # Off-pole orthonormal frame (look down +x; rays stay near the equator).
    F = np.array([1.0, 0.0, 0.0])
    R = np.array([0.0, 1.0, 0.0])
    U = np.array([0.0, 0.0, 1.0])

    def exit_angles(px, py):
        sx = (2.0 * (px + 0.5) / width - 1.0) * tan_x
        sy = (1.0 - 2.0 * (py + 0.5) / height) * tan_y
        d = F + sx * R + sy * U
        d /= np.linalg.norm(d)
        return math.acos(max(-1.0, min(1.0, d[2]))), math.atan2(d[1], d[0])

    def wrap(x):
        return x - 2.0 * math.pi * round(x / (2.0 * math.pi))

    out = []
    for py in range(0, height - 1, max(1, height // 12)):
        for px in range(0, width - 1, max(1, width // 12)):
            th, ph = exit_angles(px, py)
            thx, phx = exit_angles(px + 1, py)
            thy, phy = exit_angles(px, py + 1)
            dthx, dphx = thx - th, wrap(phx - ph)
            dthy, dphy = thy - th, wrap(phy - ph)
            src = abs(dthx * dphy - dthy * dphx) * math.sin(th)
            sx = (2.0 * (px + 0.5) / width - 1.0) * tan_x
            sy = (1.0 - 2.0 * (py + 0.5) / height) * tan_y
            d_omega = (4.0 * tan_x * tan_y / (width * height)) / (1.0 + sx * sx + sy * sy) ** 1.5
            out.append(d_omega / src)
    return np.asarray(out)


def test_magnification_normalizes_to_one_in_flat_space():
    mu = _flat_space_mu(200, 200, 60.0)
    # FD discretization leaves only a sub-percent ripple around 1.
    assert mu.mean() == pytest.approx(1.0, abs=2e-3)
    assert mu.min() > 0.98
    assert mu.max() < 1.02


# --------------------------------------------------------------------------- #
# Host: catalog cell-grid binning (Layer A gather index)
# --------------------------------------------------------------------------- #
def test_build_star_grid_csr_is_consistent():
    cols, rows = 8, 4
    # Four stars placed in known cells (θ′ row = θ′/π·rows, φ′ col = φ′/2π·cols).
    catalog = np.array([
        [0.05, 0.05, 1.0, 0.0, 0.0],                      # row 0, col 0
        [math.pi - 0.05, 2 * math.pi - 0.05, 0.0, 1.0, 0.0],  # row 3, col 7
        [math.pi / 2, math.pi, 0.0, 0.0, 1.0],            # row 2, col 4
        [math.pi / 2, math.pi, 0.5, 0.5, 0.5],            # row 2, col 4 (same cell)
    ], dtype=np.float32)

    theta, _phi, flux, starts, counts = tr._build_star_grid(catalog, cols, rows)

    assert counts.sum() == catalog.shape[0]               # every star binned once
    assert theta.shape == (4,) and flux.shape == (4, 3)
    assert starts.shape == (rows * cols,)
    # The shared cell (row 2, col 4) holds exactly two stars.
    shared = 2 * cols + 4
    assert counts[shared] == 2
    s = starts[shared]
    block = flux[s:s + counts[shared]]
    assert np.allclose(np.sort(block[:, 2]), [0.5, 1.0])  # both blue-channel stars
    # CSR offsets are a prefix sum of counts.
    assert starts[0] == 0
    assert np.array_equal(starts[1:], np.cumsum(counts)[:-1])


# --------------------------------------------------------------------------- #
# GPU: the two-layer dngr render produces finite, sharp point stars
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dngr_frame() -> dict:
    if not _CAMERA_PATH.exists():
        pytest.skip(f"camera_matrix.json not found at {_CAMERA_PATH}")
    if not _CATALOG_PATH.exists():
        pytest.skip(f"ingested catalog not found at {_CATALOG_PATH} "
                    "(run scripts/ingest_stars.py)")

    import taichi as ti
    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without CUDA
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")

    cfg = tr.load_config()
    if "diffuse_map" in cfg.get("starfield", {}):
        if not (_ROOT / cfg["starfield"]["diffuse_map"]).exists():
            pytest.skip("diffuse Milky-Way map not present")
    cfg["starfield"]["mode"] = "dngr"

    with open(_CAMERA_PATH, "r", encoding="utf-8-sig") as fh:
        frames = json.load(fh)
    cam = frames[0]

    tr.setup_renderer(cfg)
    # Disk off so the background (Layers A+B) is what we measure.
    hdr = tr.render_beauty_frame(cfg, cam, 960, 540, with_disk=False, lod_enabled=True)
    lum = hdr.sum(axis=2)
    return {"hdr": hdr, "lum": lum}


def test_dngr_render_is_finite(dngr_frame):
    hdr = dngr_frame["hdr"]
    assert int(np.isnan(hdr).sum()) == 0
    assert int(np.isinf(hdr).sum()) == 0


def test_dngr_has_sharp_point_stars(dngr_frame):
    lum = dngr_frame["lum"]
    lit = lum[lum > 0.0]
    # Point stars are energy gathers: the peak background luminance is many times
    # the diffuse median (a baked-texture starmap could never spike this far).
    assert float(lum.max()) > 20.0 * float(np.median(lit))
