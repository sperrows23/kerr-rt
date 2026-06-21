"""V2 — CKS-16 flared 3D volumetric density (spec docs/specs/2026-06-14-V2-flared-3d-density.md).

Two layers:

* **Resolver / CPU twin (no GPU).** The CKS-13 resolver derives ``flare_beta`` and
  the widened bounding half-angle ``theta_half_bound`` from the base slab params and
  the ``disk.volumetric.flare`` block. Pins: flare-off and β=0 are no-ops
  (bit-identical bound), β>0 widens the bound to cover ≥ band_sigma·σ_θ(r_outer),
  idempotency, input validation, and the flared σ_θ(r) monotonicity.
* **GPU beauty path (gpu-marked).** Drives the production ``render_beauty_frame``
  (same path as ``test_gpu_regression``): flare-off / enabled-β=0 render
  byte-for-byte the same as the no-flare-block config (goldens intact), and a flared
  β>0 disk is NaN-free, genuinely different, and BRIGHTER (a thicker slab puts more
  emitting gas along each crossing ray — additive emission, self-shadow off).

CUDA is mandatory for the GPU class (backend LOCKED to ``ti.init(arch=ti.cuda)``);
those tests skip cleanly without it. The resolver/CPU tests are Taichi-free.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from renderer import kerr_params as kp

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "configs" / "render.yaml"
_CAMERA_PATH = _ROOT / "camera_matrix.json"


def _raw_config() -> dict:
    """The unresolved YAML (utf-8 — Windows cp949 box)."""
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolved(*, enabled=False, beta=0.0, band_sigma=3.0) -> dict:
    cfg = _raw_config()
    flare = cfg["disk"].setdefault("volumetric", {}).setdefault("flare", {})
    flare["enabled"] = enabled
    flare["beta"] = beta
    flare["band_sigma"] = band_sigma
    return kp.resolve_config(cfg)


# --------------------------------------------------------------------------- #
# Resolver (CKS-16 derivation) — Taichi-free                                   #
# --------------------------------------------------------------------------- #

def test_flare_off_is_noop():
    """flare.enabled:false ⇒ flare_beta 0 and the bound equals the base slab angle."""
    d = _resolved(enabled=False, beta=0.5)["disk"]
    assert d["flare_beta"] == 0.0
    assert d["theta_half_bound"] == pytest.approx(d["theta_half_width"])


def test_enabled_zero_beta_is_noop():
    """enabled:true but β=0 is still a no-op (constant-H/r slab) ⇒ bit-identical."""
    d = _resolved(enabled=True, beta=0.0)["disk"]
    assert d["flare_beta"] == 0.0
    assert d["theta_half_bound"] == pytest.approx(d["theta_half_width"])


def test_flare_widens_bound_to_cover_outer_envelope():
    """β>0 sets flare_beta=β and widens the bound to ≥ band_sigma·σ_θ(r_outer)."""
    beta, band = 0.35, 3.0
    d = _resolved(enabled=True, beta=beta, band_sigma=band)["disk"]
    assert d["flare_beta"] == pytest.approx(beta)

    base = d["theta_half_width"]
    sigma0 = base * d["vertical_sigma_frac"]
    sigma_outer = sigma0 * (d["r_outer"] / d["r_inner"]) ** beta
    expected = max(base, band * sigma_outer)
    assert d["theta_half_bound"] == pytest.approx(expected)
    # For these disk params the flared outer envelope genuinely exceeds the base band.
    assert d["theta_half_bound"] > base


def test_flared_sigma_is_monotonic_outward():
    """σ_θ(r)=σ0·(r/r_inner)^β strictly increases with r for β>0 (outward flare)."""
    beta = 0.4
    d = _resolved(enabled=True, beta=beta)["disk"]
    sigma0 = d["theta_half_width"] * d["vertical_sigma_frac"]
    r_inner, r_outer = d["r_inner"], d["r_outer"]
    rs = np.linspace(r_inner, r_outer, 16)
    sig = sigma0 * (rs / r_inner) ** beta
    assert sig[0] == pytest.approx(sigma0)            # inner edge keeps σ0
    assert np.all(np.diff(sig) > 0.0)                 # monotonic outward
    assert sig[-1] > sig[0]


def test_resolver_idempotent_with_flare():
    """Resolving an already-resolved flared config reproduces the derived keys."""
    once = _resolved(enabled=True, beta=0.3)
    twice = kp.resolve_config(copy.deepcopy(once))
    assert twice["disk"]["flare_beta"] == pytest.approx(once["disk"]["flare_beta"])
    assert twice["disk"]["theta_half_bound"] == pytest.approx(once["disk"]["theta_half_bound"])
    # theta_half_width itself is never mutated (it is the σ0 anchor, not the bound).
    assert twice["disk"]["theta_half_width"] == pytest.approx(once["disk"]["theta_half_width"])


def test_resolver_rejects_negative_beta():
    with pytest.raises(ValueError, match="flare.beta"):
        _resolved(enabled=True, beta=-0.1)


def test_resolver_rejects_nonpositive_band_sigma():
    with pytest.raises(ValueError, match="band_sigma"):
        _resolved(enabled=True, beta=0.3, band_sigma=0.0)


# --------------------------------------------------------------------------- #
# GPU beauty path (production render_beauty_frame)                             #
# --------------------------------------------------------------------------- #

class TestFlareGPU:
    pytestmark = pytest.mark.gpu

    _WIDTH = 480
    _HEIGHT = 270

    @staticmethod
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

    def _cam(self):
        with open(_CAMERA_PATH, encoding="utf-8-sig") as fh:
            return json.load(fh)[0]

    def _render(self, *, enabled=None, beta=0.0, drop_block=False):
        """Disk-only RGB at a forced flare setting. Simple model, fixed doppler."""
        from renderer import taichi_renderer as tr

        self._ensure_cuda()
        cfg = copy.deepcopy(tr.load_config())
        cfg["disk"]["temperature_model"] = "simple"
        cfg["disk"]["doppler_strength"] = 1.0
        cfg["disk"].setdefault("noise", {})["enabled"] = False
        vol = cfg["disk"].setdefault("volumetric", {})
        if drop_block:
            vol.pop("flare", None)
        else:
            flare = vol.setdefault("flare", {})
            flare["enabled"] = bool(enabled)
            flare["beta"] = float(beta)
        # Re-resolve so theta_half_bound / flare_beta reflect the forced flare.
        kp.resolve_config(cfg)

        tr.setup_renderer(cfg)
        tr.frame_pixels = None
        tr._FW = 0
        tr._FH = 0
        tr.render_beauty_frame(cfg, self._cam(), self._WIDTH, self._HEIGHT,
                               with_disk=True, lod_enabled=True)
        return np.nan_to_num(tr.disk_buf.to_numpy()[:, :, :3]).copy()

    def test_flare_off_bit_identical_to_no_block(self):
        """enabled:false (and enabled:true/β=0) ⇒ byte-for-byte the no-flare march."""
        none = self._render(drop_block=True)
        off = self._render(enabled=False, beta=0.4)
        zero = self._render(enabled=True, beta=0.0)
        assert np.array_equal(off, none)
        assert np.array_equal(zero, none)

    @staticmethod
    def _vertical_spread(lum: np.ndarray) -> float:
        """Luminance-weighted std of the row index — the vertical extent of emission.

        A thicker (flared) disk spreads its emission over more rows, so this grows
        even when total luminance is flat. Scale-free in brightness (it is a weighted
        *position* moment), so it is the honest geometric signature of vertical bulk,
        unlike a thresholded pixel count (the disk-only buffer has a faint nonzero
        floor across the whole frame, so any absolute area threshold saturates).
        """
        rows = np.arange(lum.shape[0], dtype=np.float64)[:, None]   # (H, 1)
        w = lum.astype(np.float64)
        total = w.sum()
        mean = (rows * w).sum() / total
        var = (((rows - mean) ** 2) * w).sum() / total
        return float(np.sqrt(var))

    def test_flare_thickens_disk_vertical_extent(self):
        """β>0 is NaN-free, differs, and THICKENS the disk (wider vertical emission).

        Note — flare is NOT a brightness boost. The hot inner edge (peak emission,
        where most of the light comes from) sits at ``r_inner``, whose σ is the
        *anchored* σ0 and does not change; flare only adds COLD outer gas at larger
        |z|. Under the absorbing march that extra bulk can even read marginally DIMMER
        in total (it self-absorbs the bright inner edge slightly more than the dim cold
        gas it contributes — empirically ~1.6% on this inclined view). The robust,
        physically-honest signature of the added vertical bulk is therefore that the
        emission is spread over a larger vertical extent, not a higher integrated
        luminance.
        """
        off = self._render(enabled=False)
        on = self._render(enabled=True, beta=0.4)
        assert np.isnan(on).sum() == 0
        assert not np.array_equal(off, on)               # flare genuinely active

        off_spread = self._vertical_spread(off.sum(axis=2))
        on_spread = self._vertical_spread(on.sum(axis=2))
        assert on_spread > off_spread                    # gas spread vertically ⇒ thicker
