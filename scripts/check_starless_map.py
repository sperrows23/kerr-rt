"""Starless Layer-B acceptance check (DNGR Formula 13 / PROJECT.md §8).

The DNGR Hybrid background (SKILL.md Formula 13) draws the sky as two decoupled
layers:

  * **Layer A** — a point-star *catalog* gathered as energy and brightened by the
    per-pixel lensing magnification. Stars stay sharp (isotropic J^-1 splat).
  * **Layer B** — a low-frequency Milky-Way *diffuse* plate sampled with an
    **anisotropic EWA** footprint. Toward a wide-FOV screen corner that footprint
    is elongated (~2.3x at a 90deg corner), so ANY resolvable point source baked
    into Layer B is smeared into a directional streak — the edge-elongation
    artifact. Layer B must therefore be *genuinely star-free*: no point sources
    broader (sharper) than the corner EWA footprint can survive.

This script is the quantitative gate for that requirement. It is the tool that
disproved the old "milkyway_2020 omits the bright-star field" config claim
(milkyway_2020 and starmap_2020 measured an identical ~2% sharp-spike density)
and that validated the StarNet2 plate (star_image/starmap_final.exr): sharp >10x
spikes dropped ~46x, to ~0.045% of lit pixels.

Metric
------
Per equirectangular pixel:

    lum      = Rec.709 luminance of the linear RGB plate
    boxmean  = separable BOX x BOX running mean of `lum`
               (wraps in phi / columns, edge-clamps in theta / rows — the
                equirect topology; matches starmap.celestial_to_uv)
    ratio    = lum / (boxmean + eps)          # local contrast vs neighbourhood

A pixel is **lit** (part of the galaxy band, not black void) when its luminance
is at or above the *lit floor* = the **median over the positive (non-zero)
pixels** — i.e. the median brightness *within* the galaxy band. (The global
median is unusable here: a StarNet2-cleaned plate is mostly hard zeros, so its
global median is 0 and the floor would collapse onto near-black processing noise,
where boxmean ~ 0 makes the local-contrast ratio meaningless.) Among lit pixels
we count **point sources** at local-contrast thresholds k in {3, 5, 10}:
a pixel spikes at k when `ratio > k`, i.e. it is k-times brighter than its own
BOX-neighbourhood — the signature of an isolated star the EWA filter will streak.
Counts are reported as a percentage of lit pixels.

Acceptance bar (SKILL.md Formula-13 status note, 2026-06-07): a starless Layer-B
plate has sharp >10x spikes <= ~0.05% of lit pixels. Merely *dimming* baked stars
is NOT sufficient — they remain resolvable point sources and still streak.

Usage
-----
    uv run python scripts/check_starless_map.py
        # checks starfield.diffuse_map from configs/render.yaml

    uv run python scripts/check_starless_map.py star_image/starmap_final.exr
        # check one explicit plate

    uv run python scripts/check_starless_map.py a.exr b.exr --box 9
        # compare several plates side by side
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Make ``renderer`` importable when run directly (src layout) — mirrors thumb.py / ingest_stars.py.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from renderer.starmap import load_equirect  # noqa: E402  (reuse the OIIO equirect loader)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "configs" / "render.yaml"

# Rec.709 luminance weights (linear RGB).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

_EPS = 1.0e-6
_THRESHOLDS = (3.0, 5.0, 10.0)
_ACCEPT_PCT = 0.05  # >10x sharp-spike pass bar, % of lit pixels


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:  # utf-8: Windows cp949 box
        return yaml.safe_load(fh)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of an (H, W, 3) linear-RGB array -> (H, W) float32."""
    return np.ascontiguousarray(rgb[:, :, :3].astype(np.float32) @ _LUMA)


def _running_mean_1d(a: np.ndarray, box: int, axis: int, wrap: bool) -> np.ndarray:
    """Centered BOX-wide running mean along `axis` (pure numpy, no scipy).

    `wrap=True` treats the axis as periodic (phi / equirect columns); otherwise it
    edge-clamps (theta / equirect rows). Uses a padded cumulative-sum so cost is
    O(N) regardless of box width.
    """
    if box <= 1:
        return a.astype(np.float32, copy=True)
    pad = box // 2
    mode = "wrap" if wrap else "edge"
    pad_width = [(0, 0)] * a.ndim
    # Pad symmetrically; +1 on the trailing side so a difference of cumsum over
    # exactly `box` samples is well defined for every output index.
    pad_width[axis] = (pad, box - pad)
    padded = np.pad(a.astype(np.float64), pad_width, mode=mode)
    csum = np.cumsum(padded, axis=axis)
    # window sum at i = csum[i + box] - csum[i]
    n = a.shape[axis]
    hi = np.take(csum, np.arange(box, box + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
    return ((hi - lo) / float(box)).astype(np.float32)


def box_mean(lum: np.ndarray, box: int) -> np.ndarray:
    """Separable BOXxBOX mean over an equirect luminance map (phi wraps, theta clamps)."""
    m = _running_mean_1d(lum, box, axis=1, wrap=True)  # columns = phi (periodic)
    m = _running_mean_1d(m, box, axis=0, wrap=False)  # rows = theta (clamped)
    return m


def evaluate(path: str, box: int = 9) -> dict:
    """Run the acceptance metric on one equirect EXR plate."""
    rgb = load_equirect(path)
    lum = luminance(rgb)
    bm = box_mean(lum, box)
    ratio = lum / (bm + _EPS)

    # Floor = median *within the band*: median over positive pixels, not the whole
    # plate. A starless plate is mostly hard zeros (global median 0), so the band
    # median is what separates real galaxy signal from the black void + noise.
    positive = lum[lum > _EPS]
    lit_floor = float(np.median(positive)) if positive.size else _EPS
    lit_mask = lum >= max(lit_floor, _EPS)
    n_lit = int(lit_mask.sum())

    spikes = {}
    for k in _THRESHOLDS:
        n = int(np.count_nonzero(lit_mask & (ratio > k)))
        spikes[k] = (n, 100.0 * n / n_lit if n_lit else 0.0)

    return {
        "path": path,
        "shape": lum.shape,
        "lum_mean": float(lum.mean()),
        "lum_peak": float(lum.max()),
        "lit_floor": lit_floor,
        "n_lit": n_lit,
        "spikes": spikes,
        "pass": spikes[10.0][1] <= _ACCEPT_PCT,
    }


def _print_report(res: dict) -> None:
    print(f"\n{res['path']}")
    h, w = res["shape"]
    print(f"  {w}x{h}  lum mean={res['lum_mean']:.4g}  peak={res['lum_peak']:.4g}")
    print(f"  lit floor (band median lum)={res['lit_floor']:.4g}  lit pixels={res['n_lit']:,}")
    for k in _THRESHOLDS:
        n, pct = res["spikes"][k]
        print(f"  sharp >{k:>4.0f}x spikes: {n:>9,}  ({pct:.4f}% of lit)")
    verdict = "PASS" if res["pass"] else "FAIL"
    print(f"  >10x <= {_ACCEPT_PCT}% of lit  ->  {verdict} (starless Layer-B bar)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acceptance check that a DNGR Layer-B diffuse plate is star-free."
    )
    parser.add_argument(
        "maps",
        nargs="*",
        help="equirect EXR plate(s) to check; default: starfield.diffuse_map from render.yaml",
    )
    parser.add_argument("--box", type=int, default=9, help="neighbourhood box width (px); default 9")
    args = parser.parse_args(argv)

    maps = args.maps
    if not maps:
        cfg = load_config()
        diffuse = cfg["starfield"]["diffuse_map"]
        maps = [str(_ROOT / diffuse)]
        print(f"(no plate given; using starfield.diffuse_map = {diffuse})")

    all_pass = True
    for m in maps:
        res = evaluate(m, box=args.box)
        _print_report(res)
        all_pass = all_pass and res["pass"]

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
