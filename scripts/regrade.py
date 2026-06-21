"""Instant re-grade of a saved linear-HDR EXR (no re-render).

Loads a linear-HDR .exr (e.g. the hero_4k.py output), applies bloom + the
exposure/saturation/tint/gamma grade through the SAME tonemap path, and writes a
PNG. Use this to dial exposure/colour in <1s instead of re-rendering. Anything
baked into the radiance (temperature, Doppler, extinction, geometry) still needs
a re-render — this only changes the post-grade.

    uv run python scripts/regrade.py --exr proxy_hero.exr --saturation 1.7 --tint 1.3,0.7,0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for pth in (_SRC, _SCRIPTS):
    if str(pth) not in sys.path:
        sys.path.insert(0, str(pth))

from renderer import taichi_renderer as tr  # noqa: E402
from hero_4k import _bloom  # noqa: E402  (reuse the identical bloom kernel)


def read_rgb_exr(filename: str) -> np.ndarray:
    import OpenImageIO as oiio

    src = oiio.ImageInput.open(filename)
    if src is None:  # pragma: no cover
        raise RuntimeError(f"OpenImageIO could not open {filename}: {oiio.geterror()}")
    spec = src.spec()
    pix = np.array(src.read_image(format=oiio.FLOAT), dtype=np.float32)
    src.close()
    return pix.reshape(spec.height, spec.width, spec.nchannels)[..., :3]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Instant re-grade of a linear-HDR EXR.")
    p.add_argument("--exr", default="proxy_hero.exr")
    p.add_argument("--out", default="regrade.png")
    p.add_argument("--exposure", type=float, default=5.0)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--saturation", type=float, default=1.6)
    p.add_argument("--tint", default="1.22,0.8,0.55")
    p.add_argument("--bloom", type=float, default=0.55)
    p.add_argument("--bloom-threshold", type=float, default=1.8)
    args = p.parse_args(argv)

    beauty = read_rgb_exr(args.exr)
    if args.bloom > 0.0:
        beauty = _bloom(beauty, args.bloom, args.bloom_threshold)
    tint = [float(v) for v in args.tint.split(",")]
    pixels = tr.tonemap(beauty, args.exposure, args.gamma, saturation=args.saturation, tint=tint)

    from PIL import Image

    Image.fromarray(pixels, mode="RGB").save(args.out)
    print(f"saved {args.out}  exposure={args.exposure} saturation={args.saturation} tint={tint}")


if __name__ == "__main__":
    main()
