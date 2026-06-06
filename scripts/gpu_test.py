"""Single-frame FHD GPU beauty render — Phase 2 smoke test.

Renders ONE high-resolution frame (default 1920x1080) on the Taichi CUDA backend
combining:

  * Pipe A — gravitationally lensed 16K starmap with Formula 10 differential-ray
             mip LOD (anti-aliased background),
  * Pipe B — volumetric accretion disk (Formulas 3/4/5/8/9): orbiting + plunging
             gas, g-factor redshift, and g^4 Doppler-beamed blackbody emission.

The camera is read from ``camera_matrix.json`` (Blender export, world Cartesian).
Output is tone-mapped (Reinhard + gamma) and saved to ``scripts/gpu_test_disk.png``.

All physics follows ``skills/kerr-physics/SKILL.md`` verbatim; numerical parameters
come from ``configs/render.yaml`` (no hardcoded physics values here).

Usage
-----
    uv run python scripts/gpu_test.py                 # frame 0, 1920x1080, +disk
    uv run python scripts/gpu_test.py --frame 1
    uv run python scripts/gpu_test.py --no-disk       # Pipe A only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from renderer import taichi_renderer as tr  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FHD GPU beauty render (Pipe A + Pipe B).")
    parser.add_argument(
        "--frame", type=int, default=0, help="0-based index into camera_matrix.json (default 0)."
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--no-disk", action="store_true", help="Render Pipe A only.")
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Linear pre-tonemap exposure (default: config thumb.exposure).",
    )
    parser.add_argument("--out", default=str(_ROOT / "scripts" / "gpu_test_disk.png"))
    args = parser.parse_args(argv)

    cfg = tr.load_config()

    # camera_matrix.json is a UTF-8-with-BOM array of per-frame camera matrices.
    cam_path = _ROOT / "camera_matrix.json"
    with open(cam_path, encoding="utf-8-sig") as fh:
        frames = json.load(fh)
    if not 0 <= args.frame < len(frames):
        raise SystemExit(f"--frame {args.frame} out of range (have {len(frames)} frames)")
    cam = frames[args.frame]
    print(
        f"camera: index {args.frame} (json frame={cam['frame']})  "
        f"pos={cam['pos']}  fov={cam['fov']:.4f} rad"
    )

    print("initialising Taichi (ti.cuda) + uploading 16K starmap mip pyramid ...")
    tr.setup_renderer(cfg)

    print(f"rendering {args.width}x{args.height}  disk={'off' if args.no_disk else 'on'} ...")
    t0 = time.time()
    hdr = tr.render_beauty_frame(
        cfg, cam, args.width, args.height, with_disk=not args.no_disk, lod_enabled=True
    )
    dt = time.time() - t0
    hdr = np.nan_to_num(hdr)

    h, w = hdr.shape[:2]
    lum = hdr.sum(axis=2)
    left = float(lum[:, : w // 2].mean())
    right = float(lum[:, w // 2 :].mean())
    nonblack = float((lum > 1e-6).mean())
    print(
        f"done in {dt:.1f}s  hdr[min={hdr.min():.4g} max={hdr.max():.4g} "
        f"mean={hdr.mean():.4g}]  non-black px={nonblack * 100:.1f}%"
    )
    print(
        f"Doppler check: left-half mean lum={left:.4g}  right-half mean lum={right:.4g}  "
        f"(asymmetry ratio={max(left, right) / max(min(left, right), 1e-9):.2f}x)"
    )

    th = cfg["thumb"]
    exposure = args.exposure if args.exposure is not None else float(th.get("exposure", 1.0))
    gamma = float(th["gamma"])
    img = tr.tonemap(hdr, exposure, gamma)

    from PIL import Image  # local import so --help works without Pillow

    out_path = Path(args.out)
    Image.fromarray(img, mode="RGB").save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
