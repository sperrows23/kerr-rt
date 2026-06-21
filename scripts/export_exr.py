"""Multi-channel (RGBAZ) EXR export — Phase 5 of the Kerr pipeline.

Renders one beauty frame on the Taichi CUDA backend (Pipe A + Pipe B) plus its
transmittance-weighted Z-depth pass (guid 3.4) and writes a single multi-channel
OpenEXR with named channels ``R, G, B, Z`` — the G-buffer the Blender compositor
(Phase 3) consumes to deep-composite the spaceship against the black hole / disk.

All physics follows ``skills/kerr-physics/SKILL.md``; numeric parameters come from
``configs/render.yaml`` (output dir/prefix from the ``output`` section).

Usage
-----
    uv run python scripts/export_exr.py --frame 0
    uv run python scripts/export_exr.py --frame 12 --width 3840 --height 2160
    uv run python scripts/export_exr.py --frame 12 --motion-blur   # temporal blur (guid 4.2)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from renderer import taichi_renderer as tr  # noqa: E402


def _azimuth(pos: list[float]) -> float:
    return math.atan2(float(pos[1]), float(pos[0]))


def _shutter_arc(frames: list[dict], idx: int, shutter_fraction: float, fps: float) -> float:
    """Azimuthal travel during the shutter (guid 4.2).

    ``Δφ`` between adjacent ``camera_matrix.json`` frames is the per-inter-frame
    arc, i.e. ω·(1/fps). The shutter arc is ω·shutter_fraction (see
    ``render_beauty_frame_mb``), where ``shutter_fraction`` is the shutter *time*
    in seconds (config ``camera.shutter_fraction`` = 1/48 s). Converting the
    per-frame Δφ to ω via ``×fps`` gives ``arc = Δφ·fps·shutter_fraction``. At
    24 fps and a 1/48 s shutter this is ``Δφ·0.5`` — the legacy 180° shutter.
    Returns 0 if there is no neighbor frame."""
    if idx + 1 >= len(frames):
        return 0.0
    dphi = _azimuth(frames[idx + 1]["pos"]) - _azimuth(frames[idx]["pos"])
    # wrap into [-pi, pi]
    dphi = (dphi + math.pi) % (2.0 * math.pi) - math.pi
    return dphi * fps * shutter_fraction


def write_rgbaz_exr(filename: str, beauty_rgb: np.ndarray, depth_z: np.ndarray) -> None:
    """Write a 4-channel (R, G, B, Z) float EXR via OpenImageIO (guid 5.2)."""
    import OpenImageIO as oiio

    h, w = beauty_rgb.shape[:2]
    if depth_z.ndim == 2:
        depth_z = np.expand_dims(depth_z, axis=2)
    pixels = np.concatenate(
        [
            np.ascontiguousarray(beauty_rgb, dtype=np.float32),
            np.ascontiguousarray(depth_z, dtype=np.float32),
        ],
        axis=2,
    )
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)

    out = oiio.ImageOutput.create(filename)
    if out is None:  # pragma: no cover - environment guard
        raise RuntimeError(f"OpenImageIO could not create writer for {filename}: {oiio.geterror()}")
    spec = oiio.ImageSpec(w, h, 4, oiio.FLOAT)
    spec.channelnames = ("R", "G", "B", "Z")
    out.open(filename, spec)
    out.write_image(pixels)
    out.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multi-channel RGBAZ EXR export (Pipe A + Pipe B + Z)."
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="0-based index into camera_matrix.json (default 0)."
    )
    parser.add_argument("--width", type=int, default=None, help="default: render.width")
    parser.add_argument("--height", type=int, default=None, help="default: render.height")
    parser.add_argument("--no-disk", action="store_true", help="Render Pipe A only.")
    parser.add_argument(
        "--motion-blur",
        action="store_true",
        help="Temporal motion blur (guid 4.2) using the shutter arc to the next frame.",
    )
    parser.add_argument("--out", default=None, help="Override output path.")
    args = parser.parse_args(argv)

    cfg = tr.load_config()
    rcfg = cfg["render"]
    width = args.width if args.width is not None else int(rcfg["width"])
    height = args.height if args.height is not None else int(rcfg["height"])

    cam_path = _ROOT / "camera_matrix.json"
    with open(cam_path, encoding="utf-8-sig") as fh:
        frames = json.load(fh)
    if not 0 <= args.frame < len(frames):
        raise SystemExit(f"--frame {args.frame} out of range (have {len(frames)} frames)")
    cam = frames[args.frame]

    # D2.3 disk-animation clock (CKS-12 §2): t_disk = frame/fps · time_scale, in
    # geometric M (time_scale = period_inner_M / inner_lap_seconds, CKS-13). The
    # noise system shears/reseeds against this. No disk.dynamics block ⇒ 0 ⇒ the
    # static D2.2 path (bit-identical to the goldens).
    dyn = cfg["disk"].get("dynamics") or {}
    t_disk = args.frame / float(rcfg["fps"]) * float(dyn.get("time_scale", 0.0))

    print("initialising Taichi (ti.cuda) + uploading 16K starmap mip pyramid ...")
    tr.setup_renderer(cfg)

    print(
        f"rendering {width}x{height}  frame={args.frame}  disk={'off' if args.no_disk else 'on'}"
        f"  motion_blur={'on' if args.motion_blur else 'off'} ..."
    )
    t0 = time.time()
    if args.motion_blur:
        arc = _shutter_arc(
            frames,
            args.frame,
            float(cfg["camera"]["shutter_fraction"]),
            float(cfg["render"]["fps"]),
        )
        beauty, depth = tr.render_beauty_frame_mb(
            cfg,
            cam,
            width,
            height,
            shutter_arc=arc,
            with_disk=not args.no_disk,
            lod_enabled=True,
            return_depth=True,
            t_disk=t_disk,
        )
    else:
        beauty, depth = tr.render_beauty_frame(
            cfg, cam, width, height, with_disk=not args.no_disk, lod_enabled=True,
            return_depth=True, t_disk=t_disk,
        )
    beauty = np.nan_to_num(beauty)
    dt = time.time() - t0
    print(
        f"done in {dt:.1f}s  beauty[max={beauty.max():.4g}]  "
        f"depth[finite={float((depth < float(rcfg['depth_infinity'])).mean()) * 100:.1f}% of px]"
    )

    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_dir = _ROOT / cfg["output"]["blackhole_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg['output']['blackhole_prefix']}{args.frame:04d}.exr"

    write_rgbaz_exr(str(out_path), beauty, depth)
    print(f"Saved {out_path}  (channels: R, G, B, Z)")


if __name__ == "__main__":
    main()
