"""TEMP DIAGNOSTIC — meridian U-jump convergence sweep / UV-debug dump.

Drives the production Pipe-A physics pass at three step densities with MATCHED
total Mino-time (steps * d_lambda = 4.0) so only step density varies, not
escape coverage. For each density it dumps exit_buf as a UV-debug PNG
(R=u, G=v, B=0, linear) and prints the center-column U-jump metrics.

NOTE: this script only overrides d_lambda_pipe_a / max_steps_pipe_a in the cfg
dict. To reproduce the *fixed-step* sweep you must ALSO temporarily set the
integrator step in renderer.taichi_renderer (kernel, ~L908) to:

    local_h = d_lambda          # fixed dense step

As checked in, the kernel uses the adaptive step
(local_h = d_lambda * max(adaptive_floor, r/(r+2))), so without that edit this
sweep runs adaptive-with-overridden-d_lambda, not fixed-step.

Run from repo root with src on PYTHONPATH (PowerShell):
    $env:PYTHONPATH = "src"; python scripts/_uv_sweep.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from renderer import taichi_renderer as tr

ROOT = Path(__file__).resolve().parents[1]
CAM = ROOT / "camera_matrix.json"
W, H = 1920, 1080
TOTAL_MINO = 4.0  # steps * d_lambda held constant across the sweep

# (d_lambda, n_steps) — matched total Mino-time = 4.0
SWEEP = [
    (0.002, 2000),
    (0.001, 4000),
    (0.0005, 8000),
]


def main() -> None:
    import taichi as ti

    ti.init(arch=ti.cuda)  # LOCKED backend (CLAUDE.md)
    assert str(ti.cfg.arch) == "Arch.cuda", f"CUDA unavailable: {ti.cfg.arch}"

    cfg = tr.load_config()
    with open(CAM, "r", encoding="utf-8-sig") as fh:
        cam = json.load(fh)[0]

    tr.setup_renderer(cfg)  # one-time starmap upload

    c = W // 2  # center column = 960
    print(f"{'dlambda':>8} {'steps':>6} {'esc%':>6} "
          f"{'centerJump':>11} {'intMedian':>11} {'ratio':>8} {'jumpCol':>8}")

    for d_lambda, n_steps in SWEEP:
        cfg["render"]["d_lambda_pipe_a"] = d_lambda
        cfg["render"]["max_steps_pipe_a"] = n_steps

        # Disk off: exit_buf is written by the physics pass regardless; keeps it clean.
        tr.render_beauty_frame(cfg, cam, W, H, with_disk=False, lod_enabled=False)

        eb = tr.exit_buf.to_numpy()  # (H, W, 3): cos(theta'), phi', outcome
        cos_th = np.clip(eb[..., 0], -1.0, 1.0)
        phi = eb[..., 1]
        outcome = eb[..., 2]

        # Sampler's exact equirect mapping (taichi_renderer L1291-1293).
        u = phi / (2.0 * math.pi)
        u = u - np.floor(u)
        v = np.clip(np.arccos(cos_th) / math.pi, 0.0, 1.0)

        # UV-as-color debug image (R=u, G=v, B=0), linear -> 8-bit.
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[..., 0] = u
        rgb[..., 1] = v
        img = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        out = ROOT / "scripts" / f"starfield_uv_sweep_dl{d_lambda:.4f}.png"
        Image.fromarray(img, "RGB").save(out)

        # Metrics over ESCAPED pixels only (outcome flag; non-escaped have junk phi).
        escaped = outcome > 0.5
        esc_pct = 100.0 * escaped.mean()

        # Column-wise mean U over escaped rows, then |diff| between adjacent cols.
        # (NB: mean of frac(u) wraps where phi' crosses 0 -> the global argmax can
        # land on the prime-meridian wrap, not the center; read the image too.)
        u_masked = np.where(escaped, u, np.nan)
        with np.errstate(invalid="ignore"):
            col_u = np.nanmean(u_masked, axis=0)  # (W,)
        col_jump = np.abs(np.diff(col_u))          # (W-1,)
        col_jump = np.nan_to_num(col_jump, nan=0.0)

        center_jump = float(col_jump[c - 4:c + 4].max())
        interior = np.concatenate([col_jump[5:c - 4], col_jump[c + 4:-5]])
        int_median = float(np.median(interior))
        ratio = center_jump / max(int_median, 1e-12)
        jump_col = int(np.argmax(col_jump))

        print(f"{d_lambda:>8.4f} {n_steps:>6d} {esc_pct:>6.1f} "
              f"{center_jump:>11.5f} {int_median:>11.7f} {ratio:>8.1f} {jump_col:>8d}")

    print("\nWrote: scripts/starfield_uv_sweep_dl{0.0020,0.0010,0.0005}.png")


if __name__ == "__main__":
    main()
