"""Seam diagnostic: r_min and photon-ring glow along the center column.

Reproduces the render() camera setup and walks the vertical center column
(px = res//2), which crosses the top and bottom of the shadow where the
polar-axis "beaded seam" appeared. Reports r_min and the resulting glow
weight per row, then counts extrema (sign changes in the row-to-row diff)
of the glow. A smooth, monotonic glow approaching the shadow edge => 0
interior extrema along each monotone arm.
"""

from __future__ import annotations

import numpy as np

from thumb import (
    build_disk_params,
    camera_ray_direction,
    load_config,
    ring_glow,
    trace_pixel,
)


def main(res: int = 256) -> None:
    cfg = load_config()
    bh = cfg["black_hole"]
    rcfg = cfg["render"]
    cam = cfg["camera"]
    th = cfg["thumb"]

    a = float(bh["spin"])
    r_plus = float(bh["r_plus"])
    r_cam = float(th.get("camera_radius", cam["default_radius"]))
    fov_deg = float(th.get("fov_deg", cam["default_fov_deg"]))
    theta_cam = np.deg2rad(float(th["camera_theta_deg"]))
    phi_cam = 0.0

    n_steps = int(rcfg["max_steps_pipe_a"])
    d_lambda = float(rcfg["d_lambda_pipe_a"])
    r_max = float(rcfg["r_max"])

    ring_col = np.array(th["ring_color"], dtype=float)
    ring_gain = float(th["ring_gain"])
    ring_peak = float(th["ring_r_peak"])
    ring_sigma = float(th["ring_r_sigma"])

    tan_half_fov = np.tan(np.deg2rad(fov_deg) / 2.0)
    disk_params = build_disk_params(cfg, a, d_lambda)

    px = res // 2
    rows, outcomes, r_mins, glows = [], [], [], []
    for py in range(res):
        n = camera_ray_direction(px, py, res, tan_half_fov)
        outcome, r_min, _dc, _tr = trace_pixel(
            r_cam, theta_cam, phi_cam, a, n,
            n_steps, d_lambda, r_max, r_plus, disk_params,
        )
        # Glow brightness scalar (max channel of the ring_glow vector); 0 for
        # captured pixels which render as the black shadow.
        if outcome == "captured":
            glow = 0.0
        else:
            glow = float(np.max(ring_glow(r_min, ring_peak, ring_sigma,
                                          ring_gain, ring_col)))
        rows.append(py)
        outcomes.append(outcome)
        r_mins.append(r_min)
        glows.append(glow)

    g = np.array(glows)
    rm = np.array(r_mins)

    # Extrema = interior sign changes in the first difference of the glow
    # profile. A clean profile rises monotonically to the shadow edge then
    # falls -> exactly the turning points where it crosses the shadow, no beads.
    dg = np.diff(g)
    nz = dg[np.abs(dg) > 1e-9]
    sign_changes = int(np.sum(np.diff(np.sign(nz)) != 0)) if nz.size > 1 else 0

    print(f"center column px={px}, res={res}")
    print(f"glow extrema (interior sign changes in diff): {sign_changes}")
    print(f"glow range: {g.min():.4f} .. {g.max():.4f}")
    print()
    print(" py  outcome     r_min     glow")
    for py, oc, r_min, glow in zip(rows, outcomes, r_mins, glows):
        if 96 <= py <= 160:  # zoom on the shadow edge / seam region
            print(f"{py:3d}  {oc:9s}  {r_min:8.4f}  {glow:8.4f}")


if __name__ == "__main__":
    main()
