"""Seam-isolation diagnostics for the Pipe A background (NOT part of the render
path). Extracted from ``renderer.taichi_renderer`` to keep the production module
lean (PROJECT.md §7, item C1).

These were the Gate-2 root-cause tools for the center "static" seam (the spin-axis
meridian caustic). They drive the *production* ti.func helpers imported from
``renderer.taichi_renderer`` — no physics is re-implemented here.

Run from the repo root (``src`` on the path, e.g. ``PYTHONPATH=src``):

    python scripts/seam_diagnostics.py            # Gate-2 LOD on/off comparison
    python scripts/seam_diagnostics.py seam       # raw-starmap vs fixed-LOD seam isolation
    python scripts/seam_diagnostics.py phidump    # per-column exit-azimuth dump
"""

import math

import numpy as np
import taichi as ti

# Late-bound module reference: ``pixels`` and ``_DELTA_MIN`` are reassigned inside
# taichi_renderer (per-render allocation / config override), so the kernels below
# read them as ``tr.pixels`` / ``tr._DELTA_MIN`` to pick up the live values at JIT
# time (first call, after setup_renderer). The rest are stable and imported by name.
from renderer import taichi_renderer as tr
from renderer.taichi_renderer import (  # noqa: E402  (stable helpers / constants)
    _ROOT,
    _alloc_output,
    _delta_y,
    _horizon_constants,
    _ray_dir,
    _rk4_step,
    _sample_trilinear,
    _zamo_init,
    _CAPTURED,
    _ESCAPED,
    _RUNNING,
    load_config,
    render_pipe_a_image,
    setup_renderer,
    tonemap,
    vec3,
    vec6,
)


@ti.kernel
def render_starmap_raw(res: int, lod: ti.f32):
    """Diagnostic 1: direct equirect sky dump at a FIXED mip LOD.

    No geodesic, no lensing. Screen (px,py) maps straight to (u=φ/2π, v=θ/π)
    and samples the pyramid. If a seam shows here it lives in the starmap data /
    mip pyramid itself (the φ-wrap of the pyramid is at the u=0/1 image edges).
    """
    for py, px in ti.ndrange(res, res):
        u = (ti.cast(px, ti.f32) + 0.5) / res
        v = (ti.cast(py, ti.f32) + 0.5) / res
        col = _sample_trilinear(u, v, lod)
        tr.pixels[py, px, 0] = col[0]
        tr.pixels[py, px, 1] = col[1]
        tr.pixels[py, px, 2] = col[2]


@ti.kernel
def render_fixed_lod(res: int, tan_half_fov: float, r_cam: float,
                     theta_cam: float, phi_cam: float, a: float,
                     k_horizon: float, r_plus: float,
                     r_max: float, n_steps: int, d_lambda: float,
                     lod_fixed: ti.f32):
    """Diagnostic 2: full geodesic lensing (primary ray only) with mip LOD
    PINNED to ``lod_fixed`` for every escaped pixel.

    Removes the Jacobian entirely. If a seam appears here but not in
    Diagnostic 1, it is a ray-classification boundary (escaped vs captured),
    not a starmap/mip issue.
    """
    y_cam = r_cam - r_plus
    u_cam = ti.cos(theta_cam)
    r_capture = 2.0 - r_plus
    y_escape = r_max - r_plus
    for py, px in ti.ndrange(res, res):
        npr_r, npr_th, npr_ph = _ray_dir(ti.cast(px, ti.f32), ti.cast(py, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)
        Ep, Lp, Qp, vy_p, vu_p = _zamo_init(r_cam, theta_cam, a, k_horizon, r_plus,
                                            npr_r, npr_th, npr_ph)
        sp = vec6(y_cam, u_cam, phi_cam, 0.0, vy_p, vu_p)
        out_p = _RUNNING
        u_p_exit = u_cam
        ph_p_exit = phi_cam
        step = 0
        while step < n_steps and out_p == _RUNNING:
            if _delta_y(sp[0], k_horizon) < tr._DELTA_MIN:
                out_p = _CAPTURED
            else:
                sp = _rk4_step(sp, Ep, Lp, Qp, a, k_horizon, r_plus, d_lambda)
                if _delta_y(sp[0], k_horizon) < tr._DELTA_MIN or sp[0] < r_capture:
                    out_p = _CAPTURED
                elif sp[0] >= y_escape:
                    out_p = _ESCAPED
                    u_p_exit = sp[1]
                    ph_p_exit = sp[2]
            step += 1
        col = vec3(0.0, 0.0, 0.0)
        if out_p == _ESCAPED:
            # u = cosθ keeps θ_exit ∈ [0, π]; recover θ and wrap φ (no polar fold).
            th_p_n = ti.acos(ti.min(ti.max(u_p_exit, -1.0), 1.0))
            u = ph_p_exit / (2.0 * math.pi)
            u = u - ti.floor(u)
            v = ti.min(ti.max(th_p_n / math.pi, 0.0), 1.0)
            col = _sample_trilinear(u, v, lod_fixed)
        tr.pixels[py, px, 0] = col[0]
        tr.pixels[py, px, 1] = col[1]
        tr.pixels[py, px, 2] = col[2]


# Per-column exit-state dump buffer: [phi_exit_raw, theta_exit, outcome].
phi_dump: ti.Field = None          # type: ignore[assignment]


@ti.kernel
def dump_phi_exit(res: int, row_y: int, tan_half_fov: float, r_cam: float,
                  theta_cam: float, phi_cam: float, a: float,
                  k_horizon: float, r_plus: float,
                  r_max: float, n_steps: int, d_lambda: float):
    """Diagnostic 3: trace the PRIMARY ray for every column of a single screen row
    and record the *raw accumulated* exit azimuth ``phi_exit`` (no mod/frac), the
    exit ``theta`` (= acos(u_exit)), and the outcome code. Reveals whether adjacent
    columns wind by different multiples of 2π (the branch-cut hypothesis)."""
    y_cam = r_cam - r_plus
    u_cam = ti.cos(theta_cam)
    r_capture = 2.0 - r_plus
    y_escape = r_max - r_plus
    for px in range(res):
        npr_r, npr_th, npr_ph = _ray_dir(ti.cast(px, ti.f32), ti.cast(row_y, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)
        Ep, Lp, Qp, vy_p, vu_p = _zamo_init(r_cam, theta_cam, a, k_horizon, r_plus,
                                            npr_r, npr_th, npr_ph)
        sp = vec6(y_cam, u_cam, phi_cam, 0.0, vy_p, vu_p)
        out_p = _RUNNING
        u_p_exit = u_cam
        ph_p_exit = phi_cam
        step = 0
        while step < n_steps and out_p == _RUNNING:
            if _delta_y(sp[0], k_horizon) < tr._DELTA_MIN:
                out_p = _CAPTURED
            else:
                sp = _rk4_step(sp, Ep, Lp, Qp, a, k_horizon, r_plus, d_lambda)
                if _delta_y(sp[0], k_horizon) < tr._DELTA_MIN or sp[0] < r_capture:
                    out_p = _CAPTURED
                elif sp[0] >= y_escape:
                    out_p = _ESCAPED
                    u_p_exit = sp[1]
                    ph_p_exit = sp[2]
            step += 1
        phi_dump[px, 0] = ph_p_exit
        phi_dump[px, 1] = ti.acos(ti.min(ti.max(u_p_exit, -1.0), 1.0))
        phi_dump[px, 2] = ti.cast(out_p, ti.f32)


def _phi_dump() -> None:
    """Gate-2 root-cause probe: dump per-column exit azimuth across a screen row."""
    global phi_dump

    cfg = load_config()
    setup_renderer(cfg)

    res = int(cfg["render"]["thumb_width"])
    th = cfg["thumb"]
    a = float(cfg["black_hole"]["spin"])
    cam = cfg["camera"]
    r_cam = float(th.get("camera_radius", cam["default_radius"]))
    fov_deg = float(th.get("fov_deg", cam["default_fov_deg"]))
    theta_cam = math.radians(float(th["camera_theta_deg"]))
    n_steps = int(cfg["render"]["max_steps_pipe_a"])
    d_lambda = float(cfg["render"]["d_lambda_pipe_a"])
    r_max = float(cfg["render"]["r_max"])
    tan_half_fov = math.tan(math.radians(fov_deg) / 2.0)
    row_y = res // 2

    k_horizon, r_plus = _horizon_constants(a)
    phi_dump = ti.field(dtype=ti.f32, shape=(res, 3))
    dump_phi_exit(res, row_y, tan_half_fov, r_cam, theta_cam, 0.0, a,
                  k_horizon, r_plus, r_max, n_steps, d_lambda)
    ti.sync()
    d = phi_dump.to_numpy()

    two_pi = 2.0 * math.pi
    names = {0: "RUN", 1: "ESC", 2: "CAP"}
    print(f"phi_exit dump  row y={row_y}  (res={res})")
    print(f"{'col':>4} {'out':>4} {'phi_raw':>10} {'phi/2pi':>9} "
          f"{'frac_u':>8} {'atan2_u':>8} {'theta':>8}")
    for px in range(183, 193):
        ph = float(d[px, 0]); th_e = float(d[px, 1]); out = int(round(d[px, 2]))
        frac_u = ph / two_pi - math.floor(ph / two_pi)
        atan2_u = (math.atan2(math.sin(ph), math.cos(ph)) + math.pi) / two_pi
        print(f"{px:>4} {names.get(out, '?'):>4} {ph:>10.4f} {ph/two_pi:>9.4f} "
              f"{frac_u:>8.4f} {atan2_u:>8.4f} {th_e:>8.4f}")


def _gate2_lod_test() -> None:
    """Render the two Gate-2 LOD comparison images at 256×256."""
    from PIL import Image

    cfg = load_config()
    setup_renderer(cfg)

    res = int(cfg["render"]["thumb_width"])
    th = cfg["thumb"]
    exposure = float(th.get("exposure", 1.0)) * 3.0  # lift the LDR starmap nebulae
    gamma = float(th["gamma"])

    out_dir = _ROOT / "scripts"
    for lod_on, name in ((False, "test_lod_off.png"), (True, "test_lod_on.png")):
        hdr = render_pipe_a_image(cfg, res, lod_enabled=lod_on)
        finite = np.isfinite(hdr)
        nan_count = int((~finite).sum())
        img = tonemap(np.nan_to_num(hdr), exposure, gamma)
        Image.fromarray(img, mode="RGB").save(out_dir / name)
        nonblack = float((hdr.sum(axis=2) > 1e-6).mean())
        print(f"{name}: lod={'on' if lod_on else 'off(L=0)'}  "
              f"hdr[min={hdr.min():.4g} max={hdr.max():.4g}]  "
              f"non-black px={nonblack*100:.1f}%  NaN={nan_count}")
    print(f"Saved both LOD test images to {out_dir}")


def _probe_columns(img: np.ndarray, res: int):
    """Find the strongest interior column-to-column luma jump (the 'seam') and
    return (seam_col, jump_magnitude). img is uint8 (res,res,3)."""
    luma = img.astype(np.float32).mean(axis=2).mean(axis=0)   # per-column mean over rows
    d = np.abs(np.diff(luma))                                  # len res-1
    lo, hi = 5, res - 6
    k = lo + int(np.argmax(d[lo:hi]))                          # jump between col k and k+1
    return k + 1, float(d[k])


def _seam_diag() -> None:
    """Gate-2 seam isolation: render the two diagnostic frames and report
    where (if anywhere) a vertical seam appears."""
    from PIL import Image

    cfg = load_config()
    setup_renderer(cfg)

    res = int(cfg["render"]["thumb_width"])
    th = cfg["thumb"]
    exposure = float(th.get("exposure", 1.0)) * 3.0
    gamma = float(th["gamma"])
    out_dir = _ROOT / "scripts"
    y = res // 2

    def _save_and_probe(name: str, hdr: np.ndarray) -> None:
        img = tonemap(np.nan_to_num(hdr), exposure, gamma)
        Image.fromarray(img, mode="RGB").save(out_dir / name)
        sx, jump = _probe_columns(img, res)
        print(f"{name}: hdr[min={hdr.min():.4g} max={hdr.max():.4g}]  "
              f"seam@col={sx} (jump={jump:.1f} luma)")
        print(f"    y={y}  col{sx-1}={tuple(int(v) for v in img[y, sx-1])}  "
              f"col{sx}={tuple(int(v) for v in img[y, sx])}  "
              f"col{sx+1}={tuple(int(v) for v in img[y, sx+1])}")
        print(f"    y={y}  wrap-edges: col0={tuple(int(v) for v in img[y, 0])}  "
              f"col1={tuple(int(v) for v in img[y, 1])}  "
              f"col{res-1}={tuple(int(v) for v in img[y, res-1])}")

    _alloc_output(res)

    # Diagnostic 1: raw equirect starmap at fixed L=3 (no geodesic, no lensing).
    render_starmap_raw(res, 3.0)
    ti.sync()
    _save_and_probe("test_starmap_raw.png", tr.pixels.to_numpy())

    # Diagnostic 2: geodesic lensing, primary ray only, LOD pinned to L=3.
    a = float(cfg["black_hole"]["spin"])
    cam = cfg["camera"]
    r_cam = float(th.get("camera_radius", cam["default_radius"]))
    fov_deg = float(th.get("fov_deg", cam["default_fov_deg"]))
    theta_cam = math.radians(float(th["camera_theta_deg"]))
    n_steps = int(cfg["render"]["max_steps_pipe_a"])
    d_lambda = float(cfg["render"]["d_lambda_pipe_a"])
    r_max = float(cfg["render"]["r_max"])
    tan_half_fov = math.tan(math.radians(fov_deg) / 2.0)
    k_horizon, r_plus = _horizon_constants(a)
    render_fixed_lod(res, tan_half_fov, r_cam, theta_cam, 0.0, a,
                     k_horizon, r_plus, r_max, n_steps, d_lambda, 3.0)
    ti.sync()
    _save_and_probe("test_fixed_lod.png", tr.pixels.to_numpy())

    print(f"Saved seam diagnostics to {out_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "seam":
        _seam_diag()
    elif len(sys.argv) > 1 and sys.argv[1] == "phidump":
        _phi_dump()
    else:
        _gate2_lod_test()
