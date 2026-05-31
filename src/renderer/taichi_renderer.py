"""Production Taichi GPU renderer — Phase 2 of the Kerr pipeline.

This module currently implements **Pipe A** (4K-capable beauty pass: starmap +
gravitational lensing, no volume). Pipe B (volumetric disk march) and the
multi-part EXR writer are added in later build steps.

Physics is ported **verbatim** from the NumPy reference modules, which in turn
follow ``skills/kerr-physics/SKILL.md`` (project CRITICAL RULE: no formula is
re-derived here). The ported pieces are:

  * Formula 1  — Kerr metric components       (``renderer.metric``)
  * Formula 6  — Mino-time RK4 integration    (``renderer.geodesic``)
  * Formula 7  — ZAMO tetrad photon momentum  (``scripts.thumb.zamo_photon_momentum``)
  * Formula 10 — differential-ray mip LOD     (``skills/kerr-physics`` + ``renderer.starmap``)

GPU backend is LOCKED to ``ti.init(arch=ti.cuda)`` (never ``ti.gpu``) per CLAUDE.md.

All numerical parameters come from ``configs/render.yaml`` (no hardcoded values).
The only literals here are integration-scheme safety constants that already live
in ``renderer.geodesic`` (``_DELTA_MIN`` horizon stop, ``1e-10`` polar-axis
denominator clamp) — these are not physics and are mirrored, not re-derived.
"""

import math
from pathlib import Path

import numpy as np
import taichi as ti
import yaml

from renderer.starmap import Starmap

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "configs" / "render.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    # Explicit utf-8: this box defaults to cp949 and the config has θ/π/· bytes.
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Integration-scheme safety constants (mirrored from renderer.geodesic — NOT
# physics). Stop before the horizon where BL momenta (∝ 1/Δ) diverge; clamp the
# sin²θ denominator off the polar axis.
_DELTA_MIN = 0.05
_SIN2_MIN = 1e-10

# BL state vector layout used in the kernel: [r, θ, φ, t, v_r, v_θ].
vec6 = ti.types.vector(6, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec2 = ti.types.vector(2, ti.f32)

_TWO_PI = 2.0 * math.pi

# Ray outcome codes.
_RUNNING = 0
_ESCAPED = 1
_CAPTURED = 2


# --------------------------------------------------------------------------- #
# GPU fields (populated by setup_renderer)
# --------------------------------------------------------------------------- #
# Starmap mip pyramid, packed into one flat f16 buffer with per-level metadata.
star_flat: ti.Field = None        # type: ignore[assignment]
star_off: ti.Field = None         # type: ignore[assignment]
star_w: ti.Field = None           # type: ignore[assignment]
star_h: ti.Field = None           # type: ignore[assignment]
_N_LEVELS = 0
_MAX_LOD = 0.0
_STARMAP_WIDTH = 0

# Output image (set per render call to match resolution).
pixels: ti.Field = None           # type: ignore[assignment]
_RES = 0


def setup_renderer(cfg: dict) -> Starmap:
    """Initialise Taichi (CUDA), load the starmap, and upload its mip pyramid.

    Returns the host-side :class:`Starmap` (kept alive so the GPU upload's source
    arrays and the reference sampler remain available to callers/tests).
    """
    global star_flat, star_off, star_w, star_h
    global _N_LEVELS, _MAX_LOD, _STARMAP_WIDTH

    mem_gb = float(cfg["render"]["device_memory_gb"])
    # LOCKED backend — ti.cuda, never ti.gpu (CLAUDE.md / SKILL.md).
    ti.init(arch=ti.cuda, device_memory_GB=mem_gb, default_fp=ti.f32)

    sm = Starmap.load(str(_ROOT / cfg["starmap"]["path"]))
    _STARMAP_WIDTH = int(cfg["starmap"]["width"])
    _MAX_LOD = float(sm.max_lod)
    _N_LEVELS = len(sm.levels)

    # Pack every mip level's RGB into one contiguous f16 buffer; record each
    # level's flat offset and dimensions so the kernel can index it directly.
    offsets, ws, hs, parts = [], [], [], []
    off = 0
    for lv in sm.levels:
        h, w = lv.shape[:2]
        offsets.append(off)
        hs.append(h)
        ws.append(w)
        flat = np.ascontiguousarray(lv, dtype=np.float16).ravel()
        parts.append(flat)
        off += flat.size
    flat_all = np.concatenate(parts)

    star_flat = ti.field(dtype=ti.f16, shape=flat_all.size)
    star_off = ti.field(dtype=ti.i32, shape=_N_LEVELS)
    star_w = ti.field(dtype=ti.i32, shape=_N_LEVELS)
    star_h = ti.field(dtype=ti.i32, shape=_N_LEVELS)

    star_flat.from_numpy(flat_all)
    star_off.from_numpy(np.asarray(offsets, dtype=np.int32))
    star_w.from_numpy(np.asarray(ws, dtype=np.int32))
    star_h.from_numpy(np.asarray(hs, dtype=np.int32))

    return sm


def _alloc_output(res: int) -> None:
    global pixels, _RES
    if pixels is None or _RES != res:
        pixels = ti.field(dtype=ti.f32, shape=(res, res, 3))
        _RES = res


# --------------------------------------------------------------------------- #
# Physics @ti.func — ported verbatim from renderer.geodesic / metric / thumb
# --------------------------------------------------------------------------- #
@ti.func
def _delta(r, a):
    return r * r - 2.0 * r + a * a


@ti.func
def _radial_potential(r, E, Lz, Q, a):
    # R(r) = [E(r²+a²) − aL_z]² − Δ·[(L_z − aE)² + Q]   (Formula 6, null form)
    P = E * (r * r + a * a) - a * Lz
    B = (Lz - a * E) ** 2 + Q
    return P * P - _delta(r, a) * B


@ti.func
def _radial_potential_deriv(r, E, Lz, Q, a):
    # dR/dr = 4Er·P − (2r−2)·B   (calculus derivative of R above)
    P = E * (r * r + a * a) - a * Lz
    B = (Lz - a * E) ** 2 + Q
    return 4.0 * E * r * P - (2.0 * r - 2.0) * B


@ti.func
def _theta_potential(theta, E, Lz, Q, a):
    # Θ(θ) = Q − cos²θ·(−a²E² + L_z²/sin²θ)   (Formula 6, null form)
    cos2 = ti.cos(theta) ** 2
    sin2 = ti.max(ti.sin(theta) ** 2, _SIN2_MIN)
    return Q - cos2 * (-(a * a) * E * E + Lz * Lz / sin2)


@ti.func
def _theta_potential_deriv(theta, E, Lz, a):
    # dΘ/dθ = −a²E²·sin(2θ) + 2 L_z² cosθ / sin³θ
    sin2 = ti.max(ti.sin(theta) ** 2, _SIN2_MIN)
    c = ti.cos(theta)
    return -(a * a) * E * E * ti.sin(2.0 * theta) + 2.0 * Lz * Lz * c / (sin2 ** 1.5)


@ti.func
def _deriv(s, E, Lz, Q, a):
    # State s = [r, θ, φ, t, v_r, v_θ]; returns ds/dλ (Mino time), Formula 6.
    r = s[0]
    th = s[1]
    vr = s[4]
    vth = s[5]
    Delta = _delta(r, a)
    sin2 = ti.sin(th) ** 2
    sin2_safe = ti.max(sin2, _SIN2_MIN)
    P = E * (r * r + a * a) - a * Lz

    dr = vr
    dth = vth
    dphi = -(a * E - Lz / sin2_safe) + a * P / Delta
    dt = -a * (a * E * sin2 - Lz) + (r * r + a * a) * P / Delta
    dvr = 0.5 * _radial_potential_deriv(r, E, Lz, Q, a)
    dvth = 0.5 * _theta_potential_deriv(th, E, Lz, a)
    return vec6(dr, dth, dphi, dt, dvr, dvth)


@ti.func
def _project(s, E, Lz, Q, a):
    # Re-impose (dr/dλ)² = R, (dθ/dλ)² = Θ exactly (signs from RK4 evolution).
    R = _radial_potential(s[0], E, Lz, Q, a)
    Theta = _theta_potential(s[1], E, Lz, Q, a)
    vr_mag = ti.sqrt(ti.max(0.0, R))
    vth_mag = ti.sqrt(ti.max(0.0, Theta))
    vr = vr_mag if s[4] >= 0.0 else -vr_mag
    vth = vth_mag if s[5] >= 0.0 else -vth_mag
    return vec6(s[0], s[1], s[2], s[3], vr, vth)


@ti.func
def _rk4_step(s, E, Lz, Q, a, h):
    k1 = _deriv(s, E, Lz, Q, a)
    k2 = _deriv(s + 0.5 * h * k1, E, Lz, Q, a)
    k3 = _deriv(s + 0.5 * h * k2, E, Lz, Q, a)
    k4 = _deriv(s + h * k3, E, Lz, Q, a)
    s2 = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _project(s2, E, Lz, Q, a)


@ti.func
def _zamo_init(r, theta, a, n_r, n_th, n_ph):
    """Formula 7 ZAMO tetrad → (E, L_z, Q, v_r0, v_θ0) for the geodesic state.

    Mirrors ``scripts.thumb.zamo_photon_momentum`` + the v_r/v_θ extraction in
    ``renderer.geodesic.integrate_null_geodesic`` (v_r = Δ·p_r, v_θ = p_θ).
    """
    sin2 = ti.sin(theta) ** 2
    cos2 = ti.cos(theta) ** 2
    Sigma = r * r + a * a * cos2
    Delta = _delta(r, a)
    A = (r * r + a * a) ** 2 - a * a * Delta * sin2     # exact A (Formula 7)
    omega = 2.0 * a * r / A
    alpha = ti.sqrt(Sigma * Delta / A)
    g_phiphi = A * sin2 / Sigma

    # Contravariant momentum (Formula 7).
    p_t_con = 1.0 / alpha
    p_r_con = n_r * ti.sqrt(Delta / Sigma)
    p_th_con = n_th * (1.0 / ti.sqrt(Sigma))
    p_ph_con = omega / alpha + n_ph * (1.0 / ti.sqrt(g_phiphi))

    # Metric components (Formula 1) to lower the index.
    g_tt = -(1.0 - 2.0 * r / Sigma)
    g_tph = -2.0 * a * r * sin2 / Sigma
    g_phph = g_phiphi
    g_rr = Sigma / Delta
    g_thth = Sigma

    p_t = g_tt * p_t_con + g_tph * p_ph_con
    p_r = g_rr * p_r_con
    p_th = g_thth * p_th_con
    p_ph = g_phph * p_ph_con + g_tph * p_t_con

    E = -p_t
    Lz = p_ph
    sin2_safe = ti.max(sin2, _SIN2_MIN)
    Q = p_th * p_th + cos2 * (-(a * a) * E * E + Lz * Lz / sin2_safe)

    v_r0 = Delta * p_r       # v_r = Δ·p_r  (p_r covariant)
    v_th0 = p_th             # v_θ = p_θ
    return E, Lz, Q, v_r0, v_th0


# --------------------------------------------------------------------------- #
# Starmap sampler @ti.func — mirrors renderer.starmap.Starmap.sample
# --------------------------------------------------------------------------- #
@ti.func
def _texel(level, x, y):
    base = star_off[level]
    w = star_w[level]
    idx = base + (y * w + x) * 3
    return vec3(
        ti.cast(star_flat[idx + 0], ti.f32),
        ti.cast(star_flat[idx + 1], ti.f32),
        ti.cast(star_flat[idx + 2], ti.f32),
    )


@ti.func
def _sample_level(level, u, v):
    w = star_w[level]
    h = star_h[level]
    uu = u - ti.floor(u)                          # wrap φ (periodic)
    vv = ti.min(ti.max(v, 0.0), 1.0)              # clamp θ
    fu = uu * w - 0.5
    fv = vv * h - 0.5
    x0 = ti.cast(ti.floor(fu), ti.i32)
    y0 = ti.cast(ti.floor(fv), ti.i32)
    du = fu - ti.floor(fu)
    dv = fv - ti.floor(fv)

    ix0 = x0 % w
    if ix0 < 0:
        ix0 += w
    ix1 = (ix0 + 1) % w
    iy0 = ti.min(ti.max(y0, 0), h - 1)
    iy1 = ti.min(ti.max(y0 + 1, 0), h - 1)

    c00 = _texel(level, ix0, iy0)
    c10 = _texel(level, ix1, iy0)
    c01 = _texel(level, ix0, iy1)
    c11 = _texel(level, ix1, iy1)
    top = c00 * (1.0 - du) + c10 * du
    bot = c01 * (1.0 - du) + c11 * du
    return top * (1.0 - dv) + bot * dv


@ti.func
def _normalize_sphere(theta, phi):
    """Fold raw integrator (θ, φ) onto the standard sphere: θ ∈ [0, π].

    Mirrors ``renderer.starmap.normalize_sphere_angles`` (the host single source
    of truth). Rays with near-zero L_z on the center column push θ slightly
    negative (polar punch-through); reflecting across the pole
    (θ→|θ|, φ→φ+π) maps that back to the same physical direction in canonical
    form, so neither the UV lookup nor the Formula 10 Jacobian sees a θ outside
    [0, π]. Pure coordinate identity — no physics, no formula re-derivation.
    """
    th = theta - _TWO_PI * ti.floor(theta / _TWO_PI)   # → [0, 2π)
    ph = phi
    if th > math.pi:
        th = _TWO_PI - th                              # reflect across nearer pole
        ph = ph + math.pi                              # ... + half turn in azimuth
    ph = ph - _TWO_PI * ti.floor(ph / _TWO_PI)         # → [0, 2π)
    return vec2(th, ph)


@ti.func
def _sample_trilinear(u, v, lod):
    L = ti.min(ti.max(lod, 0.0), _MAX_LOD)
    l0 = ti.cast(ti.floor(L), ti.i32)
    l1 = ti.min(l0 + 1, _N_LEVELS - 1)
    l0 = ti.min(l0, _N_LEVELS - 1)
    f = L - ti.floor(L)
    c0 = _sample_level(l0, u, v)
    c1 = _sample_level(l1, u, v)
    return c0 * (1.0 - f) + c1 * f


# --------------------------------------------------------------------------- #
# Pipe A kernel
# --------------------------------------------------------------------------- #
@ti.func
def _ray_dir(px, py, res, tan_half_fov):
    """Local ZAMO-frame unit ray direction for screen sample (px, py)."""
    sx = (2.0 * (px + 0.5) / res - 1.0) * tan_half_fov   # right (+φ̂)
    sy = (1.0 - 2.0 * (py + 0.5) / res) * tan_half_fov   # up    (−θ̂)
    n_r = -1.0
    n_th = -sy
    n_ph = sx
    inv = 1.0 / ti.sqrt(n_r * n_r + n_th * n_th + n_ph * n_ph)
    return n_r * inv, n_th * inv, n_ph * inv


@ti.kernel
def render_pipe_a(res: int, tan_half_fov: float, r_cam: float,
                  theta_cam: float, phi_cam: float, a: float,
                  r_max: float, n_steps: int, d_lambda: float,
                  lod_enabled: int):
    """Pipe A: trace one primary + one offset ray per pixel; sample the lensed sky.

    The offset ray (screen u shifted by +1/res = one pixel) is integrated to exit
    **inside the same while loop** as the primary ray (Formula 10 requirement) so
    its exit direction yields the on-sky Jacobian J → mip LOD L. No step-count
    proxy is used.
    """
    for py, px in ti.ndrange(res, res):
        # --- primary + offset initial conditions (ZAMO tetrad, Formula 7) ---
        npr_r, npr_th, npr_ph = _ray_dir(ti.cast(px, ti.f32), ti.cast(py, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)
        nof_r, nof_th, nof_ph = _ray_dir(ti.cast(px + 1, ti.f32), ti.cast(py, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)

        Ep, Lp, Qp, vr_p, vth_p = _zamo_init(r_cam, theta_cam, a, npr_r, npr_th, npr_ph)
        Eo, Lo, Qo, vr_o, vth_o = _zamo_init(r_cam, theta_cam, a, nof_r, nof_th, nof_ph)

        sp = vec6(r_cam, theta_cam, phi_cam, 0.0, vr_p, vth_p)
        so = vec6(r_cam, theta_cam, phi_cam, 0.0, vr_o, vth_o)

        out_p = _RUNNING
        out_o = _RUNNING
        th_p_exit = theta_cam
        ph_p_exit = phi_cam
        th_o_exit = theta_cam
        ph_o_exit = phi_cam

        # --- shared integration loop: step both rays until both exit ---
        step = 0
        while step < n_steps and (out_p == _RUNNING or out_o == _RUNNING):
            if out_p == _RUNNING:
                if _delta(sp[0], a) < _DELTA_MIN:
                    out_p = _CAPTURED
                else:
                    sp = _rk4_step(sp, Ep, Lp, Qp, a, d_lambda)
                    if _delta(sp[0], a) < _DELTA_MIN or sp[0] < 2.0:
                        out_p = _CAPTURED
                    elif sp[0] >= r_max:
                        out_p = _ESCAPED
                        th_p_exit = sp[1]
                        ph_p_exit = sp[2]
            if out_o == _RUNNING:
                if _delta(so[0], a) < _DELTA_MIN:
                    out_o = _CAPTURED
                else:
                    so = _rk4_step(so, Eo, Lo, Qo, a, d_lambda)
                    if _delta(so[0], a) < _DELTA_MIN or so[0] < 2.0:
                        out_o = _CAPTURED
                    elif so[0] >= r_max:
                        out_o = _ESCAPED
                        th_o_exit = so[1]
                        ph_o_exit = so[2]
            step += 1

        # --- shade ---
        col = vec3(0.0, 0.0, 0.0)
        if out_p == _ESCAPED:
            # Fold the raw integrator exit angles back onto the standard sphere
            # BEFORE the UV lookup and the Jacobian: center-column rays push θ
            # negative (polar punch-through), which the old clamp crushed onto the
            # north-pole texel (the vertical streak). Both rays are normalized so
            # the Jacobian deltas are taken between canonical directions.
            np_ = _normalize_sphere(th_p_exit, ph_p_exit)
            th_p_n = np_[0]
            ph_p_n = np_[1]

            u = ph_p_n / (2.0 * math.pi)
            u = u - ti.floor(u)
            v = ti.min(ti.max(th_p_n / math.pi, 0.0), 1.0)

            lod = 0.0
            if lod_enabled == 1:
                if out_o == _ESCAPED:
                    # Formula 10 (v1.3): J from the offset ray's raw one-pixel
                    # exit-direction delta. φ spans 2π radians across the full
                    # 16384-texel width, so dividing by 2π maps the angular
                    # footprint to a texel footprint (without it the LOD saturates
                    # to max mip for every background pixel).
                    no_ = _normalize_sphere(th_o_exit, ph_o_exit)
                    th_o_n = no_[0]
                    ph_o_n = no_[1]
                    d_th = th_o_n - th_p_n
                    d_ph = ph_o_n - ph_p_n
                    # wrap δφ into [−π, π] (φ periodic).
                    d_ph = d_ph - 2.0 * math.pi * ti.round(d_ph / (2.0 * math.pi))
                    sin_th = ti.sin(th_p_n)
                    J = ti.sqrt(d_th * d_th + sin_th * sin_th * d_ph * d_ph)
                    lod = ti.log(_STARMAP_WIDTH * J / (2.0 * math.pi)) / ti.log(2.0)  # log2
                else:
                    # Offset ray dived into the chaotic edge → huge footprint.
                    lod = _MAX_LOD
            col = _sample_trilinear(u, v, lod)

        pixels[py, px, 0] = col[0]
        pixels[py, px, 1] = col[1]
        pixels[py, px, 2] = col[2]


# --------------------------------------------------------------------------- #
# Seam-isolation diagnostics (Gate 2 follow-up — NOT part of the render path)
# --------------------------------------------------------------------------- #
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
        pixels[py, px, 0] = col[0]
        pixels[py, px, 1] = col[1]
        pixels[py, px, 2] = col[2]


@ti.kernel
def render_fixed_lod(res: int, tan_half_fov: float, r_cam: float,
                     theta_cam: float, phi_cam: float, a: float,
                     r_max: float, n_steps: int, d_lambda: float,
                     lod_fixed: ti.f32):
    """Diagnostic 2: full geodesic lensing (primary ray only) with mip LOD
    PINNED to ``lod_fixed`` for every escaped pixel.

    Removes the Jacobian entirely. If a seam appears here but not in
    Diagnostic 1, it is a ray-classification boundary (escaped vs captured),
    not a starmap/mip issue.
    """
    for py, px in ti.ndrange(res, res):
        npr_r, npr_th, npr_ph = _ray_dir(ti.cast(px, ti.f32), ti.cast(py, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)
        Ep, Lp, Qp, vr_p, vth_p = _zamo_init(r_cam, theta_cam, a, npr_r, npr_th, npr_ph)
        sp = vec6(r_cam, theta_cam, phi_cam, 0.0, vr_p, vth_p)
        out_p = _RUNNING
        th_p_exit = theta_cam
        ph_p_exit = phi_cam
        step = 0
        while step < n_steps and out_p == _RUNNING:
            if _delta(sp[0], a) < _DELTA_MIN:
                out_p = _CAPTURED
            else:
                sp = _rk4_step(sp, Ep, Lp, Qp, a, d_lambda)
                if _delta(sp[0], a) < _DELTA_MIN or sp[0] < 2.0:
                    out_p = _CAPTURED
                elif sp[0] >= r_max:
                    out_p = _ESCAPED
                    th_p_exit = sp[1]
                    ph_p_exit = sp[2]
            step += 1
        col = vec3(0.0, 0.0, 0.0)
        if out_p == _ESCAPED:
            # Fold raw exit angles onto the standard sphere before UV lookup
            # (polar punch-through: θ<0). Same normalization as render_pipe_a.
            np_ = _normalize_sphere(th_p_exit, ph_p_exit)
            u = np_[1] / (2.0 * math.pi)
            u = u - ti.floor(u)
            v = ti.min(ti.max(np_[0] / math.pi, 0.0), 1.0)
            col = _sample_trilinear(u, v, lod_fixed)
        pixels[py, px, 0] = col[0]
        pixels[py, px, 1] = col[1]
        pixels[py, px, 2] = col[2]


# Per-column exit-state dump buffer: [phi_exit_raw, theta_exit, outcome].
phi_dump: ti.Field = None          # type: ignore[assignment]


@ti.kernel
def dump_phi_exit(res: int, row_y: int, tan_half_fov: float, r_cam: float,
                  theta_cam: float, phi_cam: float, a: float,
                  r_max: float, n_steps: int, d_lambda: float):
    """Diagnostic 3: trace the PRIMARY ray for every column of a single screen row
    and record the *raw accumulated* exit azimuth ``phi_exit`` (no mod/frac), the
    exit ``theta``, and the outcome code. Reveals whether adjacent columns wind by
    different multiples of 2π (the branch-cut hypothesis)."""
    for px in range(res):
        npr_r, npr_th, npr_ph = _ray_dir(ti.cast(px, ti.f32), ti.cast(row_y, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)
        Ep, Lp, Qp, vr_p, vth_p = _zamo_init(r_cam, theta_cam, a, npr_r, npr_th, npr_ph)
        sp = vec6(r_cam, theta_cam, phi_cam, 0.0, vr_p, vth_p)
        out_p = _RUNNING
        th_p_exit = theta_cam
        ph_p_exit = phi_cam
        step = 0
        while step < n_steps and out_p == _RUNNING:
            if _delta(sp[0], a) < _DELTA_MIN:
                out_p = _CAPTURED
            else:
                sp = _rk4_step(sp, Ep, Lp, Qp, a, d_lambda)
                if _delta(sp[0], a) < _DELTA_MIN or sp[0] < 2.0:
                    out_p = _CAPTURED
                elif sp[0] >= r_max:
                    out_p = _ESCAPED
                    th_p_exit = sp[1]
                    ph_p_exit = sp[2]
            step += 1
        phi_dump[px, 0] = ph_p_exit
        phi_dump[px, 1] = th_p_exit
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

    phi_dump = ti.field(dtype=ti.f32, shape=(res, 3))
    dump_phi_exit(res, row_y, tan_half_fov, r_cam, theta_cam, 0.0, a,
                  r_max, n_steps, d_lambda)
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


def render_pipe_a_image(cfg: dict, res: int, lod_enabled: bool) -> np.ndarray:
    """Render one Pipe A frame at ``res×res`` and return a float32 (res,res,3) HDR."""
    bh = cfg["black_hole"]
    rcfg = cfg["render"]
    cam = cfg["camera"]
    th = cfg["thumb"]

    a = float(bh["spin"])
    r_cam = float(th.get("camera_radius", cam["default_radius"]))
    fov_deg = float(th.get("fov_deg", cam["default_fov_deg"]))
    theta_cam = math.radians(float(th["camera_theta_deg"]))
    phi_cam = 0.0
    n_steps = int(rcfg["max_steps_pipe_a"])
    d_lambda = float(rcfg["d_lambda_pipe_a"])
    r_max = float(rcfg["r_max"])
    tan_half_fov = math.tan(math.radians(fov_deg) / 2.0)

    _alloc_output(res)
    render_pipe_a(res, tan_half_fov, r_cam, theta_cam, phi_cam, a,
                  r_max, n_steps, d_lambda, 1 if lod_enabled else 0)
    ti.sync()
    return pixels.to_numpy()


def tonemap(hdr: np.ndarray, exposure: float, gamma: float) -> np.ndarray:
    """Reinhard tonemap + gamma, → uint8 (matches scripts/thumb.py)."""
    img = hdr * exposure
    img = img / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / gamma)
    return (img * 255.0 + 0.5).astype(np.uint8)


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
    _save_and_probe("test_starmap_raw.png", pixels.to_numpy())

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
    render_fixed_lod(res, tan_half_fov, r_cam, theta_cam, 0.0, a,
                     r_max, n_steps, d_lambda, 3.0)
    ti.sync()
    _save_and_probe("test_fixed_lod.png", pixels.to_numpy())

    print(f"Saved seam diagnostics to {out_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "seam":
        _seam_diag()
    elif len(sys.argv) > 1 and sys.argv[1] == "phidump":
        _phi_dump()
    else:
        _gate2_lod_test()
