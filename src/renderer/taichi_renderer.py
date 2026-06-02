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
# physics). Overridden from configs/render.yaml in setup_renderer:
#   _DELTA_MIN ← render.horizon_epsilon  (Δ-capture stop before the horizon)
#   _SIN2_MIN  ← render.sin2_min         (polar guard on dφ/dλ, dt/dλ ONLY; Formula 12)
_DELTA_MIN = 0.05
_SIN2_MIN = 1e-10

# State vector layout (Phase 1.3, Formula 12): [y, u, φ, t, v_y, v_u] where
#   y = r − r₊   (horizon-relative radius; Formula 11)
#   u = cos θ    (singularity-free polar coordinate)
#   v_y = dy/dλ = Δ·p_r   (invariant migrated from v_r = Δ·p_r; dy = dr)
#   v_u = du/dλ = −sinθ·p_θ = −√(1−u²)·p_θ
vec6 = ti.types.vector(6, ti.f32)
vec4 = ti.types.vector(4, ti.f32)
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

# Beauty-pass output (non-square, set per frame render).
frame_pixels: ti.Field = None     # type: ignore[assignment]
# Kernel-split hand-off buffers (Phase 2.4): physics kernel writes, shading reads.
exit_buf: ti.Field = None         # type: ignore[assignment]  (H,W,3): u_exit, φ_exit, outcome
disk_buf: ti.Field = None         # type: ignore[assignment]  (H,W,4): disk_rgb + transmittance
depth_pixels: ti.Field = None     # type: ignore[assignment]  (H,W): transmittance-weighted Z (3.4)
_FW = 0
_FH = 0


def setup_renderer(cfg: dict) -> Starmap:
    """Initialise Taichi (CUDA), load the starmap, and upload its mip pyramid.

    Returns the host-side :class:`Starmap` (kept alive so the GPU upload's source
    arrays and the reference sampler remain available to callers/tests).
    """
    global star_flat, star_off, star_w, star_h
    global _N_LEVELS, _MAX_LOD, _STARMAP_WIDTH
    global _DELTA_MIN, _SIN2_MIN

    # Integration-scheme constants from config (baked into kernels at first JIT).
    _DELTA_MIN = float(cfg["render"]["horizon_epsilon"])
    _SIN2_MIN = float(cfg["render"]["sin2_min"])

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


def _horizon_constants(a: float) -> tuple[float, float]:
    """Precompute FP32-stable horizon constants (guid 1.1 / Formula 11).

    Returns ``(k_horizon, r_plus)`` with ``k_horizon = √(1−a²)`` and the *true*
    outer horizon ``r₊ = 1 + k_horizon``. NOTE: this is derived from ``a`` in
    Python (like E_I/L_I and tan_half_fov), NOT read from ``configs.black_hole.r_plus``
    — that config key is mislabeled (it holds k=√(1−a²)≈0.0447, and is consumed by
    scripts/thumb.py as an r_floor) so it must not be repurposed here.
    """
    k = math.sqrt(1.0 - a * a)
    return k, 1.0 + k


def _alloc_output(res: int) -> None:
    global pixels, _RES
    if pixels is None or _RES != res:
        pixels = ti.field(dtype=ti.f32, shape=(res, res, 3))
        _RES = res


def _alloc_frame(width: int, height: int) -> None:
    global frame_pixels, exit_buf, disk_buf, depth_pixels, _FW, _FH
    if frame_pixels is None or _FW != width or _FH != height:
        frame_pixels = ti.field(dtype=ti.f32, shape=(height, width, 3))
        exit_buf = ti.field(dtype=ti.f32, shape=(height, width, 3))
        disk_buf = ti.field(dtype=ti.f32, shape=(height, width, 4))
        depth_pixels = ti.field(dtype=ti.f32, shape=(height, width))
        _FW = width
        _FH = height


# --------------------------------------------------------------------------- #
# Physics @ti.func — ported verbatim from renderer.geodesic / metric / thumb
# --------------------------------------------------------------------------- #
@ti.func
def _delta(r, a):
    return r * r - 2.0 * r + a * a


@ti.func
def _delta_y(y, k):
    # Δ = y·(y + 2k)  ≡  r²−2r+a²   (Formula 11; zero catastrophic cancellation).
    # y = r − r₊,  k = √(1−a²),  r₊ = 1 + k.
    return y * (y + 2.0 * k)


@ti.func
def _radial_potential_y(y, k, r_plus, E, Lz, Q, a):
    # R = [E(r²+a²) − aL_z]² − Δ·[(L_z − aE)² + Q]   (Formula 6, null form),
    # with r = y + r₊ and Δ = _delta_y(y,k)  (Formula 11).
    r = y + r_plus
    P = E * (r * r + a * a) - a * Lz
    B = (Lz - a * E) ** 2 + Q
    return P * P - _delta_y(y, k) * B


@ti.func
def _radial_potential_deriv_y(y, k, r_plus, E, Lz, Q, a):
    # dR/dy = dR/dr = 4Er·P − (2r−2)·B   (calculus derivative; dr/dy = 1).
    r = y + r_plus
    P = E * (r * r + a * a) - a * Lz
    B = (Lz - a * E) ** 2 + Q
    return 4.0 * E * r * P - (2.0 * r - 2.0) * B


@ti.func
def _theta_potential_u(u, E, Lz, Q, a):
    # Θ_u(u) = (1−u²)(Q + a²E²u²) − L_z²u²   (Formula 12; singularity-free).
    u2 = u * u
    return (1.0 - u2) * (Q + a * a * E * E * u2) - Lz * Lz * u2


@ti.func
def _theta_potential_deriv_u(u, E, Lz, Q, a):
    # dΘ_u/du = −2u(Q + a²E²u²) + 2a²E²u(1−u²) − 2L_z²u   (Formula 12).
    u2 = u * u
    return (-2.0 * u * (Q + a * a * E * E * u2)
            + 2.0 * a * a * E * E * u * (1.0 - u2)
            - 2.0 * Lz * Lz * u)


@ti.func
def _deriv(s, E, Lz, Q, a, k, r_plus):
    # State s = [y, u, φ, t, v_y, v_u]; returns ds/dλ (Mino time), Formulas 6/11/12.
    y = s[0]
    u = s[1]
    vy = s[4]
    vu = s[5]
    r = y + r_plus
    Delta = _delta_y(y, k)
    sin2 = 1.0 - u * u
    sin2_safe = ti.max(sin2, _SIN2_MIN)   # polar guard on dφ/dλ, dt/dλ ONLY
    P = E * (r * r + a * a) - a * Lz

    dy = vy
    du = vu
    dphi = -(a * E - Lz / sin2_safe) + a * P / Delta
    dt = -a * (a * E * sin2 - Lz) + (r * r + a * a) * P / Delta
    dvy = 0.5 * _radial_potential_deriv_y(y, k, r_plus, E, Lz, Q, a)
    dvu = 0.5 * _theta_potential_deriv_u(u, E, Lz, Q, a)
    return vec6(dy, du, dphi, dt, dvy, dvu)


@ti.func
def _project(s, E, Lz, Q, a, k, r_plus):
    # Re-impose (dy/dλ)² = R, (du/dλ)² = Θ_u exactly (signs from RK4 evolution).
    R = _radial_potential_y(s[0], k, r_plus, E, Lz, Q, a)
    Theta_u = _theta_potential_u(s[1], E, Lz, Q, a)
    vy_mag = ti.sqrt(ti.max(0.0, R))
    vu_mag = ti.sqrt(ti.max(0.0, Theta_u))
    vy = vy_mag if s[4] >= 0.0 else -vy_mag
    vu = vu_mag if s[5] >= 0.0 else -vu_mag
    return vec6(s[0], s[1], s[2], s[3], vy, vu)


@ti.func
def _rk4_delta(s, E, Lz, Q, a, k, r_plus, h):
    # RK4 increment Δs (before projection); split out so the loop can apply
    # Kahan compensated summation to the state accumulation (guid 1.4).
    k1 = _deriv(s, E, Lz, Q, a, k, r_plus)
    k2 = _deriv(s + 0.5 * h * k1, E, Lz, Q, a, k, r_plus)
    k3 = _deriv(s + 0.5 * h * k2, E, Lz, Q, a, k, r_plus)
    k4 = _deriv(s + h * k3, E, Lz, Q, a, k, r_plus)
    return (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@ti.func
def _rk4_step(s, E, Lz, Q, a, k, r_plus, h):
    # Plain (uncompensated) RK4 + project — used by the diagnostics kernels.
    s2 = s + _rk4_delta(s, E, Lz, Q, a, k, r_plus, h)
    return _project(s2, E, Lz, Q, a, k, r_plus)


@ti.func
def _rk4_step_kahan(s, c, E, Lz, Q, a, k, r_plus, h):
    """Compensated (Kahan) RK4 step (guid 1.4). Returns ``(s_next, c_next)``.

    Kahan summation on the state accumulation keeps the slowly-growing position
    components (y, u, φ, t) from losing low-order bits over hundreds of small
    f32 steps. The velocity components (v_y, v_u) are re-imposed exactly by the
    projection each step, so their compensation is reset to 0.
    """
    delta = _rk4_delta(s, E, Lz, Q, a, k, r_plus, h)
    y_step = delta - c
    t = s + y_step
    c_next = (t - s) - y_step
    s_next = _project(t, E, Lz, Q, a, k, r_plus)
    # Projection overwrote v_y, v_u → clear their (now stale) compensation.
    c_next = vec6(c_next[0], c_next[1], c_next[2], c_next[3], 0.0, 0.0)
    return s_next, c_next


@ti.func
def _zamo_init(r, theta, a, k, r_plus, n_r, n_th, n_ph):
    """Formula 7 ZAMO tetrad → (E, L_z, Q, v_y0, v_u0) for the [y,u,...] state.

    Mirrors ``scripts.thumb.zamo_photon_momentum``. The state-derivative
    extraction is migrated (Formula 12): v_y = Δ·p_r (= old v_r, since dy=dr) and
    v_u = du/dλ = −sinθ·p_θ = −√(1−u²)·p_θ.
    """
    sin2 = ti.sin(theta) ** 2
    cos2 = ti.cos(theta) ** 2
    Sigma = r * r + a * a * cos2
    Delta = _delta_y(r - r_plus, k)                    # Formula 11 (FP32-stable)
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

    v_y0 = Delta * p_r              # v_y = Δ·p_r  (p_r covariant; dy = dr)
    v_u0 = -ti.sin(theta) * p_th    # v_u = −sinθ·p_θ = −√(1−u²)·p_θ
    return E, Lz, Q, v_y0, v_u0


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
# Pipe B physics @ti.func — ported verbatim from renderer.disk
# --------------------------------------------------------------------------- #
@ti.func
def _gas_four_velocity(r, theta, a, r_isco, E_I, L_I):
    """Contravariant gas 4-velocity u^μ = (u^t, u^r, u^θ, u^φ).

    r ≥ r_isco : circular orbit (Formula 3).
    r <  r_isco : plunging free-fall with frozen E_I, L_I (Formula 5).
    Mirrors ``renderer.disk.gas_four_velocity`` — no formula re-derived.
    """
    u_t = 0.0
    u_r = 0.0
    u_th = 0.0
    u_ph = 0.0
    if r >= r_isco:
        # Formula 3 — circular orbit (numerator (1 + a·r^-3/2) is mandatory).
        r15 = ti.pow(r, 1.5)
        Omega = 1.0 / (r15 + a)
        u_t = (1.0 + a / r15) / ti.sqrt(1.0 - 3.0 / r + 2.0 * a / r15)
        u_ph = Omega * u_t
    else:
        # Formula 5 — plunging region (frozen E_I, L_I; u^r must be infalling).
        cos2 = ti.cos(theta) ** 2
        Sigma = r * r + a * a * cos2
        # Review #5: factored Δ=(r−r₊)(r−r₋) (Formula 11) instead of the
        # cancellation-prone r²−2r+a², matching the migrated integrator. (Dead
        # for r_inner==r_isco today, but keeps the inner-disk gas FP32-stable if
        # disk.r_inner is ever lowered below the ISCO.)
        kk = ti.sqrt(ti.max(1.0 - a * a, 0.0))
        Delta = (r - (1.0 + kk)) * (r - (1.0 - kk))
        X = E_I * (r * r + a * a) - a * L_I
        u_r = -(1.0 / Sigma) * ti.sqrt(
            ti.max(0.0, X * X - Delta * (r * r + (L_I - a * E_I) ** 2))
        )
        u_t = (1.0 / Sigma) * ((r * r + a * a) * X / Delta - a * (a * E_I - L_I))
        u_ph = (1.0 / Sigma) * (a * X / Delta - (a * E_I - L_I))
    return vec4(u_t, u_r, u_th, u_ph)


@ti.func
def _blackbody_rgb(temp):
    """Normalized blackbody chromaticity (Formula 9 helper, verbatim).

    No T⁴ amplitude — so the pow(g, 4.0) intensity factor in the march is correct
    and not double-counted.
    """
    return vec3(
        1.0 - ti.exp(-temp / 3500.0),
        1.0 - ti.exp(-temp / 5500.0),
        1.0 - ti.exp(-temp / 9500.0),
    )


@ti.func
def _disk_emit(y, u, vy, vu, E, Lz, a, k, r_plus, r_isco, E_I, L_I,
               r_inner, r_outer, theta_half, sigma_frac, T_0, emis_c, absb_c, ds):
    """One volumetric disk sample at a geodesic point → (emission RGB, dτ).

    Returns ``vec4(emission_r, emission_g, emission_b, dtau)`` where the running
    pixel update is ``color += T·emission`` and ``T *= exp(-dtau)``. Outside the
    equatorial slab bounding box it returns zeros.

    Formula 8 g-factor (state migrated, Formula 12): the kernel state carries
    ``v_y = Δ·p_r`` (p_r covariant) and ``v_u = −√(1−u²)·p_θ``; with conserved
    ``p_t = −E`` and ``p_φ = L_z`` the covariant photon momentum is recovered as
    ``p_r = v_y/Δ`` (NOT divided again — avoids the Formula-8 known bug) and
    ``p_θ = −v_u/√(1−u²)``. Formula 9: chromaticity·g⁴ volumetric emission.
    """
    out = vec4(0.0, 0.0, 0.0, 0.0)
    r = y + r_plus
    th = ti.acos(ti.min(ti.max(u, -1.0), 1.0))
    dz = th - 0.5 * math.pi
    if (ti.abs(dz) < theta_half) and (r >= r_inner) and (r <= r_outer):
        u4 = _gas_four_velocity(r, th, a, r_isco, E_I, L_I)
        Delta = _delta_y(y, k)
        sin_th = ti.sqrt(ti.max(1.0 - u * u, _SIN2_MIN))
        p_t = -E
        p_r = vy / Delta            # covariant p_r (state stores v_y = Δ·p_r)
        p_th = -vu / sin_th         # covariant p_θ = −v_u/√(1−u²)  (Formula 12)
        p_ph = Lz
        denom = p_t * u4[0] + p_r * u4[1] + p_th * u4[2] + p_ph * u4[3]
        if ti.abs(denom) > 1e-12:
            g = -1.0 / denom    # Formula 8
            if g > 0.0:
                # Decision B temperature model: T = T_0·(6/r)^0.75.
                T_emit = T_0 * ti.pow(6.0 / r, 0.75)
                T_obs = g * T_emit
                chroma = _blackbody_rgb(T_obs)
                sigma_theta = theta_half * sigma_frac
                density = ti.exp(-0.5 * (dz / sigma_theta) ** 2)
                g4 = g * g * g * g                       # Formula 9 (3D volume: g⁴)
                emission = emis_c * density * g4 * ds
                out = vec4(emission * chroma[0], emission * chroma[1],
                           emission * chroma[2], absb_c * density * ds)
    return out


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
                  k_horizon: float, r_plus: float,
                  r_max: float, n_steps: int, d_lambda: float,
                  lod_enabled: int):
    """Pipe A: trace one primary + one offset ray per pixel; sample the lensed sky.

    The offset ray (screen u shifted by +1/res = one pixel) is integrated to exit
    **inside the same while loop** as the primary ray (Formula 10 requirement) so
    its exit direction yields the on-sky Jacobian J → mip LOD L. No step-count
    proxy is used.

    State is [y, u, φ, t, v_y, v_u] (Formula 11/12): y = r − r₊, u = cosθ.
    """
    y_cam = r_cam - r_plus
    u_cam = ti.cos(theta_cam)
    r_capture = 2.0 - r_plus      # r < 2 ⇔ y < 2 − r₊
    y_escape = r_max - r_plus     # r ≥ r_max ⇔ y ≥ r_max − r₊
    for py, px in ti.ndrange(res, res):
        # --- primary + offset initial conditions (ZAMO tetrad, Formula 7) ---
        npr_r, npr_th, npr_ph = _ray_dir(ti.cast(px, ti.f32), ti.cast(py, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)
        nof_r, nof_th, nof_ph = _ray_dir(ti.cast(px + 1, ti.f32), ti.cast(py, ti.f32),
                                         ti.cast(res, ti.f32), tan_half_fov)

        Ep, Lp, Qp, vy_p, vu_p = _zamo_init(r_cam, theta_cam, a, k_horizon, r_plus,
                                            npr_r, npr_th, npr_ph)
        Eo, Lo, Qo, vy_o, vu_o = _zamo_init(r_cam, theta_cam, a, k_horizon, r_plus,
                                            nof_r, nof_th, nof_ph)

        sp = vec6(y_cam, u_cam, phi_cam, 0.0, vy_p, vu_p)
        so = vec6(y_cam, u_cam, phi_cam, 0.0, vy_o, vu_o)
        c_sp = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # Kahan compensation (guid 1.4)
        c_so = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        out_p = _RUNNING
        out_o = _RUNNING
        u_p_exit = u_cam
        ph_p_exit = phi_cam
        u_o_exit = u_cam
        ph_o_exit = phi_cam

        # --- shared integration loop: step both rays until both exit ---
        step = 0
        while step < n_steps and (out_p == _RUNNING or out_o == _RUNNING):
            if out_p == _RUNNING:
                if _delta_y(sp[0], k_horizon) < _DELTA_MIN:
                    out_p = _CAPTURED
                else:
                    sp, c_sp = _rk4_step_kahan(sp, c_sp, Ep, Lp, Qp, a,
                                               k_horizon, r_plus, d_lambda)
                    if _delta_y(sp[0], k_horizon) < _DELTA_MIN or sp[0] < r_capture:
                        out_p = _CAPTURED
                    elif sp[0] >= y_escape:
                        out_p = _ESCAPED
                        u_p_exit = sp[1]
                        ph_p_exit = sp[2]
            if out_o == _RUNNING:
                if _delta_y(so[0], k_horizon) < _DELTA_MIN:
                    out_o = _CAPTURED
                else:
                    so, c_so = _rk4_step_kahan(so, c_so, Eo, Lo, Qo, a,
                                               k_horizon, r_plus, d_lambda)
                    if _delta_y(so[0], k_horizon) < _DELTA_MIN or so[0] < r_capture:
                        out_o = _CAPTURED
                    elif so[0] >= y_escape:
                        out_o = _ESCAPED
                        u_o_exit = so[1]
                        ph_o_exit = so[2]
            step += 1

        # --- shade ---
        col = vec3(0.0, 0.0, 0.0)
        if out_p == _ESCAPED:
            # u = cosθ keeps θ_exit = acos(u) ∈ [0, π] by construction, so the old
            # polar punch-through fold is no longer needed; just recover θ and wrap φ.
            th_p_n = ti.acos(ti.min(ti.max(u_p_exit, -1.0), 1.0))
            u = ph_p_exit / (2.0 * math.pi)
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
                    th_o_n = ti.acos(ti.min(ti.max(u_o_exit, -1.0), 1.0))
                    d_th = th_o_n - th_p_n
                    d_ph = ph_o_exit - ph_p_exit
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
# Beauty kernel — Pipe A (lensed starmap + LOD) combined with Pipe B (disk)
# --------------------------------------------------------------------------- #
@ti.kernel
def render_beauty_physics(width: int, height: int,
                          fwd_r: float, fwd_th: float, fwd_ph: float,
                          rgt_r: float, rgt_th: float, rgt_ph: float,
                          up_r: float, up_th: float, up_ph: float,
                          tan_half_x: float, tan_half_y: float,
                          r_cam: float, theta_cam: float, phi_cam: float,
                          a: float, k_horizon: float, r_plus: float,
                          r_max: float, n_steps: int, d_lambda: float,
                          adaptive_floor: float, disk_enabled: int,
                          projection_mode: int, depth_infinity: float,
                          r_isco: float, E_I: float, L_I: float,
                          r_inner: float, r_outer: float, theta_half: float,
                          bound_sin_half: float,
                          sigma_frac: float, T_0: float, emis_c: float, absb_c: float):
    """Kernel 1 (Phase 2.4 split): trace ONE primary ray per pixel — no offset ray.

    Writes the per-pixel exit state to ``exit_buf`` (u_exit, φ_exit, outcome), the
    front-to-back disk accumulation to ``disk_buf`` (disk_rgb, transmittance), and
    the transmittance-weighted Mino-affine Z to ``depth_pixels`` (guid 3.4). The
    Formula-10 LOD + background lookup are deferred to ``render_beauty_shade``,
    which differences neighbor exit directions in screen space (SKILL.md F10
    amendment v1.4). Eliminating the offset ray halves the geodesic workload.

    Adaptive Mino step (guid 2.2): ``h = d_lambda·max(adaptive_floor, y/(y+2))`` —
    full steps far out, shrinking toward the horizon (y→0).

    ``projection_mode`` (guid 4.1): 0 = perspective (camera basis + FOV), 1 =
    equirectangular 360° (px→lon, py→lat in the local ZAMO frame, for VR output).

    State is [y, u, φ, t, v_y, v_u] (Formula 11/12): y = r − r₊, u = cosθ.
    """
    y_cam = r_cam - r_plus
    u_cam = ti.cos(theta_cam)
    r_capture = 2.0 - r_plus      # r < 2 ⇔ y < 2 − r₊
    y_escape = r_max - r_plus     # r ≥ r_max ⇔ y ≥ r_max − r₊
    y_inner = r_inner - r_plus    # disk bbox in y (guid 3.3)
    y_outer = r_outer - r_plus
    for py, px in ti.ndrange(height, width):
        npr_r = fwd_r
        npr_th = fwd_th
        npr_ph = fwd_ph
        if projection_mode == 1:
            # Equirectangular 360° ray-gen (guid 4.1): screen → (lon, lat) →
            # local ZAMO-frame direction. No tan_half_fov perspective math.
            lon = (px + 0.5) / width * 2.0 * math.pi          # azimuth ∈ [0, 2π)
            lat = (py + 0.5) / height * math.pi               # polar ∈ [0, π]
            npr_r = ti.sin(lat) * ti.cos(lon)
            npr_th = ti.cos(lat)
            npr_ph = ti.sin(lat) * ti.sin(lon)
        else:
            # Perspective: screen offset along the camera right/up basis.
            sx_p = (2.0 * (px + 0.5) / width - 1.0) * tan_half_x
            sy_p = (1.0 - 2.0 * (py + 0.5) / height) * tan_half_y
            npr_r = fwd_r + sx_p * rgt_r + sy_p * up_r
            npr_th = fwd_th + sx_p * rgt_th + sy_p * up_th
            npr_ph = fwd_ph + sx_p * rgt_ph + sy_p * up_ph
        invp = 1.0 / ti.sqrt(npr_r * npr_r + npr_th * npr_th + npr_ph * npr_ph)
        npr_r *= invp
        npr_th *= invp
        npr_ph *= invp

        Ep, Lp, Qp, vy_p, vu_p = _zamo_init(r_cam, theta_cam, a, k_horizon, r_plus,
                                            npr_r, npr_th, npr_ph)

        sp = vec6(y_cam, u_cam, phi_cam, 0.0, vy_p, vu_p)
        c_sp = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # Kahan compensation (guid 1.4)

        out_p = _RUNNING
        u_p_exit = u_cam
        ph_p_exit = phi_cam

        disk_col = vec3(0.0, 0.0, 0.0)
        transm = 1.0
        ray_length = 0.0          # accumulated Mino-affine path length (depth proxy)
        weighted_depth = 0.0      # Σ ray_length·contribution  (guid 3.4)
        total_emission = 0.0      # Σ contribution

        step = 0
        while step < n_steps and out_p == _RUNNING:
            # Adaptive step (guid 2.2): shrink toward the horizon (y→0). Computed
            # BEFORE the disk emit so the same h is used as the emission path
            # element ds — otherwise the variable step desyncs the Riemann sum.
            local_h = d_lambda * ti.max(adaptive_floor, sp[0] / (sp[0] + 2.0))
            # Pipe B: accumulate disk emission at the current point (front-to-back).
            # guid 3.3 bounding-box early-out: skip the disk math entirely unless the
            # sample is inside the equatorial slab (|u|<sin θ_half) and radial band.
            if disk_enabled == 1 and ti.abs(sp[1]) < bound_sin_half \
                    and sp[0] >= y_inner and sp[0] <= y_outer:
                ev = _disk_emit(sp[0], sp[1], sp[4], sp[5], Ep, Lp, a,
                                k_horizon, r_plus, r_isco, E_I, L_I,
                                r_inner, r_outer,
                                theta_half, sigma_frac, T_0, emis_c, absb_c,
                                local_h)
                disk_col += transm * vec3(ev[0], ev[1], ev[2])
                # guid 3.4: transmittance-weighted depth (contribution = T·emission).
                contribution = transm * (ev[0] + ev[1] + ev[2])
                weighted_depth += ray_length * contribution
                total_emission += contribution
                transm *= ti.exp(-ev[3])
            if _delta_y(sp[0], k_horizon) < _DELTA_MIN:
                out_p = _CAPTURED
            else:
                sp, c_sp = _rk4_step_kahan(sp, c_sp, Ep, Lp, Qp, a,
                                           k_horizon, r_plus, local_h)
                ray_length += local_h
                if _delta_y(sp[0], k_horizon) < _DELTA_MIN or sp[0] < r_capture:
                    out_p = _CAPTURED
                elif sp[0] >= y_escape:
                    out_p = _ESCAPED
                    u_p_exit = sp[1]
                    ph_p_exit = sp[2]
            step += 1

        exit_buf[py, px, 0] = u_p_exit
        exit_buf[py, px, 1] = ph_p_exit
        exit_buf[py, px, 2] = ti.cast(out_p, ti.f32)
        disk_buf[py, px, 0] = disk_col[0]
        disk_buf[py, px, 1] = disk_col[1]
        disk_buf[py, px, 2] = disk_col[2]
        disk_buf[py, px, 3] = transm
        # guid 3.4: emission-weighted mean depth, or +∞ sentinel for empty pixels.
        if total_emission > 1e-6:
            depth_pixels[py, px] = weighted_depth / total_emission
        else:
            depth_pixels[py, px] = depth_infinity


@ti.func
def _screen_jacobian_lod(py, px, height, width, th_p, ph_p):
    """Formula 10 (amendment v1.4): LOD from screen-space neighbor exit deltas.

    Differences the primary pixel's exit direction against its +x and +y
    neighbors (backward at the far edges). If any neighbor did not ESCAPE
    (``outcome != _ESCAPED``), returns ``_MAX_LOD`` (the chaotic shadow-edge
    boundary rule). Otherwise L = log2(W·J/2π) with J the larger of the two
    axis footprints.
    """
    nx = px + 1 if px + 1 < width else px - 1
    ny = py + 1 if py + 1 < height else py - 1

    out_x = exit_buf[py, nx, 2]
    out_y = exit_buf[ny, px, 2]
    lod = _MAX_LOD
    # _ESCAPED == 1.0 exactly in f32; treat anything else as a boundary.
    if (out_x < 1.5 and out_x > 0.5) and (out_y < 1.5 and out_y > 0.5):
        th_x = ti.acos(ti.min(ti.max(exit_buf[py, nx, 0], -1.0), 1.0))
        ph_x = exit_buf[py, nx, 1]
        th_y = ti.acos(ti.min(ti.max(exit_buf[ny, px, 0], -1.0), 1.0))
        ph_y = exit_buf[ny, px, 1]
        sin_th = ti.sin(th_p)

        dphx = ph_x - ph_p
        dphx = dphx - 2.0 * math.pi * ti.round(dphx / (2.0 * math.pi))
        Jx = ti.sqrt((th_x - th_p) ** 2 + sin_th * sin_th * dphx * dphx)

        dphy = ph_y - ph_p
        dphy = dphy - 2.0 * math.pi * ti.round(dphy / (2.0 * math.pi))
        Jy = ti.sqrt((th_y - th_p) ** 2 + sin_th * sin_th * dphy * dphy)

        J = ti.max(Jx, Jy)
        lod = ti.log(_STARMAP_WIDTH * J / (2.0 * math.pi)) / ti.log(2.0)
    return lod


@ti.kernel
def render_beauty_shade(width: int, height: int, lod_enabled: int):
    """Kernel 2 (Phase 2.4 split): screen-space LOD + starmap lookup + composite.

    Reads ``exit_buf`` (this pixel + neighbors) and ``disk_buf``; computes the
    Formula-10 LOD from neighbor exit deltas; samples the lensed starmap; and
    writes ``frame_pixels = disk_rgb + transmittance·background``.
    """
    for py, px in ti.ndrange(height, width):
        out_p = exit_buf[py, px, 2]
        bg = vec3(0.0, 0.0, 0.0)
        if out_p < 1.5 and out_p > 0.5:        # escaped
            u_p_exit = exit_buf[py, px, 0]
            ph_p_exit = exit_buf[py, px, 1]
            th_p_n = ti.acos(ti.min(ti.max(u_p_exit, -1.0), 1.0))
            u = ph_p_exit / (2.0 * math.pi)
            u = u - ti.floor(u)
            v = ti.min(ti.max(th_p_n / math.pi, 0.0), 1.0)

            lod = 0.0
            if lod_enabled == 1:
                lod = _screen_jacobian_lod(py, px, height, width, th_p_n, ph_p_exit)
            bg = _sample_trilinear(u, v, lod)

        disk_col = vec3(disk_buf[py, px, 0], disk_buf[py, px, 1], disk_buf[py, px, 2])
        transm = disk_buf[py, px, 3]
        col = disk_col + transm * bg
        frame_pixels[py, px, 0] = col[0]
        frame_pixels[py, px, 1] = col[1]
        frame_pixels[py, px, 2] = col[2]


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
            if _delta_y(sp[0], k_horizon) < _DELTA_MIN:
                out_p = _CAPTURED
            else:
                sp = _rk4_step(sp, Ep, Lp, Qp, a, k_horizon, r_plus, d_lambda)
                if _delta_y(sp[0], k_horizon) < _DELTA_MIN or sp[0] < r_capture:
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
        pixels[py, px, 0] = col[0]
        pixels[py, px, 1] = col[1]
        pixels[py, px, 2] = col[2]


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
            if _delta_y(sp[0], k_horizon) < _DELTA_MIN:
                out_p = _CAPTURED
            else:
                sp = _rk4_step(sp, Ep, Lp, Qp, a, k_horizon, r_plus, d_lambda)
                if _delta_y(sp[0], k_horizon) < _DELTA_MIN or sp[0] < r_capture:
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
    k_horizon, r_plus = _horizon_constants(a)

    _alloc_output(res)
    render_pipe_a(res, tan_half_fov, r_cam, theta_cam, phi_cam, a,
                  k_horizon, r_plus,
                  r_max, n_steps, d_lambda, 1 if lod_enabled else 0)
    ti.sync()
    return pixels.to_numpy()


def render_beauty_frame(cfg: dict, cam_frame: dict, width: int, height: int,
                        with_disk: bool = True, lod_enabled: bool = True,
                        return_depth: bool = False):
    """Render one beauty frame (Pipe A + Pipe B) for a camera_matrix.json entry.

    ``cam_frame`` carries the Blender camera in **world Cartesian** coordinates
    (``pos``/``fwd``/``up``/``right`` and a vertical ``fov`` in radians, per
    ``src/blender/export_camera.py``). This converts the position to Boyer-Lindquist
    (r, θ, φ) via the spherical embedding and projects the camera axes onto the
    local (r̂, θ̂, φ̂) triad, which the ZAMO tetrad (Formula 7) consumes directly.

    Returns a float32 ``(height, width, 3)`` HDR buffer.
    """
    bh = cfg["black_hole"]
    rcfg = cfg["render"]
    d = cfg["disk"]

    a = float(bh["spin"])
    r_isco = float(bh["r_isco"])
    k_horizon, r_plus = _horizon_constants(a)

    pos = np.asarray(cam_frame["pos"], dtype=float)
    fwd = np.asarray(cam_frame["fwd"], dtype=float)
    up = np.asarray(cam_frame["up"], dtype=float)
    right = np.asarray(cam_frame["right"], dtype=float)

    # World Cartesian → Boyer-Lindquist (spherical embedding; the a²-oblateness
    # is ~0.1% at r≈18 and is neglected for the camera placement only).
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r_cam = math.sqrt(x * x + y * y + z * z)
    theta_cam = math.acos(z / r_cam)
    phi_cam = math.atan2(y, x)

    st, ct = math.sin(theta_cam), math.cos(theta_cam)
    sp_, cp_ = math.sin(phi_cam), math.cos(phi_cam)
    rhat = np.array([st * cp_, st * sp_, ct])
    thhat = np.array([ct * cp_, ct * sp_, -st])
    phhat = np.array([-sp_, cp_, 0.0])

    def to_local(vec):
        return (float(vec @ rhat), float(vec @ thhat), float(vec @ phhat))

    fwd_l = to_local(fwd)
    rgt_l = to_local(right)
    up_l = to_local(up)

    # fov is the vertical field of view (radians) per the exporter; the horizontal
    # half-angle follows from the frame aspect ratio.
    fov = float(cam_frame["fov"])
    tan_half_y = math.tan(0.5 * fov)
    tan_half_x = tan_half_y * (width / height)

    n_steps = int(rcfg["max_steps_pipe_a"])
    d_lambda = float(rcfg["d_lambda_pipe_a"])
    r_max = float(rcfg["r_max"])
    adaptive_floor = float(rcfg["adaptive_step_floor"])
    depth_infinity = float(rcfg["depth_infinity"])
    projection_mode = 1 if str(rcfg.get("projection_mode", "perspective")) == "equirect" else 0
    bound_sin_half = float(d["bounding_sin_theta_half"])

    # Formula 4 — frozen ISCO conserved quantities for the plunging gas (Pipe B).
    E_I, L_I = 0.0, 0.0
    if with_disk:
        from renderer.disk import isco_conserved_quantities
        E_I, L_I = isco_conserved_quantities(r_isco, a)

    _alloc_frame(width, height)
    # Phase 2.4 kernel split: physics pass writes exit_buf/disk_buf, shading pass
    # computes the screen-space-Jacobian LOD and composites.
    render_beauty_physics(
        width, height,
        fwd_l[0], fwd_l[1], fwd_l[2],
        rgt_l[0], rgt_l[1], rgt_l[2],
        up_l[0], up_l[1], up_l[2],
        tan_half_x, tan_half_y,
        r_cam, theta_cam, phi_cam,
        a, k_horizon, r_plus, r_max, n_steps, d_lambda,
        adaptive_floor, 1 if with_disk else 0,
        projection_mode, depth_infinity,
        r_isco, E_I, L_I,
        float(d["r_inner"]), float(d["r_outer"]), float(d["theta_half_width"]),
        bound_sin_half,
        float(d["vertical_sigma_frac"]), float(d["T_0"]),
        float(d["emission_coeff"]), float(d["absorption_coeff"]),
    )
    render_beauty_shade(width, height, 1 if lod_enabled else 0)
    ti.sync()
    if return_depth:
        # Review #3: a non-finite disk sample (RK4 overshoot at the inner edge)
        # could leave NaN/±inf in the Z pass, which nothing downstream guards
        # (export_exr only nan_to_num's beauty). Map any non-finite depth to the
        # no-hit sentinel so a poisoned pixel reads as "empty", never as a finite
        # garbage Z the compositor would trust.
        depth = np.nan_to_num(depth_pixels.to_numpy(), nan=depth_infinity,
                              posinf=depth_infinity, neginf=depth_infinity)
        return frame_pixels.to_numpy(), depth
    return frame_pixels.to_numpy()


def _rotate_z(vec, dphi: float):
    """Rotate a world Cartesian vector about the BH spin axis (z) by ``dphi``."""
    c, s = math.cos(dphi), math.sin(dphi)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return [c * x - s * y, s * x + c * y, z]


def render_beauty_frame_mb(cfg: dict, cam_frame: dict, width: int, height: int,
                           shutter_arc: float, with_disk: bool = True,
                           lod_enabled: bool = True, return_depth: bool = False):
    """Temporal motion blur (guid 4.2) by host-side averaging of jittered sub-frames.

    Renders ``render.motion_blur_samples`` copies of the frame with the camera
    rotated about the spin axis across the shutter arc ``shutter_arc`` (radians of
    azimuthal travel during the shutter, = ω·shutter_fraction; the caller derives ω
    from adjacent ``camera_matrix.json`` entries) and averages the HDR results.

    Averaging at the frame level (not inside the kernel) keeps the Phase-2.4 split
    intact: each sub-frame still has a single, well-defined exit direction per pixel
    for the screen-space Jacobian. ``samples<=1`` or ``shutter_arc==0`` → a single
    render (no blur, no extra cost).
    """
    samples = int(cfg["render"]["motion_blur_samples"])
    if samples <= 1 or shutter_arc == 0.0:
        return render_beauty_frame(cfg, cam_frame, width, height,
                                   with_disk, lod_enabled, return_depth)

    depth_inf = float(cfg["render"]["depth_infinity"])
    acc = None
    depth_sum = None   # Σ of finite (disk-hit) depths only
    depth_hits = None  # per-pixel count of sub-frames that hit the disk
    for i in range(samples):
        # Symmetric jitter across the shutter window: [−arc/2, +arc/2].
        frac = (i + 0.5) / samples - 0.5
        dphi = frac * shutter_arc
        jf = dict(cam_frame)
        jf["pos"] = _rotate_z(cam_frame["pos"], dphi)
        jf["fwd"] = _rotate_z(cam_frame["fwd"], dphi)
        jf["up"] = _rotate_z(cam_frame["up"], dphi)
        jf["right"] = _rotate_z(cam_frame["right"], dphi)
        out = render_beauty_frame(cfg, jf, width, height, with_disk, lod_enabled,
                                  return_depth)
        hdr = out[0] if return_depth else out
        acc = hdr.astype(np.float64) if acc is None else acc + hdr
        if return_depth:
            # Review #1: NEVER arithmetic-mean the depth_infinity no-hit sentinel
            # with real depths — averaging 1e5 with a finite Z yields a garbage
            # "finite" value the compositor trusts. Accumulate only the sub-frames
            # that actually hit the disk; pixels never hit keep the +∞ sentinel.
            d = out[1].astype(np.float64)
            hit = d < depth_inf
            if depth_sum is None:
                depth_sum = np.where(hit, d, 0.0)
                depth_hits = hit.astype(np.float64)
            else:
                depth_sum += np.where(hit, d, 0.0)
                depth_hits += hit

    hdr_mean = (acc / samples).astype(np.float32)
    if return_depth:
        # Masked mean over hit sub-frames; sentinel where the disk was never hit.
        depth_mean = np.where(depth_hits > 0.0,
                              depth_sum / np.maximum(depth_hits, 1.0),
                              depth_inf)
        return hdr_mean, depth_mean.astype(np.float32)
    return hdr_mean


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
    k_horizon, r_plus = _horizon_constants(a)
    render_fixed_lod(res, tan_half_fov, r_cam, theta_cam, 0.0, a,
                     k_horizon, r_plus, r_max, n_steps, d_lambda, 3.0)
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
