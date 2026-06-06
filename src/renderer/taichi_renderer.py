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
# Background-fold LOD saturation: an escaped pixel whose screen-space exit
# footprint J (rad) exceeds this is straddling a lensing fold (the spin-axis
# meridian caustic) and is collapsed to the coarsest mip. From render.j_fold.
_J_FOLD = 0.15

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

# --- DNGR background (Formula 13 / PROJECT.md §8), populated only in mode=dngr --- #
# Layer A — point-star catalog binned into an equirect (θ′,φ′) cell grid, stored
# CSR-style: stars sorted by cell, with per-cell [start, count] into the sorted
# arrays. Off-plane candidate counts stay O(1–10), so the per-pixel gather is cheap.
cat_theta: ti.Field = None        # type: ignore[assignment]  (N,) sorted by cell
cat_phi: ti.Field = None          # type: ignore[assignment]  (N,)
cat_flux: ti.Field = None         # type: ignore[assignment]  (N,3) linear RGB flux
cell_start: ti.Field = None       # type: ignore[assignment]  (rows*cols,)
cell_count: ti.Field = None       # type: ignore[assignment]  (rows*cols,)
# Layer B — second equirect mip pyramid for the diffuse Milky-Way band.
mw_flat: ti.Field = None          # type: ignore[assignment]
mw_off: ti.Field = None           # type: ignore[assignment]
mw_w: ti.Field = None             # type: ignore[assignment]
mw_h: ti.Field = None             # type: ignore[assignment]
_MW_N_LEVELS = 1
_MW_MAX_LOD = 0.0
_MW_WIDTH = 16384.0
# Formula-13 scalars (baked at setup from configs/render.yaml: starfield.*).
_DNGR = 0                  # 0 = texture (legacy F10), 1 = dngr (two-layer)
_STAR_COLS = 720
_STAR_ROWS = 360
_STAR_CELL_R = 1           # cell-neighbourhood half-width searched per pixel
_STAR_PSF = 1.3            # Gaussian PSF σ (screen pixels)
_PSF_TRUNC = 3.0           # splat truncation in units of σ
_MAG_CLIP = 50.0           # cap on lensing magnification μ
_CAUSTIC_DMIN = 1.0e-3     # δ⁻ below this ⇒ on a caustic
_EWA_MAX_TAPS = 8          # max anisotropic taps for Layer B
_G_BEAMING = 0             # volumetric g⁴ star-beaming hook (0 ⇒ g≡1)

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


def _pack_pyramid(sm: Starmap):
    """Pack every mip level's RGB into one contiguous f16 buffer + per-level meta.

    Returns ``(flat_all, offsets, ws, hs)`` — the same layout the kernel samplers
    index (``base + (y*w + x)*3``). Shared by the legacy starmap and the DNGR
    diffuse (Layer B) pyramid uploads.
    """
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
    return (np.concatenate(parts),
            np.asarray(offsets, dtype=np.int32),
            np.asarray(ws, dtype=np.int32),
            np.asarray(hs, dtype=np.int32))


def _build_star_grid(catalog: np.ndarray, cols: int, rows: int):
    """Bin a point-star catalog ``[N,5]=(θ′,φ′,r,g,b)`` into an equirect cell grid.

    Returns sorted ``(theta, phi, flux[N,3])`` plus CSR ``(cell_start, cell_count)``
    over ``rows*cols`` cells (row = θ′/π·rows, col = φ′/2π·cols), matching the
    in-kernel cell math. Stars are grouped by cell so the per-pixel gather only
    scans the few cells overlapping its footprint.
    """
    theta = catalog[:, 0].astype(np.float32)
    phi = catalog[:, 1].astype(np.float32)
    flux = catalog[:, 2:5].astype(np.float32)

    col = np.clip((phi / _TWO_PI * cols).astype(np.int64), 0, cols - 1)
    row = np.clip((theta / math.pi * rows).astype(np.int64), 0, rows - 1)
    cell = row * cols + col

    order = np.argsort(cell, kind="stable")
    cell_sorted = cell[order]
    n_cells = rows * cols
    counts = np.bincount(cell_sorted, minlength=n_cells).astype(np.int32)
    starts = np.zeros(n_cells, dtype=np.int32)
    starts[1:] = np.cumsum(counts)[:-1]
    return theta[order], phi[order], flux[order], starts, counts


def setup_renderer(cfg: dict) -> Starmap:
    """Initialise Taichi (CUDA), load the starmap, and upload its mip pyramid.

    In ``starfield.mode: dngr`` it additionally uploads the Formula-13 point-star
    cell grid (Layer A) and the diffuse Milky-Way pyramid (Layer B). In
    ``texture`` mode those fields are allocated as size-1 dummies (the kernels'
    dngr branch is never taken) so the legacy path is byte-for-byte unchanged.

    Returns the host-side :class:`Starmap` (kept alive so the GPU upload's source
    arrays and the reference sampler remain available to callers/tests).
    """
    global star_flat, star_off, star_w, star_h
    global _N_LEVELS, _MAX_LOD, _STARMAP_WIDTH
    global _DELTA_MIN, _SIN2_MIN, _J_FOLD
    global cat_theta, cat_phi, cat_flux, cell_start, cell_count
    global mw_flat, mw_off, mw_w, mw_h, _MW_N_LEVELS, _MW_MAX_LOD, _MW_WIDTH
    global _DNGR, _STAR_COLS, _STAR_ROWS, _STAR_CELL_R, _STAR_PSF, _PSF_TRUNC
    global _MAG_CLIP, _CAUSTIC_DMIN, _EWA_MAX_TAPS, _G_BEAMING

    # Integration-scheme constants from config (baked into kernels at first JIT).
    _DELTA_MIN = float(cfg["render"]["horizon_epsilon"])
    _SIN2_MIN = float(cfg["render"]["sin2_min"])
    _J_FOLD = float(cfg["render"].get("j_fold", 0.15))

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

    _setup_dngr(cfg)
    return sm


def _setup_dngr(cfg: dict) -> None:
    """Load + upload the Formula-13 DNGR fields (Layer A catalog grid, Layer B
    diffuse pyramid) when ``starfield.mode == dngr``; else allocate size-1 dummies.

    All Formula-13 scalars are baked into module globals here so the kernels JIT
    against them (the same pattern as ``_J_FOLD`` / ``_MAX_LOD``).
    """
    global cat_theta, cat_phi, cat_flux, cell_start, cell_count
    global mw_flat, mw_off, mw_w, mw_h, _MW_N_LEVELS, _MW_MAX_LOD, _MW_WIDTH
    global _DNGR, _STAR_COLS, _STAR_ROWS, _STAR_CELL_R, _STAR_PSF, _PSF_TRUNC
    global _MAG_CLIP, _CAUSTIC_DMIN, _EWA_MAX_TAPS, _G_BEAMING

    sf = cfg.get("starfield", {})
    _DNGR = 1 if str(sf.get("mode", "texture")) == "dngr" else 0
    _STAR_COLS = int(sf.get("star_grid_cols", 720))
    _STAR_ROWS = int(sf.get("star_grid_rows", 360))
    _STAR_CELL_R = int(sf.get("star_cell_radius", 1))
    _STAR_PSF = float(sf.get("star_psf_px", 1.3))
    _PSF_TRUNC = float(sf.get("psf_trunc_sigma", 3.0))
    _MAG_CLIP = float(sf.get("mag_clip", 50.0))
    _CAUSTIC_DMIN = float(sf.get("caustic_delta_min", 1.0e-3))
    _EWA_MAX_TAPS = int(sf.get("ewa_max_taps", 8))
    _G_BEAMING = 1 if bool(sf.get("g_beaming", False)) else 0

    if _DNGR == 1:
        # --- Layer A: point-star catalog → CSR cell grid ---
        cat_path = _ROOT / sf["catalog_path"]
        if not cat_path.exists():
            raise FileNotFoundError(
                f"starfield.mode=dngr needs the ingested catalog {cat_path}; "
                "run `python scripts/ingest_stars.py` first.")
        catalog = np.load(cat_path)
        th_s, ph_s, fl_s, starts, counts = _build_star_grid(
            catalog, _STAR_COLS, _STAR_ROWS)
        n_stars = int(th_s.shape[0])
        n_cells = _STAR_ROWS * _STAR_COLS

        cat_theta = ti.field(dtype=ti.f32, shape=n_stars)
        cat_phi = ti.field(dtype=ti.f32, shape=n_stars)
        cat_flux = ti.field(dtype=ti.f32, shape=(n_stars, 3))
        cell_start = ti.field(dtype=ti.i32, shape=n_cells)
        cell_count = ti.field(dtype=ti.i32, shape=n_cells)
        cat_theta.from_numpy(th_s)
        cat_phi.from_numpy(ph_s)
        cat_flux.from_numpy(np.ascontiguousarray(fl_s))
        cell_start.from_numpy(starts)
        cell_count.from_numpy(counts)

        # --- Layer B: diffuse Milky-Way mip pyramid ---
        mw = Starmap.load(str(_ROOT / sf["diffuse_map"]))
        _MW_WIDTH = float(sf.get("diffuse_width", mw.levels[0].shape[1]))
        _MW_MAX_LOD = float(mw.max_lod)
        _MW_N_LEVELS = len(mw.levels)
        mflat, moff, mws, mhs = _pack_pyramid(mw)
        mw_flat = ti.field(dtype=ti.f16, shape=mflat.size)
        mw_off = ti.field(dtype=ti.i32, shape=_MW_N_LEVELS)
        mw_w = ti.field(dtype=ti.i32, shape=_MW_N_LEVELS)
        mw_h = ti.field(dtype=ti.i32, shape=_MW_N_LEVELS)
        mw_flat.from_numpy(mflat)
        mw_off.from_numpy(moff)
        mw_w.from_numpy(mws)
        mw_h.from_numpy(mhs)
    else:
        # texture mode — size-1 dummies so the kernels JIT (dngr branch unused).
        cat_theta = ti.field(dtype=ti.f32, shape=1)
        cat_phi = ti.field(dtype=ti.f32, shape=1)
        cat_flux = ti.field(dtype=ti.f32, shape=(1, 3))
        cell_start = ti.field(dtype=ti.i32, shape=1)
        cell_count = ti.field(dtype=ti.i32, shape=1)
        mw_flat = ti.field(dtype=ti.f16, shape=1)
        mw_off = ti.field(dtype=ti.i32, shape=1)
        mw_w = ti.field(dtype=ti.i32, shape=1)
        mw_h = ti.field(dtype=ti.i32, shape=1)
        _MW_N_LEVELS = 1
        _MW_MAX_LOD = 0.0
        _MW_WIDTH = 16384.0


def _horizon_constants(a: float) -> tuple[float, float]:
    """Precompute FP32-stable horizon constants (optimization Phase 1.1 / Formula 11).

    Returns ``(k_horizon, r_plus)`` with ``k_horizon = √(1−a²)`` and the *true*
    outer horizon ``r₊ = 1 + k_horizon``. Derived from ``a`` here (like E_I/L_I and
    tan_half_fov) rather than read from ``configs.black_hole.r_plus`` so it can never
    drift out of sync with ``black_hole.spin``: r₊ is a *function* of a, so the
    config key (1.0447 for a=0.999) is a derived-value duplication that silently
    desyncs if a is edited. The config key is still consumed verbatim by the CPU
    preview path (``scripts/thumb.py`` / ``seam_diag.py``) as the
    ``radial_turning_point`` r_floor (= the horizon, which is correct at a=0.999);
    those readers should be migrated to derive r₊ too. FLAGGED — CPU path, out of
    scope for the GPU audit.
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
    # Kahan compensated summation to the state accumulation (optimization Phase 1.4).
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
    """Compensated (Kahan) RK4 step (optimization Phase 1.4). Returns ``(s_next, c_next)``.

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
def _sample_trilinear(u, v, lod):
    L = ti.min(ti.max(lod, 0.0), _MAX_LOD)
    l0 = ti.cast(ti.floor(L), ti.i32)
    l1 = ti.min(l0 + 1, _N_LEVELS - 1)
    l0 = ti.min(l0, _N_LEVELS - 1)
    f = L - ti.floor(L)
    # Smoothstep the inter-level blend weight so the reconstruction is C1 (not
    # just C0) across integer LOD boundaries. Linear f has slope 1 on both sides
    # of an integer, where the active mip pair switches, so the blend rate jumps
    # — a Mach-band kink that the smooth radial LOD field paints as concentric
    # rings. smoothstep's derivative 6f(1-f) vanishes at f=0 and f=1, matching
    # the slope across each boundary. Endpoints are unchanged (0->0, 1->1).
    f = f * f * (3.0 - 2.0 * f)
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
        c_sp = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # Kahan compensation (optimization Phase 1.4)
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
                    # Per-step φ-wrap into (−π, π] — identical to render_beauty_physics.
                    # Without it a near-pole passage (dφ/dλ = Lz/sin²θ) inflates the
                    # raw accumulated φ past f32's fractional precision, so the exit
                    # azimuth collapses to noise across the spin-axis meridian. φ is
                    # cyclic and never on the RHS of `_deriv` (Kerr is axisymmetric),
                    # so folding it each step is an exact coordinate identity.
                    phi_wp = sp[2] - _TWO_PI * ti.round(sp[2] / _TWO_PI)
                    sp = vec6(sp[0], sp[1], phi_wp, sp[3], sp[4], sp[5])
                    c_sp = vec6(c_sp[0], c_sp[1], 0.0, c_sp[3], c_sp[4], c_sp[5])
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
                    # Per-step φ-wrap into (−π, π] (see the primary ray above).
                    phi_wo = so[2] - _TWO_PI * ti.round(so[2] / _TWO_PI)
                    so = vec6(so[0], so[1], phi_wo, so[3], so[4], so[5])
                    c_so = vec6(c_so[0], c_so[1], 0.0, c_so[3], c_so[4], c_so[5])
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
                    # Fold saturation — identical to _screen_jacobian_lod: a footprint
                    # spanning the spin-axis meridian caustic (J > _J_FOLD) collapses
                    # to the uniform coarsest mip instead of an aliased mid-mip fetch.
                    if J > _J_FOLD:
                        lod = _MAX_LOD
                    else:
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
    the transmittance-weighted Mino-affine Z to ``depth_pixels`` (optimization Phase 3.4). The
    Formula-10 LOD + background lookup are deferred to ``render_beauty_shade``,
    which differences neighbor exit directions in screen space (SKILL.md F10
    amendment v1.4). Eliminating the offset ray halves the geodesic workload.

    Adaptive Mino step (optimization Phase 2.2): ``h = d_lambda·max(adaptive_floor, y/(y+2))`` —
    full steps far out, shrinking toward the horizon (y→0).

    ``projection_mode`` (optimization Phase 4.1): 0 = perspective (camera basis + FOV), 1 =
    equirectangular 360° (px→lon, py→lat in the local ZAMO frame, for VR output).

    State is [y, u, φ, t, v_y, v_u] (Formula 11/12): y = r − r₊, u = cosθ.
    """
    y_cam = r_cam - r_plus
    u_cam = ti.cos(theta_cam)
    r_capture = 2.0 - r_plus      # r < 2 ⇔ y < 2 − r₊
    y_escape = r_max - r_plus     # r ≥ r_max ⇔ y ≥ r_max − r₊
    y_inner = r_inner - r_plus    # disk bbox in y (optimization Phase 3.3)
    y_outer = r_outer - r_plus
    for py, px in ti.ndrange(height, width):
        npr_r = fwd_r
        npr_th = fwd_th
        npr_ph = fwd_ph
        if projection_mode == 1:
            # Equirectangular 360° ray-gen (optimization Phase 4.1): screen → (lon, lat) →
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
        c_sp = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # Kahan compensation (optimization Phase 1.4)

        out_p = _RUNNING
        u_p_exit = u_cam
        ph_p_exit = phi_cam

        disk_col = vec3(0.0, 0.0, 0.0)
        transm = 1.0
        ray_length = 0.0          # accumulated Mino-affine path length (depth proxy)
        weighted_depth = 0.0      # Σ ray_length·contribution  (optimization Phase 3.4)
        total_emission = 0.0      # Σ contribution

        step = 0
        while step < n_steps and out_p == _RUNNING:
            # Adaptive step (optimization Phase 2.2): shrink toward the horizon (y→0). Computed
            # BEFORE the disk emit so the same h is used as the emission path
            # element ds — otherwise the variable step desyncs the Riemann sum.
            local_h = d_lambda * ti.max(adaptive_floor, sp[0] / (sp[0] + 2.0))
            # Pipe B: accumulate disk emission at the current point (front-to-back).
            # optimization Phase 3.3 bounding-box early-out: skip the disk math entirely unless the
            # sample is inside the equatorial slab (|u|<sin θ_half) and radial band.
            if disk_enabled == 1 and ti.abs(sp[1]) < bound_sin_half \
                    and sp[0] >= y_inner and sp[0] <= y_outer:
                ev = _disk_emit(sp[0], sp[1], sp[4], sp[5], Ep, Lp, a,
                                k_horizon, r_plus, r_isco, E_I, L_I,
                                r_inner, r_outer,
                                theta_half, sigma_frac, T_0, emis_c, absb_c,
                                local_h)
                disk_col += transm * vec3(ev[0], ev[1], ev[2])
                # optimization Phase 3.4: transmittance-weighted depth (contribution = T·emission).
                contribution = transm * (ev[0] + ev[1] + ev[2])
                weighted_depth += ray_length * contribution
                total_emission += contribution
                transm *= ti.exp(-ev[3])
            if _delta_y(sp[0], k_horizon) < _DELTA_MIN:
                out_p = _CAPTURED
            else:
                y_prev = sp[0]          # pre-step state, for escape event location
                u_prev = sp[1]
                ph_prev = sp[2]
                sp, c_sp = _rk4_step_kahan(sp, c_sp, Ep, Lp, Qp, a,
                                           k_horizon, r_plus, local_h)
                ray_length += local_h
                # Axisymmetry wrap (restores the original bounded-φ behaviour).
                # φ is cyclic — it never appears on the RHS of `_deriv` (Kerr is
                # axisymmetric) — so folding it into (−π, π] every step is an
                # exact coordinate identity, not a physics change. A near-pole
                # passage drives dφ/dλ = Lz/sin²θ toward Lz/_SIN2_MIN (~1e10),
                # inflating the raw accumulated φ to ~1e6 rad; at that magnitude
                # f32 retains no fractional precision, so the exit azimuth — the
                # only physically meaningful part — collapses to noise and flips
                # sign across the symmetric center column (the "static" seam).
                # Keeping φ in (−π, π] preserves full f32 precision throughout.
                phi_w = sp[2] - _TWO_PI * ti.round(sp[2] / _TWO_PI)
                sp = vec6(sp[0], sp[1], phi_w, sp[3], sp[4], sp[5])
                c_sp = vec6(c_sp[0], c_sp[1], 0.0, c_sp[3], c_sp[4], c_sp[5])
                if _delta_y(sp[0], k_horizon) < _DELTA_MIN or sp[0] < r_capture:
                    out_p = _CAPTURED
                elif sp[0] >= y_escape:
                    out_p = _ESCAPED
                    # Linear event location: interpolate the exit angles back to
                    # the escape surface y = y_escape (r = r_max). Far out, a
                    # single Mino step advances r by ~r²·h (tens of units near
                    # r_max), so neighbor pixels overshoot the sphere by
                    # different amounts. Recording the raw overshot angle injects
                    # that step-quantization jitter into the screen-space
                    # Jacobian (Formula 10), over-coarsening the background mip.
                    # Interpolating to the exact surface removes it. (Numerical
                    # event location of the integrated geodesic — no GR formula
                    # is re-derived; SKILL.md formulas are untouched.)
                    denom = sp[0] - y_prev
                    frac = 1.0
                    if denom > 1e-12:
                        frac = (y_escape - y_prev) / denom
                    frac = ti.min(ti.max(frac, 0.0), 1.0)
                    u_p_exit = u_prev + frac * (sp[1] - u_prev)
                    # Shortest-arc φ delta: ph_prev and the per-step-wrapped sp[2]
                    # can land on opposite sides of the ±π fold, so interpolate
                    # along the wrapped increment (the geodesic turns < π over one
                    # far-field step at r≈r_max, so the short arc is the true one).
                    dph = sp[2] - ph_prev
                    dph = dph - _TWO_PI * ti.round(dph / _TWO_PI)
                    ph_p_exit = ph_prev + frac * dph
            step += 1

        exit_buf[py, px, 0] = u_p_exit
        exit_buf[py, px, 1] = ph_p_exit
        exit_buf[py, px, 2] = ti.cast(out_p, ti.f32)
        disk_buf[py, px, 0] = disk_col[0]
        disk_buf[py, px, 1] = disk_col[1]
        disk_buf[py, px, 2] = disk_col[2]
        disk_buf[py, px, 3] = transm
        # optimization Phase 3.4: emission-weighted mean depth, or +∞ sentinel for empty pixels.
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
        # Fold saturation (restores the original offset-ray behaviour). Near the
        # spin-axis meridian the lensing folds: neighbour pixels exit on opposite
        # sides of the pole, ~π apart in azimuth, so a single scalar-LOD trilinear
        # fetch lands on unrelated coarse-mip texels → the "static" seam. The old
        # offset-ray Jacobian forced _MAX_LOD whenever the companion ray dived into
        # that chaotic region; with screen-space differencing the equivalent signal
        # is a footprint J spanning a large fraction of the sky. Above _J_FOLD the
        # pixel genuinely integrates that whole fan, so collapse it to the uniform
        # coarsest mip (smooth grey) rather than an aliased mid-mip fetch.
        if J > _J_FOLD:
            lod = _MAX_LOD
        else:
            lod = ti.log(_STARMAP_WIDTH * J / (2.0 * math.pi)) / ti.log(2.0)
    return lod


# --------------------------------------------------------------------------- #
# DNGR background (Formula 13) — Layer B diffuse sampler + Layer A star gather
# --------------------------------------------------------------------------- #
@ti.func
def _wrap_pi(x):
    """Fold an angular difference into (−π, π] (φ is periodic)."""
    return x - _TWO_PI * ti.round(x / _TWO_PI)


@ti.func
def _mw_texel(level, x, y):
    base = mw_off[level]
    w = mw_w[level]
    idx = base + (y * w + x) * 3
    return vec3(
        ti.cast(mw_flat[idx + 0], ti.f32),
        ti.cast(mw_flat[idx + 1], ti.f32),
        ti.cast(mw_flat[idx + 2], ti.f32),
    )


@ti.func
def _mw_sample_level(level, u, v):
    w = mw_w[level]
    h = mw_h[level]
    uu = u - ti.floor(u)
    vv = ti.min(ti.max(v, 0.0), 1.0)
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
    c00 = _mw_texel(level, ix0, iy0)
    c10 = _mw_texel(level, ix1, iy0)
    c01 = _mw_texel(level, ix0, iy1)
    c11 = _mw_texel(level, ix1, iy1)
    top = c00 * (1.0 - du) + c10 * du
    bot = c01 * (1.0 - du) + c11 * du
    return top * (1.0 - dv) + bot * dv


@ti.func
def _mw_sample_trilinear(u, v, lod):
    L = ti.min(ti.max(lod, 0.0), _MW_MAX_LOD)
    l0 = ti.cast(ti.floor(L), ti.i32)
    l1 = ti.min(l0 + 1, _MW_N_LEVELS - 1)
    l0 = ti.min(l0, _MW_N_LEVELS - 1)
    f = L - ti.floor(L)
    f = f * f * (3.0 - 2.0 * f)
    c0 = _mw_sample_level(l0, u, v)
    c1 = _mw_sample_level(l1, u, v)
    return c0 * (1.0 - f) + c1 * f


@ti.func
def _dngr_shade(py, px, height, width, th_p, ph_p, d_omega):
    """Formula 13 two-layer background for one escaped pixel.

    Builds the screen-space 2×2 beam Jacobian J = ∂(θ′,φ′)/∂(pixel) from the +x/+y
    neighbour exit directions (same stencil as ``_screen_jacobian_lod``), then:
      * Layer B — anisotropic (EWA) fetch of the diffuse Milky-Way map along the
        footprint ellipse's major axis;
      * Layer A — gathers catalog stars in the overlapping cells and adds
        ``flux · μ · g⁴ · exp(−d²/2σ²)`` with the lensing magnification
        ``μ = dΩ_pixel / |det J · sinθ′|`` (normalized so μ→1 in flat space).
    Boundary rules (Formula 13 guard b, approved 2026-06-05): a non-ESCAPED
    neighbour or a fold footprint ``δ⁺ > j_fold`` ⇒ μ=1 and the diffuse layer
    falls back to the coarsest mip (matches the legacy seam handling). On that
    same invalid-``det J`` branch the Layer-A splat is *placed* by guard (b′)
    (approved 2026-06-06): the undeflected proper-separation footprint
    ``d² = (Δθ′² + sin²θ′·Δφ′²)/dΩ`` instead of the degenerate ``J⁻¹`` — this
    removes the spin-axis meridian star-pileup (Artifact B).
    Returns the combined background radiance (diffuse + stars).
    """
    nx = px + 1 if px + 1 < width else px - 1
    ny = py + 1 if py + 1 < height else py - 1
    sgn_x = 1.0 if px + 1 < width else -1.0
    sgn_y = 1.0 if py + 1 < height else -1.0
    out_x = exit_buf[py, nx, 2]
    out_y = exit_buf[ny, px, 2]
    valid = (out_x < 1.5 and out_x > 0.5) and (out_y < 1.5 and out_y > 0.5)

    sin_th = ti.sin(th_p)
    dthx = 0.0
    dphx = 0.0
    dthy = 0.0
    dphy = 0.0
    detJ = 0.0
    delta_plus = 0.0     # δ⁺ major ellipse axis (angular, sinθ-weighted)
    delta_minus = 0.0    # δ⁻ minor ellipse axis
    if valid:
        th_x = ti.acos(ti.min(ti.max(exit_buf[py, nx, 0], -1.0), 1.0))
        ph_x = exit_buf[py, nx, 1]
        th_y = ti.acos(ti.min(ti.max(exit_buf[ny, px, 0], -1.0), 1.0))
        ph_y = exit_buf[ny, px, 1]
        dthx = (th_x - th_p) * sgn_x
        dphx = _wrap_pi(ph_x - ph_p) * sgn_x
        dthy = (th_y - th_p) * sgn_y
        dphy = _wrap_pi(ph_y - ph_p) * sgn_y
        detJ = dthx * dphy - dthy * dphx
        # Singular values of the sinθ-weighted angular Jacobian → ellipse axes.
        cx0, cx1 = dthx, sin_th * dphx
        cy0, cy1 = dthy, sin_th * dphy
        a11 = cx0 * cx0 + cx1 * cx1
        a22 = cy0 * cy0 + cy1 * cy1
        a12 = cx0 * cy0 + cx1 * cy1
        tr = a11 + a22
        det2 = a11 * a22 - a12 * a12
        disc = ti.sqrt(ti.max(tr * tr - 4.0 * det2, 0.0))
        delta_plus = ti.sqrt(ti.max(0.5 * (tr + disc), 0.0))
        delta_minus = ti.sqrt(ti.max(0.5 * (tr - disc), 0.0))

    usable = valid and (delta_plus <= _J_FOLD)

    # --- magnification μ (Formula 13 §2; normalized by the flat-space footprint) ---
    mu = 1.0
    src = ti.abs(detJ) * sin_th
    if usable and src > 1e-20:
        mu = ti.min(d_omega / src, _MAG_CLIP)
    # caustic guard (Formula 13 §2 guard b): δ⁻→0 ⇒ critical curve ⇒ keep μ capped.
    if usable and delta_minus < _CAUSTIC_DMIN:
        mu = ti.min(mu, _MAG_CLIP)

    u = ph_p / _TWO_PI
    u = u - ti.floor(u)
    v = ti.min(ti.max(th_p / math.pi, 0.0), 1.0)

    # --- Layer B: anisotropic EWA fetch of the diffuse map ---
    diffuse = vec3(0.0, 0.0, 0.0)
    if usable:
        Wd = _MW_WIDTH
        Hd = 0.5 * _MW_WIDTH
        pxu = dphx / _TWO_PI * Wd
        pxv = dthx / math.pi * Hd
        pyu = dphy / _TWO_PI * Wd
        pyv = dthy / math.pi * Hd
        lenx = ti.sqrt(pxu * pxu + pxv * pxv)
        leny = ti.sqrt(pyu * pyu + pyv * pyv)
        maj_u, maj_v, maj, minr = pxu, pxv, lenx, leny
        if leny > lenx:
            maj_u, maj_v, maj, minr = pyu, pyv, leny, lenx
        ntaps = ti.cast(ti.min(ti.ceil(maj / ti.max(minr, 1.0)),
                               ti.cast(_EWA_MAX_TAPS, ti.f32)), ti.i32)
        if ntaps < 1:
            ntaps = 1
        lod = ti.log(ti.max(minr, 1.0)) / ti.log(2.0)
        acc = vec3(0.0, 0.0, 0.0)
        for t in range(ntaps):
            f = (ti.cast(t, ti.f32) + 0.5) / ti.cast(ntaps, ti.f32) - 0.5
            acc += _mw_sample_trilinear(u + f * maj_u / Wd, v + f * maj_v / Hd, lod)
        diffuse = acc / ti.cast(ntaps, ti.f32)
    else:
        diffuse = _mw_sample_trilinear(u, v, _MW_MAX_LOD)

    # --- Layer A: point-star energy gather (flux · μ · g⁴ · Gaussian PSF) ---
    # Placement of each catalog star's splat (its screen-space offset from the
    # pixel centre) follows Formula 13 guards (b) + (b′):
    #   * `usable` (det J a valid beam Jacobian): lensed placement
    #       (dpx, dpy) = J⁻¹ · (Δθ′, Δφ′)        — stars follow the lensing.
    #   * NOT `usable` (det J invalid: non-ESCAPED neighbour, or fold δ⁺>j_fold —
    #     the spin-axis seam): guard (b′). J⁻¹ is degenerate there (Δφ′≈±π ⇒
    #     |det J| large ⇒ J⁻¹→0), which would collapse every polar-cell star to
    #     d≈0 and pile them onto the meridian (the seam pileup). Instead place
    #     the splat by the star's TRUE proper angular separation under the
    #     undeflected footprint dΩ (the guard-(a) quantity, `d_omega`):
    #         d² = (Δθ′² + sin²θ′·Δφ′²) / dΩ   [screen-pixel²]
    #     i.e. great-circle separation / undeflected angular pixel size √dΩ, so
    #     polar stars keep their real spacing. μ is already clamped to 1 here, so
    #     seam stars stay sharp point-like at base flux. The gather now runs for
    #     every escaped pixel (the old `valid`-only gate left a star-free band
    #     around the shadow silhouette and on the seam); placement degenerates to
    #     the no-lens geometry exactly where the lensed Jacobian is unusable.
    stars = vec3(0.0, 0.0, 0.0)
    g4 = 1.0   # _G_BEAMING hook: g≡1 for a static camera at the celestial sphere
    two_sig2 = 2.0 * _STAR_PSF * _STAR_PSF
    d_max2 = (_PSF_TRUNC * _STAR_PSF) ** 2
    inv_det = 0.0
    if usable and ti.abs(detJ) > 1e-20:
        inv_det = 1.0 / detJ
    # Only gather when a placement metric is available: lensed J⁻¹ (usable) or the
    # undeflected footprint dΩ (guard b′). dΩ≈0 (degenerate pixel) ⇒ skip.
    if (usable and ti.abs(detJ) > 1e-20) or (not usable and d_omega > 1e-20):
        ci = ti.cast(u * _STAR_COLS, ti.i32)
        cj = ti.cast(v * _STAR_ROWS, ti.i32)
        for dj in range(-_STAR_CELL_R, _STAR_CELL_R + 1):
            jj = cj + dj
            if jj >= 0 and jj < _STAR_ROWS:
                for di in range(-_STAR_CELL_R, _STAR_CELL_R + 1):
                    ii = (ci + di) % _STAR_COLS
                    if ii < 0:
                        ii += _STAR_COLS
                    cell = jj * _STAR_COLS + ii
                    start = cell_start[cell]
                    cnt = cell_count[cell]
                    for sidx in range(start, start + cnt):
                        dth = cat_theta[sidx] - th_p
                        dph = _wrap_pi(cat_phi[sidx] - ph_p)
                        d2 = 0.0
                        if usable:
                            # lensed placement: screen offset = J⁻¹ · (Δθ′, Δφ′)
                            dpx = (dphy * dth - dthy * dph) * inv_det
                            dpy = (-dphx * dth + dthx * dph) * inv_det
                            d2 = dpx * dpx + dpy * dpy
                        else:
                            # guard (b′): undeflected proper-separation placement
                            d2 = (dth * dth + sin_th * sin_th * dph * dph) / d_omega
                        if d2 < d_max2:
                            wgt = ti.exp(-d2 / two_sig2) * mu * g4
                            stars += vec3(cat_flux[sidx, 0], cat_flux[sidx, 1],
                                          cat_flux[sidx, 2]) * wgt
    return diffuse + stars


@ti.kernel
def render_beauty_shade(width: int, height: int, lod_enabled: int, mode: int,
                        tan_half_x: float, tan_half_y: float, projection_mode: int):
    """Kernel 2 (Phase 2.4 split): background lookup + composite.

    ``mode == 0`` (texture): the legacy Formula-10 path — screen-space LOD from
    neighbour exit deltas, single trilinear lensed-starmap fetch (byte-for-byte
    unchanged). ``mode == 1`` (dngr): the Formula-13 two-layer background
    (``_dngr_shade``: anisotropic diffuse + magnified point-star gather). Either
    way it writes ``frame_pixels = disk_rgb + transmittance·background``.
    """
    for py, px in ti.ndrange(height, width):
        out_p = exit_buf[py, px, 2]
        bg = vec3(0.0, 0.0, 0.0)
        if out_p < 1.5 and out_p > 0.5:        # escaped
            u_p_exit = exit_buf[py, px, 0]
            ph_p_exit = exit_buf[py, px, 1]
            th_p_n = ti.acos(ti.min(ti.max(u_p_exit, -1.0), 1.0))
            if mode == 1:
                # Undeflected per-pixel solid angle dΩ_pixel (μ normalization, F13 §2a).
                d_omega = 0.0
                if projection_mode == 1:
                    lat_cam = (py + 0.5) / height * math.pi
                    d_omega = ti.sin(lat_cam) * (_TWO_PI / width) * (math.pi / height)
                else:
                    sx = (2.0 * (px + 0.5) / width - 1.0) * tan_half_x
                    sy = (1.0 - 2.0 * (py + 0.5) / height) * tan_half_y
                    inv_len = 1.0 / ti.sqrt(1.0 + sx * sx + sy * sy)
                    d_omega = (4.0 * tan_half_x * tan_half_y / (width * height)) \
                        * inv_len * inv_len * inv_len
                bg = _dngr_shade(py, px, height, width, th_p_n, ph_p_exit, d_omega)
            else:
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
    # Clamp the cosine to [-1, 1]: fp rounding can push z/r_cam slightly out of
    # domain at the poles, which would make acos return NaN. Matches the clamping
    # already applied to every in-kernel acos.
    theta_cam = math.acos(min(1.0, max(-1.0, z / r_cam)))
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
    # Disk-slab |u|=|cosθ| early-out bound, DERIVED from its source parameter
    # (single source of truth): |θ−π/2| < θ_half  ⇔  |cosθ| < sin(θ_half), exactly
    # the slab test re-checked inside `_disk_emit`. The old config literal
    # `bounding_sin_theta_half` duplicated sin(θ_half) and silently desynced the
    # bounding box whenever `disk.theta_half_width` was edited (the config-sync bug).
    bound_sin_half = math.sin(float(d["theta_half_width"]))

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
    # starfield.mode: dngr ⇒ Formula-13 two-layer background; else legacy F10 texture.
    dngr_mode = 1 if str(cfg.get("starfield", {}).get("mode", "texture")) == "dngr" else 0
    render_beauty_shade(width, height, 1 if lod_enabled else 0, dngr_mode,
                        tan_half_x, tan_half_y, projection_mode)
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
    """Temporal motion blur (optimization Phase 4.2) by host-side averaging of jittered sub-frames.

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
