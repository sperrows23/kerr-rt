"""Production Taichi GPU renderer — Phase 2 of the Kerr pipeline.

This module implements **Pipe A** (4K-capable beauty pass: starmap +
gravitational lensing) and **Pipe B** (volumetric accretion-disk march) on the
GPU, plus the Formula-13 DNGR two-layer background.

Physics is ported **verbatim** from ``skills/kerr-physics/SKILL.md`` PART II —
**Cartesian Kerr-Schild (CKS)** — mirroring the verified NumPy reference modules
``renderer.metric`` / ``renderer.geodesic`` (project CRITICAL RULE: no formula is
re-derived here). The ported pieces are:

  * Formula CKS-1/2 — implicit Kerr radius r(x,y,z); metric g = η + f l⊗l
  * Formula CKS-3   — *exact* inverse g⁻¹ = η − f l⊗l   (l is η-null)
  * Formula CKS-4   — analytic ∂r, ∂f, ∂l_i
  * Formula CKS-5   — Hamiltonian null geodesic EOM, RK4 on [t,x,y,z,p_t,p_x,p_y,p_z]
  * Formula CKS-6   — horizon capture (r ≤ r₊+ε) / escape (ρ ≥ r_max)
  * Formula CKS-7   — ZAMO-from-g⁻¹ + projected-ray photon initialization
  * Formula CKS-8   — equatorial disk gas 4-velocity (rigid rotation about +z)
  * Formula CKS-9   — g-factor (Cartesian dot product; the BL Δ-divide bug is gone)
  * Formula CKS-10  — escaped-ray celestial direction from asymptotic momentum
  * Formula 10/13   — differential-ray mip LOD + DNGR μ/PSF (coordinate-agnostic;
                      they act on the celestial direction (θ′, φ′) and are unchanged)

**Why CKS:** Boyer-Lindquist had *coordinate* singularities on the spin axis
(1/sin²θ) and at the horizon (Δ→0). The whole BL band-aid lineage — u=cosθ Θ_u,
per-step φ-wrap, ``_project`` potential re-imposition, ``normalize_sphere_angles``
punch-through, the spin-axis ``j_fold`` meridian collapse — is **deleted**: CKS is
regular on the axis and across the horizon, so the artifact class is removed at the
source. The escaped-ray direction (CKS-10) is a genuine Cartesian unit vector for
every ray, so ``exit_buf`` still stores ``(cosθ′, φ′, outcome)`` and every
downstream screen-space LOD / DNGR routine is byte-for-byte unchanged.

GPU backend is LOCKED to ``ti.init(arch=ti.cuda)`` (never ``ti.gpu``) per CLAUDE.md.

All numerical parameters come from ``configs/render.yaml`` (no hardcoded values).
The only literals here are integration-scheme safety constants (the horizon-capture
margin, mirrored from ``render.horizon_epsilon``; tiny denominator clamps) — these
are not physics.
"""

import math
from pathlib import Path

import numpy as np
import taichi as ti
import yaml

from renderer import disk_flux
from renderer.starmap import Starmap

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "configs" / "render.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    # Explicit utf-8: this box defaults to cp949 and the config has θ/π/· bytes.
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Horizon-capture margin (mirrored from render.horizon_epsilon — NOT physics).
# CKS is regular at the horizon, so capture is detected right at r₊; this ε only
# caps step count in the deep field (Formula CKS-6). Baked into kernels at JIT.
_HORIZON_EPS = 0.05
# Background-fold LOD saturation: an escaped pixel whose screen-space exit
# footprint J (rad) exceeds this straddles an equirect-texture pole or the
# chaotic shadow edge and is collapsed to the coarsest mip. From render.j_fold.
# (Under CKS the BL spin-axis seam is gone; this now only guards the texture
# poles / shadow silhouette.)
_J_FOLD = 0.15

# State vector layout (Formula CKS-5): the 8-vector [t, x, y, z, p_t, p_x, p_y, p_z]
# in Cartesian Kerr-Schild coordinates (covariant momenta). p_t is conserved (= −E).
vec8 = ti.types.vector(8, ti.f32)
vec4 = ti.types.vector(4, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec2 = ti.types.vector(2, ti.f32)

_TWO_PI = 2.0 * math.pi

# Ray outcome codes.
_RUNNING = 0
_ESCAPED = 1
_CAPTURED = 2

# Floor on the coordinate speed |dx/dλ| in the disk step cap, so a degenerate
# near-null-spatial step divides by a finite number instead of blowing up.
_DISK_STEP_V_EPS = 1e-6


# --------------------------------------------------------------------------- #
# GPU fields (populated by setup_renderer)
# --------------------------------------------------------------------------- #
# Starmap mip pyramid, packed into one flat f16 buffer with per-level metadata.
star_flat: ti.Field = None  # type: ignore[assignment]
star_off: ti.Field = None  # type: ignore[assignment]
star_w: ti.Field = None  # type: ignore[assignment]
star_h: ti.Field = None  # type: ignore[assignment]
_N_LEVELS = 0
_MAX_LOD = 0.0
_STARMAP_WIDTH = 0

# --- DNGR background (Formula 13 / PROJECT.md §8), populated only in mode=dngr --- #
# Layer A — point-star catalog binned into an equirect (θ′,φ′) cell grid, stored
# CSR-style: stars sorted by cell, with per-cell [start, count] into the sorted
# arrays. Off-plane candidate counts stay O(1–10), so the per-pixel gather is cheap.
cat_theta: ti.Field = None  # type: ignore[assignment]  (N,) sorted by cell
cat_phi: ti.Field = None  # type: ignore[assignment]  (N,)
cat_flux: ti.Field = None  # type: ignore[assignment]  (N,3) linear RGB flux
cell_start: ti.Field = None  # type: ignore[assignment]  (rows*cols,)
cell_count: ti.Field = None  # type: ignore[assignment]  (rows*cols,)
# Layer B — second equirect mip pyramid for the diffuse Milky-Way band.
mw_flat: ti.Field = None  # type: ignore[assignment]
mw_off: ti.Field = None  # type: ignore[assignment]
mw_w: ti.Field = None  # type: ignore[assignment]
mw_h: ti.Field = None  # type: ignore[assignment]
_MW_N_LEVELS = 1
_MW_MAX_LOD = 0.0
_MW_WIDTH = 16384.0
_MW_GAIN = 1.0  # Layer-B diffuse brightness multiplier (starfield.diffuse_gain)
# Formula-13 scalars (baked at setup from configs/render.yaml: starfield.*).
_DNGR = 0  # 0 = texture (legacy F10), 1 = dngr (two-layer)
_STAR_COLS = 720
_STAR_ROWS = 360
_STAR_CELL_R = 1  # cell-neighbourhood half-width searched per pixel
_STAR_PSF = 1.3  # Gaussian PSF σ (screen pixels)
_PSF_TRUNC = 3.0  # splat truncation in units of σ
_MAG_CLIP = 50.0  # cap on lensing magnification μ
_CAUSTIC_DMIN = 1.0e-3  # δ⁻ below this ⇒ on a caustic
_EWA_MAX_TAPS = 8  # max anisotropic taps for Layer B
_G_BEAMING = 0  # volumetric g⁴ star-beaming hook (0 ⇒ g≡1)

# Output image (set per render call to match resolution).
pixels: ti.Field = None  # type: ignore[assignment]
_RES = 0

# Beauty-pass output (non-square, set per frame render).
frame_pixels: ti.Field = None  # type: ignore[assignment]
# Kernel-split hand-off buffers (Phase 2.4): physics kernel writes, shading reads.
exit_buf: ti.Field = None  # type: ignore[assignment]  (H,W,3): cosθ′, φ′, outcome
disk_buf: ti.Field = None  # type: ignore[assignment]  (H,W,4): disk_rgb + transmittance
depth_pixels: ti.Field = None  # type: ignore[assignment]  (H,W): transmittance-weighted Z (3.4)
_FW = 0
_FH = 0

# --- Page-Thorne disk-flux LUT (D1; SKILL.md CKS-11) ------------------------- #
# 1-D dimensionless shape f_PT(r) = F(r)/F_max over [r_isco, r_outer], precomputed
# on the CPU by renderer.disk_flux and uploaded here. The kernel reads it with
# linear interpolation when disk.temperature_model == page_thorne; the simple
# T₀·(6/r)^0.75 path ignores it. ALWAYS built/uploaded in setup (256 floats is
# negligible) so the flag toggles per-render without a re-JIT. The _PT_LUT_*
# scalars are baked into the kernel at JIT (same pattern as _MAX_LOD).
disk_flux_lut: ti.Field = None  # type: ignore[assignment]  (n,) normalized f_PT
_PT_LUT_N = 1
_PT_LUT_R0 = 0.0
_PT_LUT_INV_DR = 0.0


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
    return (
        np.concatenate(parts),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(ws, dtype=np.int32),
        np.asarray(hs, dtype=np.int32),
    )


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
    global _HORIZON_EPS, _J_FOLD
    global cat_theta, cat_phi, cat_flux, cell_start, cell_count
    global mw_flat, mw_off, mw_w, mw_h, _MW_N_LEVELS, _MW_MAX_LOD, _MW_WIDTH
    global _DNGR, _STAR_COLS, _STAR_ROWS, _STAR_CELL_R, _STAR_PSF, _PSF_TRUNC
    global _MAG_CLIP, _CAUSTIC_DMIN, _EWA_MAX_TAPS, _G_BEAMING

    # Integration-scheme constants from config (baked into kernels at first JIT).
    _HORIZON_EPS = float(cfg["render"]["horizon_epsilon"])  # CKS-6 capture margin
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
    _setup_disk_flux(cfg)
    return sm


def _setup_dngr(cfg: dict) -> None:
    """Load + upload the Formula-13 DNGR fields (Layer A catalog grid, Layer B
    diffuse pyramid) when ``starfield.mode == dngr``; else allocate size-1 dummies.

    All Formula-13 scalars are baked into module globals here so the kernels JIT
    against them (the same pattern as ``_J_FOLD`` / ``_MAX_LOD``).
    """
    global cat_theta, cat_phi, cat_flux, cell_start, cell_count
    global mw_flat, mw_off, mw_w, mw_h, _MW_N_LEVELS, _MW_MAX_LOD, _MW_WIDTH, _MW_GAIN
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
    _MW_GAIN = float(sf.get("diffuse_gain", 1.0))  # Layer-B brightness multiplier (render-time)

    if _DNGR == 1:
        # --- Layer A: point-star catalog → CSR cell grid ---
        cat_path = _ROOT / sf["catalog_path"]
        if not cat_path.exists():
            raise FileNotFoundError(
                f"starfield.mode=dngr needs the ingested catalog {cat_path}; "
                "run `python scripts/ingest_stars.py` first."
            )
        catalog = np.load(cat_path)
        th_s, ph_s, fl_s, starts, counts = _build_star_grid(catalog, _STAR_COLS, _STAR_ROWS)
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


def _setup_disk_flux(cfg: dict) -> None:
    """Build + upload the Page-Thorne f_PT(r) LUT (D1; SKILL.md CKS-11).

    ALWAYS runs, regardless of ``disk.temperature_model`` — the LUT is tiny (256
    f32) and uploading it unconditionally lets the flag toggle per-render without a
    re-JIT (the kernel just doesn't sample it in the simple branch). The
    ``_PT_LUT_*`` index scalars are baked into module globals here so the kernel
    JITs against them (same pattern as ``_J_FOLD`` / ``_MAX_LOD``).
    """
    global disk_flux_lut, _PT_LUT_N, _PT_LUT_R0, _PT_LUT_INV_DR

    d = cfg["disk"]
    a = float(cfg["black_hole"]["spin"])
    r_outer = float(d["r_outer"])
    n = int(d.get("flux_lut_samples", 256))

    lut, r0, dr = disk_flux.build_flux_lut(a, r_outer, n)
    _PT_LUT_N = n
    _PT_LUT_R0 = r0
    _PT_LUT_INV_DR = 1.0 / dr if dr > 0.0 else 0.0
    disk_flux_lut = ti.field(dtype=ti.f32, shape=n)
    disk_flux_lut.from_numpy(lut)


def _horizon_radius(a: float) -> float:
    """Outer horizon r₊ = 1 + √(1−a²) (Formula CKS-6; r is the BL radius).

    Derived from ``a`` here (like the camera basis and disk constants) rather than
    read from ``configs.black_hole.r_plus`` so it can never drift out of sync with
    ``black_hole.spin``: r₊ is a *function* of a, so the config key (1.0447 for
    a=0.999) is a derived-value duplication that silently desyncs if a is edited.
    """
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def _alloc_output(res: int) -> None:
    global pixels, _RES
    if pixels is None or res != _RES:
        pixels = ti.field(dtype=ti.f32, shape=(res, res, 3))
        _RES = res


def _alloc_frame(width: int, height: int) -> None:
    global frame_pixels, exit_buf, disk_buf, depth_pixels, _FW, _FH
    if frame_pixels is None or width != _FW or height != _FH:
        frame_pixels = ti.field(dtype=ti.f32, shape=(height, width, 3))
        exit_buf = ti.field(dtype=ti.f32, shape=(height, width, 3))
        disk_buf = ti.field(dtype=ti.f32, shape=(height, width, 4))
        depth_pixels = ti.field(dtype=ti.f32, shape=(height, width))
        _FW = width
        _FH = height


# --------------------------------------------------------------------------- #
# CKS physics @ti.func — ported verbatim from renderer.metric / renderer.geodesic
# --------------------------------------------------------------------------- #
@ti.func
def _kerr_radius(x, y, z, a):
    """Kerr radius r(x,y,z) — Formula CKS-1 explicit positive root."""
    a2 = a * a
    rho2 = x * x + y * y + z * z
    half = 0.5 * (rho2 - a2)
    r2 = half + ti.sqrt(half * half + a2 * z * z)
    return ti.sqrt(ti.max(r2, 1e-12))


@ti.func
def _cks_f_l(x, y, z, a):
    """Scalar f and covariant null vector l_α (l_t = 1) — Formula CKS-2."""
    r = _kerr_radius(x, y, z, a)
    r2 = r * r
    a2 = a * a
    D = r2 * r2 + a2 * z * z  # r⁴ + a²z²
    S = r2 + a2  # r² + a²
    f = 2.0 * r2 * r / D
    l = vec4(1.0, (r * x + a * y) / S, (r * y - a * x) / S, z / r)
    return f, l


@ti.func
def _metric_cks(x, y, z, a):
    """Covariant metric g_αβ = η_αβ + f l_α l_β — Formula CKS-2."""
    f, l = _cks_f_l(x, y, z, a)
    eta = vec4(-1.0, 1.0, 1.0, 1.0)
    M = ti.Matrix.zero(ti.f32, 4, 4)
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            diag = eta[i] if i == j else 0.0
            M[i, j] = diag + f * l[i] * l[j]
    return M


@ti.func
def _inv_metric_cks(x, y, z, a):
    """Exact inverse g^αβ = η^αβ − f l^α l^β — Formula CKS-3 (l is η-null)."""
    f, l = _cks_f_l(x, y, z, a)
    lu = vec4(-l[0], l[1], l[2], l[3])  # l^α = η^αγ l_γ : l^t = −l_t = −1
    eta = vec4(-1.0, 1.0, 1.0, 1.0)
    M = ti.Matrix.zero(ti.f32, 4, 4)
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            diag = eta[i] if i == j else 0.0
            M[i, j] = diag - f * lu[i] * lu[j]
    return M


@ti.func
def _eom(s, a):
    """Right-hand side of the CKS-5 8-vector EOM (mirrors renderer.geodesic._eom).

    s = [t, x, y, z, p_t, p_x, p_y, p_z] (covariant momenta). Working form with
    φ_l = l^β p_β = −p_t + l_x p_x + l_y p_y + l_z p_z:

        dt/dλ  = −p_t + f φ_l
        dxⁱ/dλ = p_i − f l_i φ_l
        dp_t/dλ = 0
        dp_i/dλ = ½ (∂_i f) φ_l² + f φ_l (∂_i φ_l)
    """
    x = s[1]
    y = s[2]
    z = s[3]
    p_t = s[4]
    p_x = s[5]
    p_y = s[6]
    p_z = s[7]

    a2 = a * a
    rho2 = x * x + y * y + z * z
    half = 0.5 * (rho2 - a2)
    r = ti.sqrt(ti.max(half + ti.sqrt(half * half + a2 * z * z), 1e-12))
    r2 = r * r
    r3 = r2 * r
    D = r2 * r2 + a2 * z * z  # r⁴ + a²z²
    S = r2 + a2  # r² + a²

    f = 2.0 * r3 / D
    l_x = (r * x + a * y) / S
    l_y = (r * y - a * x) / S
    l_z = z / r

    # ∂r/∂xⁱ  (CKS-4)
    dr_x = r3 * x / D
    dr_y = r3 * y / D
    dr_z = r * z * S / D

    # ∂f/∂xⁱ = f·[3 (∂r/∂xⁱ)/r − (4 r³ ∂r/∂xⁱ + 2 a² z δ_iz)/D]  (CKS-4)
    df_x = f * (3.0 * dr_x / r - (4.0 * r3 * dr_x) / D)
    df_y = f * (3.0 * dr_y / r - (4.0 * r3 * dr_y) / D)
    df_z = f * (3.0 * dr_z / r - (4.0 * r3 * dr_z + 2.0 * a2 * z) / D)

    # ∂l_i/∂xʲ  (CKS-4); l_t constant ⇒ no t-row.
    S2 = S * S
    dlx_x = ((x * dr_x + r) * S - (r * x + a * y) * (2.0 * r * dr_x)) / S2
    dlx_y = ((x * dr_y + a) * S - (r * x + a * y) * (2.0 * r * dr_y)) / S2
    dlx_z = ((x * dr_z) * S - (r * x + a * y) * (2.0 * r * dr_z)) / S2
    dly_x = ((y * dr_x - a) * S - (r * y - a * x) * (2.0 * r * dr_x)) / S2
    dly_y = ((y * dr_y + r) * S - (r * y - a * x) * (2.0 * r * dr_y)) / S2
    dly_z = ((y * dr_z) * S - (r * y - a * x) * (2.0 * r * dr_z)) / S2
    dlz_x = -z * dr_x / r2
    dlz_y = -z * dr_y / r2
    dlz_z = 1.0 / r - z * dr_z / r2

    phi_l = -p_t + l_x * p_x + l_y * p_y + l_z * p_z

    dt = -p_t + f * phi_l
    dx = p_x - f * l_x * phi_l
    dy = p_y - f * l_y * phi_l
    dz = p_z - f * l_z * phi_l

    dphi_x = dlx_x * p_x + dly_x * p_y + dlz_x * p_z
    dphi_y = dlx_y * p_x + dly_y * p_y + dlz_y * p_z
    dphi_z = dlx_z * p_x + dly_z * p_y + dlz_z * p_z

    dpx = 0.5 * df_x * phi_l * phi_l + f * phi_l * dphi_x
    dpy = 0.5 * df_y * phi_l * phi_l + f * phi_l * dphi_y
    dpz = 0.5 * df_z * phi_l * phi_l + f * phi_l * dphi_z

    return vec8(dt, dx, dy, dz, 0.0, dpx, dpy, dpz)


@ti.func
def _rk4_step(s, a, h):
    """One RK4 step of the CKS-5 EOM (mirrors renderer.geodesic.integrate)."""
    k1 = _eom(s, a)
    k2 = _eom(s + 0.5 * h * k1, a)
    k3 = _eom(s + 0.5 * h * k2, a)
    k4 = _eom(s + h * k3, a)
    return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@ti.func
def _rk4_step_k1(s, a, h, k1):
    """RK4 step reusing an already-evaluated ``k1 = _eom(s, a)``.

    Bit-identical to :func:`_rk4_step` — only the (h-independent) first stage is
    hoisted out so the disk march can read ``dz/dλ = k1[3]`` for the slab-aware
    step cap without a second EOM evaluation (net zero extra cost per step).
    """
    k2 = _eom(s + 0.5 * h * k1, a)
    k3 = _eom(s + 0.5 * h * k2, a)
    k4 = _eom(s + h * k3, a)
    return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@ti.func
def _photon_momentum_cks(cx, cy, cz, nx, ny, nz, a):
    """Covariant photon momentum p_α at the camera — Formula CKS-7.

    ZAMO observer from g^αβ, ray coordinate direction n projected g-orthogonal to
    u_obs, normalized, p^α = u_obs^α + ŝ^α, lowered, scaled so E = −p_t = 1.
    Mirrors ``renderer.geodesic.photon_momentum_cks``.
    """
    g = _metric_cks(cx, cy, cz, a)
    gi = _inv_metric_cks(cx, cy, cz, a)

    lapse = 1.0 / ti.sqrt(-gi[0, 0])
    u_obs = vec4(0.0, 0.0, 0.0, 0.0)
    for i in ti.static(range(4)):
        u_obs[i] = -lapse * gi[0, i]  # u_obs^α = −α g^{tα}

    N = vec4(0.0, nx, ny, nz)
    gNu = N.dot(g @ u_obs)  # g_μν N^μ u_obs^ν
    Nprime = N + gNu * u_obs  # now g·(N′, u_obs) = 0
    s_hat = Nprime / ti.sqrt(Nprime.dot(g @ Nprime))
    p_up = u_obs + s_hat  # E_loc = 1; null automatically
    p_cov = g @ p_up  # lower index
    E = -p_cov[0]
    return p_cov / E  # scale so E = −p_t = 1


@ti.func
def _exit_cos_phi(s, a):
    """Escaped-ray celestial direction — Formula CKS-10.

    Asymptotically flat ⇒ the contravariant spatial momentum direction
    d = (dx/dλ, dy/dλ, dz/dλ) = (p^x, p^y, p^z) IS the celestial direction. Returns
    ``(cosθ′, φ′)`` to match the legacy ``exit_buf`` layout so every downstream
    screen-space LOD / DNGR routine is unchanged.
    """
    v = _eom(s, a)
    dx = v[1]
    dy = v[2]
    dz = v[3]
    inv = 1.0 / ti.sqrt(dx * dx + dy * dy + dz * dz)
    cos_th = ti.min(ti.max(dz * inv, -1.0), 1.0)
    phi = ti.atan2(dy * inv, dx * inv)
    return cos_th, phi


@ti.func
def _trace_to_exit(cx, cy, cz, nx, ny, nz, a, r_plus, r_max, n_steps, d_lambda, adaptive_floor):
    """Integrate one null geodesic to its outcome (capture / escape).

    Returns ``(outcome, cosθ′, φ′)``. Adaptive affine step (Formula CKS-5):
    ``h = dλ·max(adaptive_floor, (r − r₊)/r)`` — full steps far out, shrinking
    toward the horizon. Used by the Pipe-A preview kernel (the beauty kernel
    inlines the same loop with disk accumulation interleaved).
    """
    p0 = _photon_momentum_cks(cx, cy, cz, nx, ny, nz, a)
    s = vec8(0.0, cx, cy, cz, p0[0], p0[1], p0[2], p0[3])

    out = _RUNNING
    cos_exit = 0.0
    phi_exit = 0.0
    step = 0
    while step < n_steps and out == _RUNNING:
        x = s[1]
        y = s[2]
        z = s[3]
        r = _kerr_radius(x, y, z, a)
        rho = ti.sqrt(x * x + y * y + z * z)
        if r <= r_plus + _HORIZON_EPS:
            out = _CAPTURED
        elif rho >= r_max:
            out = _ESCAPED
            cos_exit, phi_exit = _exit_cos_phi(s, a)
        else:
            h = d_lambda * ti.max(adaptive_floor, (r - r_plus) / r)
            s = _rk4_step(s, a, h)
        step += 1
    return out, cos_exit, phi_exit


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
    uu = u - ti.floor(u)  # wrap φ (periodic)
    vv = ti.min(ti.max(v, 0.0), 1.0)  # clamp θ
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
# Pipe B physics @ti.func — Formula CKS-8 (gas velocity), CKS-9 (g-factor)
# --------------------------------------------------------------------------- #
@ti.func
def _gas_four_velocity_cks(x, y, z, a):
    """Equatorial disk gas 4-velocity u^μ = (u^t, u^x, u^y, u^z) — Formula CKS-8.

    Circular prograde orbit (r ≥ r_isco): a rigid rotation about +z at the BL
    angular velocity Ω (Formula 3). No BL→KS Jacobian — see SKILL.md CKS-8.
    (The plunging r < r_isco branch is never sampled: disk.r_inner = r_isco.)
    """
    r = _kerr_radius(x, y, z, a)
    r15 = ti.pow(r, 1.5)
    Omega = 1.0 / (r15 + a)  # Formula 3
    u_t = (1.0 + a / r15) / ti.sqrt(ti.max(1.0 - 3.0 / r + 2.0 * a / r15, 1e-9))
    u_x = -Omega * y * u_t
    u_y = Omega * x * u_t
    return vec4(u_t, u_x, u_y, 0.0)


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
def _pt_flux_sample(r):
    """Linear-interp the Page-Thorne f_PT(r) LUT (D1; SKILL.md CKS-11).

    ``t = (r − r0)/dr`` clamped to ``[0, n−1]``; lerp the two adjacent texels.
    Returns 0 below r_isco (lut[0]==0 by construction, the zero-torque BC).
    """
    t = (r - _PT_LUT_R0) * _PT_LUT_INV_DR
    t = ti.min(ti.max(t, 0.0), ti.cast(_PT_LUT_N - 1, ti.f32))
    i0 = ti.cast(ti.floor(t), ti.i32)
    i1 = ti.min(i0 + 1, _PT_LUT_N - 1)
    frac = t - ti.cast(i0, ti.f32)
    return disk_flux_lut[i0] * (1.0 - frac) + disk_flux_lut[i1] * frac


@ti.func
def _disk_emit_cks(
    x, y, z, p_cov, a, r_inner, r_outer, theta_half, sigma_frac, T_0, emis_c, absb_c, ds,
    disk_model, doppler_strength,
):
    """One volumetric disk sample at a CKS geodesic point → (emission RGB, dτ).

    Returns ``vec4(emission_r, emission_g, emission_b, dtau)``; the running pixel
    update is ``color += T·emission`` and ``T *= exp(-dtau)``. Outside the
    equatorial slab bounding box it returns zeros.

    Formula CKS-9 g-factor (Cartesian dot product): with the integrator's
    covariant momenta ``p_μ`` and the gas ``u^μ`` (CKS-8),
    ``g = −E / (p_t u^t + p_x u^x + p_y u^y + p_z u^z)``. The Formula-8 "divide
    p_r by Δ" bug is structurally impossible — there is no Δ and p is already
    covariant. Formula 9: chromaticity·g⁴ volumetric emission.

    ``disk_model`` (D1): 0 = simple ``T = T_0·(6/r)^0.75`` (Decision-B default,
    golden frames); 1 = Page-Thorne (SKILL.md CKS-11): sample the precomputed
    f_PT(r) shape LUT, set ``T_emit = T_0·f_PT^¼`` (T_eff=(F/σ)^¼ with T_0 carrying
    the σ/Ṁ amplitude — CKS-11 Piece 3), and fold f_PT into the bolometric
    amplitude (``emission ∝ … · f_PT``) — the f factor IS the T⁴ radial profile.

    **g-bookkeeping (Formula 9 / CKS-11 Piece 3, BOTH paths):** ``_blackbody_rgb``
    is chromaticity-only (no T⁴ amplitude), so the explicit ``g⁴`` is the ONLY g
    factor and is NOT double-counted — do not add any other g (that is the g⁸
    error Formula 9 warns about). This holds identically for simple and
    page_thorne; page_thorne only changes the *radial* profile (f_PT vs (6/r)³).

    ``doppler_strength`` (visualization-only, NOT physics): exponent scale on the
    relativistic shift, ``g_eff = g^s``. 1.0 = full physics (the ``s != 1``
    branch is skipped, so the default is bit-identical to the pre-knob kernel —
    golden frames intact); 0.0 = g_eff ≡ 1 (no beaming, no color shift — the
    Interstellar/DNGR artistic treatment, which suppressed the Doppler asymmetry
    for the film); values between blend smoothly in log-g. It scales the TOTAL
    CKS-9 g (orbital Doppler + gravitational shift combined — separating the two
    would need a new SKILL.md formula). Applied once to g_eff, which then feeds
    BOTH g⁴ and the chromaticity, so the g⁴-not-g⁸ bookkeeping is unchanged.
    """
    out = vec4(0.0, 0.0, 0.0, 0.0)
    r = _kerr_radius(x, y, z, a)
    cos_th = ti.min(ti.max(z / r, -1.0), 1.0)
    th = ti.acos(cos_th)
    dz_ang = th - 0.5 * math.pi
    if (ti.abs(dz_ang) < theta_half) and (r >= r_inner) and (r <= r_outer):
        u4 = _gas_four_velocity_cks(x, y, z, a)
        E = -p_cov[0]
        denom = p_cov[0] * u4[0] + p_cov[1] * u4[1] + p_cov[2] * u4[2] + p_cov[3] * u4[3]
        if ti.abs(denom) > 1e-12:
            g = -E / denom  # Formula CKS-9
            if g > 0.0:
                # Artistic shift dial: g_eff = g^s (s=1 ⇒ untouched physics; the
                # branch keeps the default path bit-identical to ti.pow-free code).
                g_eff = g
                if doppler_strength != 1.0:
                    g_eff = ti.pow(g, doppler_strength)
                sigma_theta = theta_half * sigma_frac
                density = ti.exp(-0.5 * (dz_ang / sigma_theta) ** 2)
                g4 = g_eff * g_eff * g_eff * g_eff  # Formula 9 (3D volume: g⁴)

                # Radial profile: simple (Decision B) or Page-Thorne (CKS-11).
                T_emit = 0.0
                emission = 0.0
                emit = 1  # 0 ⇒ Page-Thorne f_PT≤0 (inside r_ms): no emission
                if disk_model == 1:
                    f = _pt_flux_sample(r)
                    if f <= 0.0:
                        emit = 0
                    else:
                        # CKS-11 Piece 3: T_eff = (F/σ)^¼ = T_0·f_PT^¼; f IS the
                        # T⁴ bolometric radial profile, so it multiplies emission.
                        T_emit = T_0 * ti.pow(f, 0.25)
                        emission = emis_c * density * f * g4 * ds
                else:
                    # Decision B temperature model: T = T_0·(6/r)^0.75.
                    T_emit = T_0 * ti.pow(6.0 / r, 0.75)
                    emission = emis_c * density * g4 * ds

                if emit == 1:
                    chroma = _blackbody_rgb(g_eff * T_emit)
                    out = vec4(
                        emission * chroma[0],
                        emission * chroma[1],
                        emission * chroma[2],
                        absb_c * density * ds,
                    )
    return out


# --------------------------------------------------------------------------- #
# Pipe A preview kernel (square, inline offset-ray LOD)
# --------------------------------------------------------------------------- #
@ti.kernel
def render_pipe_a(
    res: int,
    tan_half_fov: float,
    cx: float,
    cy: float,
    cz: float,
    fwd_x: float,
    fwd_y: float,
    fwd_z: float,
    rgt_x: float,
    rgt_y: float,
    rgt_z: float,
    up_x: float,
    up_y: float,
    up_z: float,
    a: float,
    r_plus: float,
    r_max: float,
    n_steps: int,
    d_lambda: float,
    adaptive_floor: float,
    lod_enabled: int,
):
    """Pipe A: trace one primary + one offset ray per pixel; sample the lensed sky.

    Camera position and basis (fwd/right/up) are **world Cartesian = CKS** — no
    spherical embedding. Each ray's coordinate direction is
    ``n = normalize(fwd + sx·right + sy·up)`` (CKS-7 input). The offset ray
    (screen u shifted by +1 px) yields the on-sky Jacobian J → mip LOD (Formula 10).
    """
    for py, px in ti.ndrange(res, res):
        rf = ti.cast(res, ti.f32)
        sx_p = (2.0 * (px + 0.5) / rf - 1.0) * tan_half_fov
        sy_p = (1.0 - 2.0 * (py + 0.5) / rf) * tan_half_fov
        sx_o = (2.0 * (px + 1 + 0.5) / rf - 1.0) * tan_half_fov

        npx = fwd_x + sx_p * rgt_x + sy_p * up_x
        npy = fwd_y + sx_p * rgt_y + sy_p * up_y
        npz = fwd_z + sx_p * rgt_z + sy_p * up_z
        invp = 1.0 / ti.sqrt(npx * npx + npy * npy + npz * npz)
        npx *= invp
        npy *= invp
        npz *= invp

        nox = fwd_x + sx_o * rgt_x + sy_p * up_x
        noy = fwd_y + sx_o * rgt_y + sy_p * up_y
        noz = fwd_z + sx_o * rgt_z + sy_p * up_z
        invo = 1.0 / ti.sqrt(nox * nox + noy * noy + noz * noz)
        nox *= invo
        noy *= invo
        noz *= invo

        out_p, cos_p, ph_p = _trace_to_exit(
            cx, cy, cz, npx, npy, npz, a, r_plus, r_max, n_steps, d_lambda, adaptive_floor
        )
        out_o, cos_o, ph_o = _trace_to_exit(
            cx, cy, cz, nox, noy, noz, a, r_plus, r_max, n_steps, d_lambda, adaptive_floor
        )

        col = vec3(0.0, 0.0, 0.0)
        if out_p == _ESCAPED:
            th_p = ti.acos(cos_p)
            u = ph_p / _TWO_PI
            u = u - ti.floor(u)
            v = ti.min(ti.max(th_p / math.pi, 0.0), 1.0)

            lod = 0.0
            if lod_enabled == 1:
                if out_o == _ESCAPED:
                    th_o = ti.acos(cos_o)
                    d_th = th_o - th_p
                    d_ph = ph_o - ph_p
                    d_ph = d_ph - _TWO_PI * ti.round(d_ph / _TWO_PI)
                    sin_th = ti.sin(th_p)
                    J = ti.sqrt(d_th * d_th + sin_th * sin_th * d_ph * d_ph)
                    if J > _J_FOLD:
                        lod = _MAX_LOD
                    else:
                        lod = ti.log(_STARMAP_WIDTH * J / _TWO_PI) / ti.log(2.0)
                else:
                    lod = _MAX_LOD
            col = _sample_trilinear(u, v, lod)

        pixels[py, px, 0] = col[0]
        pixels[py, px, 1] = col[1]
        pixels[py, px, 2] = col[2]


# --------------------------------------------------------------------------- #
# Beauty kernel — Pipe A (lensed background) + Pipe B (volumetric disk)
# --------------------------------------------------------------------------- #
@ti.kernel
def render_beauty_physics(
    width: int,
    height: int,
    cx: float,
    cy: float,
    cz: float,
    fwd_x: float,
    fwd_y: float,
    fwd_z: float,
    rgt_x: float,
    rgt_y: float,
    rgt_z: float,
    up_x: float,
    up_y: float,
    up_z: float,
    tan_half_x: float,
    tan_half_y: float,
    a: float,
    r_plus: float,
    r_max: float,
    n_steps: int,
    d_lambda: float,
    adaptive_floor: float,
    disk_enabled: int,
    projection_mode: int,
    depth_infinity: float,
    r_inner: float,
    r_outer: float,
    theta_half: float,
    bound_sin_half: float,
    sigma_frac: float,
    T_0: float,
    emis_c: float,
    absb_c: float,
    disk_model: int,
    doppler_strength: float,
    max_step_vfrac: float,
):
    """Kernel 1 (Phase 2.4 split): trace ONE primary ray per pixel (no offset ray).

    Writes the per-pixel exit state to ``exit_buf`` (cosθ′, φ′, outcome), the
    front-to-back disk accumulation to ``disk_buf`` (disk_rgb, transmittance), and
    the transmittance-weighted affine Z to ``depth_pixels`` (optimization Phase 3.4).
    The Formula-10 LOD + background lookup are deferred to ``render_beauty_shade``,
    which differences neighbour exit directions in screen space.

    Camera position and basis are **world Cartesian = CKS** (no spherical
    embedding). Per-pixel ray direction n = normalize(fwd + sx·right + sy·up).
    State is the CKS 8-vector [t, x, y, z, p_t, p_x, p_y, p_z] (Formula CKS-5);
    adaptive affine step h = d_lambda·max(adaptive_floor, (r−r₊)/r).

    ``projection_mode`` (optimization Phase 4.1): 0 = perspective (camera basis +
    FOV), 1 = equirectangular 360° (px→lon, py→lat about the camera basis, for VR).
    """
    for py, px in ti.ndrange(height, width):
        nx = fwd_x
        ny = fwd_y
        nz = fwd_z
        if projection_mode == 1:
            # Equirectangular 360° ray-gen: screen → (lon, lat) → camera-basis dir.
            lon = (px + 0.5) / width * _TWO_PI  # azimuth ∈ [0, 2π)
            lat = (py + 0.5) / height * math.pi  # polar ∈ [0, π]
            d_fwd = ti.sin(lat) * ti.cos(lon)
            d_rgt = ti.sin(lat) * ti.sin(lon)
            d_up = ti.cos(lat)
            nx = d_fwd * fwd_x + d_rgt * rgt_x + d_up * up_x
            ny = d_fwd * fwd_y + d_rgt * rgt_y + d_up * up_y
            nz = d_fwd * fwd_z + d_rgt * rgt_z + d_up * up_z
        else:
            sx_p = (2.0 * (px + 0.5) / width - 1.0) * tan_half_x
            sy_p = (1.0 - 2.0 * (py + 0.5) / height) * tan_half_y
            nx = fwd_x + sx_p * rgt_x + sy_p * up_x
            ny = fwd_y + sx_p * rgt_y + sy_p * up_y
            nz = fwd_z + sx_p * rgt_z + sy_p * up_z
        invp = 1.0 / ti.sqrt(nx * nx + ny * ny + nz * nz)
        nx *= invp
        ny *= invp
        nz *= invp

        p0 = _photon_momentum_cks(cx, cy, cz, nx, ny, nz, a)
        s = vec8(0.0, cx, cy, cz, p0[0], p0[1], p0[2], p0[3])

        out_p = _RUNNING
        cos_exit = 0.0
        phi_exit = 0.0

        disk_col = vec3(0.0, 0.0, 0.0)
        transm = 1.0
        ray_length = 0.0  # accumulated affine path length (depth proxy)
        weighted_depth = 0.0  # Σ ray_length·contribution  (optimization Phase 3.4)
        total_emission = 0.0  # Σ contribution

        step = 0
        while step < n_steps and out_p == _RUNNING:
            x = s[1]
            y = s[2]
            z = s[3]
            r = _kerr_radius(x, y, z, a)
            rho = ti.sqrt(x * x + y * y + z * z)

            if r <= r_plus + _HORIZON_EPS:
                out_p = _CAPTURED
            elif rho >= r_max:
                out_p = _ESCAPED
                cos_exit, phi_exit = _exit_cos_phi(s, a)
            else:
                # Adaptive affine step (Formula CKS-5). Computed BEFORE the disk
                # emit so the same h is the emission path element ds (else the
                # variable step desyncs the Riemann sum).
                local_h = d_lambda * ti.max(adaptive_floor, (r - r_plus) / r)

                # k1 is hoisted out of the RK4 step so the disk-thickness cap below
                # can read dz/dλ = k1[3] for free (the step reuses this same k1).
                k1 = _eom(s, a)

                in_band = (
                    disk_enabled == 1
                    and ti.abs(z / r) < bound_sin_half
                    and r >= r_inner
                    and r <= r_outer
                )
                if in_band:
                    # Disk-thickness step cap. The base rule sizes h only by distance
                    # to the horizon and is blind to the disk's vertical extent, so a
                    # ray crossing the equatorial plane steeply can stride over the
                    # thin emitting layer — under-sampling the Gaussian density
                    # (Formula 9) into a moiré band on the disk face. Limit the per-
                    # step VERTICAL displacement |dz/dλ|·h to a fraction of the local
                    # scale height σ_z = r·θ_half·σ_frac so the slab is resolved. Only
                    # bites for steep crossings (large |dz/dλ|); near-in-plane / edge-
                    # on grazers (dz/dλ→0) keep the full radial step, so the cap adds
                    # no steps there and cannot push those rays into the max_steps cap.
                    sigma_z = r * theta_half * sigma_frac
                    vz = ti.abs(k1[3])
                    h_cap = max_step_vfrac * sigma_z / ti.max(vz, _DISK_STEP_V_EPS)
                    local_h = ti.min(local_h, h_cap)

                # Pipe B: accumulate disk emission at the current point.
                # optimization Phase 3.3 bounding-box early-out: |cosθ| < sin θ_half (≡ the
                # slab test) and r within the radial band.
                if in_band:
                    p_cov = vec4(s[4], s[5], s[6], s[7])
                    ev = _disk_emit_cks(
                        x,
                        y,
                        z,
                        p_cov,
                        a,
                        r_inner,
                        r_outer,
                        theta_half,
                        sigma_frac,
                        T_0,
                        emis_c,
                        absb_c,
                        local_h,
                        disk_model,
                        doppler_strength,
                    )
                    disk_col += transm * vec3(ev[0], ev[1], ev[2])
                    contribution = transm * (ev[0] + ev[1] + ev[2])
                    weighted_depth += ray_length * contribution
                    total_emission += contribution
                    transm *= ti.exp(-ev[3])

                s = _rk4_step_k1(s, a, local_h, k1)
                ray_length += local_h
            step += 1

        exit_buf[py, px, 0] = cos_exit
        exit_buf[py, px, 1] = phi_exit
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
    """Formula 10 (amendment v1.4): LOD from screen-space neighbour exit deltas.

    Differences the primary pixel's exit direction against its +x and +y
    neighbours (backward at the far edges). If any neighbour did not ESCAPE
    (``outcome != _ESCAPED``), returns ``_MAX_LOD`` (the chaotic shadow-edge
    boundary rule). Otherwise L = log2(W·J/2π) with J the larger of the two
    axis footprints. ``exit_buf[...,0]`` is cosθ′ (CKS-10), so θ′ = acos(·).
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
        # Fold saturation: a footprint J spanning a large fraction of the sky
        # straddles an equirect-texture pole or the chaotic shadow silhouette, so
        # a single scalar-LOD trilinear fetch would land on unrelated coarse-mip
        # texels. Above _J_FOLD collapse to the uniform coarsest mip (smooth grey)
        # rather than an aliased mid-mip fetch. (Under CKS the BL spin-axis seam
        # is gone; this only guards the texture poles / shadow edge now.)
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
    falls back to the coarsest mip. On that same invalid-``det J`` branch the
    Layer-A splat is *placed* by guard (b′) using the undeflected proper-separation
    footprint ``d² = (Δθ′² + sin²θ′·Δφ′²)/dΩ``. ``exit_buf[...,0]`` is cosθ′
    (CKS-10), so θ′ = acos(·). Returns the combined background radiance.
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
    delta_plus = 0.0  # δ⁺ major ellipse axis (angular, sinθ-weighted)
    delta_minus = 0.0  # δ⁻ minor ellipse axis
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
        ntaps = ti.cast(
            ti.min(ti.ceil(maj / ti.max(minr, 1.0)), ti.cast(_EWA_MAX_TAPS, ti.f32)), ti.i32
        )
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
    diffuse *= _MW_GAIN  # Layer-B brightness control (starfield.diffuse_gain; 1.0 = identity)

    # --- Layer A: point-star energy gather (flux · μ · g⁴ · Gaussian PSF) ---
    # Placement of each catalog star's splat follows Formula 13 guards (b) + (b′):
    #   * `usable` (det J valid): lensed placement (dpx,dpy) = J⁻¹·(Δθ′,Δφ′).
    #   * NOT `usable`: guard (b′) — undeflected proper-separation placement
    #       d² = (Δθ′² + sin²θ′·Δφ′²) / dΩ, μ clamped to 1 (stars stay sharp).
    stars = vec3(0.0, 0.0, 0.0)
    g4 = 1.0  # _G_BEAMING hook: g≡1 for a static camera at the celestial sphere
    two_sig2 = 2.0 * _STAR_PSF * _STAR_PSF
    d_max2 = (_PSF_TRUNC * _STAR_PSF) ** 2
    inv_det = 0.0
    if usable and ti.abs(detJ) > 1e-20:
        inv_det = 1.0 / detJ
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
                            dpx = (dphy * dth - dthy * dph) * inv_det
                            dpy = (-dphx * dth + dthx * dph) * inv_det
                            d2 = dpx * dpx + dpy * dpy
                        else:
                            d2 = (dth * dth + sin_th * sin_th * dph * dph) / d_omega
                        if d2 < d_max2:
                            wgt = ti.exp(-d2 / two_sig2) * mu * g4
                            stars += (
                                vec3(cat_flux[sidx, 0], cat_flux[sidx, 1], cat_flux[sidx, 2]) * wgt
                            )
    return diffuse + stars


@ti.kernel
def render_beauty_shade(
    width: int,
    height: int,
    lod_enabled: int,
    mode: int,
    tan_half_x: float,
    tan_half_y: float,
    projection_mode: int,
):
    """Kernel 2 (Phase 2.4 split): background lookup + composite.

    ``mode == 0`` (texture): the legacy Formula-10 path — screen-space LOD from
    neighbour exit deltas, single trilinear lensed-starmap fetch. ``mode == 1``
    (dngr): the Formula-13 two-layer background (``_dngr_shade``). Either way it
    writes ``frame_pixels = disk_rgb + transmittance·background``.
    """
    for py, px in ti.ndrange(height, width):
        out_p = exit_buf[py, px, 2]
        bg = vec3(0.0, 0.0, 0.0)
        if out_p < 1.5 and out_p > 0.5:  # escaped
            cos_exit = exit_buf[py, px, 0]
            ph_p_exit = exit_buf[py, px, 1]
            th_p_n = ti.acos(ti.min(ti.max(cos_exit, -1.0), 1.0))
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
                    d_omega = (
                        (4.0 * tan_half_x * tan_half_y / (width * height))
                        * inv_len
                        * inv_len
                        * inv_len
                    )
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


# --------------------------------------------------------------------------- #
# Host-side camera helpers + entry points
# --------------------------------------------------------------------------- #
def _camera_basis(pos, fwd, world_up=(0.0, 0.0, 1.0)):
    """Orthonormal camera basis (fwd, right, up) from a look direction.

    All in world Cartesian = CKS. ``right = normalize(fwd × world_up)``,
    ``up = right × fwd`` (right-handed). ``world_up`` defaults to the +z spin axis.
    """
    f = np.asarray(fwd, dtype=float)
    f = f / np.linalg.norm(f)
    wu = np.asarray(world_up, dtype=float)
    right = np.cross(f, wu)
    nrm = np.linalg.norm(right)
    if nrm < 1e-8:  # looking along the spin axis: pick any ⟂
        right = np.cross(f, np.array([1.0, 0.0, 0.0]))
        nrm = np.linalg.norm(right)
    right = right / nrm
    up = np.cross(right, f)
    return f, right, up


def render_pipe_a_image(cfg: dict, res: int, lod_enabled: bool) -> np.ndarray:
    """Render one Pipe A frame at ``res×res`` and return a float32 (res,res,3) HDR.

    The preview camera is placed from ``thumb`` config (spherical radius/θ/φ),
    converted to CKS Cartesian, looking at the origin.
    """
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
    adaptive_floor = float(rcfg["adaptive_step_floor"])
    r_max = float(rcfg["r_max"])
    tan_half_fov = math.tan(math.radians(fov_deg) / 2.0)
    r_plus = _horizon_radius(a)

    # Camera position in CKS Cartesian (spin axis = +z); look at the origin.
    st, ct = math.sin(theta_cam), math.cos(theta_cam)
    pos = np.array(
        [r_cam * st * math.cos(phi_cam), r_cam * st * math.sin(phi_cam), r_cam * ct], dtype=float
    )
    fwd, right, up = _camera_basis(pos, -pos)

    _alloc_output(res)
    render_pipe_a(
        res,
        tan_half_fov,
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(fwd[0]),
        float(fwd[1]),
        float(fwd[2]),
        float(right[0]),
        float(right[1]),
        float(right[2]),
        float(up[0]),
        float(up[1]),
        float(up[2]),
        a,
        r_plus,
        r_max,
        n_steps,
        d_lambda,
        adaptive_floor,
        1 if lod_enabled else 0,
    )
    ti.sync()
    return pixels.to_numpy()


def render_beauty_frame(
    cfg: dict,
    cam_frame: dict,
    width: int,
    height: int,
    with_disk: bool = True,
    lod_enabled: bool = True,
    return_depth: bool = False,
):
    """Render one beauty frame (Pipe A + Pipe B) for a camera_matrix.json entry.

    ``cam_frame`` carries the Blender camera in **world Cartesian** coordinates
    (``pos``/``fwd``/``up``/``right`` and a vertical ``fov`` in radians, per
    ``src/blender/export_camera.py``). Under CKS the world Cartesian frame **is**
    the coordinate frame (spin axis = +z), so the camera position and basis are
    used directly — no Boyer-Lindquist embedding, no (r̂,θ̂,φ̂) projection. The
    supplied basis is re-orthonormalized for numerical safety.

    Returns a float32 ``(height, width, 3)`` HDR buffer (and the depth pass if
    ``return_depth``).
    """
    bh = cfg["black_hole"]
    rcfg = cfg["render"]
    d = cfg["disk"]

    a = float(bh["spin"])
    r_plus = _horizon_radius(a)

    pos = np.asarray(cam_frame["pos"], dtype=float)
    fwd_in = np.asarray(cam_frame["fwd"], dtype=float)
    up_in = np.asarray(cam_frame["up"], dtype=float)

    # World Cartesian = CKS. Re-orthonormalize the camera basis (Gram-Schmidt the
    # supplied up against fwd) so FP drift in the exported matrix can't skew rays.
    fwd = fwd_in / np.linalg.norm(fwd_in)
    up0 = up_in - np.dot(up_in, fwd) * fwd
    nrm = np.linalg.norm(up0)
    if nrm < 1e-8:
        _, right, up = _camera_basis(pos, fwd)
    else:
        up = up0 / nrm
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)

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
    # Disk-slab |cosθ| early-out bound, DERIVED from its source parameter (single
    # source of truth): |θ−π/2| < θ_half ⇔ |cosθ| < sin(θ_half), exactly the slab
    # test re-checked inside ``_disk_emit_cks``.
    bound_sin_half = math.sin(float(d["theta_half_width"]))
    # D1 disk radial-profile selector: 0 = simple T₀·(6/r)^0.75 (Decision-B default,
    # golden frames); 1 = Page-Thorne f_PT(r) LUT (SKILL.md CKS-11). The _PT_LUT_*
    # index scalars are baked at JIT from _setup_disk_flux (always run by
    # setup_renderer first, same as _MAX_LOD), so the flag toggles with no re-JIT.
    disk_model = 1 if str(d.get("temperature_model", "simple")) == "page_thorne" else 0
    # Visualization knob (NOT physics): g_eff = g^s exponent on the CKS-9 shift.
    # 1.0 = full physics (default, golden frames); 0.0 = Doppler/redshift off
    # (Interstellar-style symmetric disk). Runtime kernel arg — no re-JIT.
    doppler_strength = float(d.get("doppler_strength", 1.0))

    _alloc_frame(width, height)
    # Phase 2.4 kernel split: physics pass writes exit_buf/disk_buf, shading pass
    # computes the screen-space-Jacobian LOD and composites.
    render_beauty_physics(
        width,
        height,
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(fwd[0]),
        float(fwd[1]),
        float(fwd[2]),
        float(right[0]),
        float(right[1]),
        float(right[2]),
        float(up[0]),
        float(up[1]),
        float(up[2]),
        tan_half_x,
        tan_half_y,
        a,
        r_plus,
        r_max,
        n_steps,
        d_lambda,
        adaptive_floor,
        1 if with_disk else 0,
        projection_mode,
        depth_infinity,
        float(d["r_inner"]),
        float(d["r_outer"]),
        float(d["theta_half_width"]),
        bound_sin_half,
        float(d["vertical_sigma_frac"]),
        float(d["T_0"]),
        float(d["emission_coeff"]),
        float(d["absorption_coeff"]),
        disk_model,
        doppler_strength,
        float(d.get("max_step_vfrac", 0.5)),
    )
    # starfield.mode: dngr ⇒ Formula-13 two-layer background; else legacy F10 texture.
    dngr_mode = 1 if str(cfg.get("starfield", {}).get("mode", "texture")) == "dngr" else 0
    render_beauty_shade(
        width, height, 1 if lod_enabled else 0, dngr_mode, tan_half_x, tan_half_y, projection_mode
    )
    ti.sync()
    if return_depth:
        # A non-finite disk sample (RK4 overshoot at the inner edge) could leave
        # NaN/±inf in the Z pass; map any non-finite depth to the no-hit sentinel
        # so a poisoned pixel reads as "empty", never as a finite garbage Z.
        depth = np.nan_to_num(
            depth_pixels.to_numpy(),
            nan=depth_infinity,
            posinf=depth_infinity,
            neginf=depth_infinity,
        )
        return frame_pixels.to_numpy(), depth
    return frame_pixels.to_numpy()


def _rotate_z(vec, dphi: float):
    """Rotate a world Cartesian vector about the BH spin axis (z) by ``dphi``."""
    c, s = math.cos(dphi), math.sin(dphi)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return [c * x - s * y, s * x + c * y, z]


def render_beauty_frame_mb(
    cfg: dict,
    cam_frame: dict,
    width: int,
    height: int,
    shutter_arc: float,
    with_disk: bool = True,
    lod_enabled: bool = True,
    return_depth: bool = False,
):
    """Temporal motion blur (optimization Phase 4.2) by host-side averaging of jittered sub-frames.

    Renders ``render.motion_blur_samples`` copies of the frame with the camera
    rotated about the spin axis across the shutter arc ``shutter_arc`` (radians of
    azimuthal travel during the shutter) and averages the HDR results. ``samples<=1``
    or ``shutter_arc==0`` → a single render (no blur, no extra cost).
    """
    samples = int(cfg["render"]["motion_blur_samples"])
    if samples <= 1 or shutter_arc == 0.0:
        return render_beauty_frame(
            cfg, cam_frame, width, height, with_disk, lod_enabled, return_depth
        )

    depth_inf = float(cfg["render"]["depth_infinity"])
    acc = None
    depth_sum = None  # Σ of finite (disk-hit) depths only
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
        out = render_beauty_frame(cfg, jf, width, height, with_disk, lod_enabled, return_depth)
        hdr = out[0] if return_depth else out
        acc = hdr.astype(np.float64) if acc is None else acc + hdr
        if return_depth:
            # NEVER arithmetic-mean the depth_infinity no-hit sentinel with real
            # depths. Accumulate only the sub-frames that actually hit the disk;
            # pixels never hit keep the +∞ sentinel.
            dd = out[1].astype(np.float64)
            hit = dd < depth_inf
            if depth_sum is None:
                depth_sum = np.where(hit, dd, 0.0)
                depth_hits = hit.astype(np.float64)
            else:
                depth_sum += np.where(hit, dd, 0.0)
                depth_hits += hit

    hdr_mean = (acc / samples).astype(np.float32)
    if return_depth:
        depth_mean = np.where(depth_hits > 0.0, depth_sum / np.maximum(depth_hits, 1.0), depth_inf)
        return hdr_mean, depth_mean.astype(np.float32)
    return hdr_mean


def tonemap(hdr: np.ndarray, exposure: float, gamma: float) -> np.ndarray:
    """Reinhard tonemap + gamma, → uint8 (matches scripts/thumb.py)."""
    img = hdr * exposure
    img = img / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / gamma)
    return (img * 255.0 + 0.5).astype(np.uint8)
