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

from renderer import disk_flux, kerr_params, noise
from renderer.starmap import Starmap

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "configs" / "render.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    # Explicit utf-8: this box defaults to cp949 and the config has θ/π/· bytes.
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Inject all spin/extent-derived parameters (r_plus, r_isco, r_inner, T_0,
    # disk.dynamics.*) — Formula CKS-13. The YAML stores base parameters only.
    return kerr_params.resolve_config(cfg)


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
disk_buf: ti.Field = None  # type: ignore[assignment]  (H,W,6): disk_rgb + vec3 transmittance (CKS-19 Task 7)
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

# --- Disk procedural noise param buffer (D2.2; SKILL.md CKS-12, spec §4/§5) --- #
# A small f32 ti.field holds ALL per-layer noise tuning params so look-dev edits
# (amp / freq / octaves …) re-upload the buffer instead of re-JITting the kernel
# (spec §6: "param buffer, not baked module constants"). ``disk.noise.enabled`` and
# ``disk.noise.seed`` are kernel ARGS (toggle/seed per render). ALWAYS built+uploaded
# in setup (tiny) so the enabled flag flips per-render with no re-JIT; when the flag
# is 0 the kernel takes a branch bit-identical to the pre-noise path (CKS-12
# constraint 6 — golden frames untouched). Index map (flat f32 layout) below; the
# _NI_* scalars are plain Python ints baked at JIT (same pattern as _PT_LUT_N).
disk_noise_params: ti.Field = None  # type: ignore[assignment]  (_NOISE_N,) f32

# CKS-19 compile-time gate. The ρ_cold (dust) path roughly triples the inlined
# `_disk_noise_m` tree (the hot blended modulator + a reseeded dust one, each a
# dual-phase pair); because the curl/flow sub-tree compiles unconditionally
# (its enables are runtime field reads), a *runtime* `if _NI_MP_EN` would compile
# that tripled body into EVERY render — LLVM's superlinear inliner then turns it
# into a multi-hour JIT even with multiphase OFF. So multiphase is gated by this
# module bool via `ti.static(...)` instead of the param buffer: OFF ⇒ the dust
# branch is not emitted at all ⇒ the default path compiles exactly as before and
# golden frames are bit-identical. `_setup_disk_noise` sets it from the config;
# setup_renderer re-runs `ti.init` (clearing the kernel cache) so the new value
# takes effect on the next compile. TRADE-OFF (deliberate): toggling
# `disk.multiphase.enabled` requires a recompile, unlike the other noise dials.
_MP_COMPILE: bool = False

# CKS-20 single-scattering compile gate (same rationale as _MP_COMPILE above): the
# scatter body (a ρ_cold re-eval, a shadow lookup, two normalizes, an HG eval) would
# bloat the JIT if emitted unconditionally, so it is gated by this module bool via
# ti.static. OFF ⇒ no bytes emitted ⇒ the default path compiles exactly as before and
# golden frames are bit-identical. `_setup_disk_noise` sets it from disk.scatter.enabled;
# setup_renderer re-runs ti.init so the new value takes effect on the next compile.
_SCATTER_COMPILE: bool = False

# global
_NI_M_MAX = 0
# L0 base streaks (fbm2 on density)
_NI_L0_EN, _NI_L0_AMP, _NI_L0_OCT, _NI_L0_LAC, _NI_L0_GAIN, _NI_L0_FU, _NI_L0_FP = range(1, 8)
# L1 clump/tear (ridged3 × voronoi_billow3, coverage-masked)
(_NI_L1_EN, _NI_L1_AMP, _NI_L1_BIAS, _NI_L1_OCT, _NI_L1_LAC, _NI_L1_GAIN, _NI_L1_FU,
 _NI_L1_FP, _NI_L1_FZ, _NI_L1_COV, _NI_L1_MFU, _NI_L1_MFP, _NI_L1_ROFF,
 _NI_L1_VK) = range(8, 22)
# L2 patchiness (fbm2 on density)
(_NI_L2_EN, _NI_L2_AMP, _NI_L2_OCT, _NI_L2_LAC, _NI_L2_GAIN, _NI_L2_FU,
 _NI_L2_FP) = range(22, 29)
# D2.3 shear advection: T = disk.dynamics.shear_period_M (CKS-13-derived reset
# period); var_preserve = disk.noise.variance_preserve (1/0). ≤ 0 T ⇒ static path.
# dynamism = disk.noise.dynamism — a NON-PHYSICAL viz gain on the CKS-12 §2 shear
# amount (φ′ = φ − dynamism·Ω·a·T). 1.0 = the physical formula (bit-identical);
# >1 exaggerates the differential winding per frame. Same dial spirit as
# disk.doppler_strength: artistic emphasis, not a metric change.
_NI_SHEAR_T = 29
_NI_VAR_PRESERVE = 30
_NI_DYNAMISM = 31
# D2.4 (CKS-12 §3) modulation block — emitted-temperature / inner-&-outer-edge /
# scale-height envelopes (each an advected [0,1] fBm, co-moving with the gas).
# _NI_MOD_EN gates the whole §3 application: 0 ⇒ the D2.3 density-only path,
# bit-identical (constraint 6). The amplitudes are the τ_amp / e_in / e_out / h_amp
# of §3; edge_soft is the smoothstep window width (in r) at each modulated edge.
# Read in BOTH _disk_noise_mod_fields (the fBm freqs) and _disk_emit_cks / the
# trace kernel (the amps, edge_soft — for the worst-case σ_z step cap + band widen).
_NI_MOD_EN = 32
_NI_MOD_OCT = 33
_NI_MOD_LAC = 34
_NI_MOD_GAIN = 35
_NI_MOD_FU = 36
_NI_MOD_FP = 37
_NI_MOD_TEMP_AMP = 38
_NI_MOD_EIN_AMP = 39
_NI_MOD_EOUT_AMP = 40
_NI_MOD_EDGE_SOFT = 41
_NI_MOD_HEIGHT_AMP = 42
# V3.0 (CKS-18) curl-flow domain warp — an in-plane, divergence-free distortion of
# the noise coordinate (u, φ) = the 2-D curl of an sfbm3 scalar potential on the
# (cosφ, sinφ, u) cylinder embedding (seamless across φ=0; ρ_c/k_u may be any real).
# _NI_CURL_EN gates it: 0 ⇒ no warp (bit-identical, constraint 6). Read at the entry
# of BOTH _disk_noise_m (density) and _mod_fbm4 (§3 envelopes) so they warp identically.
_NI_CURL_EN = 43
_NI_CURL_AMP = 44
_NI_CURL_FP = 45    # ρ_c — angular feature density (freq_phi)
_NI_CURL_FU = 46    # k_u — radial feature density (freq_u)
_NI_CURL_OCT = 47
_NI_CURL_LAC = 48
_NI_CURL_GAIN = 49
_NI_CURL_SEED = 50
_NI_CURL_EPS = 51   # central finite-difference step in the (u,φ) chart
# V3.1 (CKS-18 §2) curl-flow advection — the curl clock T_c. > 0 ⇒ ψ becomes
# time-dependent via the §2 dual-phase reset blend (the eddies boil over t_disk);
# ≤ 0 ⇒ the static V3.0 warp bit-for-bit. Independent of the §2 shear clock (B1).
_NI_CURL_FLOWP = 52
# CKS-19 multi-phase media (decoupled ρ_cold absorption). _NI_MP_EN gates it:
# 0 ⇒ ρ_cold ≡ ρ_hot, grey κ ⇒ single-phase march bit-identical (constraint 6).
_NI_MP_EN = 53
_NI_MP_CHI = 54       # χ ∈ [−1,1] dust↔plasma correlation
_NI_MP_AMP = 55       # a_cold — dust log-density gain
_NI_MP_SIGFRAC = 56   # σ_cold / σ_hot — dust slab thickness ratio
_NOISE_N = 57

# CKS-14 RTE source-function march: divide guard on dτ. Below this the source
# function S = emission/dτ is numerically undefined AND physically the optically-
# thin limit where w=1−e^{−dτ}→dτ makes the RTE term equal the legacy emission
# add — so the kernel falls back to the legacy term, with no discontinuity.
_RTE_TAU_EPS = 1e-6

# --- CKS-15 radial deep-shadow-map self-shadow (V1.2; SKILL.md CKS-15, spec §2) --- #
# A 3-D cumulative absorption-optical-depth field τ_shadow[NU, NPHI, NZ] on the
# CKS-12 noise coordinates (u=ln r/r_inner, φ=atan2(y,x), ζ=Δθ/σ_θ). Baked once per
# frame by ``bake_disk_shadow`` from the SHARED ``_disk_density_cks`` (so the shadow
# ρ can never drift from the emission ρ), then trilinearly sampled per primary disk
# sample to dim the emissivity j → j·e^{−strength·τ_s} — VISUALIZATION occlusion
# bookkeeping (a straight radial CKS ray, not a geodesic), flagged like
# ``doppler_strength``; it never touches p_μ/u^μ/g/g⁴/f_PT (CKS-15 governance). The
# grid EXTENTS (u_max=ln(r_outer/r_inner), ζ_max) are baked into module globals at
# setup (same pattern as ``_PT_LUT_R0``) so the per-sample lookup needs no extra
# kernel args; the grid RESOLUTION is the field shape, read inside the kernels. The
# field is ALWAYS allocated by ``_setup_disk_shadow`` so the kernels JIT against a
# live field; ``self_shadow==0`` ⇒ no bake, no lookup (legacy path bit-identical).
disk_shadow_tau: ti.Field = None  # type: ignore[assignment]  (NU,NPHI,NZ) f32
_SHADOW_U_MAX = 1.0
_SHADOW_ZETA_MAX = 3.0

# Layer-stack constants — sourced from noise.py (the CPU twin) so the GPU kernel
# and the CPU reference noise.noise_density_mult can never drift apart.
_NSEED_L0 = noise.NSEED_L0
_NSEED_L1_RIDGE = noise.NSEED_L1_RIDGE
_NSEED_L1_VORO = noise.NSEED_L1_VORO
_NSEED_L1_MASK = noise.NSEED_L1_MASK
_NSEED_L2 = noise.NSEED_L2
_NSEED_DUST = noise.NSEED_DUST  # CKS-19: GPU twin of noise.NSEED_DUST
_NOISE_RIDGE_FEEDBACK = noise.RIDGE_FEEDBACK
_NOISE_MASK_SOFT = noise.MASK_SOFT
_NCYC_PHASE = noise.NCYC_PHASE  # dual-phase reseed offsets (CKS-12 §2, D2.3)
_NCYC_CYCLE = noise.NCYC_CYCLE
# D2.4 §3 modulation-envelope seed offsets (decorrelate n_T/n_e/n_e'/n_h).
_NSEED_MOD_T = noise.NSEED_MOD_T
_NSEED_MOD_EIN = noise.NSEED_MOD_EIN
_NSEED_MOD_EOUT = noise.NSEED_MOD_EOUT
_NSEED_MOD_H = noise.NSEED_MOD_H
_INV_TWO_PI = 1.0 / (2.0 * math.pi)


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
    _setup_disk_noise(cfg)
    _setup_disk_shadow(cfg)
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


def _setup_disk_noise(cfg: dict) -> None:
    """Pack + upload the disk procedural-noise param buffer (D2.2; SKILL.md CKS-12).

    ALWAYS runs (like ``_setup_disk_flux``); the buffer is ~29 floats, so uploading
    it unconditionally lets ``disk.noise.enabled`` (a kernel arg) toggle per render
    with no re-JIT, and lets look-dev re-tune amplitudes/frequencies by re-running
    setup (re-upload) rather than recompiling. Missing keys fall back to the spec §5
    defaults; an absent ``disk.noise`` block leaves every layer disabled (the buffer
    is still uploaded so the kernel JITs against a live field). The values are read
    only when the ``noise_enabled`` kernel arg is 1.
    """
    global disk_noise_params

    d = cfg.get("disk", {}) or {}
    nz = d.get("noise", {}) or {}
    dyn = d.get("dynamics", {}) or {}
    layers = nz.get("layers", {}) or {}
    base = layers.get("base", {}) or {}
    clump = layers.get("clump", {}) or {}
    patch = layers.get("patch", {}) or {}

    buf = np.zeros(_NOISE_N, dtype=np.float32)
    buf[_NI_M_MAX] = float(nz.get("m_max", 2.5))

    # D2.3 shear advection (CKS-12 §2). shear_period_M is CKS-13-derived (resolver);
    # absent (no disk.dynamics block) ⇒ 0 ⇒ the kernel takes the static (D2.2) path,
    # so configs without dynamics stay bit-identical to the static stack.
    buf[_NI_SHEAR_T] = float(dyn.get("shear_period_M", 0.0))
    buf[_NI_VAR_PRESERVE] = 1.0 if nz.get("variance_preserve", True) else 0.0
    # Non-physical viz gain on the shear amount (1.0 = the CKS-12 §2 formula exactly,
    # bit-identical; >1 emphasises the per-frame differential winding).
    buf[_NI_DYNAMISM] = float(nz.get("dynamism", 1.0))

    # L0 — base streaks (fBm)
    buf[_NI_L0_EN] = 1.0 if base.get("enabled", False) else 0.0
    buf[_NI_L0_AMP] = float(base.get("amp", 0.6))
    buf[_NI_L0_OCT] = float(base.get("octaves", 5))
    buf[_NI_L0_LAC] = float(base.get("lacunarity", 2.0))
    buf[_NI_L0_GAIN] = float(base.get("gain", 0.5))
    buf[_NI_L0_FU] = float(base.get("freq_u", 6.0))
    buf[_NI_L0_FP] = float(base.get("freq_phi", 24))

    # L1 — clump/tear (ridged MF × Voronoi billow, coverage-masked)
    buf[_NI_L1_EN] = 1.0 if clump.get("enabled", False) else 0.0
    buf[_NI_L1_AMP] = float(clump.get("amp", 1.2))
    buf[_NI_L1_BIAS] = float(clump.get("bias", 0.35))
    buf[_NI_L1_OCT] = float(clump.get("octaves", 3))
    buf[_NI_L1_LAC] = float(clump.get("lacunarity", 2.0))
    buf[_NI_L1_GAIN] = float(clump.get("gain", 0.5))
    buf[_NI_L1_FU] = float(clump.get("freq_u", 3.0))
    buf[_NI_L1_FP] = float(clump.get("freq_phi", 12))
    buf[_NI_L1_FZ] = float(clump.get("freq_z", 1.0))
    buf[_NI_L1_COV] = float(clump.get("coverage", 0.45))
    buf[_NI_L1_MFU] = float(clump.get("mask_freq_u", 1.0))
    buf[_NI_L1_MFP] = float(clump.get("mask_freq_phi", 3))
    buf[_NI_L1_ROFF] = float(clump.get("ridge_offset", 1.0))
    buf[_NI_L1_VK] = float(clump.get("voronoi_k", 4.0))

    # L2 — patchiness (fBm)
    buf[_NI_L2_EN] = 1.0 if patch.get("enabled", False) else 0.0
    buf[_NI_L2_AMP] = float(patch.get("amp", 0.35))
    buf[_NI_L2_OCT] = float(patch.get("octaves", 2))
    buf[_NI_L2_LAC] = float(patch.get("lacunarity", 2.0))
    buf[_NI_L2_GAIN] = float(patch.get("gain", 0.5))
    buf[_NI_L2_FU] = float(patch.get("freq_u", 1.5))
    buf[_NI_L2_FP] = float(patch.get("freq_phi", 4))

    # D2.4 — §3 temperature / edge / scale-height modulation. enabled:false (or an
    # absent block) ⇒ _NI_MOD_EN = 0 ⇒ the kernel skips §3 entirely (the density-only
    # D2.3 path stays bit-identical). The fBm freqs feed _disk_noise_mod_fields; the
    # amps + edge_soft are read by _disk_emit_cks and the trace kernel (band widen +
    # worst-case σ_z step cap, constraint 4).
    mod = nz.get("modulation", {}) or {}
    buf[_NI_MOD_EN] = 1.0 if mod.get("enabled", False) else 0.0
    buf[_NI_MOD_OCT] = float(mod.get("octaves", 3))
    buf[_NI_MOD_LAC] = float(mod.get("lacunarity", 2.0))
    buf[_NI_MOD_GAIN] = float(mod.get("gain", 0.5))
    buf[_NI_MOD_FU] = float(mod.get("freq_u", 4.0))
    buf[_NI_MOD_FP] = float(mod.get("freq_phi", 16))
    buf[_NI_MOD_TEMP_AMP] = float(mod.get("temp_amp", 0.0))
    buf[_NI_MOD_EIN_AMP] = float(mod.get("edge_in_amp", 0.0))
    buf[_NI_MOD_EOUT_AMP] = float(mod.get("edge_out_amp", 0.0))
    buf[_NI_MOD_EDGE_SOFT] = float(mod.get("edge_softness", 0.0))
    buf[_NI_MOD_HEIGHT_AMP] = float(mod.get("height_amp", 0.0))

    # V3.0 (CKS-18) curl-flow domain warp. enabled:false (or absent) ⇒ _NI_CURL_EN = 0
    # ⇒ the warp is skipped at both stack entries (bit-identical, constraint 6). freqs
    # ρ_c/k_u may be any real (seamlessness is from the cylinder embedding, not a
    # lattice period). All base look dials — nothing derived, so no CKS-13 change.
    curl = nz.get("curl", {}) or {}
    buf[_NI_CURL_EN] = 1.0 if curl.get("enabled", False) else 0.0
    buf[_NI_CURL_AMP] = float(curl.get("amp", 0.0))
    buf[_NI_CURL_FP] = float(curl.get("freq_phi", 3.0))
    buf[_NI_CURL_FU] = float(curl.get("freq_u", 1.0))
    buf[_NI_CURL_OCT] = float(curl.get("octaves", 4))
    buf[_NI_CURL_LAC] = float(curl.get("lacunarity", 2))
    buf[_NI_CURL_GAIN] = float(curl.get("gain", 0.5))
    buf[_NI_CURL_SEED] = float(curl.get("seed", 0))
    buf[_NI_CURL_EPS] = float(curl.get("fd_eps", float(noise.CURL_FD_EPS)))
    # V3.1 (CKS-18 §2) curl-flow clock; ≤ 0 (default / absent) ⇒ static V3.0 warp.
    buf[_NI_CURL_FLOWP] = float(curl.get("flow_period_M", 0.0))

    # CKS-19 multi-phase media. Absent block ⇒ disabled, σ_cold=σ_hot, grey κ ⇒
    # ρ_cold≡ρ_hot ⇒ single-phase march bit-identical (constraint 6). `multiphase`
    # is a sibling of `noise` under `disk` (d), NOT derived ⇒ no CKS-13 change.
    mp = d.get("multiphase", {}) or {}
    mp_enabled = bool(mp.get("enabled", False))
    buf[_NI_MP_EN] = 1.0 if mp_enabled else 0.0
    buf[_NI_MP_CHI] = float(mp.get("dust_correlation", -0.6))
    buf[_NI_MP_AMP] = float(mp.get("dust_amp", 1.0))
    buf[_NI_MP_SIGFRAC] = float(mp.get("dust_sigma_frac", 1.0))
    # Compile-time gate (see _MP_COMPILE): OFF ⇒ the dust branch is not emitted, so
    # the default path keeps its original (bit-identical) JIT. Read via ti.static.
    global _MP_COMPILE
    _MP_COMPILE = mp_enabled

    # CKS-20: compile the scatter branch only when disk.scatter.enabled (see _SCATTER_COMPILE).
    global _SCATTER_COMPILE
    _SCATTER_COMPILE = bool((d.get("scatter", {}) or {}).get("enabled", False))

    disk_noise_params = ti.field(dtype=ti.f32, shape=_NOISE_N)
    disk_noise_params.from_numpy(buf)


def _setup_disk_shadow(cfg: dict) -> None:
    """Allocate the CKS-15 radial deep-shadow-map field + bake its grid extents.

    ALWAYS runs (like ``_setup_disk_noise``): the field is sized from the
    ``disk.volumetric.self_shadow`` grid config so toggling ``enabled`` per render
    needs no re-JIT (the kernel just skips the bake + lookup when off). The grid
    EXTENTS (``u_max = ln(r_outer/r_inner)``, ``ζ_max``) are baked into module
    globals here so the per-sample lookup ``_sample_shadow_tau`` needs no extra
    kernel args (same pattern as ``_PT_LUT_R0``); the grid RESOLUTION is the field
    shape, read inside the bake/lookup. The τ VALUES are filled per frame by
    ``bake_disk_shadow`` (it depends on ``t_disk``), not here — the field is just
    zeroed so a pre-bake render (or the disabled path) reads no shadow.
    """
    global disk_shadow_tau, _SHADOW_U_MAX, _SHADOW_ZETA_MAX

    d = cfg.get("disk", {}) or {}
    vol = d.get("volumetric", {}) or {}
    ss = vol.get("self_shadow", {}) or {}

    nu = int(ss.get("grid_nu", 96))
    nphi = int(ss.get("grid_nphi", 256))
    nz = int(ss.get("grid_nz", 16))
    r_inner = float(d["r_inner"])
    r_outer = float(d["r_outer"])
    # u = ln(r/r_inner) spans [0, u_max]; the slab is ~Gaussian so ±ζ_max·σ covers it.
    _SHADOW_U_MAX = math.log(r_outer / r_inner)
    _SHADOW_ZETA_MAX = float(ss.get("zeta_max", 3.0))

    disk_shadow_tau = ti.field(dtype=ti.f32, shape=(nu, nphi, nz))
    disk_shadow_tau.fill(0.0)


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
        disk_buf = ti.field(dtype=ti.f32, shape=(height, width, 6))
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
def _hg_phase(cos_theta, g):
    """Henyey-Greenstein phase (SKILL.md CKS-20) — GPU twin of disk.hg_phase.

    P = (1−g²)/[4π·denom^{3/2}], denom = 1+g²−2g·cosθ (>0 for |g|<1). Pure optics:
    no p_μ/u^μ/g/g⁴ (constraint 3). g=0 ⇒ 1/4π isotropic.
    """
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return (1.0 - g2) / (4.0 * math.pi * denom * ti.sqrt(denom))


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
def _disk_curl_warp(u, phi, t_disk):
    """In-plane CKS-18 curl warp of ``(u, φ)`` from the ``disk_noise_params`` dials;
    GPU wrapper over :func:`noise.curl_warp_ti`. Returns ``ti.Vector([u', φ'])``.
    Only called when ``_NI_CURL_EN > 0.5`` (the disabled path never touches coords).

    ``t_disk`` + ``_NI_CURL_FLOWP`` drive the CKS-18 §2 curl-flow advection (the warp
    boils over time); ``_NI_CURL_FLOWP ≤ 0`` ⇒ the static V3.0 warp bit-for-bit."""
    return noise.curl_warp_ti(
        u, phi,
        disk_noise_params[_NI_CURL_AMP],
        disk_noise_params[_NI_CURL_FP],
        disk_noise_params[_NI_CURL_FU],
        ti.cast(disk_noise_params[_NI_CURL_OCT], ti.i32),
        ti.cast(disk_noise_params[_NI_CURL_LAC], ti.i32),
        disk_noise_params[_NI_CURL_GAIN],
        ti.cast(disk_noise_params[_NI_CURL_SEED], ti.i32),
        disk_noise_params[_NI_CURL_EPS],
        t_disk,
        disk_noise_params[_NI_CURL_FLOWP],
    )


@ti.func
def _disk_noise_m(u, phi, zeta, seed, t_disk):
    """Unclamped log-density sum ``m = Σ amp·(layer − bias)`` (CKS-12 §3, spec §4);
    GPU twin of :func:`noise._noise_m_stack`. Reads the per-layer dials from
    ``disk_noise_params`` and evaluates the L0/L1/L2 stack at the disk-natural coords
    (``u = ln r/r_inner``, ``φ``, ``ζ``) BEFORE the ``m_max`` clamp and ``exp``. φ
    enters each lattice as ``y = φ/(2π)·freq_phi`` with integer period ``= freq_phi``
    (exact 2π-periodicity, constraint 5). Called once per shear-advection phase by
    :func:`_disk_noise_density_mult`.

    The CKS-18 curl warp (when ``_NI_CURL_EN``) distorts ``(u, φ)`` HERE, on the
    already-sheared per-phase ``φ`` — the eddies freeze into the gas's material frame
    and the §2 shear winds them into filaments; ``ζ`` is untouched (V3.0 in-plane).
    ``t_disk`` evolves the warp itself when ``_NI_CURL_FLOWP > 0`` (CKS-18 §2).
    """
    if disk_noise_params[_NI_CURL_EN] > 0.5:
        w = _disk_curl_warp(u, phi, t_disk)
        u = w[0]
        phi = w[1]
    m = 0.0
    phi01 = phi * _INV_TWO_PI  # φ/(2π) ∈ [−0.5, 0.5]; ×freq_phi → lattice y

    # L0 — base streaks: fBm, features long along the orbit (freq_phi ≪ freq_u).
    if disk_noise_params[_NI_L0_EN] > 0.5:
        fpf = disk_noise_params[_NI_L0_FP]
        fp = ti.cast(fpf, ti.i32)
        n0 = noise.fbm2_ti(
            u * disk_noise_params[_NI_L0_FU],
            phi01 * fpf,
            fp,
            ti.cast(disk_noise_params[_NI_L0_OCT], ti.i32),
            ti.cast(disk_noise_params[_NI_L0_LAC], ti.i32),
            disk_noise_params[_NI_L0_GAIN],
            seed + _NSEED_L0,
        )
        m += disk_noise_params[_NI_L0_AMP] * (n0 - 0.5)

    # L1 — clump/tear: ridged MF × Voronoi billow, gated by a slow coverage mask.
    if disk_noise_params[_NI_L1_EN] > 0.5:
        fpf = disk_noise_params[_NI_L1_FP]
        fp = ti.cast(fpf, ti.i32)
        xu = u * disk_noise_params[_NI_L1_FU]
        yphi = phi01 * fpf
        zz = zeta * disk_noise_params[_NI_L1_FZ]
        ridge = noise.ridged3_ti(
            xu, yphi, zz, fp,
            ti.cast(disk_noise_params[_NI_L1_OCT], ti.i32),
            ti.cast(disk_noise_params[_NI_L1_LAC], ti.i32),
            disk_noise_params[_NI_L1_GAIN],
            disk_noise_params[_NI_L1_ROFF],
            _NOISE_RIDGE_FEEDBACK,
            seed + _NSEED_L1_RIDGE,
        )
        voro = noise.voronoi_billow3_ti(
            xu, yphi, zz, fp, disk_noise_params[_NI_L1_VK], seed + _NSEED_L1_VORO,
        )
        clump = ridge * voro
        # Coverage mask M ∈ [0,1]: slow low-freq fBm, smoothstep-thresholded at
        # (1 − coverage) so clumps appear in patches, not uniformly (spec §4).
        mfpf = disk_noise_params[_NI_L1_MFP]
        mask_raw = noise.fbm2_ti(
            u * disk_noise_params[_NI_L1_MFU],
            phi01 * mfpf,
            ti.cast(mfpf, ti.i32),
            2, 2, 0.5,
            seed + _NSEED_L1_MASK,
        )
        thr = 1.0 - disk_noise_params[_NI_L1_COV]
        t = (mask_raw - (thr - _NOISE_MASK_SOFT)) / (2.0 * _NOISE_MASK_SOFT)
        t = ti.min(ti.max(t, 0.0), 1.0)
        mask = t * t * (3.0 - 2.0 * t)  # smoothstep
        m += disk_noise_params[_NI_L1_AMP] * mask * (clump - disk_noise_params[_NI_L1_BIAS])

    # L2 — patchiness: subtle large-scale fBm breaking the ring symmetry.
    if disk_noise_params[_NI_L2_EN] > 0.5:
        fpf = disk_noise_params[_NI_L2_FP]
        fp = ti.cast(fpf, ti.i32)
        n2 = noise.fbm2_ti(
            u * disk_noise_params[_NI_L2_FU],
            phi01 * fpf,
            fp,
            ti.cast(disk_noise_params[_NI_L2_OCT], ti.i32),
            ti.cast(disk_noise_params[_NI_L2_LAC], ti.i32),
            disk_noise_params[_NI_L2_GAIN],
            seed + _NSEED_L2,
        )
        m += disk_noise_params[_NI_L2_AMP] * (n2 - 0.5)

    return m


@ti.func
def _disk_blended_m(u, phi, zeta, t_disk, omega, seed):
    """Pre-clamp blended modulator m (CKS-12 §2 dual-phase shear + §4 stack),
    BEFORE the ±m_max clamp and exp. GPU twin of noise._advected_m. CKS-19 calls
    it twice (hot seed, dust seed) for the ρ_cold correlation mix."""
    T = disk_noise_params[_NI_SHEAR_T]
    m = 0.0
    if T <= 0.0:
        # Static (D2.2 / bit-identical default): no shear advection, sample at φ.
        # t_disk still drives the CKS-18 §2 curl-flow (own clock, independent of T).
        m = _disk_noise_m(u, phi, zeta, seed, t_disk)
    else:
        s = t_disk / T
        g = disk_noise_params[_NI_DYNAMISM]  # viz gain on the shear amount (1.0 = formula)
        wsq = 0.0
        # Phase k = 0.
        c0 = ti.floor(s)
        a0 = s - c0
        w0 = 1.0 - ti.abs(2.0 * a0 - 1.0)
        seed0 = seed + ti.cast(c0, ti.i32) * _NCYC_CYCLE
        m += w0 * _disk_noise_m(u, phi - g * omega * (a0 * T), zeta, seed0, t_disk)
        wsq += w0 * w0
        # Phase k = 1 (half-period staggered reset).
        ar1 = s + 0.5
        c1 = ti.floor(ar1)
        a1 = ar1 - c1
        w1 = 1.0 - ti.abs(2.0 * a1 - 1.0)
        seed1 = seed + _NCYC_PHASE + ti.cast(c1, ti.i32) * _NCYC_CYCLE
        m += w1 * _disk_noise_m(u, phi - g * omega * (a1 * T), zeta, seed1, t_disk)
        wsq += w1 * w1
        if disk_noise_params[_NI_VAR_PRESERVE] > 0.5 and wsq > 0.0:
            m = m / ti.sqrt(wsq)
    return m


@ti.func
def _disk_cold_mult_from_hot(m_hot, u, phi, zeta, t_disk, omega, seed):
    """CKS-19 cold (dust) modulator given the ALREADY-computed hot modulator
    ``m_hot`` (pre-clamp). ``m_cold = a_cold·(χ·m_hot + √(1−χ²)·m_dust)``, where
    ``m_dust`` is the same dual-phase stack reseeded by ``_NSEED_DUST`` (equal
    variance ⇒ sampled Pearson r = χ). Returns ``exp(clamp(m_cold, ±m_max))``.

    Taking ``m_hot`` as a parameter lets the production caller
    (:func:`_disk_density_cks`) reuse the hot modulator it already evaluated for
    ρ_hot, so the dust path adds only ONE more ``_disk_blended_m`` (the dust seed)
    rather than recomputing the hot one — halving the extra inlined noise tree
    (the JIT-blowup fix; see ``_MP_COMPILE``)."""
    chi = disk_noise_params[_NI_MP_CHI]
    a_cold = disk_noise_params[_NI_MP_AMP]
    m_dust = _disk_blended_m(u, phi, zeta, t_disk, omega, seed + _NSEED_DUST)
    s = ti.sqrt(1.0 - chi * chi)
    m_cold = a_cold * (chi * m_hot + s * m_dust)
    mmax = disk_noise_params[_NI_M_MAX]
    m_cold = ti.min(ti.max(m_cold, -mmax), mmax)
    return ti.exp(m_cold)


@ti.func
def _disk_dust_density_mult(u, phi, zeta, t_disk, omega, seed):
    """CKS-19 cold modulator exp(clamp(a_cold·m_cold)) — self-contained GPU twin of
    noise.dust_density_mult (computes its own m_hot, then mixes). Used by the parity
    test; the production path uses :func:`_disk_cold_mult_from_hot` to share m_hot."""
    m_hot = _disk_blended_m(u, phi, zeta, t_disk, omega, seed)
    return _disk_cold_mult_from_hot(m_hot, u, phi, zeta, t_disk, omega, seed)


@ti.func
def _disk_noise_density_mult(u, phi, zeta, t_disk, omega, seed):
    """Procedural-noise density multiplier (D2.3; SKILL.md CKS-12 §2–3, spec §4).

    GPU twin of :func:`noise.noise_density_mult`. Evaluates the L0/L1/L2 layer stack
    (:func:`_disk_noise_m`) and returns ``exp(clamp(m, ±m_max)) > 0`` — the multiplier
    on the Gaussian vertical density (feeds BOTH emission and absorption, so clumps
    self-shadow). Only called when ``noise_enabled == 1``; the disabled path never
    touches density (constraint 6, bit-identical golden frames).

    **Keplerian shear advection (CKS-12 §2).** With ``T = disk_noise_params[_NI_SHEAR_T]``
    (= ``disk.dynamics.shear_period_M``) > 0 the whole stack is evaluated at two
    staggered reset phases ``φ′_k = φ − Ω(r)·a_k·T`` (``a_k = frac(s + k/2)``,
    ``s = t_disk/T``) and crossfaded with triangle weights ``w_k = 1 − |2a_k − 1|``;
    each phase draws a per-cycle reseed (``c_k = floor(s + k/2)``) so the loop does not
    repeat with period T. ``_NI_VAR_PRESERVE`` divides the blend by ``√(w_0² + w_1²)``
    to kill the mid-crossfade contrast breathing. ``omega`` = Ω(r) (Formula 3) is
    supplied by the caller. ``_NI_DYNAMISM`` is a non-physical viz gain on the shear
    amount (``φ′ = φ − dynamism·Ω·a·T``; 1.0 reproduces the formula bit-for-bit, >1
    exaggerates the differential winding). With ``T ≤ 0`` (no ``disk.dynamics`` block)
    the field is static — sampled directly at ``φ`` — i.e. exactly the D2.2 path.
    """
    m = _disk_blended_m(u, phi, zeta, t_disk, omega, seed)
    mmax = disk_noise_params[_NI_M_MAX]
    m = ti.min(ti.max(m, -mmax), mmax)
    return ti.exp(m)


@ti.func
def _smoothstep_ti(e0, e1, x):
    """Hermite smoothstep, twin of the NumPy ``t*t*(3−2t)`` form used by the CPU
    edge windows. Returns 0 below ``e0``, 1 above ``e1``; ``e1==e0`` ⇒ a hard step."""
    t = 0.0
    if e1 > e0:
        t = ti.min(ti.max((x - e0) / (e1 - e0), 0.0), 1.0)
    else:
        t = 1.0 if x >= e0 else 0.0
    return t * t * (3.0 - 2.0 * t)


@ti.func
def _mod_fbm4(u, phi, seed, t_disk):
    """The four §3 modulation envelopes ``(n_T, n_e_in, n_e_out, n_h)`` at one
    (already-advected) phase — GPU twin of :func:`noise._mod_fbm_stack`. Each is a
    single ``fbm2`` in ``[0, 1]`` over ``(u·freq_u, φ/(2π)·freq_phi)`` with integer
    φ-period ``freq_phi`` (constraint 5), keyed by a distinct seed offset so the
    four envelopes are mutually decorrelated. Returns a ``ti.Vector`` of 4 f32.

    Applies the SAME CKS-18 curl warp as :func:`_disk_noise_m` (when ``_NI_CURL_EN``)
    so the §3 envelopes swirl coherently with the density; ``t_disk`` evolves the warp
    in lockstep when ``_NI_CURL_FLOWP > 0`` (CKS-18 §2)."""
    if disk_noise_params[_NI_CURL_EN] > 0.5:
        w = _disk_curl_warp(u, phi, t_disk)
        u = w[0]
        phi = w[1]
    fpf = disk_noise_params[_NI_MOD_FP]
    fp = ti.cast(fpf, ti.i32)
    x = u * disk_noise_params[_NI_MOD_FU]
    y = phi * _INV_TWO_PI * fpf
    oct_ = ti.cast(disk_noise_params[_NI_MOD_OCT], ti.i32)
    lac = ti.cast(disk_noise_params[_NI_MOD_LAC], ti.i32)
    gain = disk_noise_params[_NI_MOD_GAIN]
    return ti.Vector([
        noise.fbm2_ti(x, y, fp, oct_, lac, gain, seed + _NSEED_MOD_T),
        noise.fbm2_ti(x, y, fp, oct_, lac, gain, seed + _NSEED_MOD_EIN),
        noise.fbm2_ti(x, y, fp, oct_, lac, gain, seed + _NSEED_MOD_EOUT),
        noise.fbm2_ti(x, y, fp, oct_, lac, gain, seed + _NSEED_MOD_H),
    ])


@ti.func
def _disk_noise_mod_fields(u, phi, t_disk, omega, seed):
    """Advected ``[0, 1]`` envelopes ``(n_T, n_e_in, n_e_out, n_h)`` for the CKS-12
    §3 temperature / edge / scale-height modulation — GPU twin of
    :func:`noise.noise_modulation_fields`. Advected with the SAME dual-phase reset +
    ``_NI_DYNAMISM`` gain as the density field (:func:`_disk_noise_density_mult`) so
    the envelopes co-move with the gas. **No** ``variance_preserve`` divide — the
    convex triangle weights (``w_0 + w_1 ≡ 1``) keep each blend in ``[0, 1]`` so
    ``n − ½`` is a bounded ``±½`` modulation. With ``T ≤ 0`` the field is static.
    Only called inside the ``_NI_MOD_EN == 1`` branch of :func:`_disk_emit_cks`."""
    T = disk_noise_params[_NI_SHEAR_T]
    out = ti.Vector([0.5, 0.5, 0.5, 0.5])
    if T <= 0.0:
        # Static shear; t_disk still drives the CKS-18 §2 curl-flow (own clock).
        out = _mod_fbm4(u, phi, seed, t_disk)
    else:
        s = t_disk / T
        g = disk_noise_params[_NI_DYNAMISM]
        c0 = ti.floor(s)
        a0 = s - c0
        w0 = 1.0 - ti.abs(2.0 * a0 - 1.0)
        seed0 = seed + ti.cast(c0, ti.i32) * _NCYC_CYCLE
        v0 = _mod_fbm4(u, phi - g * omega * (a0 * T), seed0, t_disk)
        ar1 = s + 0.5
        c1 = ti.floor(ar1)
        a1 = ar1 - c1
        w1 = 1.0 - ti.abs(2.0 * a1 - 1.0)
        seed1 = seed + _NCYC_PHASE + ti.cast(c1, ti.i32) * _NCYC_CYCLE
        v1 = _mod_fbm4(u, phi - g * omega * (a1 * T), seed1, t_disk)
        out = w0 * v0 + w1 * v1
    return out


@ti.func
def _disk_density_cks(
    x, y, r, dz_ang, sigma_theta, flare_beta, r_inner, r_outer, r_isco,
    noise_enabled, noise_seed, t_disk, a,
):
    """Volumetric mass density ρ at a CKS equatorial-slab point (CKS-12 §3 stack).

    Single source of truth for the density used by BOTH the emission march
    (``_disk_emit_cks``) and the CKS-15 radial deep-shadow bake — extracting it
    here is what keeps the two from drifting. Returns
    ``vec3(density, density_cold, temp_factor)``:

    - ``density`` (= ρ_hot) — Gaussian envelope × §3 noise multiplier × ragged-edge
      window. Feeds EMISSION. enabled==0 ⇒ the bare Gaussian (legacy path
      bit-identical, constraint 6).
    - ``density_cold`` (= ρ_cold, CKS-19) — the decoupled ABSORPTION density: the
      variance-preserving Pearson mix of the hot modulator and a reseeded dust
      stack, on a (possibly thinner) σ_cold = σ_hot·sigfrac slab. Feeds dτ and the
      shadow optical depth. ``_NI_MP_EN==0`` (or noise off) ⇒ ρ_cold ≡ ρ_hot, so
      the grey-κ single-phase march is bit-identical (constraint 6).
    - ``temp_factor`` — the §3 emitted-temperature lump (applied to ``T_emit``
      downstream, BEFORE the g shift — constraint 2). 1.0 unless §3 modulation
      (``_NI_MOD_EN``) is on, so the non-modulated path is unaffected.

    Caller supplies ``sigma_theta = σ0 = theta_half_width · sigma_frac`` (the BASE
    inner-edge width, NOT the widened bounding angle) and the slab
    ``dz_ang = θ − π/2`` so the bake can reuse them off its grid coordinates.

    ``flare_beta`` (CKS-16, V2): radial flare of the scale height,
    ``σ_θ(r) = σ0·(r/r_inner)^β``. ``β = 0`` skips the ``ti.pow`` ⇒ ``σ_eff ≡ σ0``,
    bit-identical to the pre-V2 constant slab (the resolver sets it to 0 unless
    ``disk.volumetric.flare.enabled``). ``β > 0`` thickens the disk outward.
    """
    sigma_eff = sigma_theta
    if flare_beta != 0.0:
        sigma_eff = sigma_theta * ti.pow(r / r_inner, flare_beta)
    density = ti.exp(-0.5 * (dz_ang / sigma_eff) ** 2)
    # CKS-19: cold absorbing phase. Default ρ_cold ≡ ρ_hot (the bare Gaussian when
    # noise is off, the hot density when MP is off) ⇒ grey-κ march bit-identical.
    density_cold = density
    temp_factor = 1.0
    # D2.2 procedural turbulence (SKILL.md CKS-12 §3): multiply the Gaussian
    # density by the noise field (amplitude only — feeds BOTH emission and
    # absorption). enabled==0 skips this entirely ⇒ the legacy path is
    # bit-identical (constraint 6).
    if noise_enabled == 1:
        u_n = ti.log(r / r_inner)
        phi_n = ti.atan2(y, x)
        zeta_n = dz_ang / sigma_eff
        # Ω(r) = 1/(r^{3/2}+a) — Formula 3 (prograde), the shear-advection rate
        # (CKS-12 §2). d/dt atan2(y,x) = Ω co-moves with the CKS-8 gas.
        omega = 1.0 / (r * ti.sqrt(r) + a)
        # Hot modulator, kept pre-clamp so the CKS-19 dust path can reuse it (the
        # JIT-blowup fix; see _MP_COMPILE). exp(clamp(m_hot)) IS _disk_noise_density_mult
        # bit-for-bit, so ρ_hot — and every golden frame — is unchanged.
        m_hot = _disk_blended_m(u_n, phi_n, zeta_n, t_disk, omega, noise_seed)
        mmax = disk_noise_params[_NI_M_MAX]
        dmult = ti.exp(ti.min(ti.max(m_hot, -mmax), mmax))
        gauss = density  # base Gaussian (unmodulated σ); reassigned below
        win = 1.0        # edge window (1.0 = hard cutoff already enforced)
        # D2.4 §3 modulation (temperature / ragged edges / lumpy scale height).
        # _NI_MOD_EN==0 ⇒ skip ⇒ density = gauss·dmult exactly (the D2.3 path,
        # bit-identical — constraint 6).
        if disk_noise_params[_NI_MOD_EN] > 0.5:
            mf = _disk_noise_mod_fields(u_n, phi_n, t_disk, omega, noise_seed)
            # Lumpy scale height: re-evaluate the Gaussian at the modulated σ_θ
            # (the worst-case-thin σ is what the trace step cap guards,
            # constraint 4). n_h = mf[3].
            h_amp = disk_noise_params[_NI_MOD_HEIGHT_AMP]
            sigma_m = sigma_eff * (1.0 + h_amp * (mf[3] - 0.5))
            gauss = ti.exp(-0.5 * (dz_ang / sigma_m) ** 2)
            # Ragged edges: smoothstep windows replace the hard radial cutoffs.
            # r_in_eff ≥ r_isco (zero-torque BC, constraint 3).
            ein = disk_noise_params[_NI_MOD_EIN_AMP]
            eout = disk_noise_params[_NI_MOD_EOUT_AMP]
            soft = disk_noise_params[_NI_MOD_EDGE_SOFT]
            r_in_eff = ti.max(r_inner * (1.0 + ein * (mf[1] - 0.5)), r_isco)
            r_out_eff = r_outer * (1.0 + eout * (mf[2] - 0.5))
            win = _smoothstep_ti(r_in_eff, r_in_eff + soft, r) * (
                1.0 - _smoothstep_ti(r_out_eff - soft, r_out_eff, r)
            )
            # Emitted-temperature lumps (applied to T_emit, pre-g — n_T=mf[0]).
            t_amp = disk_noise_params[_NI_MOD_TEMP_AMP]
            temp_factor = 1.0 + t_amp * (mf[0] - 0.5)
        density = gauss * dmult * win
        # CKS-19: cold absorbing phase. ti.static gate (NOT a runtime _NI_MP_EN
        # branch) so the dust noise tree is emitted ONLY when multiphase is on —
        # otherwise it would compile into every render and blow up the JIT (see
        # _MP_COMPILE). OFF ⇒ this block vanishes ⇒ ρ_cold ≡ ρ_hot, bit-identical.
        density_cold = density
        if ti.static(_MP_COMPILE):
            # Reuse the hot modulator (only the dust seed re-evaluates the stack).
            dmult_cold = _disk_cold_mult_from_hot(
                m_hot, u_n, phi_n, zeta_n, t_disk, omega, noise_seed)
            sigma_cold = sigma_eff * disk_noise_params[_NI_MP_SIGFRAC]
            gauss_cold = ti.exp(-0.5 * (dz_ang / sigma_cold) ** 2)
            density_cold = gauss_cold * dmult_cold * win
    return vec3(density, density_cold, temp_factor)


@ti.func
def _sample_shadow_tau(u_n, phi_n, zeta_n):
    """Trilinear lookup into the CKS-15 deep-shadow-map (φ periodic).

    Returns the cumulative absorption optical depth between ``r_inner`` and the
    sample at noise coords ``(u_n, φ_n, ζ_n)``. ``u`` and ``ζ`` clamp at the grid
    edges (a sample at ``r > r_outer`` reads the outermost column; beyond ±ζ_max the
    rim cell), ``φ`` wraps (the disk is azimuthally periodic — no φ=0 seam). The
    field stores τ at each cell's INNER edge (gas strictly inward of the sample), so
    a cell never shadows itself (see ``bake_disk_shadow``).
    """
    NU = disk_shadow_tau.shape[0]
    NPHI = disk_shadow_tau.shape[1]
    NZ = disk_shadow_tau.shape[2]
    # Continuous bin coordinates (cell centres sit at index + 0.5).
    fu = u_n / _SHADOW_U_MAX * NU - 0.5
    fp = (phi_n + math.pi) / _TWO_PI * NPHI - 0.5
    fz = (zeta_n + _SHADOW_ZETA_MAX) / (2.0 * _SHADOW_ZETA_MAX) * NZ - 0.5
    # u, ζ clamp to the valid cell-centre span [0, N-1]; φ wraps below.
    fu = ti.min(ti.max(fu, 0.0), ti.cast(NU - 1, ti.f32))
    fz = ti.min(ti.max(fz, 0.0), ti.cast(NZ - 1, ti.f32))
    i0 = ti.cast(ti.floor(fu), ti.i32)
    j0 = ti.cast(ti.floor(fp), ti.i32)
    k0 = ti.cast(ti.floor(fz), ti.i32)
    tu = fu - i0
    tp = fp - j0
    tz = fz - k0
    i1 = ti.min(i0 + 1, NU - 1)
    k1 = ti.min(k0 + 1, NZ - 1)
    # φ periodic wrap on both interpolation endpoints (handles j0 = -1 or NPHI-1).
    j0w = (j0 % NPHI + NPHI) % NPHI
    j1w = ((j0 + 1) % NPHI + NPHI) % NPHI
    c00 = disk_shadow_tau[i0, j0w, k0] * (1.0 - tu) + disk_shadow_tau[i1, j0w, k0] * tu
    c01 = disk_shadow_tau[i0, j0w, k1] * (1.0 - tu) + disk_shadow_tau[i1, j0w, k1] * tu
    c10 = disk_shadow_tau[i0, j1w, k0] * (1.0 - tu) + disk_shadow_tau[i1, j1w, k0] * tu
    c11 = disk_shadow_tau[i0, j1w, k1] * (1.0 - tu) + disk_shadow_tau[i1, j1w, k1] * tu
    c0 = c00 * (1.0 - tp) + c10 * tp
    c1 = c01 * (1.0 - tp) + c11 * tp
    return c0 * (1.0 - tz) + c1 * tz


@ti.kernel
def bake_disk_shadow(
    r_inner: float,
    r_outer: float,
    r_isco: float,
    sigma_theta0: float,
    flare_beta: float,
    zeta_max: float,
    max_tau: float,
    absb_c: float,
    noise_enabled: int,
    noise_seed: int,
    t_disk: float,
    a: float,
):
    """Bake the CKS-17 3D inner-edge-ray deep-shadow-map ``τ_shadow[NU,NPHI,NZ]``.

    Generalises the CKS-15 radial column scan to a 3D shadow ray: for each target cell
    ``(i_u, φ, i_z)`` (all three loops parallelized) the ray runs from the illuminator
    at the inner edge IN THE MIDPLANE ``(u=0, ζ=0)`` to the sample ``(u_s, φ, ζ_s)`` at
    fixed ``φ``, with ``ζ(u)=(u/u_s)·ζ_s``. March the STRICTLY-inner radial cells
    ``j<i_u`` (a cell never shadows itself) accumulating the SAME absorption the
    emission march uses — ``κ·ρ·ds`` with ``κ=absb_c`` — but ``ρ`` sampled at the
    TILTED point ``ρ(u_j, φ, ζ_j)`` and ``ds`` the 3D arc length
    ``√((r_j·du)² + ΔZ_j²)``, ``ΔZ_j`` the ray's physical-height change
    ``Z(u)=r·ζ(u)·σ_θ(r)`` over the cell (CKS-16 flared ``σ_θ``). So an off-midplane
    parcel is shadowed by the dense midplane gas between it and the hot inner edge —
    the vertical self-shadow V2's 3D bulk makes physical. Clamped to ``max_tau``.

    Exact CKS-15 reduction on the midplane: at ``ζ_s=0`` the ray is flat (``ζ_j≡0``,
    ``ΔZ_j≡0``), so ``ds=r·du`` and ``ρ`` is the midplane density — term for term the
    old radial column. The radial element keeps the ``dr=r·du`` convention (not an
    endpoint ``ΔR``) precisely so this limit is bit-exact; the vertical leg is added in
    quadrature (zero on the midplane). VISUALIZATION occlusion bookkeeping — a straight
    CKS ray, NOT a geodesic, single inner-edge illuminator, single-scatter (CKS-17
    governance); never touches p_μ/u^μ/g/g⁴/f_PT.

    Density reconstruction (must match ``_disk_emit_cks`` exactly at the same noise
    coords): ``_disk_density_cks`` reads ``(x, y)`` ONLY through ``atan2(y, x)=φ`` and
    otherwise uses the passed ``r`` / ``dz_ang`` / ``σ_θ``, so feeding ``x=cos φ``,
    ``y=sin φ``, ``r=r_inner·e^u`` and ``dz_ang=ζ·σ_θ`` reproduces the identical
    ``ρ(u, φ, ζ; t)`` — including its §2 shear advection and §3 modulation.

    Cost: each target ζ_s tilts its own ray (no shared prefix), so this is O(NU) per
    cell ⇒ O(NU²·NPHI·NZ) — ~NU/2× the CKS-15 evals, parallel over all cells.
    """
    NU = disk_shadow_tau.shape[0]
    NPHI = disk_shadow_tau.shape[1]
    NZ = disk_shadow_tau.shape[2]
    u_max = ti.log(r_outer / r_inner)
    du = u_max / NU
    dzeta = 2.0 * zeta_max / NZ
    for i_u, i_phi, i_z in ti.ndrange(NU, NPHI, NZ):
        phi = -math.pi + (i_phi + 0.5) * (_TWO_PI / NPHI)
        zeta_s = -zeta_max + (i_z + 0.5) * dzeta
        u_s = (i_u + 0.5) * du
        x = ti.cos(phi)
        y = ti.sin(phi)
        # 3D inner-edge ray: accumulate τ from gas STRICTLY inward of this cell along
        # the tilted line to (u_s, ζ_s). i_u==0 ⇒ empty loop ⇒ τ=0 (inner edge unshadowed,
        # matching CKS-15). ζ_s==0 ⇒ ΔZ≡0, ds≡r·du, ρ at midplane ⇒ exactly CKS-15.
        tau = 0.0
        for j in range(i_u):
            u_c = (j + 0.5) * du
            r_c = r_inner * ti.exp(u_c)
            # Ray ζ at this radius (linear from the midplane illuminator to the sample).
            zeta_c = (u_c / u_s) * zeta_s
            # σ_θ(r) flared (CKS-16); β=0 ⇒ σ_eff ≡ σ0 (no ti.pow) ⇒ midplane bit-exact.
            sigma_c = sigma_theta0
            if flare_beta != 0.0:
                sigma_c = sigma_theta0 * ti.pow(r_c / r_inner, flare_beta)
            dz_ang = zeta_c * sigma_c
            dens = _disk_density_cks(
                x, y, r_c, dz_ang, sigma_theta0, flare_beta, r_inner, r_outer, r_isco,
                noise_enabled, noise_seed, t_disk, a,
            )[1]   # CKS-19: τ ≡ ∫κ·ρ_cold (OFF ⇒ [1]==[0] ⇒ shadow map unchanged)
            # 3D arc length over the cell: radial dr=r_c·du (CKS-15 convention, keeps the
            # midplane reduction bit-exact) + the ray's physical-height change ΔZ in
            # quadrature. Z(u)=r(u)·ζ(u)·σ_θ(r), endpoints at u_c±½du.
            u_lo = u_c - 0.5 * du
            u_hi = u_c + 0.5 * du
            r_lo = r_inner * ti.exp(u_lo)
            r_hi = r_inner * ti.exp(u_hi)
            sig_lo = sigma_theta0
            sig_hi = sigma_theta0
            if flare_beta != 0.0:
                sig_lo = sigma_theta0 * ti.pow(r_lo / r_inner, flare_beta)
                sig_hi = sigma_theta0 * ti.pow(r_hi / r_inner, flare_beta)
            z_lo = r_lo * ((u_lo / u_s) * zeta_s) * sig_lo
            z_hi = r_hi * ((u_hi / u_s) * zeta_s) * sig_hi
            d_z = z_hi - z_lo
            dr = r_c * du
            ds = ti.sqrt(dr * dr + d_z * d_z)
            tau += absb_c * dens * ds
        disk_shadow_tau[i_u, i_phi, i_z] = ti.min(tau, max_tau)


@ti.func
def _disk_emit_cks(
    x, y, z, p_cov, a, r_inner, r_outer, r_isco, theta_half_bound, sigma_theta0, flare_beta,
    T_0, emis_c, absb_c, ds,
    disk_model, doppler_strength, noise_enabled, noise_seed, t_disk, self_shadow, shadow_strength,
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
    if (ti.abs(dz_ang) < theta_half_bound) and (r >= r_inner) and (r <= r_outer):
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
                # CKS-16 flared scale height: σ_eff = σ0·(r/r_inner)^β (β=0 ⇒ σ0,
                # bit-identical). σ_eff is only needed locally for the CKS-15 shadow
                # ζ lookup below; the density func recomputes the same σ_eff from
                # (σ0, β) internally, so the Gaussian/noise stack stays single-source.
                sigma_eff = sigma_theta0
                if flare_beta != 0.0:
                    sigma_eff = sigma_theta0 * ti.pow(r / r_inner, flare_beta)
                # Density (Gaussian × §3 noise × ragged-edge window) and the §3
                # temperature lump come from the shared CKS-12 stack so the
                # emission march and the CKS-15 shadow bake can't drift. The
                # noise/modulation gating lives inside; enabled==0 ⇒ bare
                # Gaussian (legacy bit-identical).
                dens_tf = _disk_density_cks(
                    x, y, r, dz_ang, sigma_theta0, flare_beta, r_inner, r_outer, r_isco,
                    noise_enabled, noise_seed, t_disk, a,
                )
                density = dens_tf[0]          # ρ_hot — drives emission (CKS-19)
                density_cold = dens_tf[1]     # ρ_cold — drives absorption/dτ (CKS-19)
                # temp_factor carries the §3 emitted-temperature modulation forward
                # to T_emit below (BEFORE the g shift — constraint 2). 1.0 = identity.
                # Index [2]: the vec3 middle slot [1] now carries ρ_cold (CKS-19).
                temp_factor = dens_tf[2]
                g4 = g_eff * g_eff * g_eff * g_eff  # Formula 9 (3D volume: g⁴)

                # CKS-15 radial self-shadow (VISUALIZATION). Dim the EMISSIVITY j by
                # the absorption between r_inner and this sample (the deep-shadow-map
                # baked along the SAME density), BEFORE it becomes the source
                # function S — so a shadowed clump reads dark, not just dim, and
                # composes with CKS-14 (S·e^{−τ_s}). dτ (κ) is NOT attenuated: the
                # gas still occludes regardless of how lit it is. self_shadow==0 ⇒
                # shadow_atten ≡ 1 (legacy path bit-identical). It multiplies the
                # emission AMPLITUDE only — never p_μ/u^μ/g/g⁴/f_PT (CKS-15 / CKS-12
                # constraint 1); the straight-radial shadow ray is a viz approx,
                # flagged like doppler_strength.
                shadow_atten = 1.0
                if self_shadow == 1:
                    u_n = ti.log(r / r_inner)
                    phi_n = ti.atan2(y, x)
                    zeta_n = dz_ang / sigma_eff
                    tau_s = _sample_shadow_tau(u_n, phi_n, zeta_n)
                    shadow_atten = ti.exp(-shadow_strength * tau_s)

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

                # CKS-15: apply the self-shadow attenuation to the emissivity in both
                # radial-profile branches (shadow_atten ≡ 1 when self_shadow==0, so
                # bit-identical). κ/dτ below stays untouched.
                emission *= shadow_atten

                # §3 emitted-temperature modulation — applied to T_emit BEFORE the
                # g_eff shift (constraint 2: chroma(g_eff·T_emit) keeps g⁴-not-g⁸).
                # temp_factor == 1.0 unless _NI_MOD_EN is set, so bit-identical otherwise.
                T_emit *= temp_factor

                if emit == 1:
                    chroma = _blackbody_rgb(g_eff * T_emit)
                    out = vec4(
                        emission * chroma[0],
                        emission * chroma[1],
                        emission * chroma[2],
                        absb_c * density_cold * ds,   # CKS-19: κ·ρ_cold (grey κ)
                    )
    return out


@ti.func
def _disk_scatter_cks(
    x, y, z, cx, cy, cz, a, r_inner, r_outer, r_isco, theta_half_bound,
    sigma_theta0, flare_beta, noise_enabled, noise_seed, t_disk,
    self_shadow, shadow_strength, absb_c, ds, albedo, hg_g, src_rgb,
):
    """CKS-20 single-scatter source at one CKS sample → vec4(J_scat·ds RGB, σ_s·ρ_cold·ds).

    Single-scattering from the hot inner edge (the dominant illuminant):
        σ_s        = albedo · absb_c                      # ϖ·κ, grey (Decision 3)
        ρ_cold     = _disk_density_cks(...)[1]            # CKS-19 cold absorber
        ŝ_src      = normalize(x − x_inner), x_inner = (r_inner·cosφ, r_inner·sinφ, 0)
        ŝ_view     = normalize(x_cam − x)
        cosθ_s     = ŝ_src·ŝ_view                          # straight CKS rays (constraint 3)
        e^{−τ_src} = exp(−shadow_strength·τ_shadow)        # CKS-17 deep-shadow-map (=shadow_atten)
        J_scat·ds  = σ_s·ρ_cold·_hg_phase(cosθ_s, hg_g)·src_rgb·e^{−τ_src}·ds

    Returns vec4(J_r, J_g, J_b, σ_s·ρ_cold·ds). The caller adds σ_s·ρ_cold·ds to the
    grey extinction (so scattering removes forward light — constraint 2) and adds
    T⃗⊙(J·ds) to disk_col. Outside the slab band ⇒ zeros. Pure optics: no p_μ/u^μ/g/g⁴.
    Only compiled when _SCATTER_COMPILE (caller gates with ti.static); albedo=0 ⇒ zeros.
    """
    out = vec4(0.0, 0.0, 0.0, 0.0)
    r = _kerr_radius(x, y, z, a)
    cos_th = ti.min(ti.max(z / r, -1.0), 1.0)
    th = ti.acos(cos_th)
    dz_ang = th - 0.5 * math.pi
    if (ti.abs(dz_ang) < theta_half_bound) and (r >= r_inner) and (r <= r_outer):
        sigma_eff = sigma_theta0
        if flare_beta != 0.0:
            sigma_eff = sigma_theta0 * ti.pow(r / r_inner, flare_beta)
        dens = _disk_density_cks(
            x, y, r, dz_ang, sigma_theta0, flare_beta, r_inner, r_outer, r_isco,
            noise_enabled, noise_seed, t_disk, a,
        )
        rho_cold = dens[1]
        # e^{−τ_src}: the CKS-17 inner-edge-ray shadow (graceful: self_shadow==0 ⇒ 1).
        atten = 1.0
        if self_shadow == 1:
            u_n = ti.log(r / r_inner)
            phi_n = ti.atan2(y, x)
            zeta_n = dz_ang / sigma_eff
            atten = ti.exp(-shadow_strength * _sample_shadow_tau(u_n, phi_n, zeta_n))
        # Straight-CKS-ray scattering geometry (Decision 5).
        phi = ti.atan2(y, x)
        sx = x - r_inner * ti.cos(phi)
        sy = y - r_inner * ti.sin(phi)
        sz = z
        inv_s = 1.0 / ti.max(ti.sqrt(sx * sx + sy * sy + sz * sz), 1e-9)
        sx *= inv_s; sy *= inv_s; sz *= inv_s
        vx = cx - x
        vy = cy - y
        vz = cz - z
        inv_v = 1.0 / ti.max(ti.sqrt(vx * vx + vy * vy + vz * vz), 1e-9)
        vx *= inv_v; vy *= inv_v; vz *= inv_v
        cos_s = sx * vx + sy * vy + sz * vz
        phase = _hg_phase(cos_s, hg_g)
        sigma_s = albedo * absb_c                  # σ_s = ϖ·κ (grey)
        sigma_dtau = sigma_s * rho_cold * ds
        j = sigma_dtau * phase * atten             # σ_s·ρ_cold·P·e^{−τ_src}·ds (scalar)
        out = vec4(j * src_rgb[0], j * src_rgb[1], j * src_rgb[2], sigma_dtau)
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
    r_isco: float,
    theta_half_bound: float,
    bound_sin_half: float,
    sigma_theta0: float,
    flare_beta: float,
    T_0: float,
    emis_c: float,
    absb_c: float,
    ext_r: float,
    ext_g: float,
    ext_b: float,
    disk_model: int,
    doppler_strength: float,
    max_step_vfrac: float,
    noise_enabled: int,
    noise_seed: int,
    t_disk: float,
    source_function: int,
    self_shadow: int,
    shadow_strength: float,
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
        # CKS-19 Task 7: per-channel transmittance T⃗. dτ⃗ = κ⃗·ρ_cold·ds with
        # κ⃗ = absb_c·extinction_rgb; κ_B>κ_R reddens light surviving through dust.
        # Grey extinction (1,1,1) ⇒ all three components stay equal at every step ⇒
        # the scalar single-phase march is reproduced bit-for-bit (constraint 6).
        transm = vec3(1.0, 1.0, 1.0)
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

                # D2.4 §3 ragged edges (CKS-12): when modulation is on the emitting
                # band can bulge OUT to r_outer·(1+e_out/2)+edge_soft and the inner
                # edge can recede IN to r_isco (r_in_eff ≥ r_isco, constraint 3), so
                # the geometric early-out must use the WIDEST possible band or a ray
                # would skip the soft falloff region. _NI_MOD_EN==0 ⇒ r_lo/r_hi stay
                # r_inner/r_outer ⇒ bit-identical to D2.3.
                r_lo = r_inner
                r_hi = r_outer
                if disk_noise_params[_NI_MOD_EN] > 0.5:
                    r_lo = r_isco
                    r_hi = r_outer * (1.0 + 0.5 * disk_noise_params[_NI_MOD_EOUT_AMP]) \
                        + disk_noise_params[_NI_MOD_EDGE_SOFT]
                in_band = (
                    disk_enabled == 1
                    and ti.abs(z / r) < bound_sin_half
                    and r >= r_lo
                    and r <= r_hi
                )
                if in_band:
                    # Disk-thickness step cap. The base rule sizes h only by distance
                    # to the horizon and is blind to the disk's vertical extent, so a
                    # ray crossing the equatorial plane steeply can stride over the
                    # thin emitting layer — under-sampling the Gaussian density
                    # (Formula 9) into a moiré band on the disk face. Limit the per-
                    # step VERTICAL displacement |dz/dλ|·h to a fraction of the local
                    # scale height σ_z = r·σ0 so the slab is resolved. Only
                    # bites for steep crossings (large |dz/dλ|); near-in-plane / edge-
                    # on grazers (dz/dλ→0) keep the full radial step, so the cap adds
                    # no steps there and cannot push those rays into the max_steps cap.
                    # CKS-16: σ0 (NOT the flared σ_θ(r)) is intentional — flare only
                    # THICKENS the slab outward, so the inner edge σ0 is the thinnest
                    # (worst) case; capping on it is conservative at every radius.
                    sigma_z = r * sigma_theta0
                    # §3 lumpy scale height: the cap must resolve the WORST-CASE (thinnest)
                    # modulated σ, σ_z·(1 − h_amp/2), or the face-on moiré returns where a
                    # lump thins below the step (CKS-12 constraint 4). h_amp=0 ⇒ unchanged.
                    if disk_noise_params[_NI_MOD_EN] > 0.5:
                        sigma_z = sigma_z * (1.0 - 0.5 * disk_noise_params[_NI_MOD_HEIGHT_AMP])
                    # CKS-19 constraint 3: the cold slab can be thinner (σ_cold = σ_hot·sigfrac).
                    # Cap on the thinnest of the two so absorption lanes are resolved too.
                    # OFF (sigfrac defaults 1.0) ⇒ no narrowing ⇒ bit-identical.
                    if disk_noise_params[_NI_MP_EN] > 0.5:
                        sf = disk_noise_params[_NI_MP_SIGFRAC]
                        if sf < 1.0:
                            sigma_z = sigma_z * sf
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
                        r_isco,
                        theta_half_bound,
                        sigma_theta0,
                        flare_beta,
                        T_0,
                        emis_c,
                        absb_c,
                        local_h,
                        disk_model,
                        doppler_strength,
                        noise_enabled,
                        noise_seed,
                        t_disk,
                        self_shadow,
                        shadow_strength,
                    )
                    # CKS-14 radiative-transfer source-function march. The kernel
                    # still returns (emission=j·ds, dτ=κ·ds); the term ACTUALLY
                    # added this step is `added`:
                    #   legacy (source_function==0 or dτ≈0): added = emission, i.e.
                    #     dI = j·ds → disk_col += T·emission (bit-identical fallback).
                    #   RTE  (source_function==1, dτ>ε): added = w·S with
                    #     S = emission/dτ = (emis_c/absb_c)·[f_PT]·g_eff⁴·chroma
                    #     (ρ and ds cancel) and w = 1−e^{−dτ}.
                    # SAME continuum integral as legacy (transm·j·ds = transm·S·dτ);
                    # CKS-14 is just the EXACT per-step quadrature (legacy is the
                    # left-endpoint rectangle, which over-counts thick steps). Thin
                    # limit w→dτ ⇒ w·S→emission (→ legacy to O(dτ²)). Its real payoff
                    # is materialising S so CKS-15 self-shadow can dim it (S·e^{−τ_s}).
                    # g-bookkeeping unchanged: S carries g_eff⁴·chroma exactly once.
                    # CKS-19 Task 7: ev[3] is the GREY base dτ = absb_c·ρ_cold·ds; the
                    # per-channel optical depth is dτ⃗ = base·extinction_rgb. The CKS-14
                    # source-function factor f = (1−e^{−dτ})/dτ becomes per-channel (S⃗
                    # differs by channel only through dτ⃗; the emission j is unchanged).
                    # Grey extinction ⇒ dtau_v ≡ base ⇒ f⃗ ≡ scalar w ⇒ bit-identical.
                    dtau = ev[3]
                    dtau_v = vec3(dtau * ext_r, dtau * ext_g, dtau * ext_b)
                    added = vec3(ev[0], ev[1], ev[2])
                    if source_function == 1 and dtau > _RTE_TAU_EPS:
                        f = vec3(1.0, 1.0, 1.0)
                        for c in ti.static(range(3)):
                            if dtau_v[c] > _RTE_TAU_EPS:
                                f[c] = (1.0 - ti.exp(-dtau_v[c])) / dtau_v[c]
                        added = vec3(ev[0] * f[0], ev[1] * f[1], ev[2] * f[2])
                    disk_col += vec3(transm[0] * added[0], transm[1] * added[1], transm[2] * added[2])
                    # depth / total_emission key off the radiance actually contributed,
                    # so the transmittance-weighted Z stays consistent with the march.
                    # transm[0] (any channel; equal in grey) keeps the depth proxy
                    # bit-identical to the pre-Task-7 scalar march.
                    contribution = transm[0] * (added[0] + added[1] + added[2])
                    weighted_depth += ray_length * contribution
                    total_emission += contribution
                    transm[0] *= ti.exp(-dtau_v[0])
                    transm[1] *= ti.exp(-dtau_v[1])
                    transm[2] *= ti.exp(-dtau_v[2])

                s = _rk4_step_k1(s, a, local_h, k1)
                ray_length += local_h
            step += 1

        exit_buf[py, px, 0] = cos_exit
        exit_buf[py, px, 1] = phi_exit
        exit_buf[py, px, 2] = ti.cast(out_p, ti.f32)
        disk_buf[py, px, 0] = disk_col[0]
        disk_buf[py, px, 1] = disk_col[1]
        disk_buf[py, px, 2] = disk_col[2]
        disk_buf[py, px, 3] = transm[0]
        disk_buf[py, px, 4] = transm[1]
        disk_buf[py, px, 5] = transm[2]
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
        # CKS-19 Task 7: per-channel transmittance reddens the background seen through
        # cold dust (T⃗ ⊙ bg). Grey extinction ⇒ all three equal ⇒ scalar transm·bg.
        transm = vec3(disk_buf[py, px, 3], disk_buf[py, px, 4], disk_buf[py, px, 5])
        col = disk_col + vec3(transm[0] * bg[0], transm[1] * bg[1], transm[2] * bg[2])
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
    t_disk: float = 0.0,
):
    """Render one beauty frame (Pipe A + Pipe B) for a camera_matrix.json entry.

    ``cam_frame`` carries the Blender camera in **world Cartesian** coordinates
    (``pos``/``fwd``/``up``/``right`` and a vertical ``fov`` in radians, per
    ``src/blender/export_camera.py``). Under CKS the world Cartesian frame **is**
    the coordinate frame (spin axis = +z), so the camera position and basis are
    used directly — no Boyer-Lindquist embedding, no (r̂,θ̂,φ̂) projection. The
    supplied basis is re-orthonormalized for numerical safety.

    ``t_disk`` is the disk animation time in geometric M (D2.3 shear advection,
    CKS-12 §2); callers pass ``frame_index / render.fps × disk.dynamics.time_scale``
    (the CKS-13-derived ``time_scale``). It only matters when ``disk.noise.enabled``
    and a ``disk.dynamics`` block are present (else the noise is static); ``0.0`` is
    the phase-0 frame.

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
    # CKS-16 (V2) flared scale height. σ0 = theta_half_width·vertical_sigma_frac is
    # the BASE inner-edge width (anchor); ``flare_beta`` and the widened bounding
    # half-angle ``theta_half_bound`` are CKS-13-derived (resolve_config) — both fall
    # back to the no-flare values so an unresolved/legacy config is bit-identical.
    sigma_theta0 = float(d["theta_half_width"]) * float(d["vertical_sigma_frac"])
    flare_beta = float(d.get("flare_beta", 0.0))
    theta_half_bound = float(d.get("theta_half_bound", d["theta_half_width"]))
    # Disk-slab |cosθ| early-out bound, DERIVED from the (possibly flare-widened)
    # bounding half-angle: |θ−π/2| < θ_bound ⇔ |cosθ| < sin(θ_bound), exactly the
    # slab test re-checked inside ``_disk_emit_cks``.
    bound_sin_half = math.sin(theta_half_bound)
    # D1 disk radial-profile selector: 0 = simple T₀·(6/r)^0.75 (Decision-B default,
    # golden frames); 1 = Page-Thorne f_PT(r) LUT (SKILL.md CKS-11). The _PT_LUT_*
    # index scalars are baked at JIT from _setup_disk_flux (always run by
    # setup_renderer first, same as _MAX_LOD), so the flag toggles with no re-JIT.
    disk_model = 1 if str(d.get("temperature_model", "simple")) == "page_thorne" else 0
    # Visualization knob (NOT physics): g_eff = g^s exponent on the CKS-9 shift.
    # 1.0 = full physics (default, golden frames); 0.0 = Doppler/redshift off
    # (Interstellar-style symmetric disk). Runtime kernel arg — no re-JIT.
    doppler_strength = float(d.get("doppler_strength", 1.0))
    # D2.2 procedural turbulence (SKILL.md CKS-12): runtime kernel args so the flag
    # toggles per render with no re-JIT; enabled==0 takes a branch bit-identical to
    # the legacy kernel (golden frames untouched). The per-layer dials live in the
    # uploaded ``disk_noise_params`` buffer (_setup_disk_noise), not kernel args, so
    # look-dev re-tuning re-uploads instead of recompiling.
    nz = d.get("noise", {}) or {}
    noise_enabled = 1 if nz.get("enabled", False) else 0
    noise_seed = int(nz.get("seed", 1234))
    # V1 CKS-14 volumetric RTE source-function march (SKILL.md CKS-14). Runtime
    # kernel arg, no re-JIT; false ⇒ legacy ``color += T·emission`` (golden frames
    # bit-identical). true ⇒ thick gas converges to a bright emitting surface
    # (brightness ~ emission_coeff/absorption_coeff) instead of reading black.
    vol = d.get("volumetric", {}) or {}
    source_function = 1 if vol.get("source_function", False) else 0
    # V1.2 CKS-15 radial deep-shadow-map self-shadow (SKILL.md CKS-15). enabled:false
    # ⇒ no bake, no lookup (golden frames bit-identical). When on, the τ_shadow field
    # is re-baked per frame (it tracks the shear-advected density at t_disk) before
    # the trace; the per-sample lookup then dims emissivity by exp(−strength·τ_s).
    ss = vol.get("self_shadow", {}) or {}
    self_shadow = 1 if ss.get("enabled", False) else 0
    shadow_strength = float(ss.get("strength", 1.0))
    r_isco_cfg = float(cfg.get("black_hole", {}).get("r_isco", d["r_inner"]))
    # V1.3 LOD gate (spec §3.1): the per-frame bake + per-sample lookup are close-up
    # cost. When the camera sits beyond ``self_shadow.lod_max_camera_radius`` (the
    # disk subtends a small solid angle — wide/mid Gargantua), drop the self-shadow
    # so those frames pay ZERO added cost and ride the cheap legacy march. ``<= 0``
    # (default) ⇒ no gate, so the default look is unchanged. Only the (expensive,
    # bake-bearing) self-shadow is gated; the source-function march is a cheap
    # march-loop reinterpretation with no bake, so it is left always-honored to avoid
    # a look-pop between a wide and a close shot of the same scene.
    lod_max = float(ss.get("lod_max_camera_radius", 0.0))
    if self_shadow == 1 and lod_max > 0.0 and float(np.linalg.norm(pos)) > lod_max:
        self_shadow = 0
    if self_shadow == 1:
        bake_disk_shadow(
            float(d["r_inner"]),
            float(d["r_outer"]),
            r_isco_cfg,
            sigma_theta0,
            flare_beta,
            float(ss.get("zeta_max", 3.0)),
            float(ss.get("max_tau", 8.0)),
            float(d["absorption_coeff"]),
            noise_enabled,
            noise_seed,
            float(t_disk),
            a,
        )

    # CKS-19 Task 7: chromatic extinction κ⃗ = absb_c·extinction_rgb. Optional list
    # [kR,kG,kB]; absent/grey [1,1,1] ⇒ neutral, scalar-march bit-identical. κ_B>κ_R
    # reddens dust lanes (astrophysical reddening). A scalar value broadcasts to grey.
    ext_cfg = d.get("extinction_rgb", [1.0, 1.0, 1.0])
    if not isinstance(ext_cfg, (list, tuple)):
        ext_cfg = [ext_cfg, ext_cfg, ext_cfg]
    ext_rgb = (float(ext_cfg[0]), float(ext_cfg[1]), float(ext_cfg[2]))

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
        # r_isco (CKS-13-derived) — the zero-torque floor the §3 ragged inner edge
        # may recede to but never below (constraint 3). Falls back to r_inner.
        float(cfg.get("black_hole", {}).get("r_isco", d["r_inner"])),
        theta_half_bound,
        bound_sin_half,
        sigma_theta0,
        flare_beta,
        float(d["T_0"]),
        float(d["emission_coeff"]),
        float(d["absorption_coeff"]),
        # CKS-19 Task 7: per-channel extinction multiplier κ⃗ = absb_c·extinction_rgb.
        # Grey default [1,1,1] ⇒ neutral darkening, bit-identical to the scalar march.
        ext_rgb[0],
        ext_rgb[1],
        ext_rgb[2],
        disk_model,
        doppler_strength,
        float(d.get("max_step_vfrac", 0.5)),
        noise_enabled,
        noise_seed,
        float(t_disk),
        source_function,
        self_shadow,
        shadow_strength,
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
    t_disk: float = 0.0,
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
            cfg, cam_frame, width, height, with_disk, lod_enabled, return_depth, t_disk
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
        # NOTE (D2.5): the disk noise phase ``t_disk`` is held fixed across the shutter
        # here (same value for every sub-frame). Strictly the shear pattern should be
        # jittered alongside ``dphi`` (the rotate-the-camera blur trick assumes an
        # axisymmetric disk, which the noise breaks); per-sub-frame ``t_disk`` jitter
        # is the D2.5 motion-blur task. The error is bounded (the shutter is a small
        # fraction of a frame) and motion blur is off in the default/test pipeline.
        out = render_beauty_frame(
            cfg, jf, width, height, with_disk, lod_enabled, return_depth, t_disk
        )
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


def tonemap(
    hdr: np.ndarray,
    exposure: float,
    gamma: float,
    saturation: float = 1.0,
    tint=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """Reinhard tonemap + gamma, → uint8 (matches scripts/thumb.py).

    ``saturation`` / ``tint`` are a NON-PHYSICAL color grade (VISUALIZATION class,
    like ``disk.doppler_strength``) applied in linear HDR *before* the Reinhard
    compressor — they push the art-directed warm-amber look that the Formula 9
    blackbody chromaticity (intentionally desaturated) cannot reach on its own.
    They NEVER touch the rendered radiance / physics; ``saturation == 1.0`` and
    ``tint == (1,1,1)`` reproduce the ungraded tonemap bit-for-bit.

    - ``saturation``: ``rgb' = luma + saturation·(rgb − luma)`` about the Rec.709
      luminance (1 = unchanged, >1 richer, 0 = grayscale).
    - ``tint``: per-channel linear gain ``(r,g,b)`` (warm amber ≈ ``(1.15,1.0,0.8)``).
    """
    img = hdr * exposure
    # --- VISUALIZATION color grade (identity at defaults ⇒ bit-identical) ---
    if saturation != 1.0:
        luma = (img * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1, keepdims=True)
        img = luma + saturation * (img - luma)
        img = np.maximum(img, 0.0)  # saturation<1 can't create negatives, >1 can on edges
    tint = np.asarray(tint, dtype=img.dtype)
    if not np.array_equal(tint, (1.0, 1.0, 1.0)):
        img = img * tint
    # ----------------------------------------------------------------------
    img = img / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / gamma)
    return (img * 255.0 + 0.5).astype(np.uint8)
