"""Thumbnail preview renderer for the Kerr black hole — Cartesian Kerr-Schild.

Traces one null geodesic per pixel through Kerr spacetime and produces a small
PNG. Without ``--disk`` it renders Pipe A only (shadow + photon ring + gradient
background) to verify the lensing geometry. With ``--disk`` it composites the
Pipe B volumetric accretion disk on top.

This is the CPU reference twin of ``renderer.taichi_renderer``: it drives the
same Cartesian Kerr-Schild (CKS) physics through the pure-NumPy
``renderer.geodesic`` / ``renderer.disk`` so a render can be sanity-checked
without a GPU. All physics follows ``skills/kerr-physics/SKILL.md`` PART II
verbatim:

  * Formula CKS-5/6 — Hamiltonian null geodesic + capture/escape (``renderer.geodesic``)
  * Formula CKS-7   — ZAMO observer + projected-ray photon init (``renderer.geodesic``)
  * Formula CKS-8   — equatorial gas 4-velocity (``renderer.disk``)
  * Formula CKS-9   — g-factor (``renderer.disk``)
  * Formula 9       — blackbody emission (``renderer.disk`` + march_disk below)

All numerical parameters come from ``configs/render.yaml`` (project rule: no
hardcoded values in source).

Usage
-----
    uv run python scripts/thumb.py --res 256 --frame 0           # Pipe A
    uv run python scripts/thumb.py --res 256 --frame 0 --disk    # + Pipe B disk
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Make ``renderer`` importable when this script is run directly (src layout).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from renderer.disk import (  # noqa: E402
    blackbody_rgb,
    g_factor,
    gas_four_velocity_cks,
)
from renderer.geodesic import (  # noqa: E402
    _horizon_radius,
    integrate_null_geodesic,
    make_null_initial_conditions,
)
from renderer.kerr_params import resolve_config  # noqa: E402
from renderer.metric import kerr_radius  # noqa: E402
from renderer.noise import noise_density_mult, noise_modulation_fields  # noqa: E402

# CKS Cartesian coordinate index order (matches renderer.metric / geodesic).
T, X, Y, Z = 0, 1, 2, 3

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "render.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Inject spin/extent-derived parameters (r_isco, r_inner, T_0, dynamics —
    # Formula CKS-13); the YAML stores base parameters only. Taichi-free.
    return resolve_config(cfg)


# --------------------------------------------------------------------------- #
# Camera (CKS Cartesian) — matches renderer.taichi_renderer._camera_basis
# --------------------------------------------------------------------------- #
def camera_basis(
    pos: np.ndarray, world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (fwd, right, up) looking from ``pos`` toward the origin.

    All in world Cartesian = CKS. ``fwd = -normalize(pos)``,
    ``right = normalize(fwd × world_up)``, ``up = right × fwd`` (right-handed);
    ``world_up`` defaults to the +z spin axis.
    """
    f = -np.asarray(pos, dtype=float)
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


def pixel_direction(
    px: int,
    py: int,
    res: int,
    tan_half_fov: float,
    fwd: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Coordinate (CKS) unit direction the photon travels for pixel (px, py).

    The camera looks along ``fwd`` toward the hole; ``+right`` is screen-x and
    ``+up`` is screen-y. ``n = normalize(fwd + sx·right + sy·up)``.
    """
    sx = (2.0 * (px + 0.5) / res - 1.0) * tan_half_fov  # right
    sy = (1.0 - 2.0 * (py + 0.5) / res) * tan_half_fov  # up
    n = fwd + sx * right + sy * up
    return n / np.linalg.norm(n)


# --------------------------------------------------------------------------- #
# Ray classification
# --------------------------------------------------------------------------- #
def trace_pixel(
    pos3: np.ndarray,
    n: np.ndarray,
    a: float,
    n_steps: int,
    d_lambda: float,
    r_max: float,
    r_plus: float,
    horizon_eps: float,
    adaptive_floor: float,
    disk_params: dict | None = None,
) -> tuple[str, float, np.ndarray, float]:
    """Trace one photon and return (outcome, r_min, disk_color, transmittance).

    outcome:  'captured' (fell to the horizon -> shadow),
              'escaped'  (reached r_max  -> background),
              'undecided' (ran out of steps near the photon sphere).
    r_min:        closest-approach Kerr radius along the path (for the ring glow).
    disk_color:   (3,) accumulated disk emission (zeros if disk_params is None).
    transmittance: remaining transmittance through the disk (1.0 if no disk).
    """
    x0, p_cov0 = make_null_initial_conditions(pos3, n, a)

    # Near-horizon r→r₊ and far-field overflow can both trip FP warnings; the
    # integrator's CKS-6 stop conditions handle termination, so silence them.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, p = integrate_null_geodesic(
            x0,
            p_cov0,
            a,
            n_steps,
            d_lambda,
            r_max=r_max,
            horizon_eps=horizon_eps,
            adaptive_floor=adaptive_floor,
        )

    disk_color = np.zeros(3, dtype=float)
    transmittance = 1.0
    if disk_params is not None:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            disk_color, transmittance = march_disk(x, p, disk_params)

    # Kerr radius r along the (finite) path — closest approach feeds the ring glow.
    xs = x[:, X]
    ys = x[:, Y]
    zs = x[:, Z]
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if not np.any(finite):
        return "captured", r_plus, disk_color, transmittance
    r_series = np.array(
        [
            kerr_radius(float(xi), float(yi), float(zi), a)
            for xi, yi, zi in zip(xs[finite], ys[finite], zs[finite], strict=False)
        ]
    )
    r_min = float(r_series.min())

    # Escape: the final point reached r_max (rho); background light.
    rho_last = float(np.sqrt(xs[finite][-1] ** 2 + ys[finite][-1] ** 2 + zs[finite][-1] ** 2))
    r_last = float(r_series[-1])
    if rho_last >= r_max:
        return "escaped", r_min, disk_color, transmittance
    # Capture: stopped at the horizon margin.
    if r_last <= r_plus + horizon_eps + 1e-6:
        return "captured", r_plus, disk_color, transmittance
    return "undecided", r_min, disk_color, transmittance


# --------------------------------------------------------------------------- #
# Shading
# --------------------------------------------------------------------------- #
def background_color(py: int, res: int, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Vertical linear-RGB gradient (top -> bottom)."""
    t = py / max(1, res - 1)
    return (1.0 - t) * top + t * bottom


def ring_glow(
    r_min: float, peak: float, sigma: float, gain: float, color: np.ndarray
) -> np.ndarray:
    """Gaussian photon-ring glow peaked at the photon-sphere radius."""
    w = np.exp(-((r_min - peak) ** 2) / (2.0 * sigma * sigma))
    return gain * w * color


def _smoothstep(e0: float, e1: float, x: float) -> float:
    """Hermite smoothstep, twin of the GPU ``_smoothstep_ti`` (CKS-12 §3 edges)."""
    if e1 <= e0:
        t = 1.0 if x >= e0 else 0.0
    else:
        t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# Pipe B — volumetric accretion disk (CKS)
# --------------------------------------------------------------------------- #
def march_disk(x: np.ndarray, p_cov: np.ndarray, dp: dict) -> tuple[np.ndarray, float]:
    """Accumulate disk emission along an already-traced CKS geodesic.

    Implements Formulas CKS-8 (gas velocity), CKS-9 (g-factor) and 9 (emission)
    of skills/kerr-physics/SKILL.md. Marches the recorded geodesic samples; at
    each sample inside the equatorial-slab bounding box it adds redshifted
    blackbody emission and attenuates the running transmittance.

    Parameters
    ----------
    x      : (N, 4) CKS coordinates       [t, x, y, z].
    p_cov  : (N, 4) covariant photon momenta [p_t, p_x, p_y, p_z].
    dp     : dict of disk parameters (see build_disk_params()).

    Returns
    -------
    disk_color    : (3,) accumulated linear-RGB emission.
    transmittance : float remaining transmittance (1 = fully transparent).
    """
    color = np.zeros(3, dtype=float)
    transmittance = 1.0

    half_pi = 0.5 * np.pi
    a = dp["a"]
    sigma_theta = dp["theta_half_width"] * dp["vertical_sigma_frac"]
    ds = dp["d_lambda"]  # emission weight per CKS affine step

    for i in range(x.shape[0]):
        xi, yi, zi = x[i, X], x[i, Y], x[i, Z]
        if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(zi)):
            continue

        r = kerr_radius(float(xi), float(yi), float(zi), a)
        cos_th = min(max(zi / r, -1.0), 1.0)
        theta = np.arccos(cos_th)

        # Bounding box: thin equatorial slab between r_inner and r_outer. With §3
        # modulation on, widen the radial gate to the worst-case ragged band so the
        # soft-edge falloff region is still marched (mirrors the GPU trace kernel).
        dz = theta - half_pi
        if abs(dz) >= dp["theta_half_width"]:
            continue
        r_lo = dp["r_inner"]
        r_hi = dp["r_outer"]
        if dp["modulation_enabled"]:
            r_lo = dp["r_isco"]
            r_hi = dp["r_outer"] * (1.0 + 0.5 * dp["mod_edge_out_amp"]) + dp["mod_edge_soft"]
        if r < r_lo or r > r_hi:
            continue

        p = p_cov[i]
        if not np.all(np.isfinite(p)):
            continue

        # Formula CKS-8 — gas 4-velocity at this point.
        u = gas_four_velocity_cks(float(xi), float(yi), float(zi), a)

        # Formula CKS-9 — g-factor (Cartesian dot product; p_cov used as-is).
        denom = p[T] * u[T] + p[X] * u[X] + p[Y] * u[Y] + p[Z] * u[Z]
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            continue
        g = g_factor(p, u)
        if not np.isfinite(g) or g <= 0.0:
            continue

        # Formula 9 — emission: chromaticity * g^4 intensity.
        T_emit = dp["T_0"] * (6.0 / r) ** 0.75

        # Vertical Gaussian density profile within the slab (base σ_θ).
        density = np.exp(-0.5 * (dz / sigma_theta) ** 2)

        # D2.2/D2.3 procedural turbulence (SKILL.md CKS-12 §2–3): multiply the
        # density by the noise field (amplitude only — feeds emission AND
        # absorption). Uses the SAME noise.noise_density_mult the GPU kernel
        # mirrors, so a CPU thumb is a faithful look-dev preview of the beauty
        # render. With dp["shear_period"] > 0 the field is Keplerian-sheared and
        # reseeded against dp["t_disk"] (CKS-12 §2 dual-phase blend; Ω = Formula 3
        # per sample); shear_period == 0 ⇒ the static D2.2 path.
        if dp["noise_enabled"]:
            u_n = np.log(r / dp["r_inner"])
            phi_n = np.arctan2(float(yi), float(xi))
            zeta_n = dz / sigma_theta
            omega = 1.0 / (r**1.5 + a)  # Formula 3 (prograde Ω at this radius)
            density *= float(
                noise_density_mult(
                    u_n, phi_n, zeta_n, dp["noise"], dp["noise_seed"],
                    t_disk=dp["t_disk"], omega=omega, shear_period=dp["shear_period"],
                )
            )

            # D2.4 §3 modulation: emitted temperature / ragged edges / lumpy scale
            # height (twin of taichi_renderer._disk_emit_cks). Off ⇒ identity.
            if dp["modulation_enabled"]:
                nT, nein, neout, nh = (
                    float(v) for v in noise_modulation_fields(
                        u_n, phi_n, zeta_n, dp["noise"], dp["noise_seed"],
                        t_disk=dp["t_disk"], omega=omega, shear_period=dp["shear_period"],
                    )
                )
                # Scale height: re-evaluate the Gaussian at the modulated σ_θ.
                sigma_m = sigma_theta * (1.0 + dp["mod_height_amp"] * (nh - 0.5))
                density = (
                    density
                    / np.exp(-0.5 * (dz / sigma_theta) ** 2)
                    * np.exp(-0.5 * (dz / sigma_m) ** 2)
                )
                # Ragged edges: smoothstep windows; r_in_eff ≥ r_isco (constraint 3).
                r_in_eff = max(dp["r_inner"] * (1.0 + dp["mod_edge_in_amp"] * (nein - 0.5)),
                               dp["r_isco"])
                r_out_eff = dp["r_outer"] * (1.0 + dp["mod_edge_out_amp"] * (neout - 0.5))
                soft = dp["mod_edge_soft"]
                density *= _smoothstep(r_in_eff, r_in_eff + soft, r) * (
                    1.0 - _smoothstep(r_out_eff - soft, r_out_eff, r)
                )
                # Emitted-temperature lumps (BEFORE the g shift — constraint 2).
                T_emit *= 1.0 + dp["mod_temp_amp"] * (nT - 0.5)

        T_obs = g * T_emit
        chroma = blackbody_rgb(T_obs)

        emission = dp["emission_coeff"] * density * chroma * (g**4) * ds
        color += transmittance * emission
        transmittance *= np.exp(-dp["absorption_coeff"] * density * ds)

    return color, transmittance


def build_disk_params(cfg: dict, a: float, d_lambda: float, t_disk: float = 0.0) -> dict:
    """Assemble disk parameters from config (CKS-8 needs no frozen ISCO constants)."""
    d = cfg["disk"]
    nz = d.get("noise", {}) or {}
    dyn = d.get("dynamics") or {}
    mod = nz.get("modulation", {}) or {}
    return {
        "a": a,
        "r_inner": float(d["r_inner"]),
        "r_outer": float(d["r_outer"]),
        # r_isco floor for the §3 ragged inner edge (constraint 3); derived by CKS-13.
        "r_isco": float(cfg.get("black_hole", {}).get("r_isco", d["r_inner"])),
        "theta_half_width": float(d["theta_half_width"]),
        "vertical_sigma_frac": float(d["vertical_sigma_frac"]),
        "T_0": float(d["T_0"]),
        "emission_coeff": float(d["emission_coeff"]),
        "absorption_coeff": float(d["absorption_coeff"]),
        "d_lambda": d_lambda,
        # D2.2 procedural turbulence (static; SKILL.md CKS-12). Off by default
        # (disk.noise.enabled: false) ⇒ this CPU twin matches the legacy disk.
        "noise": nz,
        "noise_enabled": bool(nz.get("enabled", False)),
        "noise_seed": int(nz.get("seed", 1234)),
        # D2.3 shear advection (CKS-12 §2). shear_period == 0 (no disk.dynamics
        # block) ⇒ static D2.2 noise. t_disk is the disk clock in geometric M.
        "shear_period": float(dyn.get("shear_period_M", 0.0)),
        "t_disk": float(t_disk),
        # D2.4 §3 modulation (temperature / edges / scale height). Off ⇒ identity.
        "modulation_enabled": bool(mod.get("enabled", False)),
        "mod_temp_amp": float(mod.get("temp_amp", 0.0)),
        "mod_edge_in_amp": float(mod.get("edge_in_amp", 0.0)),
        "mod_edge_out_amp": float(mod.get("edge_out_amp", 0.0)),
        "mod_edge_soft": float(mod.get("edge_softness", 0.0)),
        "mod_height_amp": float(mod.get("height_amp", 0.0)),
    }


def render(cfg: dict, res: int, with_disk: bool = False, t_disk: float = 0.0) -> np.ndarray:
    bh = cfg["black_hole"]
    rcfg = cfg["render"]
    cam = cfg["camera"]
    th = cfg["thumb"]

    a = float(bh["spin"])
    r_plus = _horizon_radius(a)

    # Preview framing overrides the cinematic camera when present (see config).
    r_cam = float(th.get("camera_radius", cam["default_radius"]))
    fov_deg = float(th.get("fov_deg", cam["default_fov_deg"]))
    theta_cam = np.deg2rad(float(th["camera_theta_deg"]))
    phi_cam = 0.0

    n_steps = int(rcfg["max_steps_pipe_a"])
    d_lambda = float(rcfg["d_lambda_pipe_a"])
    r_max = float(rcfg["r_max"])
    horizon_eps = float(rcfg["horizon_epsilon"])
    adaptive_floor = float(rcfg["adaptive_step_floor"])

    bg_top = np.array(th["background_top"], dtype=float)
    bg_bottom = np.array(th["background_bottom"], dtype=float)
    ring_col = np.array(th["ring_color"], dtype=float)
    ring_gain = float(th["ring_gain"])
    ring_peak = float(th["ring_r_peak"])
    ring_sigma = float(th["ring_r_sigma"])
    exposure = float(th.get("exposure", 1.0))
    gamma = float(th["gamma"])

    tan_half_fov = np.tan(np.deg2rad(fov_deg) / 2.0)

    # Camera position in CKS Cartesian (spin axis = +z); look at the origin.
    st, ct = np.sin(theta_cam), np.cos(theta_cam)
    pos = np.array(
        [r_cam * st * np.cos(phi_cam), r_cam * st * np.sin(phi_cam), r_cam * ct], dtype=float
    )
    fwd, right, up = camera_basis(pos)

    disk_params = build_disk_params(cfg, a, d_lambda, t_disk) if with_disk else None

    img = np.zeros((res, res, 3), dtype=float)

    print(
        f"Tracing {res}x{res} = {res * res} rays "
        f"(a={a}, r_cam={r_cam}, theta_cam={np.rad2deg(theta_cam):.0f} deg, "
        f"fov={fov_deg} deg, disk={'on' if with_disk else 'off'})"
    )
    t0 = time.time()

    for py in range(res):
        bg = background_color(py, res, bg_top, bg_bottom)
        for px in range(res):
            n = pixel_direction(px, py, res, tan_half_fov, fwd, right, up)
            outcome, r_min, disk_color, transmittance = trace_pixel(
                pos,
                n,
                a,
                n_steps,
                d_lambda,
                r_max,
                r_plus,
                horizon_eps,
                adaptive_floor,
                disk_params,
            )

            if outcome == "captured":
                # Photon swallowed by the hole -> shadow silhouette, black.
                bg_layer = np.zeros(3, dtype=float)
            elif outcome == "escaped":
                # Background light, brightened if the ray grazed the photon
                # sphere (small r_min) -> the photon ring just outside the edge.
                bg_layer = bg + ring_glow(r_min, ring_peak, ring_sigma, ring_gain, ring_col)
            else:
                # Undecided: ran out of steps winding near the critical curve.
                # These are ring photons -> glow on a dark (behind-shadow) base.
                bg_layer = ring_glow(r_min, ring_peak, ring_sigma, ring_gain, ring_col)

            # Composite: disk emission in front, background attenuated behind it.
            color = disk_color + transmittance * bg_layer
            img[py, px] = color

        if (py + 1) % 16 == 0 or py == res - 1:
            elapsed = time.time() - t0
            print(f"  row {py + 1}/{res}  ({elapsed:.1f}s)")

    # Reinhard tone map compresses the huge g^4 Doppler dynamic range (the
    # approaching limb is far brighter than the receding one) so both limbs stay
    # visible instead of the bright side clipping to flat white. Then gamma encode.
    img = img * exposure
    img = img / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / gamma)
    return (img * 255.0 + 0.5).astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kerr black hole thumbnail (Pipe A).")
    parser.add_argument(
        "--res", type=int, default=None, help="Square resolution (default: config thumb_width)."
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Frame index (reserved; frame 0 uses config camera)."
    )
    parser.add_argument(
        "--disk",
        action="store_true",
        help="Composite the volumetric accretion disk (Pipe B); "
        "saves thumb_disk.png instead of thumb_output.png.",
    )
    parser.add_argument(
        "--t-disk",
        type=float,
        default=None,
        help="Disk clock in geometric M (CKS-12 §2 shear advection). Overrides the "
        "value derived from --frame. Needs disk.dynamics in config to have any effect.",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    res = args.res if args.res is not None else int(cfg["render"]["thumb_width"])

    if args.frame != 0:
        print(
            f"[warn] --frame {args.frame}: per-frame camera matrices are not "
            f"wired up yet; using the config camera (frame 0). The disk animation "
            f"clock (t_disk) DOES advance with --frame."
        )

    # D2.3 disk clock: t_disk = frame/fps · time_scale (geometric M, CKS-13).
    # No disk.dynamics block ⇒ time_scale absent ⇒ 0 ⇒ static D2.2 noise.
    if args.t_disk is not None:
        t_disk = args.t_disk
    else:
        dyn = cfg["disk"].get("dynamics") or {}
        t_disk = args.frame / float(cfg["render"]["fps"]) * float(dyn.get("time_scale", 0.0))

    from PIL import Image  # local import so --help works without Pillow

    pixels = render(cfg, res, with_disk=args.disk, t_disk=t_disk)
    out_name = "thumb_disk.png" if args.disk else "thumb_output.png"
    out_path = Path(__file__).resolve().parent / out_name
    Image.fromarray(pixels, mode="RGB").save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
