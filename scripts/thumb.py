"""Thumbnail preview renderer for the Kerr black hole.

Traces one null geodesic per pixel through Kerr spacetime and produces a small
PNG. Without ``--disk`` it renders Pipe A only (shadow + photon ring + gradient
background) to verify the lensing geometry. With ``--disk`` it composites the
Pipe B volumetric accretion disk on top.

All physics follows ``skills/kerr-physics/SKILL.md`` verbatim:

  * Formula 1     — Kerr metric            (``renderer.metric``)
  * Formula 6     — Mino-time integration  (``renderer.geodesic``)
  * Formula 7     — ZAMO observer tetrad   (photon momentum initialisation, below)
  * Formulas 3-5  — gas 4-velocity         (``renderer.disk``)
  * Formula 8     — g-factor               (``renderer.disk``)
  * Formula 9     — blackbody emission     (``renderer.disk`` + march_disk below)

Photon momenta are initialised with the **ZAMO tetrad** (Formula 7), *not* the
heuristic dot-product projection that the skill flags as only valid far away
with a narrow FOV.

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

from renderer.geodesic import integrate_null_geodesic  # noqa: E402
from renderer.metric import metric_bl  # noqa: E402
from renderer.disk import (  # noqa: E402
    blackbody_rgb,
    g_factor,
    gas_four_velocity,
    isco_conserved_quantities,
)

# Boyer-Lindquist coordinate index order (matches renderer.metric / geodesic).
T, R, TH, PH = 0, 1, 2, 3

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "render.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Formula 7 — ZAMO tetrad photon momentum initialisation
# --------------------------------------------------------------------------- #
def zamo_photon_momentum(r, theta, a, n_rhat, n_thetahat, n_phihat):
    """Covariant photon momentum p_mu from a local ZAMO ray direction.

    Implements Formula 7 of skills/kerr-physics/SKILL.md verbatim.

    Parameters
    ----------
    r, theta, a : float
        Boyer-Lindquist position and spin.
    n_rhat, n_thetahat, n_phihat : float
        Components of the local (orthonormal) camera ray direction in the ZAMO
        frame. Should be a unit 3-vector.

    Returns
    -------
    p_cov : (4,) ndarray
        Covariant photon momentum [p_t, p_r, p_theta, p_phi].
    """
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2

    Sigma = r * r + a * a * cos2
    Delta = r * r - 2.0 * r + a * a
    # Exact A — do NOT approximate as (r^2 + a^2)^2 (Formula 7 note).
    A = (r * r + a * a) ** 2 - a * a * Delta * sin2

    omega = 2.0 * a * r / A            # ZAMO angular velocity
    alpha = np.sqrt(Sigma * Delta / A)  # lapse function
    g_phiphi = A * sin2 / Sigma

    # Contravariant momentum: p^mu = e_(t) + n . spatial tetrad (Formula 7).
    p_t_con = 1.0 / alpha
    p_r_con = n_rhat * np.sqrt(Delta / Sigma)
    p_th_con = n_thetahat * (1.0 / np.sqrt(Sigma))
    p_ph_con = omega / alpha + n_phihat * (1.0 / np.sqrt(g_phiphi))

    # Lower the index with the metric (Formula 7).
    g = metric_bl(r, theta, a)
    p_t = g[T, T] * p_t_con + g[T, PH] * p_ph_con
    p_r = g[R, R] * p_r_con
    p_th = g[TH, TH] * p_th_con
    p_ph = g[PH, PH] * p_ph_con + g[T, PH] * p_t_con

    return np.array([p_t, p_r, p_th, p_ph], dtype=float)


def camera_ray_direction(px, py, res, tan_half_fov):
    """Local ZAMO-frame unit direction for pixel (px, py) of an res x res image.

    The camera looks inward toward the black hole (local -r_hat), with +phi_hat
    to the right and -theta_hat up (theta increases toward the south pole).
    """
    # Screen coordinates in [-1, 1], scaled by the half-FOV tangent.
    sx = (2.0 * (px + 0.5) / res - 1.0) * tan_half_fov   # right (+phi_hat)
    sy = (1.0 - 2.0 * (py + 0.5) / res) * tan_half_fov   # up    (-theta_hat)

    n_rhat = -1.0      # forward, toward the black hole
    n_thetahat = -sy   # up is decreasing theta
    n_phihat = sx
    n = np.array([n_rhat, n_thetahat, n_phihat], dtype=float)
    n /= np.linalg.norm(n)
    return n


# --------------------------------------------------------------------------- #
# Ray classification
# --------------------------------------------------------------------------- #
def trace_pixel(r_cam, theta_cam, phi_cam, a, n, n_steps, d_lambda,
                r_max, r_plus, disk_params=None):
    """Trace one photon and return (outcome, r_min, disk_color, transmittance).

    outcome:  'captured' (fell to horizon -> shadow),
              'escaped'  (reached r_max  -> background),
              'undecided' (ran out of steps near the photon sphere).
    r_min:        closest approach radius along the path (for ring glow).
    disk_color:   (3,) accumulated disk emission (zeros if disk_params is None).
    transmittance: remaining transmittance through the disk (1.0 if no disk).
    """
    p_cov = zamo_photon_momentum(r_cam, theta_cam, a, n[0], n[1], n[2])
    x0 = np.array([0.0, r_cam, theta_cam, phi_cam], dtype=float)

    # Escaping rays overflow harmlessly once they shoot past r_max (we filter to
    # finite values below); near-horizon Delta->0 also trips invalid warnings.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, p = integrate_null_geodesic(x0, p_cov, a, n_steps, d_lambda)

    disk_color = np.zeros(3, dtype=float)
    transmittance = 1.0
    if disk_params is not None:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            disk_color, transmittance = march_disk(x, p, disk_params)

    r_series = x[:, R]
    finite = np.isfinite(r_series)
    r_finite = r_series[finite]
    if r_finite.size == 0:
        return "captured", r_plus, disk_color, transmittance
    r_min = float(np.min(r_finite))

    if np.any(r_finite >= r_max):
        return "escaped", r_min, disk_color, transmittance

    r_last = float(r_finite[-1])
    # The integrator stops when Delta < its horizon threshold; a small final r
    # means the photon plunged through the horizon.
    if r_last < 2.0:
        return "captured", r_min, disk_color, transmittance
    return "undecided", r_min, disk_color, transmittance


# --------------------------------------------------------------------------- #
# Shading
# --------------------------------------------------------------------------- #
def background_color(py, res, top, bottom):
    """Vertical linear-RGB gradient (top -> bottom)."""
    t = py / max(1, res - 1)
    return (1.0 - t) * top + t * bottom


def ring_glow(r_min, peak, sigma, gain, color):
    """Gaussian photon-ring glow peaked at the photon-sphere radius."""
    w = np.exp(-((r_min - peak) ** 2) / (2.0 * sigma * sigma))
    return gain * w * color


# --------------------------------------------------------------------------- #
# Pipe B — volumetric accretion disk
# --------------------------------------------------------------------------- #
def march_disk(x, p_cov, dp):
    """Accumulate disk emission along an already-traced geodesic.

    Implements Formulas 5/3 (gas velocity), 8 (g-factor) and 9 (emission) of
    skills/kerr-physics/SKILL.md. Marches the recorded geodesic samples; at each
    sample inside the disk bounding box it adds redshifted blackbody emission and
    attenuates the running transmittance.

    Parameters
    ----------
    x      : (N, 4) Boyer-Lindquist coordinates [t, r, theta, phi].
    p_cov  : (N, 4) covariant photon momenta     [p_t, p_r, p_theta, p_phi].
    dp     : dict of disk parameters (see render()).

    Returns
    -------
    disk_color    : (3,) accumulated linear-RGB emission.
    transmittance : float remaining transmittance (1 = fully transparent).
    """
    color = np.zeros(3, dtype=float)
    transmittance = 1.0

    half_pi = 0.5 * np.pi
    sigma_theta = dp["theta_half_width"] * dp["vertical_sigma_frac"]
    ds = dp["d_lambda"]  # emission weight per Mino-time step

    for i in range(x.shape[0]):
        r = x[i, R]
        theta = x[i, TH]
        if not (np.isfinite(r) and np.isfinite(theta)):
            continue

        # Bounding box: thin equatorial slab between r_inner and r_outer.
        dz = theta - half_pi
        if abs(dz) >= dp["theta_half_width"]:
            continue
        if r < dp["r_inner"] or r > dp["r_outer"]:
            continue

        p = p_cov[i]
        if not np.all(np.isfinite(p)):
            continue

        # Formula 3/5 — gas 4-velocity at this point.
        u = gas_four_velocity(r, theta, dp["a"], dp["r_isco"],
                              dp["E_I"], dp["L_I"])

        # Formula 8 — g-factor (p_cov used as-is; NOT divided by Delta).
        denom = p[T] * u[T] + p[R] * u[R] + p[TH] * u[TH] + p[PH] * u[PH]
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            continue
        g = g_factor(p, u)
        if not np.isfinite(g) or g <= 0.0:
            continue

        # Formula 9 — emission: chromaticity * g^4 intensity.
        T_emit = dp["T_0"] * (6.0 / r) ** 0.75
        T_obs = g * T_emit
        chroma = blackbody_rgb(T_obs)

        # Vertical Gaussian density profile within the slab.
        density = np.exp(-0.5 * (dz / sigma_theta) ** 2)

        emission = dp["emission_coeff"] * density * chroma * (g ** 4) * ds
        color += transmittance * emission
        transmittance *= np.exp(-dp["absorption_coeff"] * density * ds)

    return color, transmittance


def build_disk_params(cfg: dict, a: float, d_lambda: float) -> dict:
    """Assemble disk parameters + frozen ISCO constants (Formula 4) from config."""
    bh = cfg["black_hole"]
    d = cfg["disk"]
    r_isco = float(bh["r_isco"])
    E_I, L_I = isco_conserved_quantities(r_isco, a)
    return {
        "a": a,
        "r_isco": r_isco,
        "E_I": E_I,
        "L_I": L_I,
        "r_inner": float(d["r_inner"]),
        "r_outer": float(d["r_outer"]),
        "theta_half_width": float(d["theta_half_width"]),
        "vertical_sigma_frac": float(d["vertical_sigma_frac"]),
        "T_0": float(d["T_0"]),
        "emission_coeff": float(d["emission_coeff"]),
        "absorption_coeff": float(d["absorption_coeff"]),
        "d_lambda": d_lambda,
    }


def render(cfg: dict, res: int, with_disk: bool = False) -> np.ndarray:
    bh = cfg["black_hole"]
    rcfg = cfg["render"]
    cam = cfg["camera"]
    th = cfg["thumb"]

    a = float(bh["spin"])
    r_plus = float(bh["r_plus"])

    # Preview framing overrides the cinematic camera when present (see config).
    r_cam = float(th.get("camera_radius", cam["default_radius"]))
    fov_deg = float(th.get("fov_deg", cam["default_fov_deg"]))
    theta_cam = np.deg2rad(float(th["camera_theta_deg"]))
    phi_cam = 0.0

    n_steps = int(rcfg["max_steps_pipe_a"])
    d_lambda = float(rcfg["d_lambda_pipe_a"])
    r_max = float(rcfg["r_max"])

    bg_top = np.array(th["background_top"], dtype=float)
    bg_bottom = np.array(th["background_bottom"], dtype=float)
    ring_col = np.array(th["ring_color"], dtype=float)
    ring_gain = float(th["ring_gain"])
    ring_peak = float(th["ring_r_peak"])
    ring_sigma = float(th["ring_r_sigma"])
    exposure = float(th.get("exposure", 1.0))
    gamma = float(th["gamma"])

    tan_half_fov = np.tan(np.deg2rad(fov_deg) / 2.0)

    disk_params = build_disk_params(cfg, a, d_lambda) if with_disk else None

    img = np.zeros((res, res, 3), dtype=float)

    print(f"Tracing {res}x{res} = {res * res} rays "
          f"(a={a}, r_cam={r_cam}, theta_cam={np.rad2deg(theta_cam):.0f} deg, "
          f"fov={fov_deg} deg, disk={'on' if with_disk else 'off'})")
    t0 = time.time()

    for py in range(res):
        bg = background_color(py, res, bg_top, bg_bottom)
        for px in range(res):
            n = camera_ray_direction(px, py, res, tan_half_fov)
            outcome, r_min, disk_color, transmittance = trace_pixel(
                r_cam, theta_cam, phi_cam, a, n,
                n_steps, d_lambda, r_max, r_plus, disk_params,
            )

            if outcome == "captured":
                # Photon swallowed by the hole -> shadow silhouette, black.
                bg_layer = np.zeros(3, dtype=float)
            elif outcome == "escaped":
                # Background light, brightened if the ray grazed the photon
                # sphere (small r_min) -> the photon ring just outside the edge.
                bg_layer = bg + ring_glow(r_min, ring_peak, ring_sigma,
                                          ring_gain, ring_col)
            else:
                # Undecided: ran out of steps winding near the critical curve.
                # These are ring photons -> glow on a dark (behind-shadow) base.
                bg_layer = ring_glow(r_min, ring_peak, ring_sigma,
                                     ring_gain, ring_col)

            # Composite: disk emission in front, background attenuated behind it.
            # Disk emission accumulated before the ray plunges shows the disk
            # both in front of and (via lensing) behind the shadow.
            color = disk_color + transmittance * bg_layer
            img[py, px] = color

        if (py + 1) % 16 == 0 or py == res - 1:
            elapsed = time.time() - t0
            print(f"  row {py + 1}/{res}  ({elapsed:.1f}s)")

    # Reinhard tone map compresses the huge g^4 Doppler dynamic range (the
    # approaching limb is 10-100x the receding one) so both limbs stay visible
    # instead of the bright side clipping to flat white. Then gamma encode.
    img = img * exposure
    img = img / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / gamma)
    return (img * 255.0 + 0.5).astype(np.uint8)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kerr black hole thumbnail (Pipe A).")
    parser.add_argument("--res", type=int, default=None,
                        help="Square resolution (default: config thumb_width).")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index (reserved; frame 0 uses config camera).")
    parser.add_argument("--disk", action="store_true",
                        help="Composite the volumetric accretion disk (Pipe B); "
                             "saves thumb_disk.png instead of thumb_output.png.")
    args = parser.parse_args(argv)

    cfg = load_config()
    res = args.res if args.res is not None else int(cfg["render"]["thumb_width"])

    if args.frame != 0:
        print(f"[warn] --frame {args.frame}: per-frame camera matrices are not "
              f"wired up yet; using the config camera (frame 0).")

    from PIL import Image  # local import so --help works without Pillow

    pixels = render(cfg, res, with_disk=args.disk)
    out_name = "thumb_disk.png" if args.disk else "thumb_output.png"
    out_path = Path(__file__).resolve().parent / out_name
    Image.fromarray(pixels, mode="RGB").save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
