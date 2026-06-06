"""Null-geodesic integration in Kerr spacetime — Cartesian Kerr-Schild (CKS).

All physics follows ``skills/kerr-physics/SKILL.md`` PART II **verbatim**:

  * Formula CKS-2/3/4 — metric, exact inverse, analytic derivatives
    (via :mod:`renderer.metric`).
  * Formula CKS-5 — Hamiltonian null-geodesic equations of motion, integrated
    with RK4 on the 8-vector ``[t, x, y, z, p_t, p_x, p_y, p_z]``.
  * Formula CKS-6 — horizon capture / r_max escape termination.
  * Formula CKS-7 — ZAMO-observer + projected-ray photon initialization.
  * Formula CKS-9 reuse — conserved-quantity helpers.

Coordinates are CKS Cartesian ``(t, x, y, z)`` (indices 0,1,2,3), geometric
units ``M = 1``, signature ``(- + + +)``, spin ``a`` along ``+z``.

PUBLIC API (for downstream consumers such as scripts/thumb.py)
--------------------------------------------------------------
``photon_momentum_cks(pos3, dir3, a) -> p_cov (4,)``
    Formula CKS-7. ``pos3 = (x, y, z)`` camera position, ``dir3 = (nx, ny, nz)``
    the *coordinate* unit direction the photon travels. Returns covariant
    ``p_alpha`` scaled so ``E = -p_t = 1``.
``make_null_initial_conditions(pos3, dir3, a) -> (x0 (4,), p0_cov (4,))``
    Thin wrapper: ``x0 = [0, x, y, z]``, ``p0 = photon_momentum_cks(...)``.
``integrate_null_geodesic(x0, p0_cov, a, n_steps, d_lambda, r_max=50.0,
        horizon_eps=0.05) -> (x (N,4), p_cov (N,4))``
    RK4 integration of Formula CKS-5 with the adaptive affine step and the
    CKS-6 capture/escape stop conditions.
``energy(p_cov) -> float``                 = -p_t.
``axial_angular_momentum(x, p_cov) -> float`` = x p_y - y p_x.
``null_norm(x, p_cov, a) -> float``        = p . g_inv . p  (CKS-3 inverse).
``carter_Q(x, p_cov, a) -> float``         null Carter constant, via CKS->BL.

NOTE FOR scripts/thumb.py (downstream, do NOT edited here)
----------------------------------------------------------
The legacy BL API (``radial_turning_point``, the BL ``make_null_initial_conditions``
signature ``(r, theta, E, L_z, Q, a, p_r_sign)``, Mino ``Theta(theta)`` / ``R(r)``)
is **removed**. ``radial_turning_point`` has no clean CKS analogue (it was a BL
1D-potential root) and is **NOT** provided. The new entry points are the
``pos3``/``dir3`` Cartesian functions above.
"""

from __future__ import annotations

import numpy as np

from renderer.metric import (
    T,
    X,
    Y,
    Z,
    inverse_metric_cks,
    kerr_radius,
    metric_cks,
    null_vector_cks,
    dmetric_inv_cks,
)


def _horizon_radius(a: float) -> float:
    """Outer horizon r_+ = 1 + sqrt(1 - a^2)  (Formula CKS-6; r is the BL radius)."""
    return 1.0 + np.sqrt(max(0.0, 1.0 - a * a))


# --------------------------------------------------------------------------- #
# Formula CKS-5 — equations of motion (working form)
# --------------------------------------------------------------------------- #


def _eom(state: np.ndarray, a: float) -> np.ndarray:
    """Right-hand side of the CKS-5 8-vector EOM.

    state = [t, x, y, z, p_t, p_x, p_y, p_z]   (covariant momenta).

    Working form (eta constant, g^{bg} = eta^{bg} - f l^b l^g), with
    phi_l = l^b p_b = -p_t + l_x p_x + l_y p_y + l_z p_z:

        dt/dl  = -p_t + f phi_l
        dx^i/dl = p_i - f l_i phi_l
        dp_t/dl = 0
        dp_i/dl = 1/2 (d_i f) phi_l^2 + f phi_l (d_i phi_l)
            d_i phi_l = (d_i l_x) p_x + (d_i l_y) p_y + (d_i l_z) p_z
    """
    x, y, z = state[1], state[2], state[3]
    p_t, p_x, p_y, p_z = state[4], state[5], state[6], state[7]

    d = dmetric_inv_cks(x, y, z, a)
    f = d["f"]
    l = d["l"]            # covariant l_alpha = (1, l_x, l_y, l_z)
    df = d["df"]          # (3,)   d f / d x^i
    dl = d["dl"]          # (3, 4) d l_alpha / d x^j  (j over x,y,z)

    l_x, l_y, l_z = l[X], l[Y], l[Z]

    # phi_l = l^beta p_beta with l^t = -l_t = -1, l^i = l_i:
    phi_l = -p_t + l_x * p_x + l_y * p_y + l_z * p_z

    out = np.empty(8, dtype=float)
    # dx^alpha/dlambda = g^{alpha beta} p_beta, expanded:
    out[T] = -p_t + f * phi_l                        # dt/dl  (l^t = -1)
    out[X] = p_x - f * l_x * phi_l                   # dx/dl
    out[Y] = p_y - f * l_y * phi_l                   # dy/dl
    out[Z] = p_z - f * l_z * phi_l                   # dz/dl

    # dp_t/dl = 0 (E conserved analytically).
    out[4] = 0.0

    # d_i phi_l = (d_i l_x) p_x + (d_i l_y) p_y + (d_i l_z) p_z   (l_t const).
    dphi = dl[:, X] * p_x + dl[:, Y] * p_y + dl[:, Z] * p_z       # (3,)

    # dp_i/dl = 1/2 (d_i f) phi_l^2 + f phi_l (d_i phi_l)
    dp_i = 0.5 * df * (phi_l * phi_l) + f * phi_l * dphi          # (3,)
    out[5] = dp_i[0]
    out[6] = dp_i[1]
    out[7] = dp_i[2]
    return out


# --------------------------------------------------------------------------- #
# Formula CKS-7 — photon initialization
# --------------------------------------------------------------------------- #


def photon_momentum_cks(pos3, dir3, a: float) -> np.ndarray:
    """Covariant photon momentum p_alpha at the camera — Formula CKS-7.

    Parameters
    ----------
    pos3 : (3,)  camera position (x, y, z) in CKS.
    dir3 : (3,)  coordinate direction the photon travels (need not be unit;
                 it is renormalized in the g-orthogonal projection).
    a    : spin.

    Returns
    -------
    p_cov : (4,) covariant momentum, scaled so E = -p_t = 1.
    """
    pos3 = np.asarray(pos3, dtype=float)
    dir3 = np.asarray(dir3, dtype=float)
    x, y, z = pos3

    g = metric_cks(x, y, z, a)
    g_inv = inverse_metric_cks(x, y, z, a)

    # 1. ZAMO 4-velocity directly from g^{alpha beta}:
    #    alpha = 1/sqrt(-g^{tt});  u_obs^alpha = -alpha g^{t alpha}  (u^t = 1/alpha > 0)
    lapse = 1.0 / np.sqrt(-g_inv[T, T])
    u_obs = -lapse * g_inv[T, :]                       # contravariant (4,)

    # 2. Camera ray as 4-vector, g-orthogonal to u_obs, then g-unit (+++):
    N = np.array([0.0, dir3[0], dir3[1], dir3[2]], dtype=float)
    # g_{mu nu} N^mu u_obs^nu
    gNu = N @ g @ u_obs
    Nprime = N + gNu * u_obs                           # now g.(N', u_obs) = 0
    gNN = Nprime @ g @ Nprime
    s_hat = Nprime / np.sqrt(gNN)                      # spatial unit vector

    # 3. Null photon momentum (contravariant) then lower:
    p_up = u_obs + s_hat                               # E_loc = 1; null automatically
    p_cov = g @ p_up                                   # lower index

    # Scale so E = -p_t = 1 (g uses ratios, so any positive scale is fine).
    E = -p_cov[T]
    p_cov = p_cov / E
    return p_cov


def make_null_initial_conditions(pos3, dir3, a: float):
    """Initial CKS position and covariant momentum for a null geodesic.

    Returns
    -------
    x0    : (4,) [t, x, y, z]   (t = 0)
    p0    : (4,) covariant [p_t, p_x, p_y, p_z], scaled so E = 1.
    """
    pos3 = np.asarray(pos3, dtype=float)
    x0 = np.array([0.0, pos3[0], pos3[1], pos3[2]], dtype=float)
    p0 = photon_momentum_cks(pos3, dir3, a)
    return x0, p0


# --------------------------------------------------------------------------- #
# Formula CKS-5/6 — RK4 integration with capture/escape termination
# --------------------------------------------------------------------------- #


def integrate_null_geodesic(
    x0,
    p0_cov,
    a: float,
    n_steps: int,
    d_lambda: float,
    r_max: float = 50.0,
    horizon_eps: float = 0.05,
    adaptive_floor: float = 0.005,
):
    """Integrate a CKS null geodesic with RK4 (Formula CKS-5/6).

    Parameters
    ----------
    x0     : (4,) [t, x, y, z]
    p0_cov : (4,) covariant [p_t, p_x, p_y, p_z]
    a      : spin.
    n_steps, d_lambda : max steps and base affine step.
    r_max  : escape radius (rho = sqrt(x^2+y^2+z^2) >= r_max stops).
    horizon_eps : capture margin; stop when r <= r_+ + horizon_eps.
    adaptive_floor : lower bound on the near-horizon step factor.

    Returns
    -------
    x     : (N, 4) CKS coordinates [t, x, y, z]
    p_cov : (N, 4) covariant momenta [p_t, p_x, p_y, p_z]

    Adaptive step: h = d_lambda * max(adaptive_floor, (r - r_+)/r).
    """
    s = np.asarray(x0, dtype=float).copy()
    s = np.concatenate([s, np.asarray(p0_cov, dtype=float)])     # 8-vector

    r_plus = _horizon_radius(a)

    xs = [s[:4].copy()]
    ps = [s[4:].copy()]

    for _ in range(int(n_steps)):
        x, y, z = s[1], s[2], s[3]
        r = kerr_radius(x, y, z, a)
        rho = np.sqrt(x * x + y * y + z * z)

        # Termination (CKS-6).
        if r <= r_plus + horizon_eps:
            break
        if rho >= r_max:
            break

        # Adaptive affine step.
        h = d_lambda * max(adaptive_floor, (r - r_plus) / r)

        k1 = _eom(s, a)
        k2 = _eom(s + 0.5 * h * k1, a)
        k3 = _eom(s + 0.5 * h * k2, a)
        k4 = _eom(s + h * k3, a)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        xs.append(s[:4].copy())
        ps.append(s[4:].copy())

    return np.array(xs), np.array(ps)


# --------------------------------------------------------------------------- #
# Conserved-quantity helpers (Formula CKS-5 invariants / CKS-3 norm)
# --------------------------------------------------------------------------- #


def energy(p_cov) -> float:
    """Photon energy E = -p_t (Formula CKS-5: stationary => E conserved)."""
    p_cov = np.asarray(p_cov, dtype=float)
    return float(-p_cov[..., T])


def axial_angular_momentum(x, p_cov) -> float:
    """Axial angular momentum L_z = x p_y - y p_x (CKS-5: axisymmetric => conserved)."""
    x = np.asarray(x, dtype=float)
    p_cov = np.asarray(p_cov, dtype=float)
    return float(x[..., X] * p_cov[..., Y] - x[..., Y] * p_cov[..., X])


def null_norm(x, p_cov, a: float) -> float:
    """Null condition value g^{alpha beta} p_alpha p_beta (= 2H; CKS-3 inverse)."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p_cov, dtype=float)
    g_inv = inverse_metric_cks(x[X], x[Y], x[Z], a)
    return float(p @ g_inv @ p)


def carter_Q(x, p_cov, a: float) -> float:
    """Null Carter constant Q, computed by converting CKS -> BL.

    Uses r from CKS-1, cos(theta) = z/r, and the BL p_theta obtained from the
    Cartesian momenta via the chain rule. The null Carter form (Formula 6) is:

        Q = p_theta^2 + cos^2(theta) ( -a^2 E^2 + L_z^2 / sin^2(theta) )

    BL angles from CKS (z = r cos theta):
        r       = kerr_radius(x, y, z, a)
        cos th  = z / r,   sin th = sqrt(1 - cos^2 th)
        phi_BL  = atan2(y, x) - atan2(a, r)         (KS -> BL azimuth shift)

    p_theta (covariant) via the Jacobian of (x, y, z) w.r.t. theta at fixed
    (r, phi_BL): the equatorial-to-polar tangent. For the disk plane this is
    handled cleanly; near the axis sin(theta) -> 0 and the BL form is singular
    (which is exactly why the renderer uses CKS). Q is therefore a diagnostic
    only -- use null_norm for the production null-condition check.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p_cov, dtype=float)
    xx, yy, zz = x[X], x[Y], x[Z]

    r = kerr_radius(xx, yy, zz, a)
    E = -p[T]
    L_z = xx * p[Y] - yy * p[X]

    cos_th = zz / r
    cos_th = max(-1.0, min(1.0, cos_th))
    sin2 = max(1.0 - cos_th * cos_th, 1e-12)
    sin_th = np.sqrt(sin2)

    # p_theta = sum_i p_i * (d x^i / d theta) at fixed (r, phi_BL).
    # CKS spatial position at fixed r, BL azimuth phi:
    #   x = r cos(phi) - a sin(phi) ... but the simplest exact route is the
    #   contravariant momentum: p^theta = sum_i (d theta / d x^i) p^i, then
    #   p_theta = Sigma * p^theta (BL g_thth = Sigma). Use the spatial p^i.
    # Contravariant spatial momentum:
    g_inv = inverse_metric_cks(xx, yy, zz, a)
    p_up = g_inv @ p
    pux, puy, puz = p_up[X], p_up[Y], p_up[Z]

    # theta = arccos(z / r),  r = r(x,y,z). d theta/d x^i computed analytically.
    # d(cos theta)/d x^i = (delta_iz r - z dr_i) / r^2 ; d theta = -dcos/sin.
    d = dmetric_inv_cks(xx, yy, zz, a)
    dr = d["dr"]                                  # [dr/dx, dr/dy, dr/dz]
    dcos = np.array(
        [(0.0 * r - zz * dr[0]), (0.0 * r - zz * dr[1]), (1.0 * r - zz * dr[2])]
    ) / (r * r)
    dtheta = -dcos / sin_th                       # d theta / d x^i  (3,)
    p_theta_up = dtheta[0] * pux + dtheta[1] * puy + dtheta[2] * puz   # p^theta

    Sigma = (r * r * r * r + a * a * zz * zz) / (r * r)   # CKS-1 identity
    p_theta = Sigma * p_theta_up                  # lower with BL g_thth = Sigma

    return float(
        p_theta * p_theta
        + cos_th * cos_th * (-(a * a) * E * E + L_z * L_z / sin2)
    )
