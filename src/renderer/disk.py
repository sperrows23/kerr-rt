"""Accretion-disk gas physics in Kerr spacetime.

All formulas follow ``skills/kerr-physics/SKILL.md`` **verbatim** — none are
re-derived (project CRITICAL RULE):

  * Formula 3 — circular-orbit 4-velocity        (r >= r_isco)
  * Formula 4 — ISCO conserved quantities E_I, L_I
  * Formula 5 — plunging-region 4-velocity        (r <  r_isco)
  * Formula 8 — g-factor (Doppler + gravitational redshift)
  * Formula 9 — blackbody_rgb chromaticity helper

Index order is the project convention (matches renderer.metric / geodesic):
``[t, r, theta, phi]``.
"""

from __future__ import annotations

import numpy as np

T, R, TH, PH = 0, 1, 2, 3


def isco_conserved_quantities(r_isco: float, a: float):
    """ISCO energy E_I and angular momentum L_I (Formula 4, verbatim).

    Evaluated once at r_I = r_isco and frozen; per SKILL.md these must NOT be
    recomputed at any r < r_isco.
    """
    r_I = r_isco
    denom_I = r_I * np.sqrt(r_I ** 2 - 3.0 * r_I + 2.0 * a * np.sqrt(r_I))
    E_I = (r_I ** 2 - 2.0 * r_I + a * np.sqrt(r_I)) / denom_I
    L_I = np.sqrt(r_I) * (r_I ** 2 - 2.0 * a * np.sqrt(r_I) + a ** 2) / denom_I
    return float(E_I), float(L_I)


def gas_four_velocity(r, theta, a, r_isco, E_I, L_I):
    """Contravariant gas 4-velocity u^mu = [u^t, u^r, u^theta, u^phi].

    r >= r_isco : circular orbit (Formula 3).
    r <  r_isco : plunging free-fall with frozen E_I, L_I (Formula 5).
    """
    if r >= r_isco:
        # Formula 3 — circular orbit 4-velocity (verbatim).
        Omega = 1.0 / (r ** 1.5 + a)
        u_t = (1.0 + a * r ** -1.5) / np.sqrt(1.0 - 3.0 / r + 2.0 * a * r ** -1.5)
        u_phi = Omega * u_t
        u_r = 0.0
        u_theta = 0.0
        return np.array([u_t, u_r, u_theta, u_phi], dtype=float)

    # Formula 5 — plunging region 4-velocity (verbatim).
    cos2 = np.cos(theta) ** 2
    Sigma = r * r + a * a * cos2
    Delta = r * r - 2.0 * r + a * a

    X = E_I * (r * r + a * a) - a * L_I  # intermediate quantity

    u_r = -(1.0 / Sigma) * np.sqrt(
        max(0.0, X * X - Delta * (r * r + (L_I - a * E_I) ** 2))
    )
    u_t = (1.0 / Sigma) * ((r * r + a * a) * X / Delta - a * (a * E_I - L_I))
    u_phi = (1.0 / Sigma) * (a * X / Delta - (a * E_I - L_I))
    u_theta = 0.0

    # SKILL.md sign rule: u^r must be infalling (negative). A positive value is
    # unphysical outflowing gas. The max(0.0, ...) clamp guarantees this.
    assert u_r <= 0.0, f"plunging u^r must be <= 0, got {u_r} at r={r}"

    return np.array([u_t, u_r, u_theta, u_phi], dtype=float)


def g_factor(p_cov, u_con):
    """Redshift g = E_obs / E_emit (Formula 8, verbatim).

    For a camera at rest at infinity, (p.u)_obs = -E = -1, so

        g = -1 / (p_t·u^t + p_r·u^r + p_theta·u^theta + p_phi·u^phi)

    p_cov : covariant photon momentum [p_t, p_r, p_theta, p_phi]
    u_con : contravariant gas 4-velocity [u^t, u^r, u^theta, u^phi]

    IMPORTANT: p_cov[R] is already covariant (from the integrator's Hamiltonian
    equations). Do NOT divide it by Delta — that is the known bug documented in
    Formula 8 that corrupts every Doppler color.
    """
    denom = (p_cov[T] * u_con[T] + p_cov[R] * u_con[R]
             + p_cov[TH] * u_con[TH] + p_cov[PH] * u_con[PH])
    return -1.0 / denom


def blackbody_rgb(temperature):
    """Normalized blackbody **chromaticity** (Formula 9 helper, verbatim).

    Returns RGB in [0, 1] representing the color of the spectrum at temperature
    T, with NO T^4 amplitude scaling. Because there is no built-in amplitude,
    applying pow(g, 4.0) as the intensity factor is correct and not
    double-counted (see Formula 9 warning).
    """
    r_col = 1.0 - np.exp(-temperature / 3500.0)
    g_col = 1.0 - np.exp(-temperature / 5500.0)
    b_col = 1.0 - np.exp(-temperature / 9500.0)
    return np.array([r_col, g_col, b_col], dtype=float)
