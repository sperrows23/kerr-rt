"""Accretion-disk gas physics in Kerr spacetime — Cartesian Kerr-Schild (CKS).

All formulas follow ``skills/kerr-physics/SKILL.md`` PART II **verbatim** — none
are re-derived (project CRITICAL RULE):

  * Formula 3       — circular-orbit angular velocity Ω and u^t  (r >= r_isco)
  * Formula CKS-8   — equatorial gas 4-velocity (rigid rotation about +z)
  * Formula CKS-9   — g-factor (Doppler + gravitational redshift), Cartesian
  * Formula 9       — blackbody_rgb chromaticity helper

Index order is the CKS Cartesian convention (matches renderer.metric /
renderer.geodesic): ``[t, x, y, z] = (0, 1, 2, 3)``; geometric units
``G = M = c = 1``; signature ``(- + + +)``; spin ``a`` along ``+z``. The CKS
radius ``r`` IS the Boyer-Lindquist radial coordinate, so the BL-radius
quantities Ω, u^t (Formula 3) carry over unchanged.

These are the CPU single-source-of-truth for the Taichi ``_gas_four_velocity_cks``
/ ``_disk_emit_cks`` ``@ti.func`` kernels; the two must stay in agreement.
"""

from __future__ import annotations

import numpy as np

# CKS Cartesian coordinate index order.
T, X, Y, Z = 0, 1, 2, 3


def gas_four_velocity_cks(x: float, y: float, z: float, a: float) -> np.ndarray:
    """Contravariant gas 4-velocity u^mu = [u^t, u^x, u^y, u^z] — Formula CKS-8.

    Equatorial circular prograde orbit (``r >= r_isco``): a rigid rotation about
    +z at the BL angular velocity Ω (Formula 3). No BL→KS Jacobian is needed (see
    SKILL.md CKS-8 derivation: at ``z = 0`` the BL→KS ``t``/``φ`` shifts are
    constant along a circular orbit, so the velocity is a pure +z rotation).

        Ω   = 1 / (r^{3/2} + a)
        u^t = (1 + a r^{-3/2}) / sqrt(1 - 3/r + 2 a r^{-3/2})
        u^x = -Ω y u^t,  u^y = +Ω x u^t,  u^z = 0

    The plunging ``r < r_isco`` branch is intentionally absent: the disk inner
    edge is ``r_inner = r_isco``, so it is never sampled. If it is ever required,
    transform Formula 5 with the full BL→KS Jacobian and FLAG FOR HUMAN REVIEW
    before use (SKILL.md CKS-8).
    """
    from renderer.metric import kerr_radius

    r = kerr_radius(x, y, z, a)
    r15 = r ** 1.5
    Omega = 1.0 / (r15 + a)                                   # Formula 3
    u_t = (1.0 + a / r15) / np.sqrt(max(1.0 - 3.0 / r + 2.0 * a / r15, 1e-9))
    u_x = -Omega * y * u_t
    u_y = Omega * x * u_t
    return np.array([u_t, u_x, u_y, 0.0], dtype=float)


def g_factor(p_cov, u_con) -> float:
    """Redshift g = E_obs / E_emit — Formula CKS-9.

    Observer at rest at infinity ⇒ (p·u)_obs = p_t = -E, so

        g = -E / (p_t u^t + p_x u^x + p_y u^y + p_z u^z)

    p_cov : covariant photon momentum [p_t, p_x, p_y, p_z] (already covariant from
            the integrator's Hamiltonian equations — there is NO Δ to divide by;
            the Formula-8 BL "divide p_r by Δ" bug is structurally impossible).
    u_con : contravariant gas 4-velocity [u^t, u^x, u^y, u^z].
    """
    E = -p_cov[T]
    denom = (p_cov[T] * u_con[T] + p_cov[X] * u_con[X]
             + p_cov[Y] * u_con[Y] + p_cov[Z] * u_con[Z])
    return -E / denom


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
