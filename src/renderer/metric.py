"""Kerr metric in Boyer-Lindquist coordinates.

All formulas follow skills/kerr-physics/SKILL.md verbatim (Formula 1). The
covariant metric components are written out exactly as in the skill; the inverse
is obtained by numerical matrix inversion (np.linalg.inv) rather than by
hand-deriving the closed-form inverse, so no GR formula is re-derived here.
"""

from __future__ import annotations

import numpy as np

# Boyer-Lindquist coordinate index order used throughout the renderer.
T, R, TH, PH = 0, 1, 2, 3


def metric_bl(r: float, theta: float, a: float) -> np.ndarray:
    """Covariant Kerr metric g_{mu nu} in BL coords (t, r, theta, phi).

    Formula 1 (Kerr 1963), verbatim:

        Sigma = r^2 + a^2 cos^2 theta
        Delta = r^2 - 2r + a^2

        g_tt  = -(1 - 2r/Sigma)
        g_tphi = -2 a r sin^2 theta / Sigma
        g_phiphi = (r^2 + a^2 + 2 r a^2 sin^2 theta / Sigma) sin^2 theta
        g_rr  = Sigma / Delta
        g_thth = Sigma
    """
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2

    Sigma = r * r + a * a * cos2
    Delta = r * r - 2.0 * r + a * a

    g = np.zeros((4, 4), dtype=float)
    g[T, T] = -(1.0 - 2.0 * r / Sigma)
    g[T, PH] = -2.0 * a * r * sin2 / Sigma
    g[PH, T] = g[T, PH]
    g[PH, PH] = (r * r + a * a + 2.0 * r * a * a * sin2 / Sigma) * sin2
    g[R, R] = Sigma / Delta
    g[TH, TH] = Sigma
    return g


def inverse_metric_bl(r: float, theta: float, a: float) -> np.ndarray:
    """Inverse Kerr metric g^{mu nu} in BL coords (t, r, theta, phi).

    Built from the verbatim covariant metric (Formula 1) by numerical inversion.
    """
    return np.linalg.inv(metric_bl(r, theta, a))
