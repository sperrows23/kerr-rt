"""Cartesian Kerr-Schild (CKS) Kerr metric.

All formulas follow ``skills/kerr-physics/SKILL.md`` PART II (Formulas
CKS-1 ... CKS-4) **verbatim**. Coordinate order is ``(t, x, y, z)`` = indices
``(0, 1, 2, 3)``; geometric units ``G = M = c = 1``; signature ``(- + + +)``;
the spin ``a`` points along ``+z``. Nothing here is re-derived: the metric, its
*exact* closed-form inverse (valid because ``l`` is null w.r.t. the flat
background ``eta``), the implicit Kerr radius, and the analytic coordinate
derivatives are transcribed directly from the skill file.

Public API
----------
``kerr_radius(x, y, z, a) -> float``
    CKS-1 explicit positive root of the Kerr radius ``r``.
``null_vector_cks(x, y, z, a) -> (l_cov (4,), f)``
    CKS-2 covariant null vector ``l_alpha`` (with ``l_t = 1``) and scalar ``f``.
``metric_cks(x, y, z, a) -> (4, 4)``
    CKS-2 covariant metric ``g = eta + f * l (x) l``.
``inverse_metric_cks(x, y, z, a) -> (4, 4)``
    CKS-3 *exact* inverse ``g = eta - f * l_up (x) l_up`` (no numerical inverse).
``dmetric_inv_cks(x, y, z, a) -> dict``
    CKS-4 analytic spatial derivatives needed by the geodesic force term:
    ``dr`` (3,), ``df`` (3,), ``dl`` (3, 4) = d(l_alpha)/d(x^j) for j in x,y,z.
"""

from __future__ import annotations

import numpy as np

# CKS Cartesian coordinate index order used throughout the renderer.
T, X, Y, Z = 0, 1, 2, 3

# Flat-space Minkowski background, signature (- + + +).
ETA = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(float)


def kerr_radius(x: float, y: float, z: float, a: float) -> float:
    """Kerr radius r(x, y, z) — Formula CKS-1 explicit positive root.

        rho^2 = x^2 + y^2 + z^2
        r^2   = 1/2 (rho^2 - a^2) + sqrt( 1/4 (rho^2 - a^2)^2 + a^2 z^2 )
        r     = sqrt(r^2)
    """
    rho2 = x * x + y * y + z * z
    half = 0.5 * (rho2 - a * a)
    r2 = half + np.sqrt(half * half + a * a * z * z)
    # r2 is non-negative for any real point; guard FP noise at the origin.
    return float(np.sqrt(max(0.0, r2)))


def null_vector_cks(x: float, y: float, z: float, a: float):
    """Covariant null vector l_alpha and scalar f — Formula CKS-2.

        f   = 2 r^3 / (r^4 + a^2 z^2)
        l_t = 1
        l_x = (r x + a y) / (r^2 + a^2)
        l_y = (r y - a x) / (r^2 + a^2)
        l_z = z / r

    Returns
    -------
    l_cov : (4,) ndarray   covariant l_alpha = (1, l_x, l_y, l_z)
    f     : float
    """
    r = kerr_radius(x, y, z, a)
    r2 = r * r
    D = r2 * r2 + a * a * z * z          # r^4 + a^2 z^2
    S = r2 + a * a                       # r^2 + a^2

    f = 2.0 * r2 * r / D
    l_cov = np.array(
        [
            1.0,
            (r * x + a * y) / S,
            (r * y - a * x) / S,
            z / r,
        ],
        dtype=float,
    )
    return l_cov, float(f)


def metric_cks(x: float, y: float, z: float, a: float) -> np.ndarray:
    """Covariant CKS metric g_{alpha beta} — Formula CKS-2.

        g = eta + f * l (x) l        (eta = diag(-1, 1, 1, 1))
    """
    l_cov, f = null_vector_cks(x, y, z, a)
    return ETA + f * np.outer(l_cov, l_cov)


def inverse_metric_cks(x: float, y: float, z: float, a: float) -> np.ndarray:
    """Inverse CKS metric g^{alpha beta} — Formula CKS-3 (exact closed form).

        g^{ab} = eta^{ab} - f * l^a l^b,   l^a = eta^{ag} l_g = (-1, l_x, l_y, l_z)

    Because l is eta-null (eta^{ab} l_a l_b = 0) this closed form is exact; no
    numerical matrix inversion is used.
    """
    l_cov, f = null_vector_cks(x, y, z, a)
    # Raise index with eta: l^t = -l_t = -1, l^i = l_i.
    l_up = l_cov.copy()
    l_up[T] = -l_cov[T]
    return ETA - f * np.outer(l_up, l_up)


def dmetric_inv_cks(x: float, y: float, z: float, a: float) -> dict:
    """Analytic spatial coordinate derivatives — Formula CKS-4 (verbatim).

    Returns the pieces the geodesic force term needs (everything is independent
    of ``t``; ``l_t`` is constant so dl_t = 0). Index ``j`` runs over the spatial
    coordinates (x, y, z) = (1, 2, 3).

        dr/dx^j      : CKS-4
        df/dx^j      : CKS-4
        dl_alpha/dx^j: CKS-4  (l_t derivative is 0)

    Returns
    -------
    dict with keys
        ``r``  : float  kerr radius
        ``f``  : float  scalar f
        ``l``  : (4,)   covariant l_alpha
        ``dr`` : (3,)   [dr/dx, dr/dy, dr/dz]
        ``df`` : (3,)   [df/dx, df/dy, df/dz]
        ``dl`` : (3, 4) dl[j, alpha] = d(l_alpha)/d(x^{j+1})  (j: x,y,z)
    """
    r = kerr_radius(x, y, z, a)
    r2 = r * r
    a2 = a * a
    D = r2 * r2 + a2 * z * z              # r^4 + a^2 z^2
    S = r2 + a2                           # r^2 + a^2
    l_cov, f = null_vector_cks(x, y, z, a)

    # dr/dx^j  (CKS-4)
    dr = np.array(
        [
            r2 * r * x / D,                 # dr/dx
            r2 * r * y / D,                 # dr/dy
            r * z * (r2 + a2) / D,          # dr/dz
        ],
        dtype=float,
    )

    # df/dx^j = f * [ 3 (dr/dx^j)/r - (4 r^3 (dr/dx^j) + 2 a^2 z delta_jz) / D ]
    delta_z = np.array([0.0, 0.0, 1.0])    # delta_{j z} for j = x, y, z
    df = f * (
        3.0 * dr / r
        - (4.0 * r2 * r * dr + 2.0 * a2 * z * delta_z) / D
    )

    # dl_alpha / dx^j  (CKS-4); l_t constant => row entry 0.
    # delta_{j x}, delta_{j y}, delta_{j z} as columns over j = x, y, z.
    dxj = np.array([1.0, 0.0, 0.0])        # delta_jx
    dyj = np.array([0.0, 1.0, 0.0])        # delta_jy
    dzj = np.array([0.0, 0.0, 1.0])        # delta_jz

    S2 = S * S
    # dl_x/dx^j = [ (x dr + r d_jx + a d_jy) S - (r x + a y)(2 r dr) ] / S^2
    dl_x = ((x * dr + r * dxj + a * dyj) * S - (r * x + a * y) * (2.0 * r * dr)) / S2
    # dl_y/dx^j = [ (y dr + r d_jy - a d_jx) S - (r y - a x)(2 r dr) ] / S^2
    dl_y = ((y * dr + r * dyj - a * dxj) * S - (r * y - a * x) * (2.0 * r * dr)) / S2
    # dl_z/dx^j = d_jz / r - z (dr) / r^2
    dl_z = dzj / r - z * dr / r2

    dl = np.zeros((3, 4), dtype=float)
    dl[:, T] = 0.0                          # l_t constant
    dl[:, X] = dl_x
    dl[:, Y] = dl_y
    dl[:, Z] = dl_z

    return {"r": r, "f": f, "l": l_cov, "dr": dr, "df": df, "dl": dl}
