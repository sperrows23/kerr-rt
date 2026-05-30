"""Null-geodesic integration in Kerr spacetime (Mino time).

All physics follows skills/kerr-physics/SKILL.md verbatim:

  * Formula 1  — Kerr metric (via :mod:`renderer.metric`)
  * Formula 6  — Carter constant Q (null form) and the Mino-time separated
                 equations of motion. Per SKILL.md v1.2 the radial potential
                 R(r) uses the null form (no massive-particle r^2 term).

Integration strategy
--------------------
The Mino-time separated equations give first-order forms ``dr/dλ = ±√R`` and
``dθ/dλ = ±√Θ``. We integrate their equivalent **second-order** forms

    d²r/dλ² = ½ R'(r)        d²θ/dλ² = ½ Θ'(θ)

(obtained by differentiating ``(dr/dλ)² = R``). These are algebraically the same
equations of motion but are smooth through radial/polar turning points, so no
sign-flip bookkeeping is needed. R'(r) and Θ'(θ) are ordinary calculus
derivatives of the verbatim Formula-6 potentials — no physics formula is
re-derived. The constants of motion E, L_z, Q (which fully define R and Θ) are
held fixed, so they are conserved by construction; the integrator's job is to
keep ``(dr/dλ)² = R`` and ``(dθ/dλ)² = Θ`` satisfied, which RK4 does to O(h⁴).
"""

from __future__ import annotations

import numpy as np

# BL coordinate order (matches renderer.metric).
T, R_, TH, PH = 0, 1, 2, 3

# Integration is terminated before the photon reaches the horizon, where the
# Boyer-Lindquist φ/t equations (∝ 1/Δ) and p_r = √R/Δ diverge. Δ → 0 at the
# horizon; stopping at a small positive Δ keeps the covariant momenta finite.
_DELTA_MIN = 0.05


def _delta(r: float, a: float) -> float:
    return r * r - 2.0 * r + a * a


def _radial_potential(r: float, E: float, L_z: float, Q: float, a: float) -> float:
    """R(r), Formula 6 (null form, SKILL.md v1.2)."""
    P = E * (r * r + a * a) - a * L_z
    B = (L_z - a * E) ** 2 + Q
    return P * P - _delta(r, a) * B


def _radial_potential_deriv(r: float, E: float, L_z: float, Q: float, a: float) -> float:
    """dR/dr — calculus derivative of the verbatim R(r) above.

    R = P² − Δ·B with P = E(r²+a²) − aL_z, B = (L_z−aE)²+Q (const in r),
    Δ = r²−2r+a². Hence dR/dr = 2P·P' − Δ'·B = 4Er·P − (2r−2)·B.
    """
    P = E * (r * r + a * a) - a * L_z
    B = (L_z - a * E) ** 2 + Q
    return 4.0 * E * r * P - (2.0 * r - 2.0) * B


def _theta_potential(theta: float, E: float, L_z: float, Q: float, a: float) -> float:
    """Θ(θ), Formula 6 (null form)."""
    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    return Q - cos2 * (-(a * a) * E * E + L_z * L_z / sin2)


def _theta_potential_deriv(theta: float, E: float, L_z: float, a: float) -> float:
    """dΘ/dθ — calculus derivative of the verbatim Θ(θ).

    Θ = Q + a²E²cos²θ − L_z²cos²θ/sin²θ, so
    dΘ/dθ = −a²E²·sin(2θ) + 2 L_z² cosθ / sin³θ.
    """
    s = np.sin(theta)
    c = np.cos(theta)
    return -(a * a) * E * E * np.sin(2.0 * theta) + 2.0 * L_z * L_z * c / (s ** 3)


def carter_Q(theta: float, p_theta: float, E: float, L_z: float, a: float) -> float:
    """Carter constant, null form (Formula 6):

        Q = p_θ² + cos²θ · (−a²E² + L_z²/sin²θ)
    """
    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    return p_theta ** 2 + cos2 * (-(a * a) * E * E + L_z * L_z / sin2)


def make_null_initial_conditions(r, theta, E, L_z, Q, a, p_r_sign):
    """Initial position and covariant momentum for a null geodesic.

    Returns
    -------
    x0 : (4,) ndarray   [t, r, θ, φ]
    p0 : (4,) ndarray   [p_t, p_r, p_θ, p_φ]   (covariant)

    Using Formula 6: p_t = −E, p_φ = L_z, p_θ = +√Θ, and (from
    p_r = g_rr p^r = (Σ/Δ)(√R/Σ)) p_r = p_r_sign·√R / Δ.
    """
    Delta = _delta(r, a)
    R = _radial_potential(r, E, L_z, Q, a)
    Theta = _theta_potential(theta, E, L_z, Q, a)

    p_t = -E
    p_phi = L_z
    p_theta = np.sqrt(max(0.0, Theta))
    p_r = p_r_sign * np.sqrt(max(0.0, R)) / Delta

    x0 = np.array([0.0, r, theta, 0.0], dtype=float)
    p0 = np.array([p_t, p_r, p_theta, p_phi], dtype=float)
    return x0, p0


def integrate_null_geodesic(x0, p0, a, n_steps, d_lambda):
    """Integrate a null geodesic in Mino time with RK4.

    Parameters
    ----------
    x0 : (4,) [t, r, θ, φ]
    p0 : (4,) covariant [p_t, p_r, p_θ, p_φ]
    a, n_steps, d_lambda : spin, max steps, Mino-time step.

    Returns
    -------
    x     : (N, 4) Boyer-Lindquist coordinates [t, r, θ, φ]
    p_cov : (N, 4) covariant momenta           [p_t, p_r, p_θ, p_φ]

    N ≤ n_steps + 1: integration stops early if the photon nears the horizon
    (Δ < _DELTA_MIN), where BL momenta diverge.
    """
    x0 = np.asarray(x0, dtype=float)
    p0 = np.asarray(p0, dtype=float)

    # Constants of motion (Formula 6 / Formula 7 extraction).
    E = -p0[T]
    L_z = p0[PH]
    Q = carter_Q(x0[TH], p0[TH], E, L_z, a)

    # State s = [r, θ, φ, t, v_r, v_θ] where v_r = dr/dλ, v_θ = dθ/dλ (Mino time).
    # v_r = dr/dλ = Σ·p^r = Δ·p_r  (since p_r = (Σ/Δ)p^r and dr/dλ = Σ p^r).
    r0, th0 = x0[R_], x0[TH]
    v_r0 = _delta(r0, a) * p0[R_]
    v_th0 = p0[TH]
    s = np.array([r0, th0, x0[PH], x0[T], v_r0, v_th0], dtype=float)

    def deriv(state: np.ndarray) -> np.ndarray:
        r, th, _phi, _t, vr, vth = state
        Delta = _delta(r, a)
        sin2 = np.sin(th) ** 2
        P = E * (r * r + a * a) - a * L_z

        dr = vr
        dth = vth
        dvr = 0.5 * _radial_potential_deriv(r, E, L_z, Q, a)
        dvth = 0.5 * _theta_potential_deriv(th, E, L_z, a)
        # Formula 6 verbatim:
        dphi = -(a * E - L_z / sin2) + a * P / Delta
        dt = -a * (a * E * sin2 - L_z) + (r * r + a * a) * P / Delta
        return np.array([dr, dth, dphi, dt, dvr, dvth], dtype=float)

    def record(state: np.ndarray):
        r, th, phi, t, vr, vth = state
        Delta = _delta(r, a)
        x = np.array([t, r, th, phi], dtype=float)
        # p_r = (Σ/Δ)p^r = (1/Δ)·(Σ p^r) = v_r/Δ ; p_θ = Σ p^θ = v_θ.
        p_cov = np.array([-E, vr / Delta, vth, L_z], dtype=float)
        return x, p_cov

    xs, ps = [], []
    x_rec, p_rec = record(s)
    xs.append(x_rec)
    ps.append(p_rec)

    def project(state: np.ndarray) -> np.ndarray:
        """Re-impose the exact constraints (dr/dλ)² = R, (dθ/dλ)² = Θ.

        These are conserved by the continuous flow but drift under RK4
        truncation. Where R is tiny (near the horizon) an unprojected negative
        drift of C = v_r² − R fabricates a spurious radial turning point and the
        photon bounces. Rescaling v_r, v_θ to ±√R, ±√Θ (signs from the RK4
        evolution) removes the drift and makes E, L_z, Q and the null norm exact.
        """
        r, th = state[0], state[1]
        R = _radial_potential(r, E, L_z, Q, a)
        Theta = _theta_potential(th, E, L_z, Q, a)
        state[4] = np.copysign(np.sqrt(max(0.0, R)), state[4])
        state[5] = np.copysign(np.sqrt(max(0.0, Theta)), state[5])
        return state

    h = d_lambda
    for _ in range(int(n_steps)):
        if _delta(s[0], a) < _DELTA_MIN:
            break
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * h * k1)
        k3 = deriv(s + 0.5 * h * k2)
        k4 = deriv(s + h * k3)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        s = project(s)
        if _delta(s[0], a) < _DELTA_MIN:
            break
        x_rec, p_rec = record(s)
        xs.append(x_rec)
        ps.append(p_rec)

    return np.array(xs), np.array(ps)
