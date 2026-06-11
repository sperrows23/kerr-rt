"""Self-consistency guard for the Page-Thorne disk-flux profile (SKILL.md CKS-11).

Physics policy (CLAUDE.md): every GR formula here must follow
``skills/kerr-physics/SKILL.md`` exactly; nothing is re-derived. This test pins
the Decision-B physical-disk upgrade *before* it is wired into a kernel, so a
later transcription slip in the closed form is caught immediately.

It encodes two SKILL.md expressions for the same quantity and asserts they
agree:

  1. The §1 conservation-law flux *integral*
        F_int(r) ∝ -Omega'/(E - Omega L)^2 · ∫_{r_ms}^r (E L' - L E') dr'
     built from the circular-orbit E, L, Omega of Formula 3/4.
  2. The CKS-11 *closed form* F(r) ∝ y^-7 · C^-1 · bracket(y), with the cubic
     roots y1,y2,y3 of y^3 - 3y + 2a = 0.

The closed form was verified (2026-06-12) to reproduce the integral to 5 sig
figs across r in [1.5, 28] M for a = 0.999, differing only by the overall
constant (3/2 · sqrt(-g) = 3/2 · r) the closed form drops. This test makes that
check permanent. See SKILL.md Formula CKS-11 and revision v1.10.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "render.yaml"


def _spin() -> float:
    with open(CONFIG, encoding="utf-8") as f:  # cp949 box: force utf-8
        return float(yaml.safe_load(f)["black_hole"]["spin"])


def _r_isco(a: float) -> float:
    """SKILL.md Formula 2 (prograde ISCO)."""
    Z1 = 1 + (1 - a**2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    return 3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


# --- SKILL.md Formula 3 / 4: equatorial circular-orbit quantities (M = 1) ---- #
def _ELO(r: np.ndarray, a: float):
    denom = r * np.sqrt(r**2 - 3 * r + 2 * a * np.sqrt(r))
    E = (r**2 - 2 * r + a * np.sqrt(r)) / denom
    L = np.sqrt(r) * (r**2 - 2 * a * np.sqrt(r) + a**2) / denom
    Omega = 1 / (r**1.5 + a)
    return E, L, Omega


def _roots(a: float):
    """SKILL.md CKS-11 cubic roots of y^3 - 3y + 2a = 0."""
    ac = np.arccos(a)
    y1 = 2 * np.cos((ac - np.pi) / 3)
    y2 = 2 * np.cos((ac + np.pi) / 3)
    y3 = -2 * np.cos(ac / 3)
    return y1, y2, y3


def _closed_bracket(y, y0, a):
    """SKILL.md CKS-11 three-log bracket(y)."""
    y1, y2, y3 = _roots(a)
    return (
        (y - y0)
        - 1.5 * a * np.log(y / y0)
        - 3 * (y1 - a) ** 2 / (y1 * (y1 - y2) * (y1 - y3)) * np.log((y - y1) / (y0 - y1))
        - 3 * (y2 - a) ** 2 / (y2 * (y2 - y1) * (y2 - y3)) * np.log((y - y2) / (y0 - y2))
        - 3 * (y3 - a) ** 2 / (y3 * (y3 - y1) * (y3 - y2)) * np.log((y - y3) / (y0 - y3))
    )


def test_cubic_roots_satisfy_their_equation():
    a = _spin()
    for y in _roots(a):
        assert abs(y**3 - 3 * y + 2 * a) < 1e-12


def test_zero_torque_inner_boundary():
    """F(r_ms) = 0: the bracket vanishes as y -> y0."""
    a = _spin()
    y0 = np.sqrt(_r_isco(a))
    eps = 1e-9
    assert abs(_closed_bracket(y0 + eps, y0, a)) < 1e-6


def test_closed_form_reproduces_flux_integral():
    """CKS-11 closed form == §1 conservation-law integral, up to the overall
    constant (3/2 · sqrt(-g) = 3/2 · r) the closed form drops via 'F ∝ ...'."""
    a = _spin()
    r_ms = _r_isco(a)

    r = np.linspace(r_ms + 1e-6, 30.0, 400_000)
    E, L, Om = _ELO(r, a)
    dE, dL, dOm = (np.gradient(q, r) for q in (E, L, Om))

    integrand = E * dL - L * dE
    integral = np.concatenate(
        [[0.0], np.cumsum((integrand[1:] + integrand[:-1]) * 0.5 * np.diff(r))]
    )
    F_int_core = (-dOm / (E - Om * L) ** 2) * integral  # no sqrt(-g) yet

    y = np.sqrt(r)
    y0 = np.sqrt(r_ms)
    C = 1 - 3 * y**-2 + 2 * a * y**-3
    F_closed = y**-7 * C**-1 * _closed_bracket(y, y0, a)

    # integral form carries an extra (3/2 · r) vs the closed form; divide it out
    ratio = F_int_core / F_closed / r
    probe = np.array([1.5, 2, 3, 5, 8, 12, 20, 28])
    idx = np.searchsorted(r, probe)
    vals = ratio[idx]
    # constant across all radii -> same radial profile (the dropped 3/2)
    assert np.allclose(vals, 1.5, rtol=2e-3), vals
