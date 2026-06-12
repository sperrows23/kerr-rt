"""CPU precompute of the Page-Thorne disk-flux radial shape (SKILL.md CKS-11).

This is the D1 "wire the LUT" path: a pure-NumPy (no Taichi) precompute of the
dimensionless Page-Thorne flux shape ``f_PT(r) = F(r)/F_max`` over
``[r_isco, r_outer]``, which ``renderer.taichi_renderer`` uploads as a 1-D LUT and
the GPU disk kernel reads with linear interpolation (no per-pixel logs/integral).

Physics policy (CLAUDE.md): every formula below is **transcribed verbatim** from
``skills/kerr-physics/SKILL.md`` — Formula 2 (prograde ISCO) and Formula CKS-11
(Page-Thorne closed form). Nothing is re-derived. The closed form is pinned
independently in ``tests/test_disk_flux.py`` (``_roots`` / ``_closed_bracket`` /
``_r_isco``); ``tests/test_disk_flux.py::test_module_matches_pinned_transcription``
asserts this module agrees with that pinned transcription to rtol 1e-10.

Units: geometric, G = M = c = 1; ``a`` is the dimensionless spin a/M, ``r`` is the
Boyer-Lindquist radial coordinate (= the CKS Kerr radius). The absolute flux
amplitude (σ, Ṁ) is a free calibration carried by ``disk.T_0`` exactly as in the
simple model, so this module returns only the **shape** normalized to peak 1.0.
"""
from __future__ import annotations

import numpy as np


def isco_radius(a: float) -> float:
    """Prograde ISCO radius r_isco(a) — SKILL.md Formula 2.

    Matches the pinned transcription ``tests/test_disk_flux.py._r_isco`` (Z1/Z2).
    """
    Z1 = 1 + (1 - a**2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    return float(3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)))


def _roots(a: float):
    """Cubic roots of ``y³ − 3y + 2a = 0`` — SKILL.md CKS-11.

    Matches ``tests/test_disk_flux.py._roots`` exactly.
    """
    ac = np.arccos(a)
    y1 = 2 * np.cos((ac - np.pi) / 3)
    y2 = 2 * np.cos((ac + np.pi) / 3)
    y3 = -2 * np.cos(ac / 3)
    return y1, y2, y3


def _closed_bracket(y: np.ndarray, y0: float, a: float) -> np.ndarray:
    """The CKS-11 three-log ``bracket(y)``.

    Matches ``tests/test_disk_flux.py._closed_bracket`` exactly.
    """
    y1, y2, y3 = _roots(a)
    return (
        (y - y0)
        - 1.5 * a * np.log(y / y0)
        - 3 * (y1 - a) ** 2 / (y1 * (y1 - y2) * (y1 - y3)) * np.log((y - y1) / (y0 - y1))
        - 3 * (y2 - a) ** 2 / (y2 * (y2 - y1) * (y2 - y3)) * np.log((y - y2) / (y0 - y2))
        - 3 * (y3 - a) ** 2 / (y3 * (y3 - y1) * (y3 - y2)) * np.log((y - y3) / (y0 - y3))
    )


def flux_shape(r: np.ndarray, a: float) -> np.ndarray:
    """Page-Thorne flux shape ``F(r) ∝ y⁻⁷·C⁻¹·bracket(y)`` — SKILL.md CKS-11.

    ``y = √r``, ``y₀ = √r_isco``, ``C = 1 − 3y⁻² + 2a·y⁻³``. Returns 0 for
    ``r ≤ r_isco`` (zero-torque inner BC ``F(r_ms)=0``; no emission inside — gas
    plunges). Proportionality only: the absolute amplitude is carried by T_0.
    """
    r = np.asarray(r, dtype=np.float64)
    r_isco = isco_radius(a)
    y0 = np.sqrt(r_isco)

    out = np.zeros_like(r)
    mask = r > r_isco
    if np.any(mask):
        ym = np.sqrt(r[mask])
        C = 1.0 - 3.0 * ym**-2 + 2.0 * a * ym**-3
        out[mask] = ym**-7 * C**-1 * _closed_bracket(ym, y0, a)
    return out


def build_flux_lut(a: float, r_outer: float, n: int) -> tuple[np.ndarray, float, float]:
    """Precompute the normalized 1-D ``f_PT(r)`` LUT over ``[r_isco, r_outer]``.

    Uniform grid of ``n`` samples; evaluate :func:`flux_shape`, clip non-finite /
    negative to 0, normalize so the peak is 1.0, and force ``lut[0] = 0`` (the
    zero-torque inner boundary). Returns ``(lut_float32, r0=r_isco, dr)`` for the
    GPU kernel's ``t = (r − r0)/dr`` index.
    """
    r_isco = isco_radius(a)
    r_grid = np.linspace(r_isco, r_outer, n)
    dr = float(r_grid[1] - r_grid[0])

    f = flux_shape(r_grid, a)
    # Guard: drop NaN/±inf and any negative excursion before normalizing.
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.clip(f, 0.0, None)

    peak = float(f.max())
    if peak > 0.0:
        f = f / peak
    f[0] = 0.0  # zero-torque inner BC F(r_isco) = 0
    return f.astype(np.float32), float(r_isco), dr
