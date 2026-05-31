"""Starmap UV-mapping / sphere-normalization tests.

Regression coverage for the center-column streak artifact: rays with near-zero
L_z push theta negative ("polar punch-through") during RK4 integration. The UV
mapping must fold these raw exit angles back onto the standard sphere
(theta -> |theta|, phi -> phi + pi) instead of clamping theta to v=0, which
crushed every punch-through ray onto the single north-pole texel and produced
the vertical streak.

No physics formula is touched; this is purely the celestial-sphere coordinate
normalization that precedes the SKILL.md Formula 10 starmap lookup.
"""
from __future__ import annotations

import math

import pytest

from renderer.starmap import (
    normalize_sphere_angles,
    direction_to_uv,
)

TWO_PI = 2.0 * math.pi


def test_negative_theta_reflects_across_north_pole():
    # theta slightly negative = ray punched through the north pole. The standard
    # spherical representation reflects: theta -> |theta|, phi -> phi + pi.
    theta, phi = normalize_sphere_angles(-0.1, 1.0)
    assert theta == pytest.approx(0.1, abs=1e-12)
    assert phi == pytest.approx((1.0 + math.pi) % TWO_PI, abs=1e-12)


def test_theta_beyond_pi_reflects_across_south_pole():
    theta, phi = normalize_sphere_angles(math.pi + 0.1, 1.0)
    assert theta == pytest.approx(math.pi - 0.1, abs=1e-12)
    assert phi == pytest.approx((1.0 + math.pi) % TWO_PI, abs=1e-12)


def test_theta_already_in_range_is_unchanged():
    theta, phi = normalize_sphere_angles(1.2, 0.7)
    assert theta == pytest.approx(1.2, abs=1e-12)
    assert phi == pytest.approx(0.7, abs=1e-12)


def test_large_negative_theta_folds_into_range():
    # Several wraps of punch-through must still land in [0, pi].
    theta, phi = normalize_sphere_angles(-(TWO_PI + 0.3), 0.5)
    assert 0.0 <= theta <= math.pi
    assert theta == pytest.approx(0.3, abs=1e-9)


def test_normalization_preserves_direction_vector():
    # The folded (theta, phi) must describe the SAME unit direction as the raw
    # angles (the whole point: no physical direction is altered).
    for th_raw, ph_raw in [(-0.1, 1.0), (math.pi + 0.4, 2.0), (-2.5, 5.0)]:
        th_n, ph_n = normalize_sphere_angles(th_raw, ph_raw)
        raw = (
            math.sin(th_raw) * math.cos(ph_raw),
            math.sin(th_raw) * math.sin(ph_raw),
            math.cos(th_raw),
        )
        norm = (
            math.sin(th_n) * math.cos(ph_n),
            math.sin(th_n) * math.sin(ph_n),
            math.cos(th_n),
        )
        for a, b in zip(raw, norm):
            assert a == pytest.approx(b, abs=1e-12)


def test_uv_from_negative_theta_is_not_crushed_to_pole():
    # The bug: negative theta produced v<0 -> clamped to 0.0 (north-pole texel).
    # The fix reflects it to a genuine v>0 on the opposite-meridian hemisphere.
    u, v = direction_to_uv(-0.1, 1.0)
    assert v == pytest.approx(0.1 / math.pi, abs=1e-12)
    expected_u = ((1.0 + math.pi) % TWO_PI) / TWO_PI
    assert u == pytest.approx(expected_u, abs=1e-12)


def test_uv_from_in_range_theta_unchanged():
    u, v = direction_to_uv(1.2, 0.7)
    assert v == pytest.approx(1.2 / math.pi, abs=1e-12)
    assert u == pytest.approx(0.7 / TWO_PI, abs=1e-12)
