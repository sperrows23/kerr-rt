"""Starmap celestial-direction → UV mapping tests (Cartesian Kerr-Schild).

Under CKS an escaped photon's contravariant spatial momentum IS the incoming sky
direction (SKILL.md Formula CKS-10), so the equirect lookup is a plain spherical
projection of a genuine Cartesian unit vector. The whole BL artifact class — the
center-column streak from theta-punch-through and its ``normalize_sphere_angles``
fold band-aid — is gone at the source (CKS is regular on the spin axis), so this
file now pins the CKS ``celestial_to_uv`` mapping directly.

No physics formula is touched; this is purely the celestial-sphere coordinate
projection that precedes the SKILL.md Formula 10 starmap lookup.
"""
from __future__ import annotations

import math

import pytest

from renderer.starmap import celestial_to_uv

TWO_PI = 2.0 * math.pi


def test_plus_x_maps_to_equator_zero_azimuth():
    # +x: theta'=pi/2 (equator, v=0.5), phi'=0 (u=0).
    u, v = celestial_to_uv(1.0, 0.0, 0.0)
    assert u == pytest.approx(0.0, abs=1e-12)
    assert v == pytest.approx(0.5, abs=1e-12)


def test_plus_z_maps_to_north_pole():
    # +z (spin axis): theta'=0 -> v=0. CKS is regular here (no punch-through).
    _, v = celestial_to_uv(0.0, 0.0, 1.0)
    assert v == pytest.approx(0.0, abs=1e-12)


def test_minus_z_maps_to_south_pole():
    _, v = celestial_to_uv(0.0, 0.0, -1.0)
    assert v == pytest.approx(1.0, abs=1e-12)


def test_plus_y_maps_to_quarter_azimuth():
    # +y: phi'=pi/2 -> u=0.25; equator -> v=0.5.
    u, v = celestial_to_uv(0.0, 1.0, 0.0)
    assert u == pytest.approx(0.25, abs=1e-12)
    assert v == pytest.approx(0.5, abs=1e-12)


def test_minus_x_maps_to_half_azimuth():
    u, _ = celestial_to_uv(-1.0, 0.0, 0.0)
    assert u == pytest.approx(0.5, abs=1e-12)


def test_negative_azimuth_wraps_into_unit_interval():
    # phi' just below 0 (dy<0) must wrap to u just below 1, not negative.
    u, _ = celestial_to_uv(1.0, -1e-6, 0.0)
    assert 0.0 <= u < 1.0
    assert u == pytest.approx(1.0, abs=1e-3)


def test_non_unit_direction_is_normalized():
    # An un-normalized direction maps the same as its unit version.
    u0, v0 = celestial_to_uv(2.0, 0.0, 0.0)
    u1, v1 = celestial_to_uv(1.0, 0.0, 0.0)
    assert u0 == pytest.approx(u1, abs=1e-12)
    assert v0 == pytest.approx(v1, abs=1e-12)


def test_uv_stays_in_canonical_range_over_the_sphere():
    # Sweep the sphere; every (u, v) must land in [0,1) x [0,1].
    for it in range(13):
        theta = math.pi * (it + 0.5) / 13.0
        for ip in range(17):
            phi = TWO_PI * ip / 17.0 - math.pi
            dx = math.sin(theta) * math.cos(phi)
            dy = math.sin(theta) * math.sin(phi)
            dz = math.cos(theta)
            u, v = celestial_to_uv(dx, dy, dz)
            assert 0.0 <= u < 1.0
            assert 0.0 <= v <= 1.0
            assert v == pytest.approx(theta / math.pi, abs=1e-9)
