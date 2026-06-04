"""Unit tests for the point-star catalog ingest (scripts/ingest_stars.py, Part A).

Covers the pure per-star transforms and the BSC5 fixed-column parser without
requiring the (large, externally-downloaded) bsc5.dat file: a couple of
hand-built catalogue lines exercise the byte offsets and the coordinate /
flux / colour pipeline. No GR/Kerr physics here — these are the ingest's
astronomy conversions (Pogson flux, Ballesteros B-V->T, RA/Dec->theta',phi').
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ingest_stars import (
    bv_to_temperature,
    parse_bsc5_record,
    radec_to_theta_phi,
    star_flux_rgb,
    vmag_to_flux,
)

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Pogson magnitude -> flux
# --------------------------------------------------------------------------- #
def test_flux_zero_magnitude_is_unit_at_default_zeropoint():
    assert vmag_to_flux(0.0) == pytest.approx(1.0)


def test_flux_five_magnitudes_is_a_factor_of_100():
    # 5 magnitudes fainter == 100x less flux (definition of the magnitude scale).
    assert vmag_to_flux(0.0) / vmag_to_flux(5.0) == pytest.approx(100.0)


def test_brighter_star_has_more_flux():
    assert vmag_to_flux(1.0) > vmag_to_flux(2.0)


def test_zero_point_shifts_scale_only():
    # A non-zero zero-point rescales every star by the same constant factor.
    ratio = vmag_to_flux(3.0, mag_zero_point=2.0) / vmag_to_flux(3.0, mag_zero_point=0.0)
    assert ratio == pytest.approx(10.0 ** (0.4 * 2.0))


# --------------------------------------------------------------------------- #
# Ballesteros B-V -> temperature
# --------------------------------------------------------------------------- #
def test_temperature_monotonic_decreasing_in_bv():
    # Bluer stars (smaller / negative B-V) are hotter than redder ones.
    assert bv_to_temperature(-0.3) > bv_to_temperature(0.0) > bv_to_temperature(1.5)


def test_sunlike_bv_gives_plausible_temperature():
    # The Sun (B-V ~ 0.65) should land near ~5800 K within the fit's accuracy.
    assert bv_to_temperature(0.65) == pytest.approx(5800.0, rel=0.1)


# --------------------------------------------------------------------------- #
# RA/Dec -> BL exit angles (theta', phi')
# --------------------------------------------------------------------------- #
def test_north_pole_maps_to_theta_zero():
    theta, _ = radec_to_theta_phi(0.0, 0.5 * math.pi)  # Dec = +90 deg
    assert theta == pytest.approx(0.0)


def test_south_pole_maps_to_theta_pi():
    theta, _ = radec_to_theta_phi(0.0, -0.5 * math.pi)  # Dec = -90 deg
    assert theta == pytest.approx(math.pi)


def test_equator_maps_to_theta_half_pi_and_phi_wraps():
    theta, phi = radec_to_theta_phi(-0.5, 0.0)  # negative RA must wrap into [0, 2pi)
    assert theta == pytest.approx(0.5 * math.pi)
    assert 0.0 <= phi < TWO_PI
    assert phi == pytest.approx((-0.5) % TWO_PI)


# --------------------------------------------------------------------------- #
# Combined flux RGB
# --------------------------------------------------------------------------- #
def test_flux_rgb_is_float32_triple_and_scales_with_brightness():
    bright = star_flux_rgb(1.0, 0.0)
    faint = star_flux_rgb(4.0, 0.0)
    assert bright.shape == (3,)
    assert bright.dtype == np.float32
    assert np.all(bright > faint)  # same colour, brighter star -> more flux in every channel


def test_blue_star_is_bluer_than_red_star():
    # Compare the chromaticity (normalise out brightness): hot/blue star has a
    # larger B:R ratio than a cool/red star.
    blue = star_flux_rgb(0.0, -0.3)
    red = star_flux_rgb(0.0, 1.5)
    assert blue[2] / blue[0] > red[2] / red[0]


# --------------------------------------------------------------------------- #
# BSC5 fixed-column parser
# --------------------------------------------------------------------------- #
def _bsc5_line(ra="06 45 08.9", dec="-16 42 58", vmag="-1.46", bv="+0.00") -> str:
    """Build a minimal BSC5-format line with the V/50 byte offsets populated.

    Defaults describe Sirius (alpha CMa): RA 06h45m, Dec -16deg43', V=-1.46.
    """
    line = [" "] * 114
    rah, ram, ras = ra.split()
    ded, dem, des = dec[1:].split() if dec[0] in "+-" else dec.split()
    sign = dec[0] if dec[0] in "+-" else "+"

    def put(s: str, start1: int) -> None:  # 1-indexed inclusive start
        for k, ch in enumerate(s):
            line[start1 - 1 + k] = ch

    put(rah.rjust(2), 76)
    put(ram.rjust(2), 78)
    put(ras.rjust(4), 80)
    put(sign, 84)
    put(ded.rjust(2), 85)
    put(dem.rjust(2), 87)
    put(des.rjust(2), 89)
    put(vmag.rjust(5), 103)
    put(bv.rjust(5), 110)
    return "".join(line)


def test_parse_bsc5_record_sirius():
    rec = parse_bsc5_record(_bsc5_line())
    assert rec is not None
    theta, phi, vmag, bv = rec
    assert vmag == pytest.approx(-1.46)
    assert bv == pytest.approx(0.0)
    # Dec = -16d43' -> theta' = pi/2 - Dec > pi/2 (southern hemisphere).
    assert theta > 0.5 * math.pi
    # RA 06h45m08.9s -> ~101.3 deg -> ~1.768 rad.
    expected_phi = math.radians((6 + 45 / 60 + 8.9 / 3600) * 15.0)
    assert phi == pytest.approx(expected_phi, rel=1e-6)


def test_parse_bsc5_record_missing_bv_falls_back():
    rec = parse_bsc5_record(_bsc5_line(bv="     "))
    assert rec is not None
    _, _, _, bv = rec
    assert bv == pytest.approx(0.0)  # _DEFAULT_BV


def test_parse_bsc5_record_no_position_is_skipped():
    # A line with a blank RA/Dec block (nova/cluster placeholder) yields None.
    blank = " " * 114
    assert parse_bsc5_record(blank) is None


def test_parse_bsc5_record_no_vmag_is_skipped():
    assert parse_bsc5_record(_bsc5_line(vmag="     ")) is None
