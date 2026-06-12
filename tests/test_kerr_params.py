"""CPU tests for the derived-parameter resolver (Formula CKS-13) — no Taichi/CUDA.

Physics policy (CLAUDE.md): the resolver transcribes SKILL.md Formulas 2 / CKS-6 / 3;
this file pins it against PUBLISHED anchor values instead of re-deriving anything:

  - a = 0 (Schwarzschild):     r_isco = 6 M,  r_plus = 2 M   (BPT 1972; any GR text)
  - a = 1 (extremal prograde): r_isco = 1 M,  r_plus = 1 M   (BPT 1972)
  - a = 0.999: r_isco ≈ 1.182 M, r_plus ≈ 1.0447 M (SKILL.md Formula 2 verified value
    and the retired render.yaml literals — the values the renderer shipped with).

This is the "reference table" role: exact closed forms beat an interpolated LUT, and
the literature values live here as regression anchors rather than as runtime data.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from renderer import kerr_params as kp

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "render.yaml"


def _load_raw_config() -> dict:
    """The YAML as stored — base parameters only, resolver NOT applied."""
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Closed forms vs published anchors
# --------------------------------------------------------------------------- #
def test_literature_anchors():
    # Schwarzschild endpoint (BPT 1972 / standard GR texts).
    assert kp.isco_radius(0.0) == pytest.approx(6.0, abs=1e-12)
    assert kp.horizon_radius(0.0) == pytest.approx(2.0, abs=1e-12)
    # Extremal prograde endpoint.
    assert kp.isco_radius(1.0) == pytest.approx(1.0, abs=1e-9)
    assert kp.horizon_radius(1.0) == pytest.approx(1.0, abs=1e-12)


def test_project_spin_matches_skill_verified_values():
    # SKILL.md Formula 2 "verified value for a = 0.999" and the retired YAML literals.
    assert kp.isco_radius(0.999) == pytest.approx(1.182, abs=5e-4)
    assert kp.horizon_radius(0.999) == pytest.approx(1.0447, abs=5e-5)


def test_isco_monotonic_and_outside_horizon():
    grid = [i / 50.0 for i in range(50)] + [0.999, 0.9999]
    iscos = [kp.isco_radius(a) for a in grid]
    rps = [kp.horizon_radius(a) for a in grid]
    assert all(x > y for x, y in zip(iscos, iscos[1:]))  # strictly decreasing in a
    assert all(ri > rp for ri, rp in zip(iscos, rps))  # prograde ISCO outside r+


def test_keplerian_omega_and_period():
    # Formula 3 at the Schwarzschild ISCO: Ω = 6^{-3/2}.
    assert kp.keplerian_omega(6.0, 0.0) == pytest.approx(6.0**-1.5, rel=1e-14)
    # Period is the exact 2π inverse.
    for r, a in [(6.0, 0.0), (1.182, 0.999), (25.0, 0.999)]:
        assert kp.orbital_period(r, a) == pytest.approx(
            2.0 * math.pi / kp.keplerian_omega(r, a), rel=1e-14
        )
    # One differential wrap takes longer than the inner lap, shorter than the outer.
    tw = kp.shear_wrap_time(1.182, 25.0, 0.999)
    assert kp.orbital_period(1.182, 0.999) < 2.0 * math.pi / kp.keplerian_omega(1.182, 0.999) + 1e-9
    assert tw > 0.0
    assert tw == pytest.approx(
        2.0 * math.pi
        / (kp.keplerian_omega(1.182, 0.999) - kp.keplerian_omega(25.0, 0.999)),
        rel=1e-14,
    )


# --------------------------------------------------------------------------- #
# resolve_config on the production YAML
# --------------------------------------------------------------------------- #
def test_resolve_production_config():
    cfg = kp.resolve_config(_load_raw_config())
    a = float(cfg["black_hole"]["spin"])
    d = cfg["disk"]

    assert cfg["black_hole"]["r_plus"] == pytest.approx(kp.horizon_radius(a), rel=1e-14)
    assert cfg["black_hole"]["r_isco"] == pytest.approx(kp.isco_radius(a), rel=1e-14)
    # r_inner: auto → the zero-torque inner edge.
    assert d["r_inner"] == pytest.approx(kp.isco_radius(a), rel=1e-14)

    # Production model is page_thorne: T_0 == target peak (f_PT LUT max-normalized),
    # i.e. exactly the retired T_0: 5500.0 literal — golden frames unaffected.
    assert d["temperature_model"] == "page_thorne"
    assert d["T_0"] == pytest.approx(float(d["target_peak_temperature"]), rel=1e-14)

    dyn = d["dynamics"]
    assert dyn["omega_inner"] == pytest.approx(kp.keplerian_omega(d["r_inner"], a), rel=1e-14)
    assert dyn["period_inner_M"] == pytest.approx(kp.orbital_period(d["r_inner"], a), rel=1e-14)
    assert dyn["time_scale"] == pytest.approx(
        dyn["period_inner_M"] / float(dyn["inner_lap_seconds"]), rel=1e-14
    )
    assert dyn["shear_period_M"] == pytest.approx(
        float(dyn["shear_wrap_budget"])
        * kp.shear_wrap_time(d["r_inner"], float(d["r_outer"]), a),
        rel=1e-14,
    )


def test_resolve_tracks_spin_change():
    """The original bug: editing spin must move EVERY dependent parameter."""
    cfg = _load_raw_config()
    cfg["black_hole"]["spin"] = 0.5
    cfg = kp.resolve_config(cfg)
    assert cfg["black_hole"]["r_plus"] == pytest.approx(1.0 + math.sqrt(0.75), rel=1e-12)
    assert cfg["disk"]["r_inner"] == pytest.approx(kp.isco_radius(0.5), rel=1e-14)
    # Time mapping rescales with the new inner edge.
    dyn = cfg["disk"]["dynamics"]
    assert dyn["time_scale"] == pytest.approx(
        kp.orbital_period(cfg["disk"]["r_inner"], 0.5) / float(dyn["inner_lap_seconds"]),
        rel=1e-14,
    )


def test_r_inner_override_and_clamp():
    a = 0.999
    base = _load_raw_config()

    cfg = copy.deepcopy(base)
    cfg["disk"]["r_inner"] = 3.0  # artistic override outside the ISCO: respected
    assert kp.resolve_config(cfg)["disk"]["r_inner"] == pytest.approx(3.0)

    cfg = copy.deepcopy(base)
    cfg["disk"]["r_inner"] = 0.5  # inside the ISCO: clamped up (zero-torque BC)
    assert kp.resolve_config(cfg)["disk"]["r_inner"] == pytest.approx(
        kp.isco_radius(a), rel=1e-14
    )


def test_t0_explicit_override_wins():
    cfg = _load_raw_config()
    cfg["disk"]["T_0"] = 1234.5  # legacy escape hatch
    assert kp.resolve_config(cfg)["disk"]["T_0"] == pytest.approx(1234.5)


def test_t0_simple_model_hits_target_peak_at_inner_edge():
    cfg = _load_raw_config()
    cfg["disk"]["temperature_model"] = "simple"
    cfg = kp.resolve_config(cfg)
    d = cfg["disk"]
    t_peak = float(d["target_peak_temperature"])
    # Decision B law evaluated at the inner edge must hit the target peak.
    assert d["T_0"] * (6.0 / d["r_inner"]) ** 0.75 == pytest.approx(t_peak, rel=1e-12)


def test_resolver_is_idempotent():
    once = kp.resolve_config(_load_raw_config())
    twice = kp.resolve_config(copy.deepcopy(once))
    assert twice == once


def test_validation_errors():
    cfg = _load_raw_config()
    cfg["black_hole"]["spin"] = 1.5
    with pytest.raises(ValueError):
        kp.resolve_config(cfg)

    cfg = _load_raw_config()
    cfg["disk"]["r_inner"] = "isco"  # only 'auto' or a number
    with pytest.raises(ValueError):
        kp.resolve_config(cfg)

    cfg = _load_raw_config()
    cfg["disk"]["r_outer"] = 0.5  # inside r_inner
    with pytest.raises(ValueError):
        kp.resolve_config(cfg)

    cfg = _load_raw_config()
    cfg["disk"]["dynamics"]["inner_lap_seconds"] = 0.0
    with pytest.raises(ValueError):
        kp.resolve_config(cfg)
