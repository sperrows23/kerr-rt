"""Conservation tests for the CKS null-geodesic integrator.

Physics policy (see CLAUDE.md): every GR formula used here must follow
``skills/kerr-physics/SKILL.md`` PART II (Cartesian Kerr-Schild) exactly. This
test file deliberately does NOT encode the Kerr metric. It:

  * uses only the conserved-quantity *definitions* (E = -p_t,
    L_z = x p_y - y p_x), and
  * delegates the inverse metric g^{alpha beta}, the photon initialization, the
    integrator, and the (diagnostic) Carter constant to the implementation
    modules in ``renderer`` (source of truth: the kerr-physics skill). Nothing
    is re-derived in the test.

The conserved quantities are checked along a single integrated null geodesic
(CKS harness, SKILL.md "Conservation test requirements"):

  1. Photon energy        E   = -p_t                       relative drift < 1e-4
  2. Axial ang. momentum  L_z = x p_y - y p_x              relative drift < 1e-4
  3. Null condition       H = 1/2 g^{ab} p_a p_b           |2H|          < 1e-6
  4. Carter constant Q (null form, CKS->BL)                relative drift < 1e-4

Golden numeric values are pinned with pytest-regressions (num_regression).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from renderer.geodesic import (
    axial_angular_momentum,
    carter_Q,
    energy,
    integrate_null_geodesic,
    make_null_initial_conditions,
    null_norm,
)

# --- Implementation under test ---------------------------------------------- #
# Source of truth for all formulas inside these functions: the kerr-physics skill.

# --------------------------------------------------------------------------- #
# Config / fixtures
# --------------------------------------------------------------------------- #

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "render.yaml"

# Integration controls for the test trajectory (test-only knobs, not physics).
N_STEPS = 4000
D_LAMBDA = 1.0e-2

# Probe null geodesic: camera off-axis at (12, 0, 5) so cos(theta) != 0 (the
# Carter polar term is exercised) and L_z != 0. The direction aims inward toward
# the hole with a tangential component, so the ray bends near the photon sphere.
POS0 = (12.0, 0.0, 5.0)
DIR0 = (-1.0, 0.35, -0.45)


@pytest.fixture(scope="module")
def config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def trajectory(config) -> dict:
    """Integrate one CKS null geodesic and return positions + covariant momenta.

    Returns a dict with:
      a      : spin parameter
      x      : (N, 4) CKS coords [t, x, y, z]
      p_cov  : (N, 4) covariant momenta [p_t, p_x, p_y, p_z]
    """
    a = float(config["black_hole"]["spin"])
    r_max = float(config["render"]["r_max"])
    horizon_eps = float(config["render"]["horizon_epsilon"])
    floor = float(config["render"]["adaptive_step_floor"])

    x0, p0 = make_null_initial_conditions(POS0, DIR0, a)
    x, p_cov = integrate_null_geodesic(
        x0,
        p0,
        a=a,
        n_steps=N_STEPS,
        d_lambda=D_LAMBDA,
        r_max=r_max,
        horizon_eps=horizon_eps,
        adaptive_floor=floor,
    )

    x = np.asarray(x, dtype=float)
    p_cov = np.asarray(p_cov, dtype=float)
    assert x.shape == p_cov.shape
    assert x.shape[1] == 4
    assert x.shape[0] > 10, "trajectory too short to test conservation"
    return {"a": a, "x": x, "p_cov": p_cov}


def _relative_drift(series: np.ndarray) -> float:
    """max |s - s[0]| / |s[0]| along the trajectory."""
    series = np.asarray(series, dtype=float)
    denom = abs(series[0])
    assert denom > 0.0, "initial value must be non-zero for a relative-drift check"
    return float(np.max(np.abs(series - series[0])) / denom)


# --------------------------------------------------------------------------- #
# 1. Energy:  E = -p_t
# --------------------------------------------------------------------------- #


def test_energy_conserved(trajectory):
    p_cov = trajectory["p_cov"]
    E = np.array([energy(p) for p in p_cov])
    assert _relative_drift(E) < 1e-4


# --------------------------------------------------------------------------- #
# 2. Axial angular momentum:  L_z = x p_y - y p_x
# --------------------------------------------------------------------------- #


def test_angular_momentum_conserved(trajectory):
    x = trajectory["x"]
    p_cov = trajectory["p_cov"]
    L_z = np.array([axial_angular_momentum(xi, pi) for xi, pi in zip(x, p_cov, strict=False)])
    assert _relative_drift(L_z) < 1e-4


# --------------------------------------------------------------------------- #
# 3. Null condition:  H = 1/2 g^{ab} p_a p_b == 0  (|2H| < 1e-6)
#    Inverse metric comes from the implementation (CKS-3, kerr-physics skill).
# --------------------------------------------------------------------------- #


def test_null_condition_preserved(trajectory):
    a = trajectory["a"]
    x = trajectory["x"]
    p_cov = trajectory["p_cov"]

    norms = np.array([null_norm(xi, pi, a) for xi, pi in zip(x, p_cov, strict=False)])
    # null_norm returns g^{ab} p_a p_b = 2H; require |H| < 1e-6 => |2H| < 2e-6.
    assert np.max(np.abs(0.5 * norms)) < 1e-6


# --------------------------------------------------------------------------- #
# 4. Carter constant (NULL form, CKS->BL diagnostic from the implementation).
# --------------------------------------------------------------------------- #


def test_carter_constant_conserved(trajectory):
    a = trajectory["a"]
    x = trajectory["x"]
    p_cov = trajectory["p_cov"]
    Q = np.array([carter_Q(xi, pi, a) for xi, pi in zip(x, p_cov, strict=False)])
    assert _relative_drift(Q) < 1e-4


# --------------------------------------------------------------------------- #
# Golden numeric regression: pin the conserved quantities (sampled) so future
# changes to the integrator that shift the physics get caught.
# --------------------------------------------------------------------------- #


def test_conserved_quantities_regression(trajectory, num_regression):
    a = trajectory["a"]
    x = trajectory["x"]
    p_cov = trajectory["p_cov"]

    E = np.array([energy(p) for p in p_cov])
    L_z = np.array([axial_angular_momentum(xi, pi) for xi, pi in zip(x, p_cov, strict=False)])
    Q = np.array([carter_Q(xi, pi, a) for xi, pi in zip(x, p_cov, strict=False)])

    # Sample at a fixed set of indices so the golden file is compact and stable.
    idx = np.linspace(0, x.shape[0] - 1, 11).astype(int)
    num_regression.check(
        {
            "lambda_index": idx.astype(float),
            "x": x[idx, 1],
            "y": x[idx, 2],
            "z": x[idx, 3],
            "E": E[idx],
            "L_z": L_z[idx],
            "Q": Q[idx],
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )
