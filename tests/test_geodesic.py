"""Conservation tests for the Kerr null-geodesic integrator.

Physics policy (see CLAUDE.md): every GR formula used here must follow
skills/kerr-physics/SKILL.md exactly. This test file deliberately does NOT
encode the Kerr metric. It:

  * uses only the conserved-quantity *definitions* (E = -p_t, L_z = p_phi),
  * uses the exact NULL Carter-constant form supplied in the task spec, and
  * delegates the inverse metric g^{mu nu} and the integrator to the
    implementation modules in ``renderer`` (source of truth: the kerr-physics
    skill). Nothing is re-derived in the test.

The four conserved quantities are checked along a single integrated null
geodesic:

  1. Photon energy        E   = -p_t                     relative drift < 1e-4
  2. Axial ang. momentum  L_z =  p_phi                   relative drift < 1e-4
  3. Carter constant (null form)                         relative drift < 1e-4
        Q = p_theta^2 + cos^2(theta) * (-a^2 E^2 + L_z^2 / sin^2(theta))
  4. Null condition       g^{mu nu} p_mu p_nu            |value|       < 1e-6

Golden numeric values are pinned with pytest-regressions (num_regression).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

# --- Implementation under test (does not exist yet -> RED via ImportError) ---
# Source of truth for all formulas inside these functions: the kerr-physics skill.
from renderer.metric import inverse_metric_bl
from renderer.geodesic import (
    make_null_initial_conditions,
    integrate_null_geodesic,
)

# --------------------------------------------------------------------------- #
# Config / fixtures
# --------------------------------------------------------------------------- #

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "render.yaml"

# Integration controls for the test trajectory (test-only knobs, not physics).
N_STEPS = 4000
D_LAMBDA = 1.0e-2

# Initial conditions for the probe null geodesic. theta is taken off the
# equatorial plane so cos(theta) != 0 and the Carter term is actually exercised.
R0 = 10.0
THETA0 = np.pi / 2.0 - 0.3
E0 = 1.0
LZ0 = 2.0
Q0 = 3.0
P_R_SIGN = -1.0  # ingoing


@pytest.fixture(scope="module")
def config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def trajectory(config) -> dict:
    """Integrate one null geodesic and return positions + covariant momenta.

    Returns a dict with:
      a      : spin parameter
      x      : (N, 4) Boyer-Lindquist coords [t, r, theta, phi]
      p_cov  : (N, 4) covariant momenta      [p_t, p_r, p_theta, p_phi]
    """
    a = float(config["black_hole"]["spin"])

    x0, p0 = make_null_initial_conditions(
        r=R0, theta=THETA0, E=E0, L_z=LZ0, Q=Q0, a=a, p_r_sign=P_R_SIGN
    )
    x, p_cov = integrate_null_geodesic(
        x0, p0, a=a, n_steps=N_STEPS, d_lambda=D_LAMBDA
    )

    x = np.asarray(x, dtype=float)
    p_cov = np.asarray(p_cov, dtype=float)
    assert x.shape == p_cov.shape
    assert x.shape[1] == 4
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
    E = -trajectory["p_cov"][:, 0]
    assert _relative_drift(E) < 1e-4


# --------------------------------------------------------------------------- #
# 2. Axial angular momentum:  L_z = p_phi
# --------------------------------------------------------------------------- #

def test_angular_momentum_conserved(trajectory):
    L_z = trajectory["p_cov"][:, 3]
    assert _relative_drift(L_z) < 1e-4


# --------------------------------------------------------------------------- #
# 3. Carter constant (NULL form, exact expression from spec):
#    Q = p_theta^2 + cos^2(theta) * (-a^2 E^2 + L_z^2 / sin^2(theta))
# --------------------------------------------------------------------------- #

def test_carter_constant_conserved(trajectory):
    a = trajectory["a"]
    theta = trajectory["x"][:, 2]
    p_theta = trajectory["p_cov"][:, 2]
    E = -trajectory["p_cov"][:, 0]
    L_z = trajectory["p_cov"][:, 3]

    Q = p_theta**2 + np.cos(theta) ** 2 * (
        -(a**2) * E**2 + L_z**2 / np.sin(theta) ** 2
    )
    assert _relative_drift(Q) < 1e-4


# --------------------------------------------------------------------------- #
# 4. Null condition:  g^{mu nu} p_mu p_nu == 0
#    Inverse metric comes from the implementation (kerr-physics skill).
# --------------------------------------------------------------------------- #

def test_null_condition_preserved(trajectory):
    a = trajectory["a"]
    x = trajectory["x"]
    p_cov = trajectory["p_cov"]

    norms = np.empty(x.shape[0], dtype=float)
    for i in range(x.shape[0]):
        g_inv = np.asarray(
            inverse_metric_bl(r=x[i, 1], theta=x[i, 2], a=a), dtype=float
        )
        p = p_cov[i]
        norms[i] = float(p @ g_inv @ p)

    assert np.max(np.abs(norms)) < 1e-6


# --------------------------------------------------------------------------- #
# Golden numeric regression: pin the conserved quantities (sampled) so future
# changes to the integrator that shift the physics get caught.
# --------------------------------------------------------------------------- #

def test_conserved_quantities_regression(trajectory, num_regression):
    a = trajectory["a"]
    x = trajectory["x"]
    p_cov = trajectory["p_cov"]
    theta = x[:, 2]

    E = -p_cov[:, 0]
    L_z = p_cov[:, 3]
    p_theta = p_cov[:, 2]
    Q = p_theta**2 + np.cos(theta) ** 2 * (
        -(a**2) * E**2 + L_z**2 / np.sin(theta) ** 2
    )

    # Sample at a fixed set of indices so the golden file is compact and stable.
    idx = np.linspace(0, x.shape[0] - 1, 11).astype(int)
    num_regression.check(
        {
            "lambda_index": idx.astype(float),
            "r": x[idx, 1],
            "theta": x[idx, 2],
            "E": E[idx],
            "L_z": L_z[idx],
            "Q": Q[idx],
        },
        default_tolerance=dict(atol=1e-8, rtol=1e-8),
    )
