"""GPU guard: the disk march must not stride over the thin emitting slab.

Physics policy (CLAUDE.md): this test re-derives no physics. It drives the
production entrypoint ``render_beauty_frame`` and compares the Pipe-B disk
emission buffer (``disk_buf``) under the disk-thickness step cap.

The bug
-------
The Pipe-B march accumulates a Riemann sum ``Σ j(r,θ)·ds`` over the affine step
``ds = h``. The base step rule (SKILL.md CKS-5 recommendation) sizes ``h`` only by
distance to the horizon — ``h = dλ·max(floor, (r−r₊)/r)`` — and is blind to the
disk's *vertical* extent. When a ray crosses the equatorial plane steeply
(face-on views) and the emitting layer is thin, one step can stride clean over the
Gaussian density ``∝ exp(−½(Δθ/σ)²)`` with ``σ = θ_half·σ_frac``, so the layer is
sampled 0/1/2 times almost at random across the frame — a concentric moiré band
around the disk. Shrinking the step makes it vanish (the user's observation).

The fix (``disk.max_step_vfrac``) caps the per-step *vertical* displacement
``|dz/dλ|·h`` to a fraction of the local scale height ``σ_z = r·θ_half·σ_frac``
while inside the slab, so the disk no longer depends on the global step size. It
only bites for steep crossings; near-in-plane / edge-on grazers keep the full step
(so it adds no cost and cannot push them into ``max_steps``).

This test reproduces the moiré on a thin disk seen face-on (where the production
config's thick disk would mask it) and pins that the capped production march
matches a non-truncated, finely-stepped ground truth. It FAILS if the cap is
removed/broken (the uncapped march aliases ~12% off the reference).

CUDA is mandatory (backend LOCKED to ``ti.init(arch=ti.cuda)``); skips cleanly on
a host without a working CUDA backend.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from renderer import taichi_renderer as tr

pytestmark = pytest.mark.gpu

_RES = 600

# Synthetic top-down camera: high on +z looking down the spin axis, so rays near
# frame center plunge through the equatorial plane steeply — the vertical
# under-sampling case. A small x-offset breaks the exact-pole degeneracy.
_POS = [0.6, 0.0, 30.0]
_NF = math.sqrt(sum(p * p for p in _POS))
_CAM = {
    "frame": 0,
    "pos": _POS,
    "fwd": [-p / _NF for p in _POS],
    "up": [0.0, 1.0, 0.0],
    "right": [1.0, 0.0, 0.0],
    "fov": 1.5,
}

# Thin disk: shrink the Gaussian scale height below the fixed step so the bug is
# active (the production thick disk is already well sampled and would hide it).
_THIN = {"vertical_sigma_frac": 0.05, "theta_half_width": 0.06}

# Convergence threshold. Measured (RTX 5060): capped production march ≈0.002 vs the
# ground truth; the *uncapped* march ≈0.12. 0.02 sits an order of magnitude below
# the broken value and above the converged one.
_MAX_REL_DIVERGENCE = 0.02


# A static (t_disk=0) noise block whose §3 scale-height modulation makes the slab
# LUMPY — the case CKS-12 constraint 4 protects: the Pipe-B step cap must size on the
# worst-case σ_z·(1 − h_amp/2), or the thinned lumps re-introduce the face-on moiré.
_MOD_THIN_NOISE = {
    "enabled": True, "seed": 1234, "m_max": 2.5, "dynamism": 1.0,
    "layers": {"base": {"enabled": True, "amp": 0.6, "octaves": 4, "lacunarity": 2.0,
                        "gain": 0.5, "freq_u": 6.0, "freq_phi": 24}},
    "modulation": {"enabled": True, "octaves": 3, "lacunarity": 2.0, "gain": 0.5,
                   "freq_u": 4.0, "freq_phi": 16, "temp_amp": 0.0, "edge_in_amp": 0.0,
                   "edge_out_amp": 0.0, "edge_softness": 0.0, "height_amp": 0.8},
}


def _render_disk_luminance(disk_overrides: dict, render_overrides: dict | None = None,
                           noise_override: dict | None = None) -> np.ndarray:
    cfg = copy.deepcopy(tr.load_config())
    cfg["disk"].update(_THIN)
    cfg["disk"].update(disk_overrides)
    # The production config now ships disk.noise.enabled: true (D2.4). This test
    # guards the *geometric* step cap, so force noise OFF unless a case explicitly
    # injects a block — otherwise the procedural texture would confound the
    # smooth-Gaussian convergence reference.
    cfg["disk"]["noise"] = noise_override if noise_override is not None else {"enabled": False}
    if render_overrides:
        cfg["render"].update(render_overrides)
    # setup_renderer re-runs ti.init (destroying all fields) and rebuilds the LUT;
    # reset the cached frame handles so _alloc_frame rebuilds against the fresh
    # runtime (same stale-handle workaround as test_gpu_regression).
    tr.setup_renderer(cfg)
    tr.frame_pixels = None
    tr._FW = 0
    tr._FH = 0
    tr.render_beauty_frame(cfg, _CAM, _RES, _RES, with_disk=True, lod_enabled=True)
    return np.nan_to_num(tr.disk_buf.to_numpy()[:, :, :3]).sum(axis=2)


@pytest.fixture(scope="module")
def divergence() -> float:
    import taichi as ti

    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without a CUDA driver
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")

    # Production-configured march (cap at the configured max_step_vfrac).
    prod = _render_disk_luminance({})

    # Ground truth: a tight cap AND a raised step budget so no ray truncates at the
    # finer step (truncation, not aliasing, would otherwise corrupt the reference).
    gt = _render_disk_luminance({"max_step_vfrac": 0.1}, {"max_steps_pipe_a": 16000})

    mask = gt > 0.02 * float(gt.max())
    if not np.any(mask):
        pytest.skip("no disk emission in reference frame — check camera/config")
    rel = np.abs(prod[mask] - gt[mask]) / (gt[mask] + 1e-9)
    div = float(rel.mean())
    print(f"\n[disk slab-convergence] mean relative divergence = {div:.4f} "
          f"(emitting px = {int(mask.sum())}, threshold = {_MAX_REL_DIVERGENCE})")
    return div


def test_disk_emission_resolves_thin_slab(divergence):
    assert divergence <= _MAX_REL_DIVERGENCE


# Looser than the smooth-slab bound: the lumpy §3 height field adds genuine
# high-frequency structure, so the capped march and the fine reference differ a
# touch more than for the plain Gaussian — but still an order of magnitude below the
# uncapped aliasing (which the worst-case σ_z cap is what prevents).
_MAX_REL_DIVERGENCE_MOD = 0.06


@pytest.fixture(scope="module")
def divergence_modulated() -> float:
    import taichi as ti

    try:
        ti.init(arch=ti.cuda)
    except Exception as exc:  # pragma: no cover - host without a CUDA driver
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if str(ti.cfg.arch) != "Arch.cuda":
        pytest.skip(f"CUDA backend unavailable (Taichi selected {ti.cfg.arch})")

    # Production cap, lumpy scale height ON (worst-case σ_z·(1−h_amp/2) step cap).
    prod = _render_disk_luminance({}, noise_override=_MOD_THIN_NOISE)
    # Ground truth: tight cap + raised step budget, SAME lumpy field.
    gt = _render_disk_luminance({"max_step_vfrac": 0.1}, {"max_steps_pipe_a": 16000},
                                noise_override=_MOD_THIN_NOISE)

    mask = gt > 0.02 * float(gt.max())
    if not np.any(mask):
        pytest.skip("no disk emission in modulated reference frame — check camera/config")
    rel = np.abs(prod[mask] - gt[mask]) / (gt[mask] + 1e-9)
    div = float(rel.mean())
    print(f"\n[disk slab-convergence, §3 height-modulated] mean relative divergence "
          f"= {div:.4f} (emitting px = {int(mask.sum())}, threshold = {_MAX_REL_DIVERGENCE_MOD})")
    return div


def test_disk_emission_resolves_lumpy_slab(divergence_modulated):
    """CKS-12 constraint 4: with §3 scale-height modulation the worst-case-σ_z step
    cap must still resolve the (now lumpy) thin slab — no returning moiré."""
    assert divergence_modulated <= _MAX_REL_DIVERGENCE_MOD
