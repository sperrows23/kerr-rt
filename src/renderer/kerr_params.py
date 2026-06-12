"""Derived Kerr / disk parameters — the config resolver (Formula CKS-13).

``configs/render.yaml`` stores only FOUNDATIONAL parameters (black-hole spin,
disk extent, target peak temperature, the time-mapping look targets under
``disk.dynamics``). Everything that is a *function* of those — horizon radius,
ISCO, the disk inner edge, the temperature-model amplitude ``T_0``, Keplerian
periods and the D2 noise time mapping — is computed here at load time by
:func:`resolve_config` and injected into the config dict. Editing
``black_hole.spin`` (or any other base parameter) can therefore never desync a
dependent literal again — the failure mode the old YAML ``r_isco: 1.182`` /
``r_plus: 1.0447`` comments flagged.

Physics policy (CLAUDE.md): nothing here is re-derived. Every expression is a
verbatim transcription, or a trivial algebraic inverse, of a pinned SKILL.md
formula:

  - :func:`horizon_radius`   Formula CKS-6   ``r₊ = 1 + √(1 − a²)``
  - :func:`isco_radius`      Formula 2       Bardeen–Press–Teukolsky (1972)
  - :func:`keplerian_omega`  Formula 3       ``Ω = 1/(r^{3/2} + a)`` (prograde)
  - :func:`orbital_period`   ``2π/Ω``        Formula 3 inverse (geometric M)
  - ``T_0`` derivation       Decision B / CKS-11 amplitude algebra (per model)

Why closed forms and not a NASA / literature lookup table: for Kerr orbital
quantities the BPT closed forms are EXACT — a table would only add sampling and
interpolation error on top of the very same equations. Published values are
instead pinned as test anchors (``tests/test_kerr_params.py``: a=0 → r_isco=6,
r₊=2 [Schwarzschild]; a=1 → both 1 [extremal prograde]). The one disk quantity
in this pipeline with no closed form — the Page–Thorne flux profile — already
IS a precomputed LUT (``disk_flux.build_flux_lut``, Formula CKS-11).

Taichi-free on purpose: this module must stay importable by ``scripts/thumb.py``
and the CPU tests without pulling in the CUDA runtime.
"""

from __future__ import annotations

import math

# Canonical homes of the two closed forms (do NOT duplicate the math here):
#   isco_radius      — Formula 2 transcription, pinned by tests/test_disk_flux.py
#   _horizon_radius  — Formula CKS-6 CPU reference (geodesic integrator's copy)
from renderer.disk_flux import isco_radius  # noqa: F401  (re-exported)
from renderer.geodesic import _horizon_radius

TWO_PI = 2.0 * math.pi


def horizon_radius(a: float) -> float:
    """Outer horizon ``r₊ = 1 + √(1 − a²)`` (Formula CKS-6; r is the BL radius)."""
    return float(_horizon_radius(a))


def keplerian_omega(r: float, a: float) -> float:
    """Prograde equatorial circular-orbit angular velocity (Formula 3, verbatim).

    ``Ω = 1 / (r^{3/2} + a)`` — coordinate angular velocity dφ/dt in geometric
    units (G = M = c = 1). Valid for ``r ≥ r_isco`` (the only regime the disk
    samples; ``disk.r_inner ≥ r_isco`` is enforced by :func:`resolve_config`).
    """
    return 1.0 / (r ** 1.5 + a)


def orbital_period(r: float, a: float) -> float:
    """One full revolution at BL radius ``r``: ``T_orb = 2π/Ω = 2π·(r^{3/2} + a)``.

    Trivial inverse of Formula 3; geometric units (M). For a distant observer
    this is coordinate time t per 2π of φ.
    """
    return TWO_PI * (r ** 1.5 + a)


def shear_wrap_time(r_inner: float, r_outer: float, a: float) -> float:
    """Time for one full differential 2π wrap across the disk (CKS-12 shear).

    ``t_wrap = 2π / (Ω(r_inner) − Ω(r_outer))`` — after this much disk time the
    inner edge has lapped the outer edge exactly once, i.e. a radially coherent
    pattern advected by Formula CKS-12 §2 has been sheared through one whole
    turn. The CKS-12 dual-phase reset period is expressed in these units.
    """
    d_omega = keplerian_omega(r_inner, a) - keplerian_omega(r_outer, a)
    return TWO_PI / d_omega


def resolve_config(cfg: dict) -> dict:
    """Inject all derived parameters into a freshly YAML-loaded config (in place).

    Base → derived map (Formula CKS-13):

    ``black_hole.spin`` →
      - ``black_hole.r_plus``  (CKS-6) — always overwritten, never read from YAML
      - ``black_hole.r_isco``  (Formula 2) — always overwritten

    ``disk.r_inner`` —
      - ``"auto"`` or absent → ``r_isco`` (zero-torque inner edge)
      - a number → artistic override; values below ``r_isco`` are clamped UP to
        it (CKS-11/CKS-12: no emission inside the ISCO, gas plunges)

    ``disk.target_peak_temperature`` → ``disk.T_0`` (skipped when an explicit
    legacy ``disk.T_0`` key is present — that override wins):
      - ``page_thorne``: ``T_0 = T_peak`` — the f_PT LUT is normalized to max 1,
        so ``max T_eff = T_0·f_PT^¼ = T_0`` (CKS-11).
      - ``simple``: ``T_0 = T_peak·(r_inner/6)^{3/4}`` — ``T = T_0·(6/r)^{3/4}``
        (Decision B) peaks at ``r = r_inner``, so this makes that peak hit
        ``T_peak`` regardless of where spin puts the inner edge.

    ``disk.dynamics`` (optional block; required base keys when present:
    ``inner_lap_seconds``, ``shear_wrap_budget`` — per project rule, no code
    defaults) → derived keys written into the same block:
      - ``omega_inner`` / ``omega_outer``      (Formula 3)
      - ``period_inner_M`` / ``period_outer_M`` (2π/Ω, geometric M)
      - ``wrap_time_M``                         (:func:`shear_wrap_time`)
      - ``time_scale``      = period_inner_M / inner_lap_seconds — geometric M of
        disk time per second of footage; D2 uses
        ``t_disk = frame/fps · time_scale``.
      - ``shear_period_M``  = shear_wrap_budget · wrap_time_M — the CKS-12 §2
        dual-phase reset period T.

    Idempotent: resolving an already-resolved dict reproduces the same values
    (derived keys are recomputed; ``T_0``/``r_inner`` keep their resolved
    values, which already satisfy the constraints). Resolution happens at load
    time from the YAML — mutate the YAML, not the resolved dict.
    """
    bh = cfg["black_hole"]
    a = float(bh["spin"])
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"black_hole.spin must be in [0, 1] (prograde Kerr), got {a}")
    r_plus = horizon_radius(a)
    r_isco = float(isco_radius(a))
    bh["r_plus"] = r_plus
    bh["r_isco"] = r_isco

    d = cfg["disk"]
    r_inner_raw = d.get("r_inner", "auto")
    if isinstance(r_inner_raw, str):
        if r_inner_raw != "auto":
            raise ValueError(f"disk.r_inner must be 'auto' or a number, got {r_inner_raw!r}")
        r_inner = r_isco
    else:
        # Artistic override — but never inside the ISCO (zero-torque BC; the
        # Formula-3 circular orbit and the CKS-11 flux are undefined below it).
        r_inner = max(float(r_inner_raw), r_isco)
    d["r_inner"] = r_inner

    r_outer = float(d["r_outer"])
    if r_outer <= r_inner:
        raise ValueError(f"disk.r_outer ({r_outer}) must exceed disk.r_inner ({r_inner})")

    if "T_0" not in d:
        t_peak = float(d["target_peak_temperature"])
        if t_peak <= 0.0:
            raise ValueError(f"disk.target_peak_temperature must be > 0, got {t_peak}")
        model = str(d.get("temperature_model", "simple"))
        if model == "page_thorne":
            d["T_0"] = t_peak
        else:
            d["T_0"] = t_peak * (r_inner / 6.0) ** 0.75

    dyn = d.get("dynamics")
    if dyn is not None:
        lap_s = float(dyn["inner_lap_seconds"])
        budget = float(dyn["shear_wrap_budget"])
        if lap_s <= 0.0 or budget <= 0.0:
            raise ValueError("disk.dynamics: inner_lap_seconds and shear_wrap_budget must be > 0")
        dyn["omega_inner"] = keplerian_omega(r_inner, a)
        dyn["omega_outer"] = keplerian_omega(r_outer, a)
        dyn["period_inner_M"] = orbital_period(r_inner, a)
        dyn["period_outer_M"] = orbital_period(r_outer, a)
        dyn["wrap_time_M"] = shear_wrap_time(r_inner, r_outer, a)
        dyn["time_scale"] = dyn["period_inner_M"] / lap_s
        dyn["shear_period_M"] = budget * dyn["wrap_time_M"]

    return cfg


def derived_report(cfg: dict) -> str:
    """Human-readable summary of what :func:`resolve_config` derived (for logs)."""
    bh, d = cfg["black_hole"], cfg["disk"]
    lines = [
        f"a = {bh['spin']}:  r_plus = {bh['r_plus']:.6f} M   r_isco = {bh['r_isco']:.6f} M",
        f"disk: r_inner = {d['r_inner']:.6f} M   r_outer = {float(d['r_outer']):.3f} M   "
        f"T_0 = {float(d['T_0']):.1f} K ({d.get('temperature_model', 'simple')})",
    ]
    dyn = d.get("dynamics")
    if dyn is not None:
        lines.append(
            f"dynamics: T_orb(inner) = {dyn['period_inner_M']:.3f} M   "
            f"T_orb(outer) = {dyn['period_outer_M']:.1f} M   "
            f"wrap = {dyn['wrap_time_M']:.3f} M"
        )
        lines.append(
            f"          time_scale = {dyn['time_scale']:.4f} M/s   "
            f"shear_period = {dyn['shear_period_M']:.3f} M"
        )
    return "\n".join(lines)
