# V2 — Vertical self-shadow: 3D inner-edge ray (Formula CKS-17)

**Status:** ✅ IMPLEMENTED 2026-06-14, default OFF (bit-identical to V1.2).
**Math of record:** SKILL.md **Formula CKS-17**. **Owner decision:** 3D inner-edge ray
(over a separable top-down column), 2026-06-14.

## Problem

CKS-15's deep-shadow-map marches a **radial** ray at constant `(φ, ζ)`: it captures gas
casting wakes *outward* but cannot capture **vertical** occlusion — an off-midplane parcel
shadowed by the dense midplane gas between it and the hot inner edge. SKILL.md CKS-15
explicitly deferred this until "the V2 3D bulk" existed. V2 (CKS-16 flared density) now
supplies that bulk, so vertical self-shadow is unblocked.

## Model — 3D inner-edge ray

Keep CKS-15's premise (the hot inner edge is THE illuminator) but make the shadow ray
**3D**: from the illuminator at the inner edge **in the midplane** `(u=0, ζ=0)` to the
sample `(u_s, φ, ζ_s)`, at fixed `φ` (azimuthal bending ignored, as in CKS-15).

```
ζ(u) = (u / u_s) · ζ_s              # ζ(0)=0 at inner edge, ζ(u_s)=ζ_s at the sample
r(u) = r_inner · e^u
Z(u) = r(u) · ζ(u) · σ_θ(r(u))      # near-equator physical height; σ_θ = CKS-16 flared
```

Bake (same field/grid/lookup as CKS-15), for target cell `(i_u, φ, i_z)`, marching the
strictly-inner radial cells `j < i_u` (a cell never shadows itself):

```
ζ_j  = (u_j / u_s) · ζ_s            u_j = (j+½)·du,  u_s = (i_u+½)·du
ρ_j  = _disk_density_cks(φ, r_j, dz_ang = ζ_j·σ_θ(r_j))     # tilted sample, shared ρ
ds_j = √( (r_j·du)² + ΔZ_j² )       ΔZ_j = Z(u_j+½du) − Z(u_j−½du)
τ_shadow(i_u,φ,i_z) = min( Σ_{j<i_u} absb_c · ρ_j · ds_j , max_tau )
```

**CKS-15 is the ζ=0 limit.** At `ζ_s=0` the ray is flat (`ζ_j≡0`, `ΔZ_j≡0`), so `ds=r·du`
and `ρ` is the midplane density — CKS-15's radial column term for term. The radial element
keeps the `dr=r·du` convention (not an endpoint `ΔR`) precisely so this reduction is
bit-exact; CKS-15 is not a separate code path.

**Off-midplane changes.** For `ζ_s≠0` the ray tilts toward the midplane going inward
(`ζ_j < ζ_s`), traversing **denser** gas than CKS-15's constant-`ζ_s` column. Offline
check at the shipped config: at `ζ≈2.8` (off-plane top) the 3D τ is ~10× the radial τ —
the off-plane gas is now correctly shadowed by the bright midplane slab.

## Lookup + application — UNCHANGED

Trilinear (φ-periodic) sample of `τ_shadow` at the primary sample's `(u, φ, ζ)`, then
`emission *= exp(−shadow_strength·τ_s)` on the EMISSIVITY only (`κ`/`dτ` untouched;
composes with CKS-14 so `S` inherits `e^{−τ_s}`). Only `bake_disk_shadow`'s ray geometry
changed — `disk_shadow_tau`, `_sample_shadow_tau`, and the `_disk_emit_cks` application are
identical. **No new config, field, or flag** — same `disk.volumetric.self_shadow.enabled`.

## Cost

Each target `ζ_s` tilts its own ray (no shared prefix), so the bake is `O(NU)` per cell ⇒
`O(NU²·NPHI·NZ)` overall, ~`NU/2`× more density evals than CKS-15, parallelised over all
cells on the GPU. Accepted for the offline bake (owner chose the 3D ray knowing it is the
heavier model).

## Governance — VISUALIZATION, not GR

Same class as CKS-15: straight CKS shadow ray (not a geodesic), single inner-edge
illuminator (midplane), single-scatter, no re-emission along the shadow march. Multiplies
the emission amplitude only (CKS-12 constraint 1) — never `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT`/
chroma-form. A physical transport (geodesic rays, multi-scatter, phase function) would
STOP for skill extension (CLAUDE.md policy).

## Tests

`tests/test_disk_self_shadow.py` (gpu-marked):

- **flag-off bit-identity**, **outward-steepening dimming**, **noise-on contrast-rise** —
  carry over unchanged (relational checks).
- **`test_bake_matches_analytic_3d_ray_integral`** — re-derived from the CKS-15 constant-ζ
  radial closed form to the CKS-17 tilted-ray line integral (mirrors the bake term for
  term at f32 tolerance). The old radial expectation was the CKS-15 model, superseded
  off-midplane.
- `test_gpu_regression.py` unchanged (default-off ⇒ goldens bit-identical).

## Deferred

A dedicated vertical-self-shadow golden frame (relational, V5 sign-off); V3 curl-flow.
