# V3.0 — Static curl domain warp (Formula CKS-18)

**Status:** SPEC — owner-approved design 2026-06-14, default OFF (bit-identical to V2).
**Math of record:** SKILL.md **Formula CKS-18**. **Owner decisions (2026-06-14):**
(1) stage V3 as *static warp first* (V3.0) then *curl-flow advection* (V3.1) — the
D2.2→D2.3 split; (2) the V3.0 displacement is **in-plane `(u,φ)` only** (ζ untouched —
3D vertical churn deferred to a later increment).

## Problem

CKS-12 §2 is the disk's only motion: Keplerian **shear** advection winds the noise
azimuthally at `Ω(r)` with a dual-phase reset. That is *laminar* — it spirals, but has
no eddies, vortices, or organic billowing. V3 adds divergence-free turbulent structure
by warping the noise **sample coordinate** with the curl of a simplex potential. This is
exactly what the V1.5 isotropic simplex basis (`snoise2/3`, `sfbm2/3`) was built and
parked for: it has no axis-aligned grid bias (which would show up as directional streaks
in a flow field), and sampled on the `(cosφ, sinφ, u)` cylinder it is seamless across
φ=0 — the "exact φ-seamlessness obtained at the V3 integration point" the §3.6 note
promised.

V3.0 is the **static** half: a fixed divergence-free coordinate distortion (eddies/
billows frozen into the gas's material frame). V3.1 will make the same potential
time-dependent and reset-blended (true curl-flow advection).

## Model — in-plane static curl warp

A scalar potential on the **cylinder embedding** of the φ-axis:

```
P(u,φ) = ( cos φ · ρ_c , sin φ · ρ_c , u · k_u )        # ρ_c, k_u = angular / radial freq
ψ(u,φ) = sfbm3( P ; octaves, lacunarity, gain, curl.seed )   # isotropic simplex fBm
```

In-plane **divergence-free** displacement = curl of the scalar potential in the `(u,φ)`
chart, by central finite difference (simplex exposes no analytic gradient):

```
∂ψ/∂u ≈ ( ψ(u+ε,φ) − ψ(u−ε,φ) ) / 2ε
∂ψ/∂φ ≈ ( ψ(u,φ+ε) − ψ(u,φ−ε) ) / 2ε
δu = +∂ψ/∂φ        δφ = −∂ψ/∂u          # ∇·(δu, δφ) ≡ 0 by construction
u' = u + A·δu       φ' = φ + A·δφ        # A = curl.amp
```

Both `δu` and `δφ` are exactly 2π-periodic in φ (they are built on `cos φ`, `sin φ`), so
`φ'` is continuous across the seam — **CKS-12 constraint 5 is preserved** even though
classic simplex is not lattice-periodic. Seamlessness comes from the *embedding*, not a
lattice period, so `ρ_c` and `k_u` may be any real (no integer-period restriction, unlike
the φ-periodic density stack).

`ε` is a fixed small finite-difference step in the `(u,φ)` chart (`curl.fd_eps`,
default `1e-3`), independent of the frequencies.

## Integration — material-frame warp, two entry points

The warp is applied at the entry of `_disk_noise_m` (density stack) and `_mod_fbm4`
(§3 modulation envelopes) — the two functions that receive the **already-sheared**
per-phase `φ′_k` from CKS-12 §2. So the eddies are frozen into the gas's *material*
frame and the Keplerian shear winds them into filaments (the physically-sensible
composition; eddies advect with the gas, not the lab frame). Density and modulation
therefore swirl coherently because they share one warp at one coordinate.

The warp uses a **fixed `curl.seed`** (NOT the per-cycle reseed `c_k` that the density
field draws), so V3.0 is genuinely static: only the §2 winding animates over `t_disk`.
V3.1 will reintroduce time dependence in the potential itself.

In the static (`T ≤ 0`) path the warp acts on `φ` directly (material = lab when there is
no shear) — same code, same helper.

## Lookup / emission / metric — UNCHANGED

The warp moves only the noise sample coordinate. `ρ = gauss(ζ)·exp(clamp(m))`,
`emission ∝ ρ·g⁴·…`, `dτ = absb·ρ·ds`, the CKS-9 Doppler `g`, the CKS-11 `f_PT` shape,
the CKS-14/15/17 source-function / self-shadow machinery — all identical. The warp is a
pure texture-domain operation upstream of the layer-stack evaluation.

## Config — `disk.noise.curl` (base dials, no resolver change)

```yaml
disk:
  noise:
    curl:
      enabled: false      # default OFF ⇒ bit-identical to V2 (constraint 6)
      amp: 0.0            # A — displacement amplitude (look-dev dial; 0 ⇒ no warp)
      freq_phi: 3.0       # ρ_c — angular feature density (any real, not integer)
      freq_u: 1.0         # k_u — radial feature density
      octaves: 4
      lacunarity: 2
      gain: 0.5
      seed: 1337
      fd_eps: 0.001       # central-difference step in the (u,φ) chart
```

These are **base look dials** — nothing is a function of other parameters, so the CKS-13
resolver (`kerr_params.resolve_config`) is unchanged. The values flow through the
`_setup_disk_noise` parameter buffer (which grows past the current `_NOISE_N = 43`).

## Cost

Central-difference gradient = 4 `sfbm3` evaluations × `octaves`, per phase, for density
AND for modulation. Heavier than §2's plain fBm but parallel per sample and offline —
accepted. An analytic-gradient optimization (Gustavson simplex has closed-form
derivatives) is deferred; if the bake time is unacceptable, the fallback is a 3-eval
forward difference.

## Governance — VISUALIZATION, not GR

Same class as CKS-12 §2/§3 and CKS-14/15/17: the warp multiplies/relocates **amplitude**
(texture) quantities only. It must NEVER touch `p_μ`, `u^μ` (CKS-8), `g` (CKS-9), the
`g⁴` exponent (Formula 9), the blackbody chromaticity form, or the `f_PT` radial shape
(CKS-11). The divergence-free construction is the principled choice (an incompressible
flow neither piles up nor evacuates the texture), but it remains texturing, not transport
of a conserved field. A genuine fluid solve (advecting a real density with continuity)
would STOP for skill extension (CLAUDE.md policy).

## Tests

`tests/test_noise.py` (CPU) + `tests/test_noise_gpu.py` / `tests/test_disk_noise.py`
(GPU twin):

- **Divergence-free** — FD-verify `∇·(δu, δφ) ≈ 0` on a `(u,φ)` grid to FD tolerance
  (the defining property of the curl construction).
- **Seamlessness** — the warped coordinate is continuous across φ=0 ≡ 2π
  (`δu`, `δφ` agree at the seam; constraint 5).
- **Determinism** — same `curl.seed` ⇒ identical warp; no `ti.random` (constraint 7).
- **CPU↔GPU twin parity** — `curl_warp` NumPy vs `@ti.func` agree to a derived
  `amp·_SATOL/fd_eps` bound, NOT `_SATOL` itself: the warp is a *derivative* of the noise
  (`amp·(ψ₊−ψ₋)/(2ε)`), so the ~1e-5 `sfbm3` GPU/CPU twin gap (FMA / transcendental
  ordering) is amplified ×1/(2ε); observed ~6.5e-5 at `fd_eps=1e-3`, `amp=0.15`.
- **curl-off bit-identity** — `curl.enabled: false` (and `amp: 0`) take a branch
  bit-identical to the pre-warp kernel; existing `test_gpu_regression.py` goldens
  unchanged (default OFF).
- **curl-on smoke** — warped density field is finite / NaN-free.

## Deferred

- **V3.1** — animate `ψ` into time-dependent curl-flow advection (dual-phase reset +
  `dynamism`-style gain), composed additively with the §2 Keplerian shear.
- **3D curl** — a vector potential `A(cosφ,sinφ,u,ζ)` whose curl also displaces ζ, so the
  V2 bulk churns vertically (~3× the evals).
- **Analytic simplex gradient** — replace the 4-eval central difference.
- A dedicated curl-on golden frame (relational, V5 sign-off).
