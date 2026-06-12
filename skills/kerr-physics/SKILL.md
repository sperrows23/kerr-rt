# kerr-physics

## When to use this skill

Load this skill whenever the task involves:
- Kerr black hole geodesic integration
- Accretion disk gas velocity or emission
- Observer/camera setup in curved spacetime
- Photon momentum initialization (tetrad)
- Doppler beaming, redshift, or g-factor computation
- Anti-aliasing / mipmap LOD for the starmap
- Point-star magnification, ray-bundle Jacobian, or star PSF (Formula 13)
- Disk procedural turbulence / noise shear-advection (Formula CKS-12)
- Any formula involving `r_isco`, `E_I`, `L_I`, `u^t`, `u^r`, `u^phi`, `g-factor`, `Carter Q`

---

## CRITICAL RULE

**Do NOT re-derive any formula in this skill from scratch.**
LLM re-derivation introduces sign errors, index mismatches, and normalization mistakes.
Use the formulas below **verbatim**. If a formula seems wrong, flag it for human review —
do not silently substitute a re-derived version.

---

## Unit and coordinate conventions

| Convention | Value |
|---|---|
| Units | Geometric: G = M = c = 1 |
| Black hole mass | M = 1 (all distances in units of M) |
| Spin parameter | a = 0.999 (near-maximal prograde) |
| Geodesic type | **Null** (photons, μ = 0) — affects Carter constant sign |
| Coordinates | **Cartesian Kerr-Schild (CKS): (t, x, y, z) — active renderer path (see PART II)**. Boyer-Lindquist (BL): (t, r, θ, φ) — retired/CPU-reference only. |
| Metric signature | (− + + +) |
| GPU backend | `ti.init(arch=ti.cuda)` — locked, do not change to `ti.gpu` |
| Orbit direction | Prograde (co-rotating with spin) everywhere |

---

## Formula 1 — Kerr metric in Boyer-Lindquist coordinates

**Source:** Kerr (1963). Decision locked — do not substitute Kerr-Schild.

```
Σ = r² + a²·cos²θ
Δ = r² − 2r + a²

g_tt  = −(1 − 2r/Σ)
g_tφ  = −2ar·sin²θ / Σ
g_φφ  = (r² + a² + 2r·a²·sin²θ/Σ) · sin²θ
g_rr  = Σ / Δ
g_θθ  = Σ
```

All other metric components are zero.

---

## Formula 2 — ISCO radius (prograde)

**Source:** Bardeen–Press–Teukolsky (1972). Use prograde (co-rotating) form only.

```
Z₁ = 1 + (1 − a²)^{1/3} · [(1 + a)^{1/3} + (1 − a)^{1/3}]
Z₂ = sqrt(3·a² + Z₁²)
r_isco = 3 + Z₂ − sqrt((3 − Z₁)(3 + Z₁ + 2·Z₂))
```

**Verified value for a = 0.999:**
```
Z₁ ≈ 1.1713,  Z₂ ≈ 2.0895,  r_isco ≈ 1.182 M
```

Do not use the retrograde formula. Do not use a numerical root-finder.

---

## Formula 3 — Circular orbit 4-velocity (r ≥ r_isco)

**Source:** Bardeen (1970).

```
Ω   = 1 / (r^{3/2} + a)

u^t = (1 + a·r^{−3/2}) / sqrt(1 − 3/r + 2·a·r^{−3/2})

u^φ = Ω · u^t

u^r = 0
u^θ = 0
```

**Critical:** the numerator `(1 + a·r^{-3/2})` in `u^t` must be present.
Using `u^t = 1/sqrt(...)` (i.e. numerator = 1) is wrong and produces ~8% error
in Doppler colors near ISCO for a = 0.999.

---

## Formula 4 — ISCO conserved quantities E_I and L_I

**Source:** Cunningham (1975). Evaluated at r_I = r_isco.

```
denom_I = r_I · sqrt(r_I² − 3·r_I + 2·a·sqrt(r_I))

E_I = (r_I² − 2·r_I + a·sqrt(r_I)) / denom_I

L_I = sqrt(r_I) · (r_I² − 2·a·sqrt(r_I) + a²) / denom_I
```

These are frozen at the ISCO boundary and carried unchanged into the plunging region.
Do not recompute E_I or L_I at any r < r_isco.

---

## Formula 5 — Plunging region 4-velocity (r < r_isco)

**Source:** Cunningham (1975). Free-fall with conserved E_I, L_I from Formula 4.

```
Σ = r² + a²·cos²θ
Δ = r² − 2r + a²

X  = E_I·(r² + a²) − a·L_I        # intermediate quantity

u^r = −(1/Σ) · sqrt(max(0.0, X² − Δ·(r² + (L_I − a·E_I)²)))

u^t = (1/Σ) · ((r² + a²)·X/Δ − a·(a·E_I − L_I))

u^φ = (1/Σ) · (a·X/Δ − (a·E_I − L_I))

u^θ = 0
```

**Sign rule:** `u^r` must be **negative** (infalling). A positive sign means
unphysical outflowing gas — check the sign before proceeding.

The `max(0.0, ...)` clamp inside the sqrt prevents NaN from floating-point
noise near the horizon. Keep it.

---

## Formula 6 — Carter constant Q and Mino-time geodesic equations

**Source:** Carter (1968) for Q; Mino (2003) for the time substitution.

### ⚠ NULL GEODESIC FORM (μ = 0) — photons only

This pipeline traces photons, not massive particles. The Carter constant
differs between the two cases:

```
# CORRECT — null geodesic (photons, μ = 0):
Q = p_θ² + cos²θ · (−a²·E² + L_z²/sin²θ)

# WRONG for this pipeline — massive particle (μ = 1):
# Q = p_θ² + cos²θ · (a²·(1−E²) + L_z²/sin²θ)   ← DO NOT USE
```

Using the massive-particle form causes Q to drift by `a²·cos²θ` as θ changes
along a null geodesic, which will **fail the pytest conservation harness**.

### Mino-time substitution

```
dλ = dτ / Σ       (Mino affine parameter)
```

### Separated equations of motion

```
R(r) = [E·(r² + a²) − a·L_z]² − Δ·[(L_z − a·E)² + Q]   ← null form (μ = 0)
Θ(θ) = Q − cos²θ · (−a²·E² + L_z²/sin²θ)          ← null form

dr/dλ  = ±sqrt(R(r))
dθ/dλ  = ±sqrt(Θ(θ))

dφ/dλ  = −(a·E − L_z/sin²θ) + a·[E·(r²+a²) − a·L_z] / Δ

dt/dλ  = −a·(a·E·sin²θ − L_z) + (r²+a²)·[E·(r²+a²) − a·L_z] / Δ
```

**Null `R(r)`:** the radial potential above is the **null** form. The general
potential contains a `−Δ·μ²·r²` term; for photons (μ = 0) that term is **absent**.
Including it makes `g^{μν} p_μ p_ν = −r²/Σ ≠ 0` and fails the null-condition test.
(See revision history v1.2.)

**Why Mino time over direct τ integration:** decouples r and θ into independent
1D potentials; handles Δ→0 near the horizon naturally. The original code's
ad hoc `Σ·dτ` steps do not have this property and become unstable near the
photon sphere. Upgrade the integrator to use these equations.

---

## Formula 7 — Observer tetrad (ZAMO frame)

**Source:** Bardeen et al. (1972). DNGR Appendix A.1/A.2. Chosen frame: ZAMO.

**Why ZAMO:** always well-defined outside the event horizon; avoids the static
observer singularity inside the ergosphere (important for a = 0.999 where
the ergosphere is large). Used by DNGR.

### ZAMO quantities — exact formulation

```
Σ = r² + a²·cos²θ
Δ = r² − 2r + a²

A = (r² + a²)² − a²·Δ·sin²θ       # ← exact; do not approximate as (r²+a²)²

ω = 2ar / A                         # ZAMO angular velocity (= −g_tφ/g_φφ)

α = sqrt(Σ·Δ / A)                   # lapse function — exact form
```

**Why the exact A matters:** approximating `A ≈ (r²+a²)²` (dropping `a²·Δ·sin²θ`)
introduces ~2% error in α near the equatorial plane at a=0.999, r≈2.
Since the accretion disk lives at the equator, this directly affects the tetrad
vectors that initialize photon momenta from disk-crossing rays.
Use the exact formula above.

### Tetrad basis vectors (local orthonormal frame)

```
e^μ_{(t̂)} : components  (1/α,  0,  0,  ω/α)   in (t, r, θ, φ)
e^μ_{(r̂)} : components  (0,  sqrt(Δ/Σ),  0,  0)
e^μ_{(θ̂)} : components  (0,  0,  1/sqrt(Σ),  0)
e^μ_{(φ̂)} : components  (0,  0,  0,  1/sqrt(g_φφ))
```

where `g_φφ = A·sin²θ/Σ`.

### Mapping camera ray direction to covariant photon momenta

Given a local camera ray direction unit vector `n = (n^r̂, n^θ̂, n^φ̂)`:

```
p^μ = e^μ_{(t̂)} + n^r̂·e^μ_{(r̂)} + n^θ̂·e^μ_{(θ̂)} + n^φ̂·e^μ_{(φ̂)}

# Lower index using metric:
p_t  = g_tt·p^t + g_tφ·p^φ
p_r  = g_rr·p^r          =  (Σ/Δ)·p^r
p_θ  = g_θθ·p^θ          =  Σ·p^θ
p_φ  = g_φφ·p^φ + g_tφ·p^t

# Extract conserved quantities:
E    = −p_t
L_z  = p_φ
Q    = p_θ² + cos²θ · (−a²·E² + L_z²/sin²θ)   ← null geodesic form
```

This replaces the heuristic dot-product projection used in the original code.
The original approach is only valid far from the black hole with a narrow FOV.

---

## Formula 8 — g-factor (relativistic Doppler + gravitational redshift)

**Source:** Standard GR. Penrose (1966).

```
g = E_obs / E_emit = (p_μ · u^μ)_obs / (p_μ · u^μ)_emit
```

For a camera/observer at rest at spatial infinity:
```
(p_μ · u^μ)_obs = −E = −1

g = −1 / (p_t·u^t + p_r·u^r + p_θ·u^θ + p_φ·u^φ)
```

where all `p_μ` are **covariant** and all `u^μ` are **contravariant**.

### KNOWN BUG in original code

```python
# WRONG (original code):
p_r_cov = p_r / Delta

# WHY WRONG:
# If p_r is already covariant (from the Hamiltonian equations),
# dividing by Δ converts it to the wrong thing.
# The correct relationship is:
#   p^r (contravariant) = (Δ/Σ) · p_r (covariant)
#   p_r (covariant)     = (Σ/Δ) · p^r (contravariant)
# Inverting this corrupts every Doppler color in the disk.

# CORRECT: determine which form p_r is in your integration loop,
# then use it as-is (covariant) or convert properly.
```

---

## Formula 9 — Doppler beaming intensity

**Source:** Misner–Thorne–Wheeler §22.6. Lorentz invariant: `I_ν / ν³ = const`.

```
# 2D surface emitter:
I_obs = g³ · I_emit

# 3D volumetric emitter (use this — pipeline uses volume marching):
I_obs = g⁴ · I_emit
```

### blackbody_rgb — chromaticity only, NOT absolute intensity

The `blackbody_rgb(T)` helper in this codebase returns **normalized chromaticity**:
RGB values in [0, 1] representing the color of the spectrum at temperature T,
with no T⁴ amplitude scaling. It is implemented as:

```python
r_col = 1.0 - exp(-T / 3500.0)
g_col = 1.0 - exp(-T / 5500.0)
b_col = 1.0 - exp(-T / 9500.0)
```

Because the output has no built-in T⁴ amplitude, applying `pow(g, 4.0)` as
the intensity factor is **correct and not double-counted**.

**Warning:** if `blackbody_rgb` is ever replaced with a physically calibrated
Planck spectrum that includes Stefan-Boltzmann T⁴ scaling, the `pow(g, 4.0)`
multiplier must be **removed** — the T = g·T_emit substitution already carries
the g⁴ factor implicitly through T⁴. Failing to remove it would scale intensity
by g⁸ instead of g⁴.

### Required fix for the original code

```python
# Original code (incomplete — applies g to temperature only):
T_obs = g * T_emit
j_obs = blackbody_rgb(T_obs)
PixelColor += Transmittance * emission

# Correct (apply g⁴ to intensity as well):
T_obs = g * T_emit
j_obs = blackbody_rgb(T_obs)       # chromaticity only
PixelColor += Transmittance * emission * pow(g, 4.0)
```

Without the `g⁴` factor, both sides of the disk appear nearly symmetric.
With it, the approaching limb is 10–100× brighter for a = 0.999.
DNGR suppressed this intentionally for artistic reasons.
This pipeline includes it; to suppress, clamp g to 1.0 in the intensity factor only.

---

## Formula 10 — Differential ray Jacobian for mipmap LOD

**Source:** DNGR James et al. (2015) §4. Single-ray approximation of ray-bundle technique.

### Why this solves starmap flickering

Near the photon ring, one pixel subtends a large region of the sky. Without
mipmap LOD, individual star texels smaller than a pixel flicker as the camera
moves. The Jacobian measures how much sky area a pixel covers after lensing
and selects the appropriate mipmap level to blur it away.

### Implementation

```
# Step 1: Integrate primary ray
#   → record exit direction (θ_exit, φ_exit)

# Step 2: Integrate offset ray in parallel
#   Initial direction: shift u by +1/WIDTH (one pixel step)
#   → record exit direction (θ_exit + δθ, φ_exit + δφ)

# Step 3: Compute Jacobian on celestial sphere (raw per-pixel exit deltas)
J = sqrt( δθ² + sin²(θ_exit) · δφ² )

# Step 4: Mipmap level (starmap_width = 16384; /2π maps angle → texels)
L = clamp(log2(16384.0 * J / (2π)),  0.0,  log2(16384.0))

# Step 5: Sample mipmap pyramid trilinearly at level L
color = sample_starmap_mip(θ_exit, φ_exit, L)
```

### What the original code did wrong

1. `ray_dir_diff` was initialized but never integrated to exit.
2. LOD was estimated as `bending_factor = steps / 100` — uncorrelated with
   sky footprint area; produces wrong LOD values everywhere.
3. Even that LOD was discarded: `sample_starmap_mip` performed simple
   bilinear at full resolution regardless of the `lod` argument.

**The fix is adding ~5 lines:** integrate the offset ray inside the existing
while loop, compute J at exit, pass L to the sampler. This simultaneously
fixes aliasing AND improves texture cache locality (reducing VRAM bandwidth
pressure).

### Amendment v1.4 — screen-space Jacobian (offset ray eliminated)

**Source:** DNGR §4 ray-bundle, screen-space finite-difference variant. Approved
2026-06-02 for the 4K performance budget.

The Jacobian `J` may equivalently be estimated **in screen space** by
finite-differencing the stored exit directions of the pixel's 4-neighborhood,
instead of integrating a second (offset) ray:

```
# Kernel 1 (physics) writes per-pixel exit state to a field:
#   exit[py, px] = (u_exit = cosθ_exit,  φ_exit,  outcome)
#
# Kernel 2 (shading) reads neighbors and differences them:
δθ = θ(py, px+1) − θ(py, px)          # θ = acos(clamp(u,−1,1))
δφ = wrap_pi( φ(py, px+1) − φ(py, px) )
J  = sqrt( δθ² + sin²(θ) · δφ² )       # identical to the offset-ray J
L  = clamp(log2(16384.0 * J / (2π)), 0.0, log2(16384.0))
```

`J`, `L`, and the 2π texel normalization are **unchanged** from the offset-ray
method above — only the source of the (δθ, δφ) pair differs.

**Boundary rule (mandatory):** if any neighbor used in the difference did **not**
ESCAPE (it was CAPTURED, or fell off the screen edge), clamp `L = max_lod`. This
mirrors the offset-ray method's `out_o != ESCAPED → L = max_lod` branch and
prevents the escaped/captured discontinuity at the photon ring from producing a
spurious tiny footprint (over-sharp aliasing at the shadow edge).

This eliminates the per-pixel offset ray (halving the geodesic workload) at the
cost of one extra (cheap) shading pass over a 2D field.

### Fidelity note — texture-mip LOD vs. DNGR ray bundles (FLAGGED, no code change)

**Logged 2026-06-04 after comparing the implementation against
`REFERENCE_dngr_paper.md` (DNGR,
James et al. 2015, Appendix A.2 ray bundles + A.3.1 spatial filtering). This is
an architectural caveat for human review — it does NOT change Formula 10, which
is correctly implemented and faithful to the paper *as a single-ray AA filter*.**

What matches the paper (confirmed correct, do not "fix"):

- Our celestial coordinate is the Boyer-Lindquist exit angle pair {θ′, φ′}.
  That is exactly the paper's local-sky→celestial-sphere map (their step v).
  There is no separate "asymptotic direction" to recover — the BL exit angles
  *are* the celestial coordinate.
- `J = √(δθ² + sin²θ·δφ²)`, `L = log2(W·J/2π)` is a faithful scalar reduction of
  the ray bundle's solid-angle change, and is the correct filter for **extended**
  background (the paper filters disks / nebulae / dust this way — A.3.1 bullet 3,
  A.6).

Where our architecture structurally diverges from DNGR (the part to review):

1. **Point stars must not blur.** DNGR's #1 stated benefit (A.3.1): *"images of
   our unresolved stars remain small; they don't stretch when magnified by
   gravitational lensing"* — magnification is converted to **brightness**, not
   blur (bullet 2), by keeping a **point-star catalog** and collecting each star
   into a finite beam. We bake stars into a 16K equirect **texture** and mip-blur
   it, so magnification necessarily smears stars into arcs and dims them (visible
   as arc-smearing of the lensed star field). This is a data-model choice
   (baked texture vs. point catalog), NOT a Formula-10 error.
2. **Anisotropy.** DNGR tracks the full beam ellipse (major δ⁺, minor δ⁻,
   orientation µ). We collapse to the scalar `J = max(Jx, Jy)`. Near critical
   curves magnification is highly anisotropic (large tangential, tiny radial),
   so the scalar over-blurs radially. The within-architecture analog is an
   anisotropic / EWA texture filter — but note EWA still blurs point stars, so
   it only helps extended structure, not point-star fidelity (item 1).
3. **Finite `r_max`.** We read exit angles at `r = r_max` (50); the paper's
   celestial sphere is `r = ∞`. Small truncation residual; raise `r_max` or
   extrapolate if {θ′, φ′} convergence ever matters. Not the dominant effect.

> **Status note (2026-06-07):** divergences 1 & 2 above describe the *legacy
> `texture` mode* and are **resolved in the shipped `dngr` mode** (Formula 13,
> §8). Stars now come from the point-star catalog (Layer A) and are brightened by
> µ, not blurred — they stay sharp. Layer B (the EWA-filtered diffuse plate) is
> now a **genuinely starless** map (`starmap_final.exr`); EWA only ever sees
> low-frequency dust/galaxy light, so the "EWA still blurs point stars" caveat in
> item 2 no longer bites in practice. The acceptance bar for a Layer-B plate is
> **no resolvable point sources**: every feature must be broader than the widest
> per-pixel EWA footprint (≈ the 90°-corner minor axis), i.e. no isolated local
> maxima above the smooth diffuse band (measured: sharp >10× spikes ≤ ~0.05% of
> lit pixels). A plate that merely *dims* stars is NOT acceptable — the smear
> depends on a point source being present, not on its brightness. This bar is
> enforced by `scripts/check_starless_map.py` (run it on any candidate Layer-B
> plate; it exits non-zero unless the >10× spike fraction clears 0.05%).

Verified separately: the brown "starless" wash in `scripts/gpu_test_disk.png`
is the **lensed/embedded accretion disk** (camera at `r=18`, inside `r_outer=25`,
near the equatorial plane → nearly every ray accumulates disk emission), NOT
LOD-coarsened stars. Disk-off renders show the full lensed star field is present.
The exit-interpolation fix (see `_screen_jacobian_lod` / `render_beauty_physics`
escape branch) removed step-overshoot jitter and brought the undeflected-corner
LOD from ≈4.2 down to ≈2.3 (≈ ideal 1.74 + real geometric magnification).

---

## Formula 11 — FP32-stable factored discriminant (variable transform)

**Source:** algebraic identity of Formula 1's Δ. Added 2026-06-02 for FP32
horizon stability (optimization Phase 1.2/1.3; see PROJECT.md §6). **Not new physics** — a factoring that
removes catastrophic cancellation in `r²−2r+a²` near the horizon.

```
k   = sqrt(1 − a²)                 # k_horizon
r±  = 1 ± k                        # outer/inner horizon radii (r₊ = config r_plus)
y   = r − r₊                       # horizon-relative radial coordinate

Δ   = (r − r₊)(r − r₋) = y·(y + 2k)    # ≡ r² − 2r + a², zero cancellation
```

**Verification:** since `r₊ + r₋ = 2` and `r₊·r₋ = a²`,
`y(y+2k) = (r−r₊)(r−r₋) = r² − (r₊+r₋)r + r₊r₋ = r² − 2r + a²`. ✓

Use `_delta_y(y, k) = y*(y + 2.0*k)` wherever Δ is needed in the y-state
integrator. Recover `r = y + r₊` for potentials that need `r` explicitly.

---

## Formula 12 — Singularity-free polar potential (u = cosθ substitution of F6 Θ)

**Source:** Mino-time μ = cosθ substitution of the verbatim Formula 6 Θ(θ).
Standard reduction (cf. DNGR Appendix A; Carter 1968). Added 2026-06-02 for the
North-pole FP32 blowout fix (optimization Phase 1.3; see PROJECT.md §6). **This is a coordinate
substitution of an existing formula, not a new physical law** — but it is entered
here per the project rule that any new factored/substituted form gets a number.

```
u   = cosθ,   sin²θ = 1 − u²
v_u ≡ du/dλ = −sinθ · (dθ/dλ) = −sqrt(1−u²) · v_θ

# Multiply Formula 6's (dθ/dλ)² = Θ(θ) by sin²θ; the 1/sin²θ pole CANCELS:
(du/dλ)² = sin²θ · Θ(θ) = (1 − u²)·(Q + a²E²u²) − L_z²·u²   ≡   Θ_u(u)

# Second-order equation of motion (matches the geodesic.py R'/Θ' pattern):
d²u/dλ² = ½ · dΘ_u/du
dΘ_u/du = −2u·(Q + a²E²u²) + 2a²E²u·(1 − u²) − 2L_z²·u

# Covariant momentum recovery (for the Formula 8 g-factor in the disk march):
p_θ = v_θ = −v_u / sqrt(1 − u²)
```

**State-vector migration (Key Invariant `v_r = Δ·p_r`):** the state becomes
`[y, u, φ, t, v_y, v_u]`. Because `dy/dλ = dr/dλ`, the invariant is **renamed not
broken**: `v_y = Δ·p_r` (Δ via Formula 11), so `p_r = v_y/Δ` (NOT `v_y/Δ²` —
Formula 8 known bug still applies). The θ-side recovery changes to
`p_θ = −v_u/√(1−u²)` as above.

**⚠ Polar guard (approved 2026-06-02):** Θ_u(u) itself is singularity-free, so the
`_SIN2_MIN` clamp is dropped from the angular potential. BUT the Formula 6
`dφ/dλ` and `dt/dλ` equations still contain `L_z/sin²θ = L_z/(1−u²)`, which
diverges as `u→±1` for rays that pass *near* but do not reach the axis. **Keep a
numerical clamp `sin²θ = max(1−u², ε)` on the dφ/dλ and dt/dλ denominators only**
(not on Θ_u) to prevent NaN during RK4 overshoots. Rays that truly cross the axis
have `L_z→0`, making the term removable there.

---

## Formula 13 — Hybrid DNGR: ray-bundle Jacobian, magnification, and point-star PSF

**Source:** Adapted from James et al. (2015) §2.2, Appendix A.2 (ray-bundle
solid-angle propagation) and A.3.1 (point-star handling). The DNGR paper derives
the bundle's solid angle from the geodesic-deviation ellipse `(δ⁺, δ⁻, µ)` via the
Pineault–Roeder equations; this formula instead estimates the **same** solid-angle
change with the **screen-space finite-difference Jacobian** already approved in
Formula 10 amendment v1.4. **Not a re-derivation** — it is the 2×2 matrix
generalization of Formula 10's scalar `J = max(Jx, Jy)`, and it resolves the two
architectural divergences that the Formula 10 *fidelity note* flagged for review:
(#1 point stars must brighten, not blur; #2 anisotropic magnification). Verified
against `REFERENCE_dngr_paper.md` 2026-06-04.

**Purpose:** Render background catalog stars as 0-D points whose **flux brightens**
under gravitational lensing (rather than smearing into arcs as the baked-texture
starmap of Formula 10 does). Stars use a point catalog; the equirect texture +
mip-LOD of Formula 10 remains for *extended* background (galactic dust / nebulae).

### 1. Screen-space lensing Jacobian J

Reuse the stored per-pixel exit angles `(θ′, φ′)` (the celestial-sphere BL exit
direction — Formula 7 / `exit_buf`). Finite-difference against the +x and +y
neighbor pixels (identical source data to `_screen_jacobian_lod`):

```
        ⎡ ∂θ′/∂x   ∂θ′/∂y ⎤
J  =    ⎢                  ⎥        # 2×2, columns = +x and +y neighbor deltas
        ⎣ ∂φ′/∂x   ∂φ′/∂y ⎦

∂θ′/∂x = θ′(py, px+1) − θ′(py, px)          θ′ = acos(clamp(u_exit, −1, 1))
∂φ′/∂x = wrap_pi( φ′(py, px+1) − φ′(py, px) )    # wrap_pi as in Formula 10 v1.4
   (and likewise the +y column with the (py+1, px) neighbor)
```

J maps a screen-pixel area element to its distorted footprint on the celestial
sphere at `r = r_max` (the same finite-`r_max` truncation noted in Formula 10).

### 2. Gravitational magnification μ

The magnification is the ratio of image (camera-pixel) solid angle to source
(celestial-sphere) solid angle. The source area element is `sinθ′·dθ′·dφ′`, so the
source footprint per pixel is `sinθ′·|det J|`:

```
det J = (∂θ′/∂x)(∂φ′/∂y) − (∂θ′/∂y)(∂φ′/∂x)

μ = |det J₀ · sinθ′₀|  /  |det J · sinθ′|        # ← normalized form (use this)
```

where `(det J₀, sinθ′₀)` is the **flat-space (undeflected) per-pixel footprint**.
At critical curves `det J → 0` (paper's `δ⁻ → 0`) so `μ → ∞`; clamp `min(μ, MAG_MAX)`.

**✓ Refinement notes — APPROVED by the project owner 2026-06-05** (resolved before
the §8 render-path landing; PROJECT.md §6/§8):

- **(a) Normalization (the `det J₀` term) — APPROVED.** The bare
  `μ = 1/|det J·sinθ′|` equals 1 in flat space *only* if `(x, y)` carry true
  local-sky solid-angle units. With raw **pixel-index** differences, `det J` carries
  a constant `(rad/pixel)²` factor and a finite-FOV geometric baseline — the same
  effect Formula 10 records as "undeflected-corner LOD ≈ 2.3 (≈ ideal 1.74 +
  geometric)". Divide by the flat-space footprint so `μ → 1` undeflected. **Resolution:**
  compute `det J₀·sinθ′₀` with the **same finite-difference estimator** applied to the
  *undeflected camera-ray celestial directions* (the straight-ray exit map), so the
  `(rad/pixel)²` + geometric baseline cancels exactly and `μ → 1` with no black hole.
- **(b) Boundary rules (inherit Formula 10 v1.4 verbatim) — APPROVED.** If any neighbor
  used in the difference did **not** `ESCAPE`, or if the footprint straddles the
  spin-axis seam (`J > j_fold`), `det J` is invalid → clamp `μ = 1` (do not brighten).
  This mirrors the `outcome != ESCAPED → max_lod` and `J > _J_FOLD` guards already in
  the LOD path. **Resolution:** additionally treat the minor ellipse axis
  `δ⁻ < caustic_delta_min` as on-caustic and clamp `μ = min(μ, mag_clip)` so a
  critical curve cannot produce an unbounded splat.
- **(b′) Layer-A splat placement when `det J` is invalid (R2 — APPROVED owner 2026-06-06).**
  When guard (b) marks `det J` invalid (a non-ESCAPED neighbour, or a fold footprint
  `J > j_fold` / `δ⁺ > j_fold`), the star's screen-space offset must **NOT** be computed
  from `J⁻¹`. On the spin-axis seam neighbour pixels straddle the celestial pole, so
  `Δφ′ ≈ ±π` ⇒ `|det J|` is large ⇒ `J⁻¹ → 0`, which collapses **every** polar-cell
  star to `d ≈ 0`, piling them onto the meridian (the observed seam pileup). Instead
  place the splat by the star's **true proper angular separation under the undeflected
  exit map**, scaled by the flat-space per-pixel footprint `dΩ = |det J₀·sinθ′₀|`
  already computed for the guard-(a) μ normalization:

  ```
  d² = ( Δθ′² + sin²θ′·Δφ′² ) / dΩ        # screen-pixel², isotropic undeflected footprint
       where Δθ′ = θ′_star − θ′,  Δφ′ = wrap_pi(φ′_star − φ′)
  ```

  i.e. the great-circle separation divided by the undeflected angular pixel size
  `√dΩ`. On-axis stars then keep their real angular spacing (only genuinely-near stars
  splat), and μ is already clamped to 1 here by guard (b), so seam stars stay sharp
  point-like at base flux. This makes the polar gather degenerate gracefully to the
  no-lens geometry exactly where the lensed Jacobian is unusable. It is the principled
  replacement for the Formula-10 `j_fold` coarse-mip collapse (PROJECT.md §8, `δ⁻→0`
  caustic marker); porting the same regularization into the Formula-10 LOD path is a
  separate, not-yet-applied follow-up tracked in docs/specs/2026-06-06-dngr-artifact-remediation.md §7.2.

### 3. Energy-conserving point flux and truncated Gaussian PSF

```
I_final     = I_base · μ · g⁴                       # bolometric point-source flux
I_pixel(d)  = I_final · exp( −d² / (2σ²) )           # truncated Gaussian splat, |d| < d_max
```

- **`μ · g⁴` is not double-counted:** μ is the geometric solid-angle magnification
  (lensing); `g⁴` is the relativistic Doppler/redshift beaming of bolometric
  specific intensity (Formula 9, **volumetric** exponent — the correct one for an
  unresolved point-source *flux*). They are physically independent, exactly as the
  paper keeps frequency shift (ray-trace step vii) separate from beam solid angle
  (A.2). For a **static** camera with stars at the celestial sphere, `g ≈ 1`; the
  factor only bites under camera motion — keep it as a hook. *(g⁴ choice **APPROVED**
  2026-06-05: volumetric g⁴ exponent, exposed as a per-pixel hook that defaults to a
  no-op (g≡1) until a moving-observer g-factor lands — config `starfield.g_beaming`.)*
- **`d`** is the screen-space distance from the pixel center to the star's projected
  center; `σ` is config-driven (paper sets the beam's initial radius to twice the
  pixel separation, targeting a ≤2% peak-to-trough flicker). The truncation `|d| < d_max`
  keeps the splat local. This is the A.3.1 anti-flicker filter, verbatim in intent.

**Net effect:** stars stay sharp, circular, and point-like while their *brightness*
tracks the lensing magnification — the A.3.1 / A.7 behavior the baked-texture
Formula 10 path structurally cannot achieve for point stars.

---

# PART II — CARTESIAN KERR-SCHILD (CKS) COORDINATES  *(active renderer path, 2026-06-06)*

**Why this part exists.** Boyer-Lindquist has *coordinate* singularities on the spin
axis (θ = 0, π: the 1/sin²θ pole) and at the horizon (Δ → 0). Part I (Formulas
1/6/7/11/12) fought the axis pole with band-aids — `u = cosθ`, `Θ_u`, per-step φ-wrap,
`j_fold` mip collapse, `normalize_sphere_angles` punch-through, F13 guard (b′). The
**Cartesian Kerr-Schild** chart is *regular on the axis and across the horizon*, so the
entire artifact class is removed at the source. As of 2026-06-06 the renderer geodesic
path, photon initialization, disk g-factor, and escaped-ray celestial direction use CKS.

**Authoritative sources (verified 2026-06-06):** Chan, Psaltis & Özel, *GRay2* (ApJ 2018,
arXiv:1706.07062); SpECTRE `gr::Solutions::KerrSchild`; Visser, *The Kerr spacetime*
(arXiv:0706.0622). The metric, inverse, implicit radius, and analytic derivatives below
were cross-checked across these and confirmed self-consistent (l is η-null ⇒ the inverse
is exact; ∂r forms verified against the quartic).

**Conventions:** geometric `G = M = c = 1`; signature `(− + + +)`; spin `a` along **+z**;
coordinates `(t, x, y, z)`. The CKS radius `r` **is the Boyer-Lindquist radial coordinate**
(`z = r cosθ`), so all BL-radius quantities (ISCO, Ω, E_I/L_I) carry over unchanged.

**Status of the Part I formulas under CKS:**

| Formula | Status under CKS |
|---|---|
| 1 (BL metric), 6 (Mino BL geodesic), 7 (BL ZAMO tetrad), 11 (y=r−r₊), 12 (u=cosθ Θ_u) | **SUPERSEDED for the renderer** — they describe the retired BL path. Kept for history / the CPU `[r,θ,…]` reference only. |
| 2 (ISCO), 3 (Ω, u^t), 4 (E_I,L_I) | **REUSED unchanged** (BL-radius quantities; r is identical). |
| 8 (g-factor), 9 (g⁴ beaming, blackbody chroma) | **REUSED** — structure unchanged; see CKS-9 (now a Cartesian dot product; the BL Δ-divide bug is structurally impossible). |
| 10 (mip LOD), 13 (DNGR μ/PSF) | **REUSED unchanged** — they act on the celestial direction `(θ′, φ′)` (CKS-10), which is coordinate-agnostic. |

---

## Formula CKS-1 — Kerr radius r(x, y, z)  [implicit]

`r` is the same BL radial coordinate, defined implicitly (`ρ² = x²+y²+z²`):

```
r⁴ − (ρ² − a²) r² − a² z² = 0

# explicit positive root:
r² = ½(ρ² − a²) + sqrt( ¼(ρ² − a²)² + a² z² )
r  = sqrt(r²)
```

Identity used below: `Σ ≡ r² + a² z²/r² = (r⁴ + a² z²)/r²`  (= BL `r²+a²cos²θ`). At the
equator `z = 0` (the disk plane): `r = sqrt(x² + y² − a²)`.

---

## Formula CKS-2 — Kerr-Schild metric (Cartesian)

```
g_αβ = η_αβ + f · l_α l_β,        η = diag(−1, 1, 1, 1)

f   = 2 r³ / (r⁴ + a² z²)         (= 2 M r / Σ with M = 1)

l_α = ( 1,
        (r x + a y)/(r² + a²),
        (r y − a x)/(r² + a²),
        z / r )                   # covariant; l_t = 1
```

`l` is null w.r.t. **both** η and g. Every `l_α` is finite on the spin axis (x=y=0 ⇒
l_x=l_y=0, l_z=z/r=±1) and across the horizon — the whole point of CKS.

---

## Formula CKS-3 — Inverse metric (exact, NO numerical inversion)

Because `l` is η-null (`η^αβ l_α l_β = 0`, verified):

```
g^αβ = η^αβ − f · l^α l^β,   where  l^α = η^αγ l_γ = (−1, l_x, l_y, l_z)
                                    (l^t = −l_t = −1,  l^i = l_i)
```

Do **not** call a matrix inverter; use this closed form.

---

## Formula CKS-4 — Coordinate derivatives (analytic; let D = r⁴ + a² z²)

```
∂r/∂x = r³ x / D
∂r/∂y = r³ y / D
∂r/∂z = r z (r² + a²) / D

∂f/∂xⁱ = f · [ 3·(∂r/∂xⁱ)/r − (4 r³·(∂r/∂xⁱ) + 2 a² z·δ_iz) / D ]
∂l_t/∂xⁱ = 0
```

Differentiate the spatial `l_i` directly (unambiguous; `S = r² + a²`):

```
∂l_x/∂xʲ = [ (x·∂r/∂xʲ + r·δ_jx + a·δ_jy)·S − (r x + a y)·(2 r·∂r/∂xʲ) ] / S²
∂l_y/∂xʲ = [ (y·∂r/∂xʲ + r·δ_jy − a·δ_jx)·S − (r y − a x)·(2 r·∂r/∂xʲ) ] / S²
∂l_z/∂xʲ = δ_jz / r − z·(∂r/∂xʲ) / r²
```

---

## Formula CKS-5 — Hamiltonian geodesic equations of motion

Null photon Hamiltonian (affine parameter λ):

```
H = ½ g^αβ p_α p_β = 0          # enforced by init; monitored as drift
dx^α/dλ = ∂H/∂p_α = g^αβ p_β
dp_α/dλ = −∂H/∂x^α = −½ (∂_α g^βγ) p_β p_γ
```

Stationary + axisymmetric ⇒ **E = −p_t** and **L_z = x p_y − y p_x** are conserved.
Working form (η constant, `g^βγ = η^βγ − f l^β l^γ`), with `φ_l ≡ l^β p_β =
−p_t + l_x p_x + l_y p_y + l_z p_z`:

```
dt/dλ  = −p_t + f φ_l                        # = η^{tβ}p_β − f l^t φ_l,  l^t=−1
dxⁱ/dλ = p_i − f l_i φ_l                      # η^{ii}=+1, l^i=l_i

dp_t/dλ = 0                                   # E conserved
dp_i/dλ = ½ (∂_i f) φ_l² + f φ_l (∂_i φ_l)
          where ∂_i φ_l = (∂_i l_x)p_x + (∂_i l_y)p_y + (∂_i l_z)p_z
```

Integrate the 8-vector `[t, x, y, z, p_t, p_x, p_y, p_z]` with RK4. `p_t` is constant
analytically — a free per-step error monitor. Recommended affine step: constant `dλ`
far out, shrunk near the horizon, e.g. `h = dλ · max(step_floor, (r − r₊)/r)`.

This step rule is numerical, not physics — size it however keeps the integrand
resolved. In particular it knows only the horizon distance, so inside the Pipe-B
disk slab (where the emission integrand has a thin vertical Gaussian of scale height
`σ_z = r·θ_half·σ_frac`, Formula 9) it must additionally be capped so a steep
equatorial crossing cannot step over the layer: bound the vertical displacement
`|dz/dλ|·h ≤ vfrac·σ_z` (config `disk.max_step_vfrac`). It only bites for steep
crossings — in-plane grazers keep the full step — so it never starves the
`max_steps` budget. Under-resolving it aliases the disk into a concentric moiré.

---

## Formula CKS-6 — Horizon capture and escape

```
r₊ = 1 + sqrt(1 − a²)            # outer horizon (r is the BL radius)
capture  when  r ≤ r₊ + ε_h      # ε_h = config render.horizon_epsilon (cost bound only)
escape   when  ρ = sqrt(x²+y²+z²) ≥ r_max    # config render.r_max
```

CKS is regular at the horizon, so capture is detected right at `r₊` with no Δ→0 blowup;
`ε_h` merely caps step count in the deep field.

---

## Formula CKS-7 — Photon initialization (ZAMO observer + projected ray direction)

Preserves **Decision A (ZAMO)**, built coordinate-cleanly from the inverse metric.

```
# 1. ZAMO 4-velocity at the camera (x,y,z) — directly from g^{αβ}:
α        = 1 / sqrt(−g^{tt})          # lapse
u_obs^α  = −α · g^{tα}                 # ⇒ u_obs^t = 1/α > 0 ; zero angular momentum

# 2. Camera ray coordinate direction n=(nx,ny,nz) (unit) from pixel + FOV
#    (n = normalize(fwd + sx·right + sy·up); fwd/right/up are the world=CKS basis).
#    Make it a 4-vector, g-orthogonal to u_obs, then g-unit:
N^α   = (0, nx, ny, nz)
N'^α  = N^α + (g_μν N^μ u_obs^ν) · u_obs^α        # now g·(N', u_obs) = 0
ŝ^α   = N'^α / sqrt(g_μν N'^μ N'^ν)               # spatial unit (+++)

# 3. Null photon momentum (contravariant), then lower:
p^α = E_loc · ( u_obs^α + ŝ^α )                    # null automatically
p_α = g_αβ p^β
E   = −p_t,   L_z = x p_y − y p_x                  # set E_loc so E=1 (any scale; g uses ratios)
```

At `r ≳ 6` the camera is far outside the ergosphere (`r_ergo ≤ 2`), so ZAMO ≈ static and
the construction is well-conditioned (`g ≈ η`). This replaces the BL closed-form ZAMO
tetrad (Formula 7); the single-direction projection avoids needing a full tetrad.

---

## Formula CKS-8 — Accretion-disk gas 4-velocity (CKS, equatorial)

Disk plane `z = 0`; `r = sqrt(x²+y²−a²)` (CKS-1). For a circular orbit (`r ≥ r_isco`),
`dr = 0` along the orbit ⇒ the BL→KS `t`/`φ` shifts are constant ⇒ the velocity is a
**rigid rotation about +z** at the BL angular velocity Ω (Formula 3):

```
Ω   = 1 / (r^{3/2} + a)                                   # Formula 3
u^t = (1 + a r^{−3/2}) / sqrt(1 − 3/r + 2 a r^{−3/2})     # Formula 3 (numerator mandatory)
u^x = −Ω y u^t,   u^y = +Ω x u^t,   u^z = 0
```

Prograde (co-rotating with +z spin): at `(R,0,0)`, `u^y > 0` (counter-clockwise).
*Derivation:* at `z=0`, `x = r cosφ̃ + a sinφ̃`, `y = r sinφ̃ − a cosφ̃` (φ̃ = KS azimuth);
`∂x/∂φ̃ = −y`, `∂y/∂φ̃ = x`, and `dr=0` ⇒ `u^x = (∂x/∂φ̃)u^φ̃ = −yΩu^t`, `u^y = xΩu^t`. No
BL→KS Jacobian needed. *(Plunging `r < r_isco` is below the disk inner edge
`r_inner = r_isco` and is never sampled; if ever required, transform Formula 5 with the
full BL→KS Jacobian and **flag for human review** before use.)*

---

## Formula CKS-9 — g-factor (CKS)

Observer at rest at infinity ⇒ `(p·u)_obs = p_t = −E`. With the integrator's covariant
CKS momenta `p_μ` and the gas `u^μ` from CKS-8:

```
g = E_obs/E_emit = −E / ( p_t u^t + p_x u^x + p_y u^y + p_z u^z )
```

Then emission follows **Formula 9 verbatim** (chromaticity · g⁴, volumetric). The
Formula-8 "divide p_r by Δ" bug is impossible here: there is no Δ and `p_μ` is already
covariant.

> **Visualization dial (NOT physics — do not "fix"):** the GPU kernel applies
> `g_eff = g^s` with `s = disk.doppler_strength` (default **1.0** = this formula
> verbatim; the `s≠1` branch is skipped, bit-identical). `s<1` artistically mutes
> the shift — `s=0` ⇒ `g_eff≡1`, the Interstellar/DNGR treatment that suppressed the
> disk's Doppler asymmetry for the film. `g_eff` feeds **both** the Formula-9 g⁴ and
> the chromaticity (single application — the g⁴-not-g⁸ rule is unaffected). It scales
> the TOTAL g; splitting orbital Doppler from gravitational redshift would need a new
> formula here first (static-observer redshift) — do not improvise one in code.

---

## Formula CKS-10 — Escaped-ray celestial direction (no spin-axis seam)

When a photon escapes (`ρ ≥ r_max`) the spacetime is asymptotically flat, so the
contravariant spatial momentum direction **is** the celestial direction:

```
d   = (dx/dλ, dy/dλ, dz/dλ) normalized   # = (p^x,p^y,p^z)/|·|, the incoming sky dir
θ′  = acos( clamp(d_z, −1, 1) )
φ′  = atan2(d_y, d_x)
equirect:  u = wrap(φ′/2π),   v = clamp(θ′/π, 0, 1)
```

`d` is a genuine Cartesian unit vector for **every** ray ⇒ the BL spin-axis seam, the
φ-accumulation blowup, `normalize_sphere_angles` punch-through, and the `j_fold` /
guard-(b′) meridian band-aids are **all removed**. Formula 10 (LOD) and Formula 13 (DNGR
μ/PSF) act unchanged on screen-space neighbour differences of `(θ′, φ′)`; the only
residual pole effect is the ordinary equirect-texture coordinate at `θ′ = 0, π`, handled
by the standard φ-wrap already in the samplers.

---

## Conservation test requirements

**CKS harness (active path).** Verify along every integrated null geodesic:

| Quantity | Tolerance | How to check |
|---|---|---|
| Photon energy `E = −p_t` | relative drift < 1e-4 | `abs((E−E₀)/E₀)` |
| Axial ang. mom. `L_z = x p_y − y p_x` | relative drift < 1e-4 | `abs((Lz−Lz₀)/Lz₀)` |
| Null condition `H = ½ g^αβ p_α p_β` | `abs(H)` < 1e-6 | CKS-3 inverse, direct eval |
| Carter `Q` (null form, optional but recommended) | relative drift < 1e-4 | convert CKS→BL (`r` from CKS-1, `cosθ=z/r`, `p_θ` via Jacobian) then the BL null-Q below |

The legacy BL harness below remains valid for the retired CPU `[r,θ,…]` path.

### Legacy BL harness

The pytest harness must verify all three of the following along every
integrated null geodesic:

| Quantity | Tolerance | How to check |
|---|---|---|
| Photon energy `E = −p_t` | relative drift < 1e-4 | `abs((E_final − E_init)/E_init)` |
| Angular momentum `L_z = p_φ` | relative drift < 1e-4 | `abs((Lz_final − Lz_init)/Lz_init)` |
| Carter constant `Q` (null form) | relative drift < 1e-4 | `abs((Q_final − Q_init)/Q_init)` |
| Null condition `g^{μν} p_μ p_ν` | absolute < 1e-6 | direct evaluation |

**Use the null geodesic Q formula** in the test harness:
```python
# Correct test harness formula (null geodesic):
Q = p_theta**2 + cos_theta**2 * (-a**2 * E**2 + L_z**2 / sin_theta**2)
```

If any of these fail, the integrator, tetrad init, or metric is wrong.
Do not proceed to rendering until all four pass.

---

## Open decisions (fill in before implementing)

**Decision A — Tetrad observer type**
- [x] ZAMO (recommended, see Formula 7)
- [ ] Circular orbit observer (more complex, marginally more accurate in-disk)

**Decision B — Disk temperature model**
- [x] Simple: `T = T_0 · (6/r)^{0.75}` — fast, already in original code (ACTIVE)
- [x] Page-Thorne (1974) flux profile — physically correct (spec below, **source-VERIFIED 2026-06-12; WIRED 2026-06-12 behind `disk.temperature_model: page_thorne`, default `simple`**)

Record the chosen options here once decided and reference them in `CLAUDE.md`.

### Formula CKS-11 — Page-Thorne disk flux profile (VERIFIED; Decision-B upgrade)

> **Status (2026-06-12):** Source-VERIFIED and **Wired 2026-06-12 behind
> `disk.temperature_model` flag (default `simple`)** — CPU `f_PT(r)` LUT in
> `src/renderer/disk_flux.py`, sampled by the GPU disk kernel. Supersedes the
> earlier ⛔ PROVISIONAL transcription (which used a
> different, unverified `Q/(B C^{1/2} D^{1/2})` parametrization — discarded).
> **Source:** Page & Thorne (1974) ApJ 191:499, restated in Abramowicz & Fragile
> (2013) Living Rev. Rel.; supplied as `paper/1104.5499v3.md`. **Verification:** the
> §3 closed form below was numerically confirmed to reproduce the §1 conservation-law
> flux integral `F = −Ṁ/(4π√−g)·Ω′/(Ẽ−ΩL̃)²·∫_{r_ms}^r(Ẽ−ΩL̃)L̃′dr` to 5 sig figs
> across r∈[1.5,28] M for a=0.999, using SKILL.md Formula 3/4 for Ẽ,L̃,Ω; the two
> forms differ only by the overall constant (`3/2` and √−g = r) that the closed form
> drops by writing `F ∝ …`. The cubic roots satisfy `y³−3y+2a=0` to machine
> precision; the bracket → 0 at `y=y₀` (zero-torque BC). This is the standard
> Page-Thorne function, structure cross-checked. See `tests/test_disk_flux.py`.

Use `y = √(r/M)`, `a_* = a/M`, `y₀ = √(r_ms/M)` with `r_ms = r_isco` (Formula 2).

**Cubic roots** of `y³ − 3y + 2a_* = 0`:
```
y₁ = 2·cos[ (arccos(a_*) − π)/3 ]
y₂ = 2·cos[ (arccos(a_*) + π)/3 ]
y₃ = −2·cos[ arccos(a_*)/3 ]
```

**Correction functions** (only B, C needed for the closed form; D not required):
```
B = 1 + a_* y^{-3}
C = 1 − 3 y^{-2} + 2 a_* y^{-3}        # note C = (y³ − 3y + 2a_*)/y³
```

**Closed-form flux** (proportionality — absolute amplitude is a free calibration,
carried by `T_0` / `Ṁ`, exactly as in the simple model):
```
F(r) ∝ y^{-7} · C^{-1} · bracket(y)        # equivalently r^{-3}·B^{-1}·C^{-1/2}·Q

bracket(y) = (y − y₀) − (3/2)·a_*·ln(y/y₀)
           − [ 3(y₁−a_*)² / (y₁(y₁−y₂)(y₁−y₃)) ]·ln((y−y₁)/(y₀−y₁))
           − [ 3(y₂−a_*)² / (y₂(y₂−y₁)(y₂−y₃)) ]·ln((y−y₂)/(y₀−y₂))
           − [ 3(y₃−a_*)² / (y₃(y₃−y₁)(y₃−y₂)) ]·ln((y−y₃)/(y₀−y₃))
```
**Zero-torque inner BC:** `F(r_ms)=0` (bracket → 0 as `y → y₀`); emission ≡ 0 inside
`r_ms` (gas plunges, does not radiate). **Implementation plan:** precompute the
dimensionless shape `f_PT(r) = F(r)/F_max` as a 1-D CPU LUT indexed by `r` for fixed
`a`; the GPU shader reads the LUT (no per-pixel integral or logs).

**Piece 3 — Spectrum & g-bookkeeping** (SAFE — standard, already constrained by
Formula 9): `T_eff(r) = (F(r)/σ)^{1/4}`, emit a physical Planck `B_ν(T_eff)`.
**Critical interaction with Formula 9:** the active code multiplies by `pow(g,4)`
*because* `blackbody_rgb` is chromaticity-only (no T⁴ amplitude). If a physical
Planck `B_ν` with Stefan-Boltzmann T⁴ amplitude replaces it, the `g⁴` must be
applied via the `T_obs = g·T_emit` substitution **OR** as an explicit factor —
**never both** (that is the g⁸ double-count Formula 9 warns about). For the
volumetric march the bolometric scaling is **g⁴** (Formula 9: 3D volume), not the
g³ that applies only to a 2D monochromatic surface.

---

## Formula CKS-12 — Disk procedural turbulence: noise coordinates, Keplerian shear advection, modulation bookkeeping (VISUALIZATION)

> **Status (2026-06-13): owner-approved design, NOT YET WIRED** (backlog **D2**;
> design spec `docs/specs/2026-06-13-disk-noise-turbulence.md`). **Not new GR.**
> The only physics input is Ω from **Formula 3, reused verbatim**. Everything else
> is procedural texturing that multiplies *amplitude* quantities (density, emitted
> temperature, edge/height windows). The noise primitives themselves (fBm, ridged
> multifractal, Worley/Voronoi) are texturing functions, not physics — they are
> specified in the design spec, with `src/renderer/noise.py` (CPU NumPy) as their
> implementation source of truth and `@ti.func` twins held to it by tests.

### 1. Noise coordinates (per disk sample, from CKS `(x, y, z)`)

```
r  = kerr_radius(x, y, z)            # CKS-1 (already computed in the disk kernel)
φ  = atan2(y, x)
u  = ln(r / r_inner)                 # log-radial: feature size scales with r (self-similar disk)
ζ  = (θ − π/2) / (θ_half · σ_frac)   # vertical position in local Gaussian scale heights
                                     #   (= the kernel's existing dz_ang / σ_theta)
```

- **Advection consistency:** under the CKS-8 gas field (`u^x = −Ωy·u^t`,
  `u^y = +Ωx·u^t`), `d/dt atan2(y, x) = Ω` exactly — so advecting noise in this φ
  is exactly co-moving with the same velocity field that drives the CKS-9 Doppler.
- **φ is NOT the KS azimuth φ̃** (CKS-8: `x = r cosφ̃ + a sinφ̃ …` ⇒
  `φ = φ̃ − arctan(a/r)`). The difference is a static, r-dependent twist of the
  noise domain — visually harmless. Do not "fix" it by converting to φ̃.

### 2. Keplerian shear advection with dual-phase reset

```
Ω(r) = 1 / (r^{3/2} + a)                        # Formula 3 — verbatim, prograde
```

Naive advection `φ′ = φ − Ω(r)·t` shears any pattern into infinitely thin spirals
as t grows (relative shear rate dΩ/dr). Standard fix — two pattern phases with
staggered resets, crossfaded (Neyret-style advected texture):

```
s    = t_disk / T                       # T = config disk.noise.shear_period
a_k  = fract(s + k/2),  k ∈ {0, 1}      # each phase's age fraction ∈ [0, 1)
c_k  = floor(s + k/2)                   # phase-k cycle index
w_k  = 1 − |2·a_k − 1|                  # triangle weights; w_0 + w_1 ≡ 1

φ′_k = φ − Ω(r) · (a_k · T)             # each phase sheared for at most T

n(u, φ, ζ; t) = w_0·N(u, φ′_0, ζ; hash(seed, k=0, c_0))
              + w_1·N(u, φ′_1, ζ; hash(seed, k=1, c_1))
```

- `t_disk` is the disk animation time in geometric units; callers compute it as
  `frame_index / render.fps × disk.noise.time_scale`.
- **Per-cycle reseed** (the `c_k` term in the hash, or equivalently a hashed
  per-cycle domain offset) is mandatory — without it the whole animation repeats
  with period T.
- **Optional variance preservation:** the crossfade lowers contrast mid-blend
  (`w² sum < 1`); dividing the blend by `sqrt(w_0² + w_1²)` removes the periodic
  contrast "breathing" (config `variance_preserve`).
- T is a look dial: long T → long Interstellar-style filaments; short T → choppier.

### 3. Modulation bookkeeping — where noise MAY and MAY NOT enter

With `m = Σ_i amp_i·(n_i − bias_i)` over the layer stack (spec §4):

```
density_mult = exp( clamp(m, −m_max, +m_max) )          # > 0 by construction
ρ            = gauss(ζ) · density_mult                   # feeds BOTH emission and absorption
emission     ∝ emis_c · ρ · [f_PT or 1] · g_eff⁴ · ds    # CKS-11 / Formula 9 unchanged
dτ           = absb_c · ρ · ds                           # clumps self-shadow

T_emit  ← T_emit · (1 + τ_amp·(n_T − ½))     # BEFORE the g shift: chroma(g_eff · T_emit)
r_in_eff(φ,t)  = max( r_inner·(1 + e_in·(n_e − ½)),  r_isco )   # zero-torque BC kept
r_out_eff(φ,t) = r_outer·(1 + e_out·(n_e' − ½))
ρ ← ρ · smoothstep windows on [r_in_eff, r_out_eff]      # replaces the hard cutoffs
σ_θ ← σ_θ · (1 + h_amp·(n_h − ½))                        # lumpy scale height
```

**Hard constraints (violating any of these is a physics bug, not a style choice):**

1. Noise must NEVER touch `p_μ`, `u^μ` (CKS-8), `g` (CKS-9), the `g⁴` exponent
   (Formula 9), the blackbody chromaticity form, or the `f_PT` radial shape
   (CKS-11). Amplitude quantities only.
2. Temperature modulation applies to the **emitted** `T_emit` before the `g_eff`
   shift — the g⁴-not-g⁸ bookkeeping of Formula 9 / CKS-11 Piece 3 is unaffected.
3. `r_in_eff ≥ r_isco` always (CKS-11 zero-torque BC; no emission from the plunge).
4. The CKS-5 Pipe-B vertical step cap must use the **worst-case modulated** scale
   height `σ_z·(1 − h_amp/2)`, or the face-on moiré that `disk.max_step_vfrac`
   fixed returns.
5. Every noise lattice is **integer-periodic in φ** (period 2π ⇒ `freq_phi ∈ ℤ`) —
   no seam at φ = 0.
6. `disk.noise.enabled: false` must take a branch **bit-identical** to the
   pre-noise kernel (the `doppler_strength == 1.0` pattern) — golden frames stay
   valid.
7. Deterministic integer hashing only (seed from config); **no `ti.random`** —
   same seed + same `t_disk` ⇒ identical frame.

---

## File locations (project conventions)

```
skills/kerr-physics/SKILL.md     ← this file
src/renderer/geodesic.py         ← Formulas 1, 6, 7
src/renderer/disk.py             ← Formulas 2, 3, 4, 5, 8, 9
src/renderer/noise.py            ← (planned, D2) Formula CKS-12 noise primitives — CPU source of truth
src/renderer/starmap.py          ← Formula 10
src/renderer/taichi_renderer.py  ← Formulas 10, 13 (screen-space Jacobian, μ, star splat)
scripts/ingest_stars.py          ← Formula 13 catalog pre-processing (HYG/ATHYG csv or BSC5 → {θ′, φ′, flux_rgb}.npy; I_base·chroma folded into flux)
tests/test_geodesic.py           ← Conservation tests (Formula 6 conserved quantities)
configs/render.yaml              ← a, r_isco, WIDTH, HEIGHT, step counts, stars:*
```

---

## Revision history

| Version | Change |
|---|---|
| v1.0 | Initial release |
| v1.1 | **F6:** Corrected Carter constant to null geodesic form (−a²E², not a²(1−E²)). **F7:** Corrected lapse α to exact form using A = (r²+a²)²−a²Δsin²θ. **F9:** Documented that blackbody_rgb returns chromaticity only; clarified g⁴ is not double-counted, but will be if a physical Planck spectrum is substituted. |
| v1.2 | **F6:** Removed the leftover massive-particle `μ²r²` term from the radial potential `R(r)`; the null (μ=0) form drops it. The previous form gave `g^{μν}p_μp_ν = −r²/Σ`, breaking the null-condition conservation test. |
| v1.3 | **F10:** Added 2π normalization to the LOD formula — φ spans 2π radians across the 16384-texel starmap width, so dividing by 2π correctly maps the angular footprint to a texel footprint. Also switched to raw per-pixel exit deltas (δθ, δφ) rather than dividing by δu=1/WIDTH. The missing factor caused LOD to saturate at max mip for all background pixels, collapsing the LOD-on render to near-black. |
| v1.4 | **F10:** Added the screen-space Jacobian amendment (eliminate the offset ray; difference exit directions of neighbor pixels in a second shading kernel; same J/L; captured-neighbor ⇒ max_lod). **F11 (new):** FP32-stable factored discriminant Δ = y(y+2k). **F12 (new):** singularity-free polar potential Θ_u(u) for the u=cosθ state transform, with the `v_r=Δ·p_r → v_y=Δ·p_r` invariant migration, `p_θ=−v_u/√(1−u²)` recovery, and the approved polar guard on dφ/dt only. All three approved by the project owner 2026-06-02 for the renderer optimization (PROJECT.md §6). |
| v1.5 | **F13 (new):** Hybrid DNGR point-star magnification — screen-space ray-bundle Jacobian J (2×2 generalization of F10's scalar J), magnification μ = \|det J₀·sinθ′₀\|/\|det J·sinθ′\|, and energy-conserving point flux `I_base·μ·g⁴` with a truncated-Gaussian PSF. Verified against `REFERENCE_dngr_paper.md` (James et al. 2015, A.2/A.3.1/A.7) on 2026-06-04. Resolves the F10 fidelity-note divergences #1 (point-star blur) and #2 (anisotropy). **⚠ Three guards FLAGGED pending owner approval:** (a) μ normalization by the flat-space footprint `det J₀·sinθ′₀`; (b) ESCAPE/`j_fold` boundary clamp `μ=1` (inherited from F10 v1.4); (c) g⁴ exponent choice for stars. |
| v1.6 | **F13 guards APPROVED (owner, 2026-06-05)** and the DNGR render path landed (PROJECT.md §8 Phases 2–5): (a) μ normalized by the FD undeflected-reference footprint so μ→1 in flat space; (b) boundary clamp μ=1 on non-ESCAPED neighbours / `J>j_fold`, plus `δ⁻<caustic_delta_min ⇒ μ=min(μ,mag_clip)`; (c) volumetric g⁴ as a `starfield.g_beaming` hook (default g≡1). Two decoupled sky layers in `taichi_renderer.py`: Layer A point-star energy gather (`flux·μ·g⁴·PSF`, cell-grid candidate query) and Layer B anisotropic-EWA diffuse Milky-Way fetch; gated by `starfield.mode: texture\|dngr` (texture default reproduces v1.4 golden frames bit-for-bit). |
| v1.8 | **PART II — Cartesian Kerr-Schild (CKS) ADDED + APPROVED (owner, 2026-06-06):** the renderer geodesic path migrates BL → CKS to remove the spin-axis (1/sin²θ) and horizon (Δ→0) *coordinate* singularities at the source (the root cause of the user-reported gray polar line and the whole seam-band-aid lineage). New Formulas CKS-1…CKS-10: implicit radius `r(x,y,z)`; metric `g=η+f l⊗l`; **exact** inverse `g=η−f l⊗l` (l is η-null); analytic ∂r/∂f/∂l; Hamiltonian geodesic EOM (`dx=g·p`, `dp=−½∂g·pp`); ZAMO-from-`g^{αβ}` + projected-ray photon init (preserves Decision A); equatorial disk gas velocity `u^x=−Ωy u^t, u^y=Ωx u^t` (no BL→KS Jacobian); CKS g-factor (Δ-bug impossible); seam-free escaped-ray celestial direction. BL Formulas 1/6/7/11/12 marked SUPERSEDED-for-renderer; 2/3/4/8/9/10/13 reused. Verified against GRay2 (arXiv:1706.07062), SpECTRE, Visser (arXiv:0706.0622). |
| v1.7 | **F13 guard (b′) ADDED + APPROVED (owner, 2026-06-06):** Layer-A splat *placement* rule when `det J` is invalid (spin-axis seam / non-ESCAPED neighbour). The shipped code positioned the splat with the degenerate `J⁻¹`, collapsing all polar-cell stars onto the meridian (the "Artifact B" seam pileup, ≈15× the off-seam jump). (b′) places the splat by the star's true proper angular separation `d² = (Δθ′²+sin²θ′·Δφ′²)/dΩ` under the undeflected footprint `dΩ=|det J₀·sinθ′₀|` (the guard-(a) quantity), so seam stars keep real angular spacing at μ=1. Resolves the dngr-default seam (`test_no_spin_axis_seam`, `test_background_has_no_vertical_seam_stripe`). The matching Formula-10 `texture`-LOD regularization is a separate follow-up (spec §7.2), **not** part of this revision. |
| v1.9 | **Decision B — physical disk upgrade DRAFTED (PROVISIONAL, owner review pending, 2026-06-11):** added a flagged spec for moving off the simple `(6/r)^0.75` law to the NT/Page-Thorne flux. Piece 1 (NT correction functions B/C/D/F/G) and Piece 3 (physical Planck `B_ν(T_eff)` + the g⁴-not-g⁸ bookkeeping vs Formula 9) are standard/safe; **Piece 2 — the time-averaged flux `F(r)` and `Q(r)` integral — is ⛔ BLOCKED on source verification** (local Page-Thorne `1974ApJ...191.md` is image-dropped, 59 formula-not-decoded; NT `II-48.md` is OCR-garbled), so it is transcribed from SYNTHESIS §4 only and **must not be implemented until confirmed against a clean Page-Thorne 1974 source + owner sign-off** (recalled-formula caution, cf. the GRay coefficient correction same day). No code path changed; ACTIVE disk remains Decision-B-simple. |
| v1.10 | **Decision B Piece 2 — Page-Thorne flux VERIFIED & UNBLOCKED (2026-06-12).** Owner supplied a clean equation-intact source (`paper/1104.5499v3.md`, Page-Thorne 1974 via Abramowicz-Fragile 2013). The ⛔ PROVISIONAL `Q/(B C^{1/2} D^{1/2})` transcription was **discarded** (different, unverified parametrization) and replaced by the canonical closed-form **Formula CKS-11**: cubic roots `y₁,y₂,y₃` of `y³−3y+2a=0`, correction functions B/C, and the three-log `bracket(y)`. **Verified numerically:** the closed form reproduces the §1 conservation-law flux integral (using Formula 3/4 Ẽ,L̃,Ω) to 5 sig figs over r∈[1.5,28] M at a=0.999, differing only by the overall `3/2·√−g` constant the closed form drops; roots satisfy the cubic to machine precision; zero-torque BC `F(r_ms)=0` holds. Regression guard added: `tests/test_disk_flux.py`. D function not needed (folded into the closed form). Owner-approved to implement behind a config flag; **ACTIVE disk still Decision-B-simple — kernel not yet wired.** |
| v1.11 | **Decision B Piece 2 — Page-Thorne flux WIRED (2026-06-12, D1).** The verified CKS-11 closed form is now live behind the runtime flag `disk.temperature_model` (default `simple`, so golden frames / the pinned GPU regression are unchanged). Path: `src/renderer/disk_flux.py` precomputes the normalized dimensionless shape `f_PT(r)=F/F_max` as a 1-D CPU LUT over `[r_isco, r_outer]` (`flux_lut_samples`, default 256; `lut[0]=0` zero-torque BC); `taichi_renderer._setup_disk_flux` always builds+uploads it (tiny → flag toggles per-render with no re-JIT); the disk kernel linear-interpolates it and sets `T_eff=T₀·f_PT^{1/4}` with emission amplitude ×`f_PT`. **g-bookkeeping preserved:** the explicit `g⁴` is kept and NOT doubled (`_blackbody_rgb` is chromaticity-only — the g⁸ error Formula 9 / CKS-11 Piece 3 warns about is avoided in both branches). Guards: `tests/test_disk_flux.py` (module vs pinned transcription + LUT properties) and a gpu-marked `tests/test_gpu_regression.py` page_thorne render check. `T₀` stays the amplitude knob. |
| v1.12 | **`disk.doppler_strength` visualization dial documented (2026-06-12) — NOT a physics revision.** The kernel applies `g_eff = g^s` to the CKS-9 g-factor before Formula 9 (`s=1` default = formulas verbatim, branch skipped, bit-identical — verified Doppler 4.317×/peak 6.1665 vs goldens 4.32×/6.1667; `s=0` ⇒ shift fully off, the Interstellar/DNGR artistic treatment). Single application feeding both g⁴ and the chromaticity — the g⁴-not-g⁸ rule is unaffected. Scales the TOTAL g; an orbital-vs-gravitational split would require a new verified static-observer redshift formula first. GPU guard: `test_doppler_strength_zero_symmetrizes_disk` (s=0 ⇒ disk-only L/R ratio < 1.5). See the dial note under Formula CKS-9. |
| v1.13 | **Formula CKS-12 ADDED — disk procedural turbulence (owner-approved 2026-06-13; NOT YET WIRED, backlog D2).** Visualization math for the layered-noise disk: disk-natural noise coordinates `(u=ln r/r_inner, φ=atan2(y,x), ζ=Δθ/σ_θ)` with the proof that this φ is advected at exactly Ω by the CKS-8 gas field (and is a static `arctan(a/r)` twist away from the KS azimuth — do not "fix"); Keplerian shear advection `φ′ = φ − Ω(r)·t_disk` (Ω = Formula 3 verbatim) with dual-phase triangle-weight reset blending, mandatory per-cycle reseed, optional variance-preserving normalization; and the modulation bookkeeping (noise multiplies density/T_emit/edge/height **amplitudes only** — never p_μ, u^μ, g, g⁴, chroma form, or f_PT; T-modulation pre-g so g⁴-not-g⁸ holds; `r_in_eff ≥ r_isco`; step-cap uses worst-case σ_z; integer φ-periodicity; `enabled:false` bit-identical; deterministic hash, no `ti.random`). Noise primitives (fBm/ridged/Voronoi) are texturing, not physics — specified in `docs/specs/2026-06-13-disk-noise-turbulence.md` with `src/renderer/noise.py` (planned) as the CPU source of truth. |

*Last verified: 2026-06-06 (F13 guard (b′) Layer-A splat-placement rule approved +
landed; (b′) is a placement regularization derived from the already-verified guard-(a)
undeflected footprint, not a new physics formula — F13 μ/PSF still match
`REFERENCE_dngr_paper.md` A.2/A.3.1/A.7). Do not update formulas without re-verifying
against primary sources listed in each section.*