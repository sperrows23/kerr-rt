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
| Coordinates | Boyer-Lindquist (BL): (t, r, θ, φ) |
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

## Conservation test requirements

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
- [x] Simple: `T = T_0 · (6/r)^{0.75}` — fast, already in original code
- [ ] Novikov-Thorne (1973) flux profile — physically correct, more work

Record the chosen options here once decided and reference them in `CLAUDE.md`.

---

## File locations (project conventions)

```
skills/kerr-physics/SKILL.md     ← this file
src/renderer/geodesic.py         ← Formulas 1, 6, 7
src/renderer/disk.py             ← Formulas 2, 3, 4, 5, 8, 9
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
| v1.7 | **F13 guard (b′) ADDED + APPROVED (owner, 2026-06-06):** Layer-A splat *placement* rule when `det J` is invalid (spin-axis seam / non-ESCAPED neighbour). The shipped code positioned the splat with the degenerate `J⁻¹`, collapsing all polar-cell stars onto the meridian (the "Artifact B" seam pileup, ≈15× the off-seam jump). (b′) places the splat by the star's true proper angular separation `d² = (Δθ′²+sin²θ′·Δφ′²)/dΩ` under the undeflected footprint `dΩ=|det J₀·sinθ′₀|` (the guard-(a) quantity), so seam stars keep real angular spacing at μ=1. Resolves the dngr-default seam (`test_no_spin_axis_seam`, `test_background_has_no_vertical_seam_stripe`). The matching Formula-10 `texture`-LOD regularization is a separate follow-up (spec §7.2), **not** part of this revision. |

*Last verified: 2026-06-06 (F13 guard (b′) Layer-A splat-placement rule approved +
landed; (b′) is a placement regularization derived from the already-verified guard-(a)
undeflected footprint, not a new physics formula — F13 μ/PSF still match
`REFERENCE_dngr_paper.md` A.2/A.3.1/A.7). Do not update formulas without re-verifying
against primary sources listed in each section.*