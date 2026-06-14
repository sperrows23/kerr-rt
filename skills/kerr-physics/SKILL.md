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
- Derived config parameters — r_plus/r_isco/r_inner/T_0/orbital periods from base spin etc. (Formula CKS-13)
- Volumetric disk radiative transfer / source-function march (Formula CKS-14)
- Disk radial self-shadow / deep-shadow-map (Formula CKS-15 — VISUALIZATION)
- Flared 3D disk scale height σ_θ(r) (Formula CKS-16 — GEOMETRY/TEXTURE)
- Disk 3D inner-edge-ray self-shadow (radial + vertical) (Formula CKS-17 — VISUALIZATION)
- Disk curl-flow domain warp / divergence-free noise-coordinate distortion (Formula CKS-18 — VISUALIZATION)
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

> **Display color grade (NOT physics — do not "fix" the chromaticity to chase it):**
> the warm amber of the showcase renders does **not** come from Formula 9.
> `blackbody_rgb` is chromaticity-only and trends sepia/white (the blue channel
> never drops far enough for a saturated amber) — that is correct and must be left
> alone. The amber is produced **downstream of all physics** by a post-process
> color grade in `renderer.tonemap` (config `render.color_grade`: `saturation`
> luma-based + `tint` per-channel linear gain), applied in linear HDR **before**
> the Reinhard compressor `img/(1+img)`. It is a VISUALIZATION dial of the same
> governance class as `disk.doppler_strength` (CKS-9) — it touches no `T`, `g`,
> `g⁴`, or chromaticity quantity. Identity defaults (`saturation=1.0`,
> `tint=[1,1,1]`) are bit-identical to the ungraded path. If a future "the disk
> looks too sepia" request arrives, tune the grade — never recalibrate Formula 9.

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

> **Status (2026-06-13): owner-approved; D2.1 (primitives) + D2.2 (static density
> modulation) + D2.3 (§2 Keplerian shear advection) + D2.4 (§3 temperature / inner+outer
> edge / scale-height modulation) ALL WIRED. Procedural noise is now ON in the shipped
> `configs/render.yaml` (`disk.noise.enabled: true`, `modulation.enabled: true`).** The
> GPU beauty path applies §3 on density AND on emission amplitudes: the four §3
> envelopes (`n_T`, `n_e_in`, `n_e_out`, `n_h`) are advected with the SAME §2 dual-phase
> reset blend + `dynamism` gain as the density field (`taichi_renderer._disk_noise_mod_fields`
> ↔ CPU `noise.noise_modulation_fields`), then applied per CKS-12 §3 constraints:
> `T_emit ← T_emit·(1+τ·(n_T−½))` BEFORE the g-shift (constraint 2, keeps g⁴-not-g⁸);
> `r_in_eff = max(r_inner·(1+e_in·(n_e−½)), r_isco)` (constraint 3); `r_out_eff =
> r_outer·(1+e_out·(n_e'−½))`; smoothstep edge windows replace the hard radial cutoffs;
> `σ_θ ← σ_θ·(1+h·(n_h−½))` (lumpy scale height, with the Pipe-B step cap sized on the
> worst-case `σ_z·(1−h/2)` — constraint 4). The envelopes are single [0,1] fBm decorrelated
> by distinct seed offsets (`NSEED_MOD_T/EIN/EOUT/H`) and carry NO variance-preserve divide
> (convex triangle weights keep them in range). CPU look-dev in `thumb.py`. Tests:
> `tests/test_noise.py §3` (4: disabled-is-½, unit-range, decorrelation, advect+determinism),
> `tests/test_disk_noise.py::test_mod_fields_match_cpu_reference` (GPU↔CPU), and
> `tests/test_disk_step_convergence.py::test_disk_emission_resolves_lumpy_slab`
> (constraint-4 worst-case-σ_z cap on a §3-lumpy thin slab). `t_disk` threaded through
> `render_beauty_frame{,_mb}` and `export_exr.py` (`frame/fps · time_scale`). `disk.noise.enabled:
> false` (and `modulation.enabled: false`) keep the legacy bit-identical branch (constraint 6);
> the GR/calibration guards (`test_gpu_regression.py`, base `test_disk_step_convergence`) force
> noise OFF so the global enable does not shift the pinned goldens. **Not new GR.**
> The only physics input is Ω from **Formula 3, reused verbatim**. Everything else
> is procedural texturing that multiplies *amplitude* quantities (density, emitted
> temperature, edge/height windows). The noise primitives themselves (fBm, ridged
> multifractal, Worley/Voronoi) are texturing functions, not physics — they live in
> `src/renderer/noise.py` (CPU NumPy source of truth) with `@ti.func` twins in the
> same file held to it by `tests/test_noise{,_gpu}.py` (CPU↔GPU agreement,
> φ-periodicity, determinism). The hard constraints below (integer φ-period,
> deterministic hashing, `enabled:false` bit-identity) are pinned by those tests.

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
s    = t_disk / T                       # T = disk.dynamics.shear_period_M (DERIVED, CKS-13)
a_k  = fract(s + k/2),  k ∈ {0, 1}      # each phase's age fraction ∈ [0, 1)
c_k  = floor(s + k/2)                   # phase-k cycle index
w_k  = 1 − |2·a_k − 1|                  # triangle weights; w_0 + w_1 ≡ 1

φ′_k = φ − Ω(r) · (a_k · T)             # each phase sheared for at most T

n(u, φ, ζ; t) = w_0·N(u, φ′_0, ζ; hash(seed, k=0, c_0))
              + w_1·N(u, φ′_1, ζ; hash(seed, k=1, c_1))
```

- `t_disk` is the disk animation time in geometric units; callers compute it as
  `frame_index / render.fps × disk.dynamics.time_scale` (`time_scale` is DERIVED
  by the CKS-13 resolver from `disk.dynamics.inner_lap_seconds`).
- **Per-cycle reseed** (the `c_k` term in the hash, or equivalently a hashed
  per-cycle domain offset) is mandatory — without it the whole animation repeats
  with period T.
- **Optional variance preservation:** the crossfade lowers contrast mid-blend
  (`w² sum < 1`); dividing the blend by `sqrt(w_0² + w_1²)` removes the periodic
  contrast "breathing" (config `variance_preserve`).
- T is a look dial: long T → long Interstellar-style filaments; short T → choppier.
- **Non-physical viz dial — `disk.noise.dynamism` (default 1.0):** the renderer
  multiplies the shear amount by this gain, `φ′_k = φ − dynamism·Ω(r)·(a_k·T)`.
  `dynamism = 1` reproduces the formula above bit-for-bit (the default path is
  unchanged); `> 1` exaggerates the per-frame differential winding (the swirl) for a
  given `t_disk` **without** changing the reset cadence (the `c_k`/reseed structure is
  unaffected, so C0-continuity at resets still holds — `w_k = 0` at each reset
  regardless of the gain). This is artistic emphasis only — the same dial spirit as
  `disk.doppler_strength` (Formula CKS-9) — and touches no metric/g/g⁴ quantity.

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

## Formula CKS-13 — Derived disk/orbit parameters (the config resolver; owner-approved 2026-06-13)

**Implemented in `src/renderer/kerr_params.py` (`resolve_config`), called by every
config loader.** `configs/render.yaml` stores **base** parameters only (spin,
disk extent, target peak temperature, `disk.dynamics` look targets); everything
that is a function of them is derived at load time so no dependent literal can
desync when a base parameter is edited (the old `r_isco: 1.182` failure mode).

Nothing below is new physics — each line is a pinned formula or its trivial
algebraic inverse:

```
r_plus  = 1 + sqrt(1 − a²)                       # Formula CKS-6, verbatim
r_isco  = BPT closed form                        # Formula 2, verbatim (disk_flux.isco_radius)
Ω(r)    = 1 / (r^{3/2} + a)                      # Formula 3, verbatim
T_orb(r) = 2π/Ω = 2π·(r^{3/2} + a)               # Formula 3 inverse (geometric M)
t_wrap  = 2π / (Ω(r_inner) − Ω(r_outer))         # one full differential 2π shear wrap

# Derived config values:
disk.r_inner          = r_isco            # 'auto'; numeric override clamped to ≥ r_isco
                                          # (CKS-11 zero-torque BC / CKS-12 constraint 3)
disk.T_0 (page_thorne) = T_peak           # f_PT LUT is max-normalized ⇒ max T_eff = T_0
disk.T_0 (simple)      = T_peak·(r_inner/6)^{3/4}   # Decision-B law peaks at r_inner
dynamics.time_scale    = T_orb(r_inner) / inner_lap_seconds    # M per footage second
dynamics.shear_period_M = shear_wrap_budget · t_wrap           # CKS-12 §2 reset period T
```

**Closed forms, not lookup tables:** for Kerr orbital quantities the
Bardeen–Press–Teukolsky (1972) expressions are exact — an external table would
only add interpolation error on top of the same equations. Published values are
pinned as **test anchors** instead (`tests/test_kerr_params.py`: a=0 → r_isco=6,
r₊=2; a=1 → both 1; a=0.999 → 1.182/1.0447, the SKILL.md Formula-2 verified
value). The one profile with no closed form, the Page–Thorne flux, is already a
precomputed LUT (Formula CKS-11, `disk_flux.build_flux_lut`).

**Override semantics:** an explicit numeric `disk.r_inner` (≥ r_isco) or a legacy
`disk.T_0` key wins over derivation — artistic escape hatches, resolved at load,
idempotent on re-resolve. `black_hole.r_isco` / `black_hole.r_plus` are ALWAYS
overwritten and must not be stored in the YAML.

**Addendum (V2, 2026-06-14) — flared θ-band coverage `theta_half_bound`.** When the
CKS-16 flare is on, the disk envelope is thicker at `r_outer`, so the photon trace
band must widen to avoid hard-clipping the outer Gaussian. The resolver derives a
SEPARATE key `theta_half_bound` (the *bounding* half-angle the kernels test
`|θ−π/2|` against) and `flare_beta`, leaving the base `theta_half_width` untouched:

```
σ0           = theta_half_width · vertical_sigma_frac      # r=r_inner angular width
flare_beta   = β  if flare.enabled else 0                  # CKS-16 exponent
theta_half_bound = max(theta_half_width,                   # ≥ band_sigma·σ_θ(r_outer)
                       band_sigma · σ0 · (r_outer/r_inner)^β)   if (enabled and β>0)
                 = theta_half_width                            otherwise
```

`band_sigma` (`disk.volumetric.flare.band_sigma`, default 3.0 ≈ ±3σ) is the coverage
factor. **`theta_half_width` is the σ0 anchor and is NEVER mutated** — deriving a
separate `theta_half_bound` is what keeps `σ0` pinned to the base and makes the
resolver idempotent for free (re-resolving recomputes the same bound). `flare.enabled
false` OR `β=0` ⇒ `theta_half_bound == theta_half_width` and `flare_beta == 0`
(bit-identical, no widening). Validation: `β < 0` and `band_sigma ≤ 0` are rejected
(disks flare OUTWARD; the band factor is positive). Patch BOTH loaders
(`taichi_renderer`, `thumb.py`) per the dual-loader rule.

---

## Formula CKS-14 — Volumetric RTE source-function march (owner-approved 2026-06-13; NO new GR)

> **Status:** standard emission/absorption radiative transfer, NOT a metric or
> geodesic change. Assembled entirely from already-verified terms — CKS-9 `g_eff`,
> CKS-11 `f_PT`, Formula-9 chromaticity·g⁴. Gated behind `disk.volumetric.
> source_function` (default `false` ⇒ the legacy emission-only sum, golden frames
> intact). Spec: `docs/specs/2026-06-13-V1-self-shadow-source-function.md`.

The Pipe-B disk march integrates the radiative transfer equation along the photon
path. In optical-depth form, per step:

```
dI = (S − I) dτ            S = j/κ   (source function = emissivity / absorption)
```

`_disk_emit_cks` already returns the two quantities this needs: the emission RGB
`= j·ds` and `dτ = κ·ds`. The source function is therefore their **ratio**, in
which the mass density ρ and the step length `ds` cancel **exactly**:

```
S = (j·ds)/(κ·ds) = emission / dτ
  = (emis_c · ρ · [f_PT] · g_eff⁴ · chroma · ds) / (absb_c · ρ · ds)
  = (emis_c / absb_c) · [f_PT] · g_eff⁴ · chroma          ← ρ and ds gone
```

`S` is density-independent: the colour/brightness a *fully opaque* parcel of gas
would show. The analytic front-to-back update of `dI=(S−I)dτ` over one step is

```
w         = 1 − exp(−dτ)
disk_col += transm · w · S
transm   *= exp(−dτ)
```

**Back-compatibility (thin limit).** As `dτ→0`, `w = 1−e^{−dτ} → dτ`, so
`transm·w·S → transm·dτ·(j/κ) = transm·(κ·ds)·(j/κ) = transm·j·ds = transm·emission`
— **exactly** the legacy `disk_col += transm·emission` (Formula 9), to first order
in `dτ`. The two differ only at O(dτ²), so this is flag-gated, not bit-identical;
goldens stay on the legacy branch. The implementation falls back to the legacy term
when `dτ ≤ _RTE_TAU_EPS` (≈1e-6), so there is no divide blow-up and no discontinuity.

**Thick limit & what CKS-14 actually buys (be precise — verified 2026-06-13).** The
legacy emission march and the CKS-14 source-function march integrate the **same
continuum quantity** `I = ∫ S e^{−τ} dτ` — because `transm·j·ds = transm·S·(κ·ds) =
transm·S·dτ`. They differ **only in quadrature**: legacy is the left-endpoint
rectangle rule (`transm·S·dτ` per step), CKS-14 is the exact per-step solution for
piecewise-constant `S` (`transm·S·(1−e^{−dτ})`). In the thin limit they agree
(`1−e^{−dτ}→dτ`); in the **thick** limit (`dτ` per step ≫ 1) the legacy rule
**over-counts** each opaque step by `dτ/(1−e^{−dτ}) > 1`, so CKS-14 is *dimmer and
more accurate* there (empirically ≈10% on the edge-on disk at `absorption=8`). CKS-14
does **not** by itself turn a black disk bright — the standalone gains are (i)
removing that thick-step over-count, and (ii) **materialising `S`** (the opaque-
parcel colour), which is exactly the object CKS-15 self-shadow attenuates
(`emission *= e^{−τ_shadow}` ⇒ `S·e^{−τ_shadow}`) to carve the dark voids. The
glowing-gas-with-voids look therefore needs **CKS-14 + CKS-15 together**, not CKS-14
alone. (Guarded by `tests/test_disk_source_function.py`:
`test_source_function_changes_thick_disk`.)

**g-bookkeeping unchanged.** `S` carries `g_eff⁴·chroma` exactly once — the same
single application as Formula 9 / CKS-11 Piece 3 (`_blackbody_rgb` is chromaticity-
only; no g⁸ double-count). CKS-14 does NOT touch p_μ, u^μ, g, g⁴, or f_PT.

**Implementation:** march-loop reinterpretation in `render_beauty_physics` only;
`_disk_emit_cks` is unchanged (still returns `vec4(emission_rgb, dτ)`). Guard:
`tests/test_disk_source_function.py` (optically-thin equivalence to the legacy
frame) + the unchanged `test_gpu_regression.py` (flag-off ⇒ goldens bit-identical).

---

## Formula CKS-15 — Radial deep-shadow-map self-shadow (owner-approved 2026-06-13; VISUALIZATION, NOT a metric)

> **Status:** a VISUALIZATION occlusion model, flagged exactly like
> `disk.doppler_strength` (CKS-12 constraint 1) — it multiplies the emission
> *amplitude* only and never touches `p_μ`, `u^μ`, `g`, `g⁴`, `f_PT`, or the
> chromaticity form. The shadow ray is a **straight radial line in CKS, not a
> geodesic**. Gated behind `disk.volumetric.self_shadow.enabled` (default `false` ⇒
> no bake, no lookup, golden frames bit-identical). If a *physical* shadow transport
> (geodesic shadow rays, multi-scatter) is ever wanted, STOP and extend this skill
> first (CLAUDE.md policy). Spec: `docs/specs/2026-06-13-V1-self-shadow-source-function.md`.

The dominant illuminator of the disk is its own **hot inner edge** (peak
`T_eff = T_0·f_PT^¼` near `r_inner`, strongest `g⁴` beaming). Gas at larger `r` is
shadowed by all the gas between it and the inner edge at the same `(φ, ζ)`. CKS-15
captures this **in-plane (radial) self-shadowing** — clumps casting dark wakes
*outward* — the dominant void mechanism for the 2.5D slab. (Vertical self-shadowing,
top gas shadowing the midplane, needs the V2 3D bulk and is out of V1 scope — it is
now provided by **Formula CKS-17**, which generalises this radial column scan to a
3D inner-edge ray and *supersedes the bake below* when `self_shadow` is on. CKS-15
remains the ζ=0 midplane limit of CKS-17.)

**The deep-shadow-map (baked once per frame).** A 3-D cumulative absorption optical
depth `τ_shadow[NU, NPHI, NZ]` on the CKS-12 noise coordinates
`(u = ln r/r_inner, φ = atan2(y,x), ζ = Δθ/σ_θ)` — dense where the gas is. For each
`(φ, ζ)` column, march `u` **outward from u=0**:

```
τ_shadow(u, φ, ζ) = Σ_{u'=0..u}  absb_c · ρ(u', φ, ζ; t_disk) · (r' · Δu)
```

with `dr = r·du` (since `u = ln r/r_inner`) and `ρ` the **identical** density the
emission march uses — the shared `@ti.func _disk_density_cks` (CKS-14 V1.0
extraction) is called by BOTH the bake and the emit so they can never drift,
including the §2 shear advection and §3 modulation at the current `t_disk`. `κ` here
is the same `absb_c` as `dτ`. Each cell stores τ from gas **strictly inward** of it
(the running sum *before* its own contribution) so a cell never shadows itself; the
total is clamped to `max_tau` (overflow / caustic safety). Re-baked per frame (it
tracks `t_disk`).

**The lookup (per primary sample).** Trilinear-sample `τ_shadow` at the sample's
`(u, φ, ζ)` (φ periodic — no φ=0 seam; u, ζ clamp at the grid edges) and dim the
**emissivity `j`** before it becomes the source function:

```
τ_s        = trilinear(τ_shadow; u, φ, ζ)
emission  *= exp(−shadow_strength · τ_s)        # j → j·e^{−τ_s};  κ/dτ UNCHANGED
```

Only `j` is attenuated; the absorption `κ`/`dτ` is **not** (the gas still occludes
behind it regardless of how lit it is). Composes exactly with CKS-14:
`S = emission/dτ` inherits the `e^{−τ_s}` factor, so a shadowed thick parcel reads
**dark** (the deep void), not merely dim. Works with the legacy march too (it just
dims `emission`), but the glowing-gas-**with-voids** look needs **CKS-14 + CKS-15
together** — CKS-14 materialises `S`, CKS-15 carves it.

**Governance (why this is a viz approximation, not GR).** Straight radial CKS shadow
ray (not a geodesic — the inner-to-sample bending is small at close-up scale, accepted
like `doppler_strength`'s non-physical shift); single illuminator direction (radially
inward); single-scatter; no emission along the shadow march — occlusion bookkeeping,
not a transport solve. It multiplies the emission amplitude only (CKS-12 constraint 1).

**Implementation:** module field `disk_shadow_tau[NU,NPHI,NZ]` (always allocated by
`_setup_disk_shadow`; extents `u_max`, `ζ_max` baked as module globals so the lookup
needs no extra args); `@ti.kernel bake_disk_shadow` (radial scan off `_disk_density_cks`);
`@ti.func _sample_shadow_tau` (trilinear, φ-periodic); the lookup + `emission *=
exp(−strength·τ_s)` in `_disk_emit_cks` behind the `self_shadow`/`shadow_strength`
kernel args of `render_beauty_physics`, threaded from `disk.volumetric.self_shadow`.
Guards: `tests/test_disk_self_shadow.py` (flag-off bit-identity; GPU bake vs the
analytic Gaussian column; outward-steepening dimming; noise-on contrast rise) + the
unchanged `test_gpu_regression.py` (flag-off ⇒ goldens bit-identical).

---

## Formula CKS-16 — Flared disk scale height σ_θ(r) (owner-approved 2026-06-14; GEOMETRY/TEXTURE, NOT GR)

> **Status:** a GEOMETRY/TEXTURE shape, NOT a metric, geodesic, or emission-physics
> change — it sets *where the gas is*, exactly like the CKS-12 §3 scale-height term it
> extends. It modulates the density **envelope** only and never touches `p_μ`, `u^μ`,
> `g`, `g⁴`, `f_PT`, or the chromaticity form. Gated behind `disk.volumetric.flare.
> enabled` (default `false` ⇒ constant-σ slab, golden frames bit-identical). Spec:
> `docs/specs/2026-06-14-V2-flared-3d-density.md`. If review finds this implies a new
> GR derivation, STOP and ask the human (CLAUDE.md policy).

The V1 disk uses a **constant** angular scale height `σ_θ = theta_half_width ·
vertical_sigma_frac`, which squashes the noise stack's `ζ = Δθ/σ_θ` coordinate into a
thin sheet. CKS-16 gives the slab radius-varying vertical bulk:

```
σ_θ(r) = σ0 · (r / r_inner)^β          σ0 ≡ theta_half_width · vertical_sigma_frac
```

- `β = 0` ⇒ `σ_θ(r) ≡ σ0` ⇒ **exactly the V1 constant slab, bit-for-bit** (the
  implementation skips `ti.pow` when `flare_beta == 0`, so it is the same instruction
  stream, not just numerically close).
- `σ0` is the `r = r_inner` value — the **inner edge keeps today's thickness**; `β > 0`
  thickens the disk **outward** (astrophysical flare, H/r growing with radius). `β < 0`
  is rejected at config-resolve (disks flare outward).
- **Why the angular envelope is the right object to flare:** near the equator
  `z = r cosθ ≈ −r·dz_ang`, so a *constant* angular `σ_θ` already means a *constant
  H/r* (physical height ∝ r) disk; `β > 0` makes H/r itself grow outward.

**Composition with CKS-12 §3 (order preserved).** The §3 lumpy-scale-height envelope
multiplies on top of the flared base — `σ_θ(r)` replaces the constant `σ0` as the base
that the `(1 + h_amp·(n_h−½))` factor perturbs:

```
σ_eff = σ_θ(r) · (1 + h_amp·(n_h − ½))          # §3 constraint 4, unchanged form
```

The Gaussian density `exp(−½(dz_ang/σ_eff)²)` and the noise coordinate `ζ = dz_ang/σ_eff`
both use `σ_eff`, so **genuine 3D falls out for free**: the existing `ridged3`/`fbm3`
stack already consumes `ζ`, and a real radius-varying thickness simply un-squashes it.
No new noise primitive is added (the V1.5 simplex basis stays unwired, reserved for V3
curl-flow).

**Single source of truth.** The flare lives once in the shared `@ti.func
_disk_density_cks` (signature gained `flare_beta`, `r_inner`); the emission march
(`_disk_emit_cks`) and the CKS-15 shadow bake (`bake_disk_shadow`) both call it, so they
inherit the same `σ_θ(r)` automatically — the V1.0 "extract shared density" refactor is
what makes this a one-point change.

**Two knock-on fixes.**

1. **θ bounding band** — a flared envelope is thicker at `r_outer` and would be
   hard-clipped by the V1 `theta_half_width` cutoff. Handled by the CKS-13 resolver
   addendum above: a separate derived `theta_half_bound ≥ band_sigma·σ_θ(r_outer)` is
   what the kernels test `|θ−π/2|` against, while `theta_half_width` stays the σ0 anchor.
2. **Pipe-B vertical step cap** (`max_step_vfrac`) — flare only thickens *outward*, so
   the thinnest slab is the inner edge `σ0`, today's worst case, which the existing cap
   already sizes for (`sigma_z = r·σ0`). **No cap change**; verified (not assumed) by the
   flared-slab convergence test.

**Implementation:** `flare_beta` + `theta_half_bound` + `sigma_theta0` threaded through
`render_beauty_physics` / `render_beauty_frame` (host computes `σ0` and reads the derived
keys); `_disk_density_cks` applies `σ_eff = σ_θ·ti.pow(r/r_inner, flare_beta)` (skipped at
β=0); bound checks use `theta_half_bound`, the step cap uses `σ0`. CPU twin in
`scripts/thumb.py::_march_disk`. Gated by `disk.volumetric.flare.enabled` (default
`false`). Guards: `tests/test_disk_flare.py` (resolver no-op/widen/monotone/idempotent/
validation + GPU flag-off bit-identity vs the no-flare march + a β>0 disk that is
NaN-free, differs, and thickens — a *wider emitting silhouette*, NOT higher total
luminance: see the note below) + the unchanged `test_gpu_regression.py` /
`test_disk_step_convergence.py` (default-off ⇒ goldens bit-identical).

**Flare is geometry, not a brightness boost (verified 2026-06-14).** The GPU test
measured a flared disk reading marginally *dimmer* in total (~1.6% on the inclined
showcase view), not brighter. This is physically correct and worth recording: the hot
inner edge — where peak emission lives — sits at `r_inner`, whose σ is the *anchored*
σ0 and **does not change** under flare; flare only adds **cold** outer gas at larger
`|z|`. Under the absorbing march (`transm *= e^{−dτ}`) that extra bulk self-absorbs the
bright inner edge slightly more than the dim cold gas it contributes. So the honest,
inclination-robust signature of the added vertical bulk is a **larger emitting
silhouette** (gas spreading into previously-dark off-plane pixels), which is what the
test asserts — not integrated luminance. (Trust-the-math: the look emerges from the
correct envelope + absorption, it is not tuned for brightness.)

---

## Formula CKS-17 — 3D inner-edge-ray self-shadow (radial + vertical) (owner-approved 2026-06-14; VISUALIZATION, NOT a metric)

> **Status:** the SAME VISUALIZATION occlusion class as CKS-15 — it multiplies the
> emission *amplitude* only and never touches `p_μ`, `u^μ`, `g`, `g⁴`, `f_PT`, or the
> chromaticity form. The shadow ray is a **straight line in CKS, not a geodesic**;
> single illuminator (the hot inner edge), single-scatter, no emission along the
> shadow march — occlusion bookkeeping, not a transport solve. It stays inside the
> CKS-15 governance envelope (straight ray / single-scatter / amplitude-only), so it
> is an *extension of CKS-15*, not the "physical shadow transport" that section says
> to stop for. Gated by the SAME flag `disk.volumetric.self_shadow.enabled` (default
> `false` ⇒ no bake, no lookup, golden frames bit-identical). Owner picked the 3D-ray
> model over a separable top-down column on 2026-06-14. Spec:
> `docs/specs/2026-06-14-V2-vertical-self-shadow.md`.

CKS-15 shadows each sample along a **radial** ray at constant `(φ, ζ)` — it captures
gas casting wakes *outward* but cannot capture the **vertical** occlusion the V2 3D
bulk makes physical: an off-midplane parcel is shadowed by the dense midplane gas
lying between it and the hot inner edge. CKS-17 unifies both by making the shadow ray
**3D**: from the illuminator at the inner edge **in the midplane** `(u=0, ζ=0)` to the
sample `(u_s, φ, ζ_s)`, at fixed `φ` (azimuthal bending ignored, as in CKS-15).

**The ray (fixed `φ`, parameterised by `u ∈ [0, u_s]`).** The vertical coordinate
interpolates linearly from the midplane illuminator to the sample:

```
ζ(u) = (u / u_s) · ζ_s          # ζ(0)=0 at the inner edge, ζ(u_s)=ζ_s at the sample
r(u) = r_inner · e^u
Z(u) = r(u) · ζ(u) · σ_θ(r(u))  # near-equator physical height; σ_θ = CKS-16 flared σ
```

**The baked optical depth** (still `τ_shadow[NU, NPHI, NZ]`, same field/grid/lookup as
CKS-15). For target cell `(i_u, φ, i_z)` march the strictly-inner radial cells
`j = 0 … i_u−1` (a cell never shadows itself — the inner-edge accumulation rule is
unchanged) and accumulate the SAME absorption the emission march uses, `κ·ρ·ds`, but
now along the tilted ray:

```
ζ_j   = (u_j / u_s) · ζ_s          u_j = (j+½)·du,  u_s = (i_u+½)·du
ρ_j   = _disk_density_cks(φ, r_j, dz_ang = ζ_j·σ_θ(r_j))     # tilted sample, shared ρ
ds_j  = sqrt( (r_j·du)² + (ΔZ_j)² )                          # 3D arc length
ΔZ_j  = Z(u_j+½du) − Z(u_j−½du)                              # ray height change over the cell
τ_shadow(i_u,φ,i_z) = min( Σ_{j<i_u} absb_c · ρ_j · ds_j , max_tau )
```

- **Exact CKS-15 reduction on the midplane.** At `ζ_s = 0` the ray is flat: `ζ_j ≡ 0`,
  `ΔZ_j ≡ 0`, so `ds_j = r_j·du` and `ρ_j = ρ(u_j, φ, 0)` — the integrand becomes
  CKS-15's radial column **term for term**. The radial element keeps the `dr = r·du`
  convention (NOT an endpoint `ΔR`) precisely so this reduction is bit-exact, and the
  vertical leg `ΔZ` is added in quadrature (zero on the midplane). CKS-15 is the
  `ζ→0` limit of CKS-17, not a separate code path.
- **Why off-midplane changes.** For `ζ_s ≠ 0` the ray tilts toward the midplane going
  inward (`ζ_j < ζ_s`), so it traverses **denser** gas than CKS-15's constant-`ζ_s`
  column — an off-plane parcel is now correctly shadowed by the bright midplane slab
  between it and the inner edge. This is the entire point: vertical self-shadow.
- **`σ_θ(r)` is the CKS-16 flared base** (`σ0·(r/r_inner)^β`, `β=0 ⇒ σ0` with no
  `ti.pow`), so on a flared disk the ray height `Z` follows the real envelope.

**The lookup and application are UNCHANGED from CKS-15.** Trilinear (φ-periodic) sample
of `τ_shadow` at the primary sample's `(u, φ, ζ)`, then `emission *= exp(−shadow_strength
· τ_s)` on the EMISSIVITY only (`κ`/`dτ` untouched; composes with CKS-14 so `S` inherits
`e^{−τ_s}`). Only `bake_disk_shadow`'s *ray geometry* changed — the field shape,
`_sample_shadow_tau`, and the `_disk_emit_cks` application are identical.

**Cost.** The 3D ray is not a prefix sum (each target `ζ_s` tilts its own ray), so the
bake is `O(NU)` per cell ⇒ `O(NU²·NPHI·NZ)` overall vs CKS-15's `O(NU·NPHI·NZ)` — ~`NU/2`×
more density evals per frame, parallelised over all cells on the GPU. Accepted for the
offline bake (owner chose the 3D ray knowing it is the heavier model).

**Governance (why this is still a viz approximation, not GR).** Straight CKS shadow ray
(not a geodesic — the inner-to-sample bending is small at close-up scale, accepted like
`doppler_strength`); single illuminator (the inner edge, midplane); single-scatter; no
re-emission along the shadow march. It multiplies the emission amplitude only (CKS-12
constraint 1). If a *physical* shadow transport (geodesic shadow rays, multi-scatter,
an anisotropic phase function) is ever wanted, STOP and extend this skill first
(CLAUDE.md policy).

**Implementation:** `bake_disk_shadow` in `taichi_renderer.py` rewritten from the radial
column scan to the per-cell 3D ray march (same signature — it already takes `r_inner`,
`r_outer`, `sigma_theta0`, `flare_beta`, `zeta_max`, `max_tau`, `absb_c`); no new config,
no new field, no kernel-arg change in `render_beauty_physics`. Guards: the CKS-15
`tests/test_disk_self_shadow.py` carries over — flag-off bit-identity, outward-steepening
dimming, and noise-on contrast-rise are unchanged relational checks; only
`test_bake_matches_analytic_gaussian_column` is re-derived to the 3D-ray line integral
(the constant-`ζ` radial closed form was the CKS-15 model and is now superseded
off-midplane). `test_gpu_regression.py` (default-off ⇒ goldens bit-identical) unchanged.

---

## Formula CKS-18 — Curl-flow domain warp (owner-approved 2026-06-14; VISUALIZATION, NOT a metric)

> **Status:** the SAME VISUALIZATION class as CKS-12 §2/§3 — it relocates the noise
> **sample coordinate** only (texturing), never a metric/transport quantity. Owner
> decisions 2026-06-14: (1) stage as **V3.0 static warp** first, then V3.1 curl-flow
> *advection* (the D2.2→D2.3 split); (2) V3.0 displacement is **in-plane `(u,φ)`** only
> (ζ untouched). Spec: `docs/specs/2026-06-14-V3-curl-domain-warp.md`.

CKS-12 §2 winds the noise azimuthally at `Ω(r)` — *laminar* shear, no eddies. CKS-18
adds divergence-free turbulent structure by warping the noise coordinate with the curl
of a scalar potential built on the **V1.5 isotropic simplex basis** (§3.6), which has no
axis-aligned grid bias (a curl field on the square-lattice `gnoise` would show directional
streaks) and, sampled on the cylinder embedding of the φ-axis, is **seamless across φ=0**
— the "exact φ-seamlessness at the V3 integration point" §3.6 promised.

### Potential and in-plane curl (V3.0 static)

```
P(u,φ) = ( cos φ · ρ_c , sin φ · ρ_c , u · k_u )         # ρ_c, k_u = angular / radial freq
ψ(u,φ) = sfbm3( P ; octaves, lacunarity, gain, curl_seed )

∂ψ/∂u ≈ (ψ(u+ε,φ) − ψ(u−ε,φ)) / 2ε      ∂ψ/∂φ ≈ (ψ(u,φ+ε) − ψ(u,φ−ε)) / 2ε   # central FD, simplex has no analytic grad here
δu = +∂ψ/∂φ        δφ = −∂ψ/∂u           # ∇·(δu, δφ) ≡ 0 by construction (curl of a scalar)
u' = u + A·δu       φ' = φ + A·δφ          # A = curl amp; ε = curl_fd_eps (chart step)
```

- **Divergence-free** is the defining property: the displacement is the 2D curl of a
  scalar potential, so the warp neither piles up nor evacuates the texture (the
  incompressible-flow look). Pinned by a numeric `∇·(δu,δφ) ≈ 0` test.
- **Seamless in φ (CKS-12 constraint 5 preserved).** `δu`, `δφ` are built on `cos φ`,
  `sin φ` ⇒ exactly 2π-periodic in φ ⇒ `φ'` is continuous across the seam, even though
  classic simplex is *not* lattice-periodic. Seamlessness comes from the **embedding**,
  not a lattice period, so `ρ_c`, `k_u` may be **any real** (no integer-period restriction
  the φ-periodic density stack has).

### Integration — material-frame warp, two entry points

Apply the warp at the entry of `_disk_noise_m` (density layer stack) and `_mod_fbm4`
(§3 modulation envelopes) — the two evaluators that receive the **already-sheared**
per-phase `φ′_k` from CKS-12 §2. The eddies are thus frozen into the gas's *material*
frame and the Keplerian shear winds them into filaments (the physically-sensible
composition — eddies advect with the gas, not the lab frame); density and modulation
share one warp so they swirl coherently. The warp uses a **fixed `curl_seed`** (NOT the
per-cycle reseed `c_k` the density draws), so V3.0 is genuinely static — only the §2
winding animates over `t_disk`. In the static `T ≤ 0` path the warp acts on `φ` directly
(material = lab when there is no shear). V3.1 will make `ψ` itself time-dependent.

### Governance — VISUALIZATION, not GR

The warp moves only the noise sample coordinate; `ρ = gauss(ζ)·exp(clamp(m))`,
`emission ∝ ρ·g⁴·…`, `dτ = absb·ρ·ds`, the CKS-9 `g`, the CKS-11 `f_PT` shape, and the
CKS-14/15/17 source-function / self-shadow machinery are all **identical**. It must NEVER
touch `p_μ`, `u^μ` (CKS-8), `g` (CKS-9), the `g⁴` exponent (Formula 9), the chromaticity
form, or `f_PT` (CKS-11) — amplitude/texture only (CKS-12 constraint 1). The
divergence-free construction is the principled choice but it is still texturing, not
transport of a conserved field; a genuine fluid solve (advecting a real density with a
continuity equation) would STOP for skill extension (CLAUDE.md policy).

### Implementation / guards

`src/renderer/noise.py`: `curl_warp` (CPU NumPy source of truth, `sfbm3`-based) +
`@ti.func` twin `curl_warp_ti`. `src/renderer/taichi_renderer.py`: the warp is called at
the entry of `_disk_noise_m` / `_mod_fbm4` behind the curl-enabled param slot; new
`disk.noise.curl` config dials flow through the `_setup_disk_noise` param buffer (grows
past `_NOISE_N = 43`). **No CKS-13 resolver change** (all base look dials, nothing
derived). Cost: central-diff gradient = 4 `sfbm3` evals × octaves, per phase, for density
AND modulation — heavier than §2 but parallel and offline (analytic-gradient or 3-eval
forward-diff is the deferred fallback). Gated by `disk.noise.curl.enabled` (default
`false`, and `amp = 0` ⇒ identity) ⇒ bit-identical to V2 (constraint 6). Guards:
`tests/test_noise.py` (divergence-free, seamlessness, determinism) +
`tests/test_noise_gpu.py` (`curl_warp_ti` CPU↔GPU twin parity) +
`tests/test_disk_noise.py` (default-off wiring, curl-on finite/NaN-free) + unchanged
`test_gpu_regression.py` (default-off ⇒ goldens bit-identical). **GPU-verified 2026-06-14**
(noise_gpu 15, disk_noise + gpu_regression, noise 44 CPU). NOTE the twin tolerance is NOT
`_SATOL`=1e-5 but a derived `amp·_SATOL/fd_eps`: the warp is a *derivative*
(`amp·(ψ₊−ψ₋)/(2ε)`), so the ~1e-5 `sfbm3` twin gap is amplified ×1/(2ε) (obs ~6.5e-5) —
same f32 FD-amplification family as the divergence/seam test tolerances.

### §2 — Curl-flow advection (V3.1: time-dependent ψ; owner-approved 2026-06-14 — Option A + clock B1)

V3.0 (§1) freezes ψ (fixed `curl_seed`): the eddies are static in the gas's material
frame and only the CKS-12 §2 Keplerian shear winds them. V3.1 makes **ψ itself evolve over
`t_disk`** so the eddies boil (form / stretch / merge / dissipate), composed *additively* on
top of the §2 shear — §2 is the laminar bulk-orbit winding, V3.1 is the in-place turbulent
evolution. **Mechanism = Option A: mirror the CKS-12 §2 dual-phase reset blend, applied to
the curl potential ψ** (not the density domain):

```
T_c  = disk.noise.curl.flow_period_M          # curl-flow clock (geometric M); ≤ 0 ⇒ V3.0 static
s_c  = t_disk / T_c
α_k  = fract(s_c + k/2),  k ∈ {0,1}           # phase age
γ_k  = floor(s_c + k/2)                        # phase cycle index (→ reseed)
ω_k  = 1 − |2·α_k − 1|                         # triangle weights, ω_0 + ω_1 ≡ 1
ψ_k(u,φ) = sfbm3( P(u,φ) ; curl_seed + k·NCYC_PHASE + γ_k·NCYC_CYCLE )   # P = §1 cylinder embedding
ψ(u,φ;t) = ω_0·ψ_0 + ω_1·ψ_1
δu = +∂ψ/∂φ,  δφ = −∂ψ/∂u                      # central FD of the BLENDED ψ, exactly as §1
```

- **Divergence-free survives the blend.** Curl is linear ⇒ `curl(ω_0ψ_0+ω_1ψ_1) =
  ω_0·curl(ψ_0)+ω_1·curl(ψ_1)` — a convex combination of divergence-free fields is
  divergence-free. (Equivalently: blend ψ first, then central-difference — the
  implementation does this, one FD stencil over the blended potential, so the §1 4-eval
  cost becomes 2·4 = 8 `sfbm3` evals at `T_c>0`.)
- **Seamless in φ** is inherited per-phase from the cylinder embedding (§1), unchanged.
- **C0-continuous through reseeds.** `ω_k → 0` *exactly* at its own reset (`α_k = 0`), so the
  warp is continuous as `γ_k` steps and the potential reseeds — the proven §2 property, now
  on the time axis. **Reuses the §2 reseed strides `NCYC_PHASE`/`NCYC_CYCLE`** so each cycle
  draws a decorrelated ψ (no period-`T_c` repeat).
- **Static fallback = the V3.0 warp bit-for-bit.** `T_c ≤ 0` ⇒ a single fixed-`curl_seed`
  phase ≡ §1 — the exact mirror of §2's "`shear_period ≤ 0` ⇒ static path bit-identical".
  This is the regression hook: V3.0 / default-off goldens stay valid, and the curl-on V3.0
  look is preserved exactly when the clock is off.
- **Clock B1 — `T_c` is a NEW independent base dial** `disk.noise.curl.flow_period_M`, NOT
  derived from `disk.dynamics.shear_period_M` (eddy turnover and bulk-orbit winding are
  physically independent timescales) ⇒ **no CKS-13 resolver change** (V3.0's
  no-resolver-touch property preserved; the value flows straight through `_setup_disk_noise`).

**Composition with §2 (unchanged from V3.0):** the warp is still applied at the entry of
`_disk_noise_m` / `_mod_fbm4` on the already-sheared per-phase `φ′_k`, so the now-time-varying
eddies live in the material frame and the shear winds them; density and modulation share one
warp. Only the `(δu,δφ)` displacement gained a `t_disk` dependence.

> **⚠ No `flow_dynamism` gain (deferred — flagged, not silently substituted).** The CKS-12
> §2 `dynamism` works because it scales a *continuous* shear displacement `Ω·a_k·T`
> independently of the weights `ω_k(α_k)` and the reseed cadence. A pure reset-blend
> potential has **no** analogous continuous displacement term — its only rate is `1/T_c`,
> which sets boil speed and reseed cadence *together* — so a C0-preserving per-frame
> boil-emphasis decoupled from the reseed cadence is **not separable** in Option A. A boil
> rate independent of reseed cadence requires the **Option B** continuous 4-D time axis
> (`sfbm4(cosφ·ρ_c, sinφ·ρ_c, u·k_u, t·k_t·dynamism)`, the deferred basis), where `t·k_t` is
> a genuine separable scale. `flow_period_M` alone is V3.1; `flow_dynamism` is **not** added.

**Governance (identical to §1):** texturing only — moves the noise sample coordinate, never
`p_μ`/`u^μ` (CKS-8), `g` (CKS-9), the `g⁴` exponent, the chroma form, or `f_PT` (CKS-11). The
divergence-free blend is the principled choice but is still a time-varying *texture-coordinate*
displacement (the Neyret advected-texture sense, exactly as §2), NOT a semi-Lagrangian
continuity solve of a conserved density — a genuine fluid transport STOPs for skill extension.

**Implementation / guards (extend the §1 suite):** `curl_warp` / `curl_warp_ti` gain
`(t_disk, flow_period)` (default `0.0` ⇒ the §1 static path bit-for-bit); the dual-phase blend
wraps the `_psi` stencil. New dial `disk.noise.curl.flow_period_M` through `_setup_disk_noise`
(buffer grows past `_NOISE_N = 52`). Tests: `tests/test_noise.py` (§2 curl — divergence-free
**at several `t_disk`**, seamless at each `t`, reset C0-continuity in time, `flow_period ≤ 0`
static-fallback **bit-identity**, determinism, evolution-smoke) + `tests/test_noise_gpu.py`
(`curl_warp_ti` time-blend twin parity to the same derived `amp·_SATOL/fd_eps` bound) +
`tests/test_disk_noise.py` (default-off wiring, flow-on finite/NaN-free) + unchanged
`test_gpu_regression.py` (default-off ⇒ goldens bit-identical). Spec:
`docs/specs/2026-06-14-V3.1-curl-flow-advection.md`.

---

## File locations (project conventions)

```
skills/kerr-physics/SKILL.md     ← this file
src/renderer/geodesic.py         ← Formulas 1, 6, 7
src/renderer/disk.py             ← Formulas 2, 3, 4, 5, 8, 9
src/renderer/noise.py            ← (D2.1–D2.4, 2026-06-13) CKS-12 noise primitives + noise_density_mult stack (§2 shear advection) + noise_modulation_fields (§3 T/edge/height envelopes) — CPU source of truth + @ti.func twins; (V1.5) §3.6 isotropic simplex basis snoise2/3 + sfbm2/3 (Perlin/Gustavson; non-periodic; the V3 curl potential basis); (V3.0, CKS-18) curl_warp + curl_warp_ti — in-plane divergence-free domain warp of the noise coordinate (sfbm3 scalar potential on the (cosφ,sinφ,u) cylinder embedding, central-FD curl), applied at the entry of _disk_noise_m / _mod_fbm4 behind disk.noise.curl.enabled; (V3.1, CKS-18 §2) curl_warp/curl_warp_ti gain (t_disk, flow_period) + _curl_psi_ti — time-dependent ψ via the §2 dual-phase reset blend (disk.noise.curl.flow_period_M; ≤0 ⇒ static V3.0)
src/renderer/taichi_renderer.py  ← (D2.3+D2.4) _disk_noise_density_mult (§2 density advection) + _disk_noise_mod_fields (§3 vec4 envelopes) + _smoothstep_ti edge windows + _setup_disk_noise param buffer (_NOISE_N=43); _disk_emit_cks / render_beauty_physics gained r_isco; t_disk threaded through render_beauty_frame{,_mb}; (V3.1, CKS-18 §2) t_disk threaded into _disk_noise_m / _mod_fbm4 → _disk_curl_warp + _NI_CURL_FLOWP slot (_NOISE_N=53) for curl-flow advection
src/renderer/kerr_params.py      ← Formula CKS-13 config resolver (derived r_plus/r_isco/r_inner/T_0/dynamics; V2 CKS-16 derives flare_beta + theta_half_bound)
src/renderer/taichi_renderer.py  ← (V2, CKS-16) flared σ_θ(r)=σ0·(r/r_inner)^β in the shared _disk_density_cks (skipped at β=0); sigma_theta0/flare_beta/theta_half_bound threaded through _disk_emit_cks / bake_disk_shadow / render_beauty_physics / render_beauty_frame; behind disk.volumetric.flare.enabled
src/renderer/taichi_renderer.py  ← (V1.0) shared @ti.func _disk_density_cks (Gaussian×§3 noise×edge window — single source for the emit march AND the CKS-15 shadow bake); (V1.1, CKS-14) source-function march in render_beauty_physics behind disk.volumetric.source_function (_RTE_TAU_EPS divide guard); (V1.2, CKS-15) disk_shadow_tau field + bake_disk_shadow kernel + _sample_shadow_tau trilinear lookup behind disk.volumetric.self_shadow.enabled (_setup_disk_shadow allocates; _SHADOW_U_MAX/_SHADOW_ZETA_MAX baked extents); (V2, CKS-17) bake_disk_shadow rewritten to a 3D inner-edge-ray march (radial+vertical self-shadow; CKS-15 is its ζ=0 limit) — same field/lookup/application, same flag
src/renderer/starmap.py          ← Formula 10
src/renderer/taichi_renderer.py  ← Formulas 10, 13 (screen-space Jacobian, μ, star splat)
scripts/ingest_stars.py          ← Formula 13 catalog pre-processing (HYG/ATHYG csv or BSC5 → {θ′, φ′, flux_rgb}.npy; I_base·chroma folded into flux)
tests/test_geodesic.py           ← Conservation tests (Formula 6 conserved quantities)
configs/render.yaml              ← BASE params only: a, WIDTH, HEIGHT, step counts, stars:* (r_isco/r_plus/r_inner/T_0 derived at load — CKS-13)
```

---

## Revision history

| Version | Change |
|---|---|
| v1.0 | Initial release |
| v1.28 | **Formula CKS-18 §2 ADDED + WIRED — curl-flow advection (owner-approved 2026-06-14, V epoch V3.1). NOT a physics revision — VISUALIZATION, same governance class as CKS-18 §1 / CKS-12 §2.** Makes the V3.0 static curl potential ψ **time-dependent** so the eddies boil over `t_disk`, composed additively on the §2 Keplerian shear. **Owner decisions: Option A** (mirror the CKS-12 §2 dual-phase reset blend, applied to ψ) **+ clock B1** (new independent base dial `disk.noise.curl.flow_period_M = T_c`, no CKS-13 resolver change). Mechanism: `ψ = ω_0ψ_0 + ω_1ψ_1`, `ω_k = 1−|2α_k−1|`, `α_k = frac(s_c+k/2)`, `s_c = t_disk/T_c`, per-cycle reseed `seed + k·NCYC_PHASE + γ_k·NCYC_CYCLE` (reusing the §2 strides); the central difference runs over the blended ψ. **All three V3.0 invariants survive:** divergence-free (curl linear ⇒ convex combo of div-free fields), seamless per phase (cylinder embedding), C0-continuous through reseeds (`ω_k → 0` at each reset — the §2 property on the time axis). **`flow_period_M ≤ 0` (default/absent) ⇒ the V3.0 static warp bit-for-bit** (regression hook; mirror of §2's `shear_period ≤ 0`), and `T_c` decoupled from the shear clock ⇒ the curl boils even on the static-shear path. **No `flow_dynamism` gain (flagged, NOT shipped):** a pure reset-blend has no continuous displacement to scale C0-safely independent of the reseed cadence (unlike §2's separable `Ω·a_k·T`); a boil-rate-vs-cadence dial needs the deferred Option-B 4-D `sfbm4` time axis. Texturing only — never `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT`/chroma-form; a continuity-equation fluid solve still STOPs for skill extension. New code: `noise.py` `curl_warp`/`curl_warp_ti` gain `(t_disk, flow_period)` + `_curl_psi_ti` (the dual-phase potential); `taichi_renderer.py` threads `t_disk` through `_disk_noise_m`/`_mod_fbm4` → `_disk_curl_warp` + new `_NI_CURL_FLOWP` slot (`_NOISE_N` 52→53). Gated by `disk.noise.curl.enabled` + `flow_period_M > 0` ⇒ bit-identical to V3.0 when off. Guards: `tests/test_noise.py` §2 (div-free at several t, seamless at each t, C0 through resets, static-fallback bit-identity, evolution+determinism — 5 new, CPU 49) + `tests/test_noise_gpu.py` `test_curl_flow_twin_matches_reference` (time-blend twin parity to `amp·_SATOL/fd_eps`) + `tests/test_disk_noise.py` `test_curl_flow_advection_matches_cpu_and_animates` (end-to-end `t_disk → _disk_curl_warp` threading + animates) + unchanged `test_gpu_regression.py` (default-off ⇒ goldens bit-identical). Spec: `docs/specs/2026-06-14-V3.1-curl-flow-advection.md`. |
| v1.27 | **`render.color_grade` display color grade documented (2026-06-14) — NOT a physics revision.** The warm-amber showcase look is a post-process grade in `renderer.tonemap`, **downstream of all physics**, NOT a Formula-9 chromaticity change. `tonemap` gained `saturation` (luma-based, `luma=Rec.709·img`, `img=luma+s·(img−luma)` clamped ≥0) and `tint` (per-channel linear gain), applied in linear HDR **before** the Reinhard compressor `img/(1+img)`. Config `render.color_grade.{saturation,tint}`; CLI `--saturation/--tint/--amber` in `scripts/showcase_disk.py` (`--amber` ⇒ `saturation=1.6, tint=[1.18,1.0,0.74]`). Same VISUALIZATION governance class as `disk.doppler_strength` (v1.12) — touches no `T`/`g`/`g⁴`/`f_PT`/chroma-form. **Identity defaults (`saturation=1.0`, `tint=[1,1,1]`) are bit-identical** to the ungraded path (`np.array_equal` verified; both the `saturation!=1.0` and `tint!=(1,1,1)` branches skipped). Rationale captured in the Formula-9 dial note: `blackbody_rgb` trends sepia/white by design (blue channel never drops far enough for saturated amber) and must NOT be recalibrated to chase the look — tune the grade instead. No GR formula, kernel, or golden frame touched (display-space only). |
| v1.26 | **Formula CKS-18 ADDED — curl-flow domain warp, owner-approved 2026-06-14, V epoch V3.0. NOT a physics revision — VISUALIZATION, same governance class as CKS-12 §2/§3.** Adds divergence-free turbulent structure to the disk noise by warping the **sample coordinate** with the in-plane curl of a scalar potential `ψ = sfbm3(cosφ·ρ_c, sinφ·ρ_c, u·k_u)` built on the V1.5 isotropic simplex basis: `δu=+∂ψ/∂φ`, `δφ=−∂ψ/∂u` (central FD, simplex has no analytic gradient here), `u'=u+A·δu`, `φ'=φ+A·δφ`. **Divergence-free by construction** (curl of a scalar ⇒ incompressible look); **seamless across φ=0** because `δu`/`δφ` are built on `cos φ`/`sin φ` (CKS-12 constraint 5 preserved even though classic simplex is not lattice-periodic — seamlessness from the cylinder *embedding*, not a lattice period, so `ρ_c`/`k_u` may be any real). Owner decisions: stage V3 as **static warp (V3.0) → curl-flow advection (V3.1)** (the D2.2→D2.3 split); V3.0 displacement is **in-plane `(u,φ)`** only (ζ untouched). Integration: warp applied at the entry of `_disk_noise_m` / `_mod_fbm4` on the already-sheared per-phase `φ′_k` (material-frame — eddies freeze into the gas and §2 winds them into filaments; density + modulation share one warp), using a **fixed `curl_seed`** (not the per-cycle reseed) so V3.0 is genuinely static — only the §2 winding animates. Texturing only — never touches `p_μ`/`u^μ`/`g`/`g⁴`/`f_PT`/chroma-form (a real continuity-equation fluid solve would STOP for skill extension). New code: `noise.py` `curl_warp`/`curl_warp_ti`; `taichi_renderer.py` warp at `_disk_noise_m`/`_mod_fbm4` entry + `disk.noise.curl` dials through `_setup_disk_noise` (buffer grows past `_NOISE_N=43`); **no CKS-13 resolver change** (all base look dials). Gated by `disk.noise.curl.enabled` (default `false`, `amp=0` ⇒ identity) ⇒ bit-identical to V2. Guards: `tests/test_noise.py` (divergence-free, seamlessness, determinism) + `tests/test_noise_gpu.py` (`curl_warp_ti` CPU↔GPU twin parity to a derived `amp·_SATOL/fd_eps` bound — the warp is a derivative so the ~1e-5 `sfbm3` twin gap is amplified ×1/(2ε), obs ~6.5e-5) + `tests/test_disk_noise.py` (default-off wiring, curl-on finite) + unchanged `test_gpu_regression.py`. **GPU-verified 2026-06-14:** noise_gpu 15, disk_noise + gpu_regression pass, noise 44 (CPU). Spec: `docs/specs/2026-06-14-V3-curl-domain-warp.md`. |
| v1.25 | **Formula CKS-17 ADDED + WIRED — 3D inner-edge-ray self-shadow (radial + vertical), owner-approved 2026-06-14, V epoch vertical-self-shadow. NOT a physics revision — VISUALIZATION, same governance class as CKS-15.** Generalises the CKS-15 radial column scan to a 3D shadow ray from the inner edge **in the midplane** `(u=0, ζ=0)` to the sample `(u_s, φ, ζ_s)` at fixed φ, so an off-midplane parcel is shadowed by the dense midplane gas between it and the hot inner edge (the vertical self-shadow that V2's 3D bulk makes physical). Ray: `ζ(u)=(u/u_s)·ζ_s`, `Z(u)=r·ζ(u)·σ_θ(r)` (CKS-16 flared σ); bake accumulates `Σ_{j<i_u} absb_c·ρ_j·ds_j` with the **tilted** sample `ρ_j=ρ(u_j,φ,ζ_j)` and 3D arc length `ds_j=√((r_j·du)²+ΔZ_j²)`. **Exact CKS-15 reduction at ζ=0** (`ΔZ≡0 ⇒ ds=r·du`, `ρ` at the midplane) — the radial element keeps the `dr=r·du` convention precisely so the midplane limit is bit-exact; CKS-15 is the `ζ→0` limit, not a separate path. **Lookup + application UNCHANGED** (same `disk_shadow_tau` field, same `_sample_shadow_tau` trilinear lookup, same `emission *= exp(−strength·τ_s)` on emissivity only — κ/dτ untouched, composes with CKS-14). Only `bake_disk_shadow`'s ray geometry changed; same kernel signature, **no new config / field / flag** — still `disk.volumetric.self_shadow.enabled` (default `false` ⇒ no bake, golden frames bit-identical). Cost: `O(NU²·NPHI·NZ)` (each ζ_s tilts its own ray ⇒ no prefix sum), ~NU/2× the CKS-15 evals, parallelised over cells (owner chose the 3D ray knowing it is heavier). Straight CKS ray / single inner-edge illuminator / single-scatter / amplitude-only — never p_μ/u^μ/g/g⁴/f_PT/chroma-form; a physical transport (geodesic rays, multi-scatter) still STOPs for skill extension. Guards: `tests/test_disk_self_shadow.py` — flag-off bit-identity / outward-steepening dimming / noise-on contrast-rise carry over unchanged; `test_bake_matches_analytic_gaussian_column` re-derived to the 3D-ray line integral (the constant-ζ radial closed form was the CKS-15 model, superseded off-midplane); `test_gpu_regression.py` unchanged. Spec: `docs/specs/2026-06-14-V2-vertical-self-shadow.md`. |
| v1.24 | **Formula CKS-16 ADDED + WIRED — flared 3D disk scale height (owner-approved 2026-06-14, V epoch V2). NOT a physics revision — GEOMETRY/TEXTURE, flagged like CKS-12 §3.** The constant angular scale height becomes radius-flared `σ_θ(r) = σ0·(r/r_inner)^β` (`σ0 ≡ theta_half_width·vertical_sigma_frac`, the r_inner width): `β=0` ⇒ today's constant-H/r slab bit-for-bit (the kernel skips `ti.pow` at `flare_beta==0`), `β>0` thickens the disk OUTWARD (H/r grows with radius); the §3 `(1+h_amp·(n_h−½))` lumpy term multiplies on top, order preserved. Genuine 3D for free: the `ridged3`/`fbm3` stack already consumes `ζ=dz_ang/σ_eff`, so a real radius-varying thickness un-squashes it — no new noise primitive (V1.5 simplex stays unwired). Single source of truth: the flare lives in the shared `@ti.func _disk_density_cks` (gained `flare_beta`, `r_inner`), so the emission march AND the CKS-15 shadow bake inherit it. Two knock-on fixes: (A) the CKS-13 resolver derives a SEPARATE `theta_half_bound ≥ band_sigma·σ_θ(r_outer)` (default `band_sigma=3.0`) as the photon trace band so the flared outer envelope is not hard-clipped, leaving `theta_half_width` as the un-mutated σ0 anchor (⇒ idempotent); (B) the Pipe-B vertical step cap is unchanged — flare only thickens outward so the thinnest slab is still the inner edge σ0 (`sigma_z=r·σ0`), verified (not assumed) by the flared-slab convergence test. Resolver also adds `flare_beta` and rejects `β<0` / `band_sigma≤0`. Gated by `disk.volumetric.flare.enabled` (default `false` ⇒ `theta_half_bound==theta_half_width`, `flare_beta==0`, golden frames bit-identical); `enabled:true,β=0` is also bit-identical. Amplitude/geometry-only — no p_μ/u^μ/g/g⁴/f_PT/chroma-form touched. New code: `kerr_params.resolve_config` CKS-16 block; `taichi_renderer.py` (`_disk_density_cks`/`_disk_emit_cks`/`bake_disk_shadow`/`render_beauty_physics`/`render_beauty_frame` signatures); `scripts/thumb.py` CPU twin. Guards: `tests/test_disk_flare.py` (7 resolver/CPU + 2 GPU) + unchanged `test_gpu_regression.py` / `test_disk_step_convergence.py`. Spec: `docs/specs/2026-06-14-V2-flared-3d-density.md`. |
| v1.19 | **`disk.noise.dynamism` visualization dial ADDED (2026-06-13) — NOT a physics revision.** A non-physical gain on the CKS-12 §2 shear amount: `φ′_k = φ − dynamism·Ω(r)·(a_k·T)` in BOTH twins (`noise.noise_density_mult` reads `nz["dynamism"]`; GPU `_disk_noise_density_mult` reads param slot `_NI_DYNAMISM=31`, buffer grew 31→32). Motivation: in the first reset cycle the visible winding reduces to `Ω·t_disk` (T cancels), so per-frame swirl was only tunable via the *physical* `inner_lap_seconds` (which also speeds reseeding) — this dial emphasises the differential winding for a given frame without touching the reset cadence or C0-continuity (`w_k=0` at each reset regardless of gain). `dynamism=1.0` (and an omitted key) is **bit-identical** to v1.18 — guarded by `tests/test_noise.py::test_dynamism_unit_gain_is_bit_identical` + the unchanged advected agreement test; effect + GPU↔CPU agreement at gain≠1 by `test_dynamism_gain_emphasises_winding` (CPU) and `test_disk_noise.py::test_dynamism_gain_matches_cpu_and_changes_shear` (CUDA). Amplitude/φ-only, no GR/g/g⁴ touched. Same dial spirit as `disk.doppler_strength` (v1.12). |
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
| v1.15 | **CKS-12 D2.1 noise primitive library SHIPPED (2026-06-13; still NOT wired into the renderer — backlog D2.2+).** Doc-only change to the CKS-12 status block + file-locations: `src/renderer/noise.py` now exists as the CPU NumPy source of truth for the §3 primitives (PCG-hash, Perlin gradient noise, fBm, billow/turbulence, Musgrave ridged-MF, Worley F1/F2, voronoi-billow, cell-wall) plus their `@ti.func` twins (same file, `_ti` suffix). Held to the reference by `tests/test_noise.py` (16 CPU tests) + `tests/test_noise_gpu.py` (10 CUDA agreement tests, ~1e-6) — pins the CKS-12 hard constraints (integer φ-periodicity ⇒ no φ=0 seam, deterministic integer hashing / no `ti.random`, f32-exact CPU↔GPU). No GR formula touched; no renderer/golden-frame impact (module is standalone until D2.2). |
| v1.18 | **CKS-12 D2.3 — Keplerian shear advection WIRED (2026-06-13). NOT a physics revision.** The §2 dual-phase reset blend now advects the density-modulation field: `noise.noise_density_mult` gained `(t_disk, omega, shear_period)` and wraps the log-density `m`-stack twice — `φ′_k = φ − Ω(r)·(a_k·T)`, triangle weights `w_k=1−|2a_k−1|`, per-cycle integer reseed `seed + k·NCYC_PHASE + c_k·NCYC_CYCLE`, optional `variance_preserve` ÷√(Σw²). GPU twin `taichi_renderer._disk_noise_density_mult` wraps `_disk_noise_m` identically (held to CPU by `tests/test_disk_noise.py::test_advected_stack_matches_cpu_reference`, rtol 1e-3). `Ω` is **Formula 3 verbatim** (`1/(r^{3/2}+a)`), computed per disk sample in `_disk_emit_cks`; `t_disk = frame/fps·time_scale` threaded through `render_beauty_frame{,_mb}`, `export_exr.py`, and `thumb.py --frame/--t-disk`. **`shear_period ≤ 0` (no `disk.dynamics`) ⇒ the static D2.2 path, bit-identical** — so existing goldens and the GPU stack-agreement test are untouched (each phase's `w_k=0` exactly at its own reset ⇒ C0-continuous reseed). New CPU tests: `tests/test_noise.py §2` (5: static-fallback, evolution, determinism, reset-continuity, variance-preserve). Amplitude-only (density), so no GR/g/g⁴ touched. D2.4 (T/edge/height modulation) still pending. |
| v1.17 | **CKS-12 D2.2 — static density modulation WIRED (2026-06-13). NOT a physics revision.** The §3 layer stack now multiplies the disk Gaussian density in the GPU beauty path: `noise.noise_density_mult` (CPU source of truth, combined L0/L1/L2) + its GPU twin `taichi_renderer._disk_noise_density_mult` (held to the CPU by `tests/test_disk_noise.py`), fed by the `disk.noise` config block via the `_setup_disk_noise` param buffer (look-dev re-tunes by re-upload, no re-JIT). `thumb.py` uses the same CPU reference for look-dev. **Static only** (`t_disk = 0`, density only — no §2 shear advection, no T/edge/height; those are D2.3+). `disk.noise.enabled: false` verified bit-identical (constraint 6): the re-anchored GPU regression (v1.16) still passes unchanged with the noise code present. Amplitude-only, so no GR formula, g-factor, or g⁴ bookkeeping is touched. |
| v1.20 | **CKS-12 D2.4 — §3 temperature / inner+outer edge / scale-height modulation WIRED + noise enabled globally (2026-06-13). NOT a physics revision.** The disk emission amplitudes are now modulated by four advected [0,1] fBm envelopes (`n_T, n_e_in, n_e_out, n_h`), co-moving via the SAME §2 dual-phase reset blend + `dynamism` gain as the density field: CPU source of truth `noise.noise_modulation_fields` + GPU twin `taichi_renderer._disk_noise_mod_fields` (vec4; held to CPU by `tests/test_disk_noise.py::test_mod_fields_match_cpu_reference`, rtol 1e-3). Applied per the CKS-12 §3 constraints: `T_emit ← T_emit·(1+τ_amp·(n_T−½))` **before** the CKS-9 g-shift (constraint 2 — preserves g⁴-not-g⁸); `r_in_eff = max(r_inner·(1+e_in·(n_e_in−½)), r_isco)` (constraint 3, zero-torque BC floor); `r_out_eff = r_outer·(1+e_out·(n_e_out−½))`; hard radial cutoffs replaced by `_smoothstep_ti` edge windows; `σ_θ ← σ_θ·(1+h_amp·(n_h−½))` lumpy scale height, with the Pipe-B vertical step cap sized on the **worst-case** `σ_z·(1−h_amp/2)` (constraint 4) — guarded against returning face-on moiré by `tests/test_disk_step_convergence.py::test_disk_emission_resolves_lumpy_slab` (≤0.06 rel divergence). To avoid a σ→σ circular dependency the noise/mod fields sample at the UNMODULATED σ, then the Gaussian is re-evaluated at σ_m. The four envelopes are single fBm in [0,1] decorrelated by distinct seed offsets (`NSEED_MOD_T/EIN/EOUT/H = 503/601/701/809`) and carry **NO** variance-preserve divide (convex triangle weights `w_0+w_1≡1` keep a [0,1] fBm in range). `_setup_disk_noise` param buffer grew 32→43 (`_NI_MOD_*` slots 32-42); `_disk_emit_cks` + `render_beauty_physics` gained an `r_isco` arg (CKS-13-derived) and the trace band widened to `[r_isco, r_outer·(1+½·e_out)+soft]` when modulation is on. **Applied globally:** shipped `configs/render.yaml` now has `disk.noise.enabled: true` + `disk.noise.modulation.enabled: true`; because that would shift the pinned goldens, the GR/calibration guards (`test_gpu_regression.py`, the base smooth-slab `test_disk_step_convergence`) force `disk.noise.enabled=False` so they stay pure physics guards (noise is art, not the GR check). `enabled:false` / `modulation.enabled:false` remain bit-identical to the D2.3 density-only path (constraint 6). New CPU tests `tests/test_noise.py §3` (disabled-is-½, unit-range, decorrelation, advect+determinism). Amplitude-only — no p_μ/u^μ/g/g⁴/chroma-form/f_PT touched; the only physics input is Ω (Formula 3 verbatim). Completes the D2 turbulence backlog. |
| v1.16 | **GPU regression goldens RE-ANCHORED + made dynamic in `doppler_strength` (2026-06-13) — NOT a physics revision.** `tests/test_gpu_regression.py`: the Doppler / disk-peak guards no longer pin a single s=1.0 band (silently invalidated by the v1.14 CKS-13 peak-temperature re-keying — see that row's correction). They now render frame 0 at forced s ∈ {0, 0.5, 1.0} (simple model, disk-only `disk_buf` metrics) and assert the g_eff=g^s beaming RESPONSE: near-symmetric at s=0 (< 1.5), monotone non-decreasing in s, and matching the re-measured s=1.0 goldens (Doppler 5.15× ±10%, disk peak 14.45 ±8%). `test_page_thorne_disk_model_renders` forces s=1.0 so the YAML's s=0.1 can't suppress the > 2× beaming check. No GR formula, kernel, or config touched — test-only re-anchor against the existing render. |
| v1.14 | **Formula CKS-13 ADDED + WIRED — derived-parameter config resolver (owner-approved 2026-06-13).** `src/renderer/kerr_params.resolve_config` (called by every config loader: `taichi_renderer.load_config`, `scripts/thumb.py`) derives all spin/extent-dependent parameters at load: `r_plus` (CKS-6), `r_isco` (Formula 2), `disk.r_inner` (`auto` → r_isco; numeric override clamped ≥ r_isco), `disk.T_0` from the new base `disk.target_peak_temperature` (page_thorne: T_0=T_peak, LUT max-normalized; simple: T_0=T_peak·(r_inner/6)^¾), and the `disk.dynamics` time mapping (`T_orb=2π(r^{3/2}+a)` Formula-3 inverse; `t_wrap=2π/ΔΩ`; `time_scale=T_orb(r_in)/inner_lap_seconds`; `shear_period_M=budget·t_wrap` — the CKS-12 §2 reset period). No new physics: every line is a pinned formula or its trivial inverse. The YAML `r_isco`/`r_plus`/`r_inner`/`T_0` literals were REMOVED (the desync failure mode is gone); literature anchors (a=0→6/2, a=1→1/1, a=0.999→1.182/1.0447) pinned in `tests/test_kerr_params.py` instead of an external LUT — BPT closed forms are exact, only CKS-11 f_PT needs tabulation. Render-path impact: r_inner 1.182→1.181765 (exact ISCO). ⚠️ **CORRECTION (2026-06-13):** the original claim here — "GPU regression metrics bit-identical except Doppler ratio Δ5e-6" — was **wrong**. Re-keying `T_0: 5500` (old simple-model *inner-reference* temperature, peak T_eff ≈ 18,600 K) to `target_peak_temperature: 5500` (peak T_eff = **5500 K**) shifted the blackbody chroma magnitude and therefore the disk emission peak (6.17→14.45) and the chroma-weighted half-frame Doppler ratio (4.32→5.15) at simple/s=1.0. This is an intended warm-peak look change, not a physics-formula error, but the `test_gpu_regression.py` goldens were never re-pinned — which is why the disk-peak / Doppler-band guards failed until v1.16. |
| v1.22 | **Formula CKS-15 ADDED + WIRED — radial deep-shadow-map self-shadow (owner-approved 2026-06-13, V epoch V1.2). NOT a physics revision — VISUALIZATION, flagged like `doppler_strength`.** A per-frame baked 3-D cumulative absorption optical depth `τ_shadow[NU,NPHI,NZ]` on the CKS-12 noise coords `(u=ln r/r_inner, φ, ζ=Δθ/σ_θ)`: each `(φ,ζ)` column integrates `Σ absb_c·ρ·(r·du)` OUTWARD from `r_inner` (`dr=r·du`), `ρ` the SHARED `_disk_density_cks` (so shadow ρ ≡ emission ρ, incl. §2 shear + §3 modulation at `t_disk`), each cell storing τ from STRICTLY-inner gas (no self-shadow within a cell), clamped to `max_tau`. Per primary sample the trilinear (φ-periodic) lookup dims the EMISSIVITY only: `emission *= exp(−strength·τ_s)` — `κ`/`dτ` untouched (gas still occludes); composes with CKS-14 so `S=emission/dτ` inherits `e^{−τ_s}` ⇒ shadowed thick parcels read DARK (the voids). **The glowing-gas-with-voids look needs CKS-14 + CKS-15 together.** Straight radial CKS shadow ray (not a geodesic), single inward illuminator, single-scatter — occlusion bookkeeping, not a transport solve; multiplies amplitude only (never p_μ/u^μ/g/g⁴/f_PT/chroma-form — CKS-12 constraint 1). Gated by `disk.volumetric.self_shadow.enabled` (default `false` ⇒ no bake, no lookup, golden frames bit-identical). New code in `taichi_renderer.py`: `disk_shadow_tau` field + `_setup_disk_shadow` (always allocates from the grid config; bakes `u_max=ln(r_outer/r_inner)`, `ζ_max` into module globals so the lookup needs no extra kernel args), `@ti.kernel bake_disk_shadow`, `@ti.func _sample_shadow_tau`, `self_shadow`/`shadow_strength` kernel args threaded through `render_beauty_physics` + `render_beauty_frame`. Guards: `tests/test_disk_self_shadow.py` (flag-off bit-identity; GPU bake vs the analytic Gaussian column ρ=exp(−½ζ²) at rtol 2e-4; outward-steepening dimming; noise-on disk-contrast rise) + unchanged `test_gpu_regression.py`. Spec: `docs/specs/2026-06-13-V1-self-shadow-source-function.md`. Completes the V1 self-shadow + source-function pair (V1.3 showcase flags / V1.4 PROJECT+golden / V1.5 Simplex follow). |
| v1.23 | **CKS-12 §3.6 isotropic simplex basis SHIPPED (2026-06-14, V epoch V1.5; NOT wired into the renderer — library addition only). NOT a physics revision — texturing, not GR.** `src/renderer/noise.py` gained the Perlin/Gustavson skewed-simplex basis: `snoise2`/`snoise3` (CPU NumPy source of truth) + `sfbm2`/`sfbm3` (reusing the shared `_octaves` machinery) + their `@ti.func` twins (`snoise2_ti`/`snoise3_ti`/`sfbm2_ti`/`sfbm3_ti`), reusing this file's PCG corner hash and the Perlin-2002 12-gradient `_grad3` (which already returns grad·d); the radial kernel is `(r₀²−|d|²)₊⁴·grad·d` (r₀²=0.5/0.6, 70/32 normalizers), float32, no transcendentals on the lattice path. **Motivation:** the square-lattice `gnoise*` basis leaks a faint axis-aligned grid bias (a 4-fold-symmetric power spectrum); the hexagonal simplex lattice does not — `tests/test_noise.py::test_simplex_more_isotropic_than_perlin` measures the m=4 angular anisotropy and finds simplex ~12× smaller (Perlin ≈0.52, simplex ≈0.04). **Scope (volumetric spec §1a / V3 step 7, decision D-V4 → "add Simplex"):** this is the basis for the V3 **curl-flow potential**, NOT a drop-in for the φ-periodic disk density stack — classic simplex is **not** lattice-periodic (the input skew couples the axes ⇒ a 2π φ-period is not a lattice period; CKS-12 constraint 5), so it carries no φ-periodicity guard and is not wired into any render path. **Every golden frame is therefore bit-identical** (pure library addition, exactly as the D2.1 primitives preceded D2.2). Tests: `tests/test_noise.py` (8 CPU: range, determinism, seed-sensitivity, fBm-single-octave≡base, C2-continuity, the isotropy guard) + `tests/test_noise_gpu.py` (4 CUDA twin-parity/determinism, atol 1e-5). No GR formula, g-factor, or g⁴ bookkeeping touched. |
| v1.21 | **Formula CKS-14 ADDED + WIRED — volumetric RTE source-function march (owner-approved 2026-06-13, V epoch V1.1). NOT a physics revision — no new GR.** The Pipe-B disk march can now integrate `dI=(S−I)dτ` with the source function `S = j/κ = emission/dτ = (emis_c/absb_c)·[f_PT]·g_eff⁴·chroma` reconstructed from the values `_disk_emit_cks` already returns (ρ and ds cancel exactly). Update: `w=1−e^{−dτ}; disk_col += transm·w·S; transm *= e^{−dτ}`. Reduces to the legacy `disk_col += transm·emission` (Formula 9) in the optically-thin limit (`w→dτ`, differs only at O(dτ²)). **Same continuum integral as legacy** (`∫S e^{−τ}dτ`); CKS-14 is its exact per-step quadrature, so in the thick regime it is *dimmer & more accurate* (legacy left-endpoint rectangle over-counts opaque steps by `dτ/(1−e^{−dτ})`) — NOT a brightness boost. Standalone value: removes that over-count and **materialises `S`** for CKS-15 self-shadow (`S·e^{−τ_shadow}`); the void look needs CKS-14+CKS-15 together. Gated by `disk.volumetric.source_function` (default `false` ⇒ legacy branch, golden frames bit-identical); falls back to the legacy term when `dτ ≤ _RTE_TAU_EPS≈1e-6` (divide guard, no discontinuity). **g-bookkeeping unchanged:** `S` carries `g_eff⁴·chroma` exactly once (no g⁸). March-loop reinterpretation in `render_beauty_physics` only — `_disk_emit_cks` untouched (still returns `vec4(emission, dτ)`); the V1.0 prerequisite extracted the density stack into the shared `@ti.func _disk_density_cks` (bit-identical, golden-guarded). Spec: `docs/specs/2026-06-13-V1-self-shadow-source-function.md`. CKS-15 (radial deep-shadow self-shadow) is the V1.2 follow-up. |

*Last verified: 2026-06-06 (F13 guard (b′) Layer-A splat-placement rule approved +
landed; (b′) is a placement regularization derived from the already-verified guard-(a)
undeflected footprint, not a new physics formula — F13 μ/PSF still match
`REFERENCE_dngr_paper.md` A.2/A.3.1/A.7). Do not update formulas without re-verifying
against primary sources listed in each section.*