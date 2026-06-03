# kerr-physics

## When to use this skill

Load this skill whenever the task involves:
- Kerr black hole geodesic integration
- Accretion disk gas velocity or emission
- Observer/camera setup in curved spacetime
- Photon momentum initialization (tetrad)
- Doppler beaming, redshift, or g-factor computation
- Anti-aliasing / mipmap LOD for the starmap
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

**Logged 2026-06-04 after comparing the implementation against `pdf.md` (DNGR,
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
horizon stability (guid.md Phase 1.2/1.3). **Not new physics** — a factoring that
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
North-pole FP32 blowout fix (guid.md Phase 1.3). **This is a coordinate
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
tests/test_geodesic.py           ← Conservation tests (Formula 6 conserved quantities)
configs/render.yaml              ← a, r_isco, WIDTH, HEIGHT, step counts
```

---

## Revision history

| Version | Change |
|---|---|
| v1.0 | Initial release |
| v1.1 | **F6:** Corrected Carter constant to null geodesic form (−a²E², not a²(1−E²)). **F7:** Corrected lapse α to exact form using A = (r²+a²)²−a²Δsin²θ. **F9:** Documented that blackbody_rgb returns chromaticity only; clarified g⁴ is not double-counted, but will be if a physical Planck spectrum is substituted. |
| v1.2 | **F6:** Removed the leftover massive-particle `μ²r²` term from the radial potential `R(r)`; the null (μ=0) form drops it. The previous form gave `g^{μν}p_μp_ν = −r²/Σ`, breaking the null-condition conservation test. |
| v1.3 | **F10:** Added 2π normalization to the LOD formula — φ spans 2π radians across the 16384-texel starmap width, so dividing by 2π correctly maps the angular footprint to a texel footprint. Also switched to raw per-pixel exit deltas (δθ, δφ) rather than dividing by δu=1/WIDTH. The missing factor caused LOD to saturate at max mip for all background pixels, collapsing the LOD-on render to near-black. |
| v1.4 | **F10:** Added the screen-space Jacobian amendment (eliminate the offset ray; difference exit directions of neighbor pixels in a second shading kernel; same J/L; captured-neighbor ⇒ max_lod). **F11 (new):** FP32-stable factored discriminant Δ = y(y+2k). **F12 (new):** singularity-free polar potential Θ_u(u) for the u=cosθ state transform, with the `v_r=Δ·p_r → v_y=Δ·p_r` invariant migration, `p_θ=−v_u/√(1−u²)` recovery, and the approved polar guard on dφ/dt only. All three approved by the project owner 2026-06-02 for the guid.md optimization. |

*Last verified: 2026-05. Do not update formulas without re-verifying against
primary sources listed in each section.*