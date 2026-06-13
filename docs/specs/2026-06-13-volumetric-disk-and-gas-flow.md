# Volumetric Self-Shadowing Disk & Curl-Flow Gas — Design Plan

**Status:** PLAN ONLY — no code yet (owner asked to plan first, 2026-06-13).
**Goal:** reach the Interstellar *close-up* accretion-gas look (`whatiwant1.png`,
`whatiwant2.png`) — glowing turbulent gas with **deep black voids** and real 3D
bulk — which the current renderer cannot produce by any config setting.
**Companion to:** `2026-06-13-disk-noise-turbulence.md` (the D2 noise stack this
builds on). This is the next epoch; tag it **V (volumetric)** to separate it from
the D-series.

---

## 0. Why config alone fails (empirically demonstrated 2026-06-13)

The Pipe-B march (`taichi_renderer._disk_emission_sample` + the integrator loop) is:

```
color += T · emission ;   T *= exp(-dτ)
emission = emis_c · density · f_PT · g⁴ · chroma · ds      (no source function)
dτ       = absb_c · density · ds
```

This is **pure emission + extinction**. Consequences, both observed in renders:

| Regime tried | Result | Cause |
|---|---|---|
| absorption ↑ (5.0), thick slab | near-black (max ≈ 6e-4) | `T` collapses before the bright inner gas contributes → we see only the cold outer skin |
| absorption ↓, thick, noise ×2.8 | smooth bright haze | emission only **adds** along the ray → no black gaps; noise averages out over the line of sight |
| camera dropped onto the disk | black sky + stars | a Gaussian equatorial **slab** has no 3D bulk to be inside of |

The defining feature of both reference shots — **bright gas with black voids punched
through it** — is a *self-shadowing* effect (dense gas shadows other gas). Our single
forward march computes no shadow term, so the look is unreachable by construction, not
by tuning.

---

## 1. Gap analysis

### 1a. Procedural texture / gas flow (owner's specific question: fBm, Perlin/Simplex, Curl/Flow)

| Technique | Interstellar/DNGR use | Us today | Verdict |
|---|---|---|---|
| **fBm** | base multi-octave structure | ✅ `fbm2/3` | HAVE |
| **Perlin gradient** | lattice basis | ✅ `gnoise2/3` (Perlin 2002, φ-periodic) | HAVE |
| **Simplex** | cheaper, fewer axis artifacts in ≥3D | ❌ classic Perlin only | LACK (minor — Perlin adequate; Simplex is an optimization/quality upgrade, not a blocker) |
| **Billow / ridged / Worley** | cloud cusps, filament ridges, cellular tears | ✅ `billow*`, `ridged*`, `worley*`, `voronoi_billow*`, `cell_wall*` | HAVE (strong set) |
| **Curl noise (divergence-free flow)** | **the** gas-flow tool — advect texture along an incompressible ∇×ψ field so filaments swirl/eddy like fluid | ❌ only **rigid Keplerian shear** `φ' = φ − Ω(r)·t` (CKS-12 §2) | **LACK — critical gap** |
| **Flow noise / domain warp** | warp the noise lookup by a (curl) field → wispy stretched filaments | ❌ only the φ-shear special case | LACK (follows from curl) |
| **3D coherent vertical structure** | full volumetric density | 🟡 2.5D (`z` only in the clump layer) | PARTIAL |

**Headline:** our "motion" is *rigid differential rotation*. Interstellar's gas
*flows* — and the flow is **curl noise**: a velocity field `v = ∇×ψ` (ψ = vector of
fBm potentials) that is divergence-free (∇·v = 0, no sources/sinks), so advecting the
density by it produces the curling, vortical, incompressible filaments. We have
zero curl machinery today.

### 1b. Volumetric rendering (what actually makes the black voids)

| Capability | Needed for | Us today | Verdict |
|---|---|---|---|
| **3D volumetric density** (thick, optionally flared torus) | the cloud *bulk* & parallax depth | thin Gaussian slab (`σ_θ`) | LACK |
| **Self-shadowing** (optical depth from each sample toward the bright inner emitter) | **the black voids** | none | **LACK — critical gap** |
| **Emissive source function** `S = j/κ` so thick gas → bright surface (not black) | usable high-opacity regime | emission+extinction only | LACK |
| **Free 6-DOF camera** | fly-through / skim framing | showcase cam only looks at origin (Phase-1 Blender has full poses) | PARTIAL |

---

## 2. Mapping to Interstellar's three models

| Their model | Shot | Our status after this plan |
|---|---|---|
| Thin disk | wide Gargantua (`wantblackhole.png`) | ✅ **already shipped** (`showcase_gargantua_4k.png`) — lensing halo + photon ring |
| Volumetric disk | mid clouds | **V1+V2** (self-shadow march + 3D density) |
| Close-up procedural | `whatiwant1/2` | **V1+V2+V3** (above + curl-flow texture) |

---

## 3. Phased plan

Ordering = highest-value-first and by dependency. Each phase ships behind a flag;
the thin-slab default and all D2 goldens stay bit-identical.

### V1 — Volumetric self-shadowing + source function  *(the voids; highest value)*
The lighting change that turns flat haze into gas-with-holes.

1. **Source-function march.** Replace `color += T·emission` with the
   radiative-transfer form `dI = (S − I)·κ·density·ds`, `S = (emis_c/absb_c)·f_PT·g⁴·chroma`.
   - Optically thin → reduces to today's emission-only sum (back-compat).
   - Optically thick → `I → S` = a bright emitting **surface** instead of black.
   - New SKILL.md formula **CKS-14** (volumetric RTE source function along a CKS geodesic;
     uses the existing CKS-9 `g`, CKS-11 `f_PT`, Formula-9 chroma — **no new GR derivation**).
2. **Self-shadow term.** Attenuate each sample's emission by the optical depth between
   it and the bright inner disk. Three implementation options (decision D-V1 below):
   - (a) **Deep-shadow map** *(recommended)*: once per frame bake a coarse cumulative-τ
     field (e.g. cylindrical `(r, φ, z)` grid) integrated radially inward from the inner
     edge; each primary sample does one cheap texture lookup → `emission *= exp(−τ_shadow)`.
     Best quality/perf; O(N) primary + O(grid) bake.
   - (b) **Short shadow ray** per sample toward the inner edge (straight, local-frame):
     truest look, O(N·M), expensive; closeup-only.
   - (c) **Gradient AO**: cheap `exp(−k·∇density·inward)` fake occlusion; weakest, cheapest.
   - The shadow path is a **non-geodesic viz approximation** (gravitational bending over
     the short shadow span is small at closeup scale). Document & gate exactly like
     `doppler_strength` / `dynamism`; new SKILL.md **CKS-15** marked *visualization
     approximation, not a metric*. Never touches the primary geodesic / `g` / `g⁴` / `f_PT`.

### V2 — 3D volumetric density (thick, flared disk)
3. Generalize the slab: density = `vertical_envelope(z; σ(r)) · noise3D_mult`, with a
   radius-flared scale height `σ(r) = σ0·(r/r_in)^β` and a real 3D noise multiplier
   (full `fbm3`/`ridged3`/`worley3` with genuine `z` variation, not the 2.5D clump-only z).
   Thin-disk = the `β→0`, small-`σ0` limit (golden-frame default preserved behind a flag).
4. Widen the Pipe-B slab early-out + the `max_step_vfrac` cap to the new vertical extent
   (it already keys off `theta_half_width`·`vertical_sigma_frac`; extend to `σ(r)`).

### V3 — Curl-noise flow advection  *(the gas-flow look; owner's ask)*
5. **Curl noise primitive** in `noise.py` (CPU source of truth + `@ti.func` twin, same
   integer-hash determinism, no `ti.random`):
   - 2D streamfunction `ψ` (fBm) → `v = (∂ψ/∂y, −∂ψ/∂x)` via analytic/finite-difference
     gradient; 3D → `v = ∇×(ψx, ψy, ψz)`. Divergence-free by construction.
6. **Domain warp** the density/texture lookup: `p_warp = p + ε·v(p)` (optional 2nd-order),
   layered **on top of** the existing CKS-12 §2 Keplerian shear (bulk rotation) — so the
   gas = global rotation × local incompressible turbulence. This is a spec/texture
   extension to CKS-12 (a domain-warp on the sample coords), **not** new GR.
7. ✅ **Simplex** basis for the curl potential to cut grid artifacts — **SHIPPED V1.5
   (2026-06-14)** as `noise.snoise2/3` + `sfbm2/3` (+ GPU twins), decision D-V4 resolved
   "add Simplex." Library basis only (NOT yet wired): classic simplex is not φ-periodic,
   so the V3 curl integration is where it meets the disk (cylinder embedding of the
   periodic axis). Measured ~12× lower 4-fold (axis-aligned) anisotropy than the Perlin
   `gnoise` basis (`test_noise.py::test_simplex_more_isotropic_than_perlin`).

### V4 — Camera + look-dev + presets
8. Free showcase camera (position + look-at, up) replacing the look-at-origin-only orbit
   cam; presets for the three shot types. Soft glow stays the **Phase-3 Blender** job
   (the `--bloom` in `showcase_disk.py` is the standalone stand-in).

### V5 — Perf, tests, docs, sign-off
9. **LOD:** full self-shadow + 3D march is closeup-only; wide/mid shots keep the cheap
   thin-slab path (gate on camera distance / a quality flag). Step budgets; the bake grid
   resolution is a config knob.
10. **Tests:** CPU/GPU twin parity for curl noise & the source-function march; a
    self-shadow regression (a known dense clump must cast a measurable dark lane); thin-slab
    + `enabled:false` bit-identity vs current goldens; new volumetric golden frame.
11. **Docs sync (mandatory, same task as code):** SKILL.md CKS-14/CKS-15 + revision row;
    PROJECT.md §6/§7/§10 + roadmap rows; this spec's checkboxes; memory.

---

## 4. Governance / constraints (carry the CKS-12 discipline forward)

- **No GR re-derivation.** CKS-14 (RTE source function) is assembled from existing
  CKS-9/CKS-11/Formula-9; CKS-15 (self-shadow) is an explicit *visualization
  approximation*, flagged like `doppler_strength`. If anything needs a genuinely new GR
  result, STOP and ask the human to extend SKILL.md first (CLAUDE.md policy).
- **Curl/flow noise is texturing, not physics** → lives in `noise.py` + this spec, same as
  the rest of CKS-12's primitives.
- **Determinism:** curl noise uses the same PCG integer-hash lattice (no `ti.random`),
  CPU source of truth + GPU twin held to ~1e-6, φ-periodicity preserved.
- **Goldens:** every new path behind a flag; thin-slab default + `enabled:false` stay
  bit-identical; new features get their own golden frames.
- **Config-driven:** all new knobs (`disk.volumetric.*`, `disk.flow.*`, shadow-grid res,
  flare β) in `render.yaml`; derived values via `kerr_params.resolve_config` (CKS-13).
- **Perf honesty:** self-shadow multiplies cost; LOD + deep-shadow-map bake keep wide/mid
  shots at current speed. Closeups are expected to be slower — budget it explicitly.

---

## 5. Open decisions (need owner input before V1 code)

- **D-V1 — self-shadow method:** (a) deep-shadow-map *(recommended: best perf/quality)*,
  (b) per-sample shadow ray *(truest, slowest)*, (c) gradient AO *(cheapest, weakest)*.
- **D-V2 — scope/sequencing:** all of V1–V3 for a full closeup, or V1 first (prove the
  voids on the existing slab) then V2/V3?
- **D-V3 — glow ownership:** soft halo via Phase-3 Blender compositor (production path), or
  also keep the in-renderer FFT bloom stand-in for stills?
- **D-V4 — Simplex:** ✅ RESOLVED "add it" — shipped as the §3.6 `snoise*` basis in V1.5
  (2026-06-14); it will replace Perlin for the V3 curl potential (~12× less axis-aligned
  grid bias, measured). Perlin stays the basis for the φ-periodic disk density stack.

---

## 6. Out of scope (explicitly not this plan)
- Magnetohydrodynamic / physically-evolved gas (we advect *texture*, not solve fluid).
- Multiple scattering beyond a single self-shadow term (possible V6 if the look needs it).
- Changing the wide-Gargantua thin-disk path (already shipped & good).
