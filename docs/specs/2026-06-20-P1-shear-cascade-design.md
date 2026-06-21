# P1 — Scale-dependent shear cascade (CKS-21, VISUALIZATION)

**Status:** RATIFIED (owner, 2026-06-20) — design approved, all open decisions resolved
with the recommended options (§F), and the owner authorized promoting **CKS-21** from
"reserved" to a full SKILL.md formula. Next step: the implementation plan
(`docs/specs/2026-06-20-P1-shear-cascade-plan.md`). **No kernel code until Task 1 writes
CKS-21 into SKILL.md and it is owner-approved** (governance §B).

This is the last of the five `tip.md` pillars (the two PHYSICS headliners **P2 multi-phase
(CKS-19)** and **P3 single-scatter (CKS-20)** and the two refinements **P4 KH erosion
(CKS-22)** / **P5 LOD cascade (CKS-23)** are shipped). P1 is the one *genuinely new* idea in
`tip.md` Pillar 1 — it is **NOT** the already-shipped CKS-12 §2 dual-phase shear reset or the
CKS-18 §2 curl-flow boil (both long landed + GPU-verified), with which it is easily
conflated. P1 is a **frequency-dependent shear transfer** that protects high-frequency
octaves from winding into the same spiral as the coarse structure (Kolmogorov-like). Frames
the design per `docs/specs/2026-06-16-cinematic-volumetric-multiphase-scattering-design.md`
Part C.2.

---

# PART A — Goal & acceptance

## A.1 The problem (mapped to live code)

CKS-12 §2 applies the Keplerian shear `φ′_k = φ − Ω(r)·a_k·T` **uniformly to the entire
fBm**: `noise._advected_m` computes ONE `φ′_k` per reset phase, bakes `y = (φ′_k/2π)·f_base`
**once**, and the `_octaves`/`fbm2`/`ridged3` loop merely scales that baked `y` by `freq` per
octave (linear-Perlin `gnoise`, integer-period wrap — no cylinder embedding; that is only for
the simplex KH/curl layers). So every octave winds at the same azimuthal rate. Over a long
reset window `T` the *fine* detail laminarizes into the *same*
concentric spirals as the coarse structure — the "infinite winding" / laminarization of
`tip.md` §1.1. The dual-phase reset (CKS-12 §2) bounds *how long* any pattern shears before
it reseeds; it does **not** decouple the shear *rate across scales*. That decoupling is P1.

## A.2 The paradigm shift (tip.md §1.2)

Physical disks (MRI + hydrodynamic turbulence, Kolmogorov cascade) inject localized eddies
that resist bulk laminarization. We approximate this with a **scale-dependent shear
transfer**: large-scale octaves are fully sheared into filaments; high-frequency octaves are
progressively *protected*, so micro-vortices regenerate before the bulk flow smears them.
This is a **texturing approximation**, not a fluid/MHD continuity solve (roadmap invariant 6).

## A.3 Defining acceptance

**Calibration lesson (from P5/CKS-23, memory `project_p4p5_edge_lod`):** do NOT assert an
*emergent render-level* spectral property. At a fixed radius the §2 shear is a pure
φ-translation, which leaves the φ-power-spectrum invariant — laminarization is a **radial**
effect (differential winding `dΩ/dr`), so a naive φ-spectrum test cannot distinguish ON from
OFF and would mis-calibrate. Prove the **cascade math directly at the primitive level**, and
let the render guard assert only sound facts (OFF bit-identical; ON re-textures).

- **Primitive proof (the real acceptance):** for a single-octave fBm of base frequency
  `f_base`, the sheared evaluation equals the unsheared fBm sampled at φ displaced by exactly
  `(1 − S(f_base))·shear_k` — i.e. octave 0 (low `f`) is displaced ~fully, a high-`f` octave
  ~not at all. A 2-octave probe shows octave 1's displacement is strictly **smaller** than
  octave 0's (the differential). Deterministic, exact (§G).
- **Render guard (ON re-textures, OFF bit-identical):** at a long `T`, the cascade-ON disk
  field **differs** from OFF (it re-textures); with the cascade OFF (or `shear_cutoff → ∞`)
  the frame is **bit-identical** to the current uniform-shear golden (constraint 6).

---

# PART B — Governance & guard rails

**Class: VISUALIZATION** — identical class to CKS-12 §2 and CKS-18. It warps the noise
**coordinate only** (a per-octave azimuthal offset). It may **not** touch `p_μ`, `u^μ`
(CKS-8), `g` (CKS-9), the `g⁴` exponent (Formula 9), the `f_PT` radial shape (CKS-11), or the
blackbody chroma form — the CKS-12 §3 hard-constraint list applies verbatim. Amplitude/
coordinate quantities only.

| Pillar | Touches j_ν / κ_ν / RTE / extinction? | Class | SKILL artifact |
|---|---|---|---|
| **P1** shear cascade | No — per-octave noise-coordinate shear | **VISUALIZATION** | **CKS-21** (promote from reserved) |

Per the roadmap split, CKS-21 is promoted from "reserved" to a full SKILL.md entry with the
math of record below, **owner-approved before any kernel code** (governance flow §G), exactly
as CKS-18 was before V3.0. No re-derivation in code (CLAUDE.md CRITICAL RULE); the CPU twin is
the source of truth, the GPU twin is ported verbatim.

---

# PART C — The math of record (proposed CKS-21)

Amends CKS-12 §2. Per octave `o ∈ {0 … octaves−1}` of frequency `f_o`, scale the shear amount
by a **frequency transfer** `S(f_o)`. **Intuitive form** (no curl warp):

```
f_o      = f_base · lac^o                                  # octave o's true spatial frequency (per layer)
S(f)     = 1 / (1 + (f / f_c)^p)                           # transfer: low f → 1, high f → 0 (Butterworth-like)
φ′_{o,k} = φ − S(f_o) · shear_k ,   shear_k = dynamism · Ω(r) · (a_k · T)     # per-octave shear

n(u,φ,ζ;t) = Σ_k w_k · [ Σ_o gain^o · N( (φ′_{o,k}/2π)·f_o , u·f_u·lac^o , ζ·f_z·lac^o ; reseed_k ) ]
```

**Implementation form (the de-shear correction — composes with the CKS-18 curl warp and is
bit-identical when `S≡1`).** Because the curl warp `curl_φ` is nonlinear and is applied to the
**already-sheared** `φ_k` today (`_noise_m_stack` / `_disk_noise_m`), the order **must** stay
shear→curl to preserve the golden. So the cascade is expressed as a per-octave *add-back* of
the protected (un-sheared) fraction, applied **inside** the lifted octave loop **after** curl:

```
φ_k       = φ − shear_k                              # FULL §2 shear, before curl (UNCHANGED)
φ_c       = curl_φ(u, φ_k)                           # CKS-18 warp on the fully-sheared φ (UNCHANGED order)
φ′_{o,k}  = φ_c + (1 − S(f_o)) · shear_k              # per-octave de-shear correction (the cascade)
```

With no curl (`curl_φ(x)=x`) this reduces to the intuitive `φ′_{o,k} = φ − S(f_o)·shear_k`.
With `S(f_o) ≡ 1` (cascade off / `f_c → ∞`) the correction is `0` ⇒ `φ′_{o,k} = φ_c` for every
octave ⇒ the **current single-φ evaluation, bit-for-bit** (§C.1, §D).

The CKS-12 §2 dual-phase bookkeeping is **unchanged**: `s = t_disk/T`, `a_k = fract(s + k/2)`,
`c_k = floor(s + k/2)`, `w_k = 1 − |2a_k − 1|`, `w_0 + w_1 ≡ 1`, per-cycle reseed via `c_k`,
optional variance-preserve divide by `√(w_0² + w_1²)`.

- `Ω(r) = 1/(r^{3/2} + a)` (Formula 3, verbatim) and `dynamism` (CKS-12 §2 non-physical
  viz gain) are unchanged; the cascade only inserts the `S(f_o)` factor.
- **`f_o` per layer** — each density layer uses its **own** `f_base·lac^o` so the cutoff `f_c`
  is measured against the layer's real spatial frequency, not an octave index.
- **Density φ is linear-Perlin, not cylinder-embedded.** The density octave stacks
  (`fbm2`/`ridged3` on `gnoise2/3`) enter φ as `y = (φ/2π)·f_base` with integer-period lattice
  wrapping (constraint 5) — **no `cos/sin`**. The per-octave correction is a constant-in-φ
  additive offset, so seam-freeness is preserved (`y_o(2π) − y_o(0) = f_base·lac^o ∈ ℤ`) and
  **no trig is added** (the cylinder embedding is only used by simplex KH/curl, which v1 does
  not cascade).

### C.1 Limiting cases (the bit-identity hook)
- `f_c → ∞` (or `shear_cascade.enabled: false`) ⇒ `S(f_o) ≡ 1` ∀o ⇒ the de-shear correction
  `(1−S)·shear_k = 0` for every octave ⇒ the octave loop sums exactly the current single-φ
  evaluation **bit-for-bit** (§D), i.e. CKS-12 §2 uniform shear exactly.
- `T ≤ 0` (no `disk.dynamics`) ⇒ static field, sampled at `φ` directly (cascade is a no-op,
  same as today).

### C.2 C0-continuity at resets (preserved)
The reset crossfade weight `w_k → 0` at each phase reset **regardless of `S(f_o)`** (the
weights are independent of the shear amount). So the per-cycle reseed `c_k` step stays C0 —
the proven CKS-12 §2 / CKS-18 §2 property carries over unchanged. The cascade scales *how far*
each octave shears, never *when* it reseeds.

---

# PART D — The core architectural change (the hard part)

CKS-23 (LOD) **already lifted** the density octave loop into a per-octave-parameterized fBm
(`noise.fbm2_lod` / `taichi_renderer`'s `fbm2_lod_ti`, threading a per-octave `n_oct` gate).
CKS-21 threads a **second per-octave modifier** — the de-shear correction — through the SAME
lifted loop. So this is *not* a from-scratch loop lift; it is an extension of the existing LOD
primitive, plus the analogous lift for the L1 ridged stack.

Today the shear is "consumed" before the stack: `_advected_m`/`_disk_blended_m` form
`φ_k = φ − shear_k` and pass the already-sheared `φ_k` down; `_noise_m_stack`/`_disk_noise_m`
curl-warp it and bake `y = (φ_k/2π)·f_base`; `fbm2_lod` then scales `y·lac^o` per octave. The
shear amount `shear_k` is discarded. CKS-21 instead **passes `shear_k` (and the cascade dials)
down to the octave loop**, where each octave adds back `(1 − S(f_o))·shear_k` to its φ:

```
# in the lifted fBm octave loop (CPU fbm2_lod / GPU fbm2_lod_ti), per octave o:
f_o   = f_base · lac^o                       # f_base = the layer's freq_phi (the y-period at o=0)
S_o   = 1 / (1 + (f_o / f_c)^p)              # transfer; sentinel f_c ⇒ S_o ≡ 1
y_o   = ( y_base + (1 − S_o)·shear_k/(2π)·f_base ) · lac^o      # y_base = (φ_c/2π)·f_base (as today)
x_o   = x_base · lac^o
n    += g_o · gain^o · gnoise2( x_o , y_o , period = f_base·lac^o , seed+o )    # g_o = the CKS-23 gate
```

When `S_o ≡ 1` the add-back is `0`, `y_o = y_base·lac^o` exactly as today ⇒ **bit-identical**
(constraint 6). The added offset is **constant in φ** (`shear_k` depends on `r/u`, not φ), so
the integer-period seam (constraint 5) is preserved: `y_o(φ=2π) − y_o(φ=0) = f_base·lac^o ∈ ℤ`.

**Cost:** one `pow`/`(f_o/f_c)^p` + a few adds per octave (no trig). Modest; accepted for v1
(§F-5).

## D.1 Touch-list (no code yet)

1. **`noise.py` — extend the lifted fBm** (`fbm2_lod` → add `shear_k`, `f_c`, `p`; add the
   `(1−S_o)·shear_k` correction per octave) and add the analogous **`ridged3` shear lift** for
   the L1 ridged stack. A `_shear_transfer(f, f_c, p)` helper holds `S(f)`. CPU **source of
   truth**.
2. **`noise._noise_m_stack`** stops pre-baking the full shear into φ for the cascaded layers:
   it passes `φ_c` (curl-warped, fully-sheared — unchanged) **plus** `shear_k` into the fBm
   calls so the loop can add the protected fraction back. `_advected_m` already computes
   `shear_k = g·ω·a_k·T` per phase; it now forwards it instead of only applying it.
3. **Scope = density octave stacks: L0 (`fbm2`), L2 (`fbm2`), and L1's `ridged3`.** The L1
   **Voronoi** (single-frequency cellular — *no octaves to cascade*) and the L1 **coverage
   mask** (a structural 2-octave gate, not turbulence) keep the current uniform shear
   (`shear_k = 0` passed ⇒ no correction). Modulation envelopes (`n_T`, `n_e_in`, `n_e_out`,
   `n_h`) are **out of scope** (density-only, §F) — they keep uniform shear.
4. **`taichi_renderer.py` — GPU twins** ported **verbatim**: extend `fbm2_lod_ti` + the ridged
   loop with `(shear_k, shear_fc, shear_p)` **args** (these `@ti.func`s live in `noise.py` and
   cannot read `disk_noise_params`, so f_c/p are passed, not read); thread `shear_k` from
   `_disk_blended_m` → `_disk_noise_m` → the layer calls. **No `ti.static` `_SC_COMPILE` gate**
   — follow the **CKS-23 precedent**: a `shear_k = 0.0` sentinel **default** makes the
   correction exactly `0` (`(1−S)·0 = 0` in f32, `y+0 = y` exact), so the primitive is
   always-compiled and bit-for-bit when the production caller omits the args (the disabled
   path). The added cost is a `pow` per density octave, accepted like CKS-23's per-octave clamp.
5. **CKS-18 curl-warp ordering UNCHANGED** — curl still warps the already-sheared `φ_k` at the
   stack entry (`_disk_noise_m:1190` / `_noise_m_stack:713`); the per-octave add-back happens
   **after** curl, inside the loop. This keeps the shear→curl order, so divergence-free + the
   φ-seam are preserved and `S≡1` is bit-identical (§F-3).
6. **`_setup_disk_noise` param buffer** grows by `enabled`, `shear_cutoff (f_c)`,
   `shear_falloff (p)`; `_NOISE_N` bumps from 69 (post-CKS-23 — confirm at impl) with new
   `_NI_SC_*` index constants. **No CKS-13 resolver change** — all base dials.
7. **No GR / RTE field changes** — `φ′` is a coordinate; emission / `dτ⃗` forms
   (CKS-12 / CKS-19 / CKS-20) are structurally untouched.

---

# PART E — Config schema (additions to `disk.noise` in `render.yaml`)

```yaml
disk:
  noise:
    shear_cascade:           # CKS-21 — frequency-dependent shear transfer. Default OFF ⇒ bit-identical.
      enabled: false         # false (or shear_cutoff huge) ⇒ S≡1 ⇒ CKS-12 §2 uniform shear, bit-for-bit.
      shear_cutoff: 0.0      # f_c (cycles). 0 ⇒ auto = +inf sentinel ⇒ S≡1 (no protection). >0 ⇒ octaves
                             #   above f_c are progressively protected from the bulk shear.
      shear_falloff: 2.0     # p — transfer steepness; larger ⇒ sharper low/high split around f_c.
```

`enabled:false` (or an absent block, or `shear_cutoff` resolving to the +inf sentinel) ⇒
`S(f_o) ≡ 1` ⇒ zero golden-frame movement. Nested under `disk.noise` as a sibling of
`disk.noise.curl` (§F-4) because it is a noise-coordinate transform, not a separate subsystem.
All base look dials ⇒ no CKS-13 resolver change.

---

# PART F — Open owner decisions — RESOLVED (owner, 2026-06-20)

All resolved with the recommended option.

1. **Transfer form → `S(f) = 1/(1 + (f/f_c)^p)`** (F-1a). Butterworth-like: smooth monotone
   roll-off, low-f → 1, high-f → 0, two intuitive dials (`f_c` cutoff, `p` steepness).
2. **`f_o` → per-layer true spatial frequency** `f_base·lac^o` (F-2a). The cutoff is measured
   against each density layer's real frequency, not a shared octave index, so L0/L1/L2 with
   different base frequencies are protected consistently.
3. **Curl-warp order → UNCHANGED (shear→curl), cascade as a per-octave de-shear add-back**
   (F-3a). The curl warp keeps operating on the already-sheared `φ_k` exactly as today
   (`_disk_noise_m:1190` / `_noise_m_stack:713`); the cascade adds `(1−S(f_o))·shear_k` back
   per octave **after** curl (§C, §D). This preserves CKS-18 divergence-free + the φ-seam (the
   warp is untouched) and keeps `S≡1` bit-identical. *(Curl-on-base-then-shear would break
   bit-identity because curl is nonlinear — rejected.)*
4. **Config home → nest under `disk.noise.shear_cascade`** (F-4a), sibling of
   `disk.noise.curl` — it is a noise-coordinate transform, not a distinct subsystem.
5. **Perf → accept one `pow` per octave in v1; defer optimization** (F-5a). The density stacks
   are linear-Perlin (no trig); the cascade adds only the `(f_o/f_c)^p` transfer + adds per
   octave. Optimize only if the mega-kernel cold compile or warm render regresses materially.

---

# PART G — Tests (TDD)

- **`S≡1` collapse / bit-identity** (`tests/test_noise.py`): with `shear_k=0` or `f_c` at the
  sentinel, the extended `fbm2_lod` / `ridged3` equal the current primitives **byte-for-byte**
  (the §C.1/§D hook) — the default-arg guarantee for every existing caller.
- **Transfer helper** (`tests/test_noise.py`): `_shear_transfer(f, f_c, p)` is
  monotone-decreasing in `f`, `=1` at `f=0` and at the sentinel `f_c`, `→0` as `f≫f_c`.
- **Per-octave displacement (the primitive proof, `tests/test_noise.py`):** a 1-octave
  `fbm2_lod(…, shear_k=Δ, f_c, p)` equals `fbm2_lod(…)` evaluated at `y` displaced by
  `(1−S(f_base))·Δ·(1/2π)·period`; a 2-octave probe shows octave 1's displacement strictly
  smaller than octave 0's (the §A.3 differential). Deterministic, exact.
- **CPU/GPU twin parity** (`tests/test_noise_gpu.py`): the extended `fbm2_lod_ti` / `ridged3_ti`
  and the routed density field (`_disk_blended_m`) agree with the CPU within `_SATOL`.
- **C0-at-reset** (`tests/test_disk_noise.py`): the blended density field is continuous across
  a reset boundary with the cascade ON (the CKS-12 §2 / CKS-18 §2 guard, now per-octave).
- **Bit-identity regression** (`tests/test_gpu_regression.py`): `shear_cascade.enabled:false`
  ⇒ goldens **bit-identical** (the key constraint-6 guard; backed by the `shear_k=0` sentinel
  default ⇒ `(1−S)·0 = 0` exact, the CKS-23 always-compiled precedent — no `_SC_COMPILE` gate).
- **Render ON re-textures** (new `tests/test_disk_shear_cascade.py`): at a long `T`, the
  cascade-ON disk field differs measurably from OFF (it re-textures); no emergent-spectrum
  assertion (§A.3 calibration lesson).
- **New golden:** one long-`T` beauty frame with the cascade ON (filaments + intact fine detail).

---

# PART H — Phasing & docs-sync

Mirrors every prior increment; expanded into a TDD task breakdown by the `writing-plans` step
(`docs/specs/2026-06-20-P1-shear-cascade-plan.md`).

1. **Promote CKS-21 to a full SKILL.md entry** with the §C math of record + the §F resolved
   decisions; bump SKILL.md version + revision history + "When to use" list;
   **owner review before kernel code** (governance §B).
2. **CPU twin first (TDD):** the sheared-fBm primitive in `noise.py` + the `S≡1` collapse and
   monotone-transfer parity tests, before touching the GPU.
3. **Route the density stack** (`_advected_m` / `noise_density_mult`) through it; density-only.
4. **GPU twin:** port verbatim into `taichi_renderer.py`'s unrolled density octave loop;
   param buffer / `_NOISE_N` growth; curl-warp ordering (§F-3).
5. **Config + (no) resolver:** add `disk.noise.shear_cascade`; confirm no CKS-13 change.
6. **Goldens:** the bit-identity regression (OFF path) + one cascade-ON beauty frame.
7. **Docs-sync (same task):** SKILL.md (CKS-21) + PROJECT.md §6/§7 updated with the landed
   change (Docs Sync Policy).

## H.1 Risk / effort / payoff

| Effort | Risk | Payoff | Principal risk |
|---|---|---|---|
| Med | Low–Med | Med | Threading `shear_k` + the de-shear add-back through the CKS-23-lifted fBm and the L1 `ridged3` loop across CPU+GPU twins while keeping `S≡1` byte-identical; matching float op-order so the OFF golden is bit-for-bit (§D). |
