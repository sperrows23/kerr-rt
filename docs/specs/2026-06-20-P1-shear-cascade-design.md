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
fBm**: `noise._advected_m` computes ONE `φ′_k` per reset phase, builds the CKS-18 cylinder
embedding `(cos φ′_k, sin φ′_k)·f_base` **once**, and the `_octaves`/`sfbm3` loop merely
scales that precomputed embedding by `freq` per octave. So every octave winds at the same
azimuthal rate. Over a long reset window `T` the *fine* detail laminarizes into the *same*
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

- **Cascade ON, long `T`:** large-scale filaments carry **intact high-frequency structure**
  riding on them — the fine octaves are NOT wound into the same spiral as the coarse ones.
  Operationally: the **radial power spectrum of the sampled density field retains its
  high-frequency energy under shear** (cascade ON) versus that energy being smeared/
  suppressed (cascade OFF) at the same `t_disk`.
- **Cascade OFF (or `shear_cutoff → ∞`):** the frame is **bit-identical** to the current
  uniform-shear golden (constraint 6).

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
by a **frequency transfer** `S(f_o)`:

```
f_o      = f_base · lac^o                                  # octave o's true spatial frequency (per layer)
S(f)     = 1 / (1 + (f / f_c)^p)                           # transfer: low f → 1, high f → 0 (Butterworth-like)
φ′_{o,k} = φ − S(f_o) · dynamism · Ω(r) · (a_k · T)        # per-octave, per-reset-phase shear amount

n(u,φ,ζ;t) = Σ_k w_k · [ Σ_o gain^o · N( embed(φ′_{o,k})·f_o , u·f_o , ζ·f_z ; reseed_k ) ]
```

where the CKS-12 §2 dual-phase bookkeeping is **unchanged**:
`s = t_disk/T`, `a_k = fract(s + k/2)`, `c_k = floor(s + k/2)`, `w_k = 1 − |2a_k − 1|`,
`w_0 + w_1 ≡ 1`, per-cycle reseed via `c_k`, optional variance-preserve divide by
`√(w_0² + w_1²)`.

- `Ω(r) = 1/(r^{3/2} + a)` (Formula 3, verbatim) and `dynamism` (CKS-12 §2 non-physical
  viz gain) are unchanged; the cascade only inserts the `S(f_o)` factor.
- **`f_o` per layer** — each density layer (L0/L1/L2) uses its **own** `f_base·lac^o` so the
  cutoff `f_c` is measured against the layer's real spatial frequency, not an octave index.
- `embed(φ) = (cos φ, sin φ)` — the CKS-18 **cylinder embedding** (constraint 5 seam-free),
  which is why per-octave shear forces a per-octave re-embed (§D).

### C.1 Limiting cases (the bit-identity hook)
- `f_c → ∞` (or `shear_cascade.enabled: false`) ⇒ `S(f_o) ≡ 1` ∀o ⇒ `φ′_{o,k} = φ′_k` for
  every octave ⇒ the per-octave re-embed reproduces the current single-embed-then-scale
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

The shear must move **from outside the fBm into the octave loop.** Today:

```
φ′_k = φ − Ω·a_k·T                  # ONE value per phase
x = cos(φ′_k)·f_base ;  y = sin(φ′_k)·f_base       # embed ONCE
n = sfbm3(x, y, u·f_u + ζ·f_z, octaves=…)          # internal loop scales (x,y) by freq per octave
```

CKS-21 needs each octave to re-embed with its **own** `φ′_{o,k}`:

```
for o in range(octaves):
    φ′_o = φ − S(f_base·lac^o)·Ω·a_k·T
    x_o = cos(φ′_o)·f_base·lac^o ;  y_o = sin(φ′_o)·f_base·lac^o
    n  += gain^o · snoise3(x_o, y_o, (u·f_u + ζ·f_z)·lac^o ; reseed)
```

So the octave loop is **lifted out of `sfbm3`** into a new **sheared-fBm primitive** that owns
the per-octave embed. When `S ≡ 1`, `φ′_o = φ′_k` for all `o`, `cos(φ′_o) = cos(φ′_k)` is the
same float, and `x_o = cos(φ′_k)·f_base·lac^o` equals the current `(cos(φ′_k)·f_base)·lac^o`
(same operations, same order) ⇒ **bit-identical**. The float op-order must be matched exactly
to preserve constraint 6.

**Cost:** trig (`cos`/`sin`) now runs **per octave** instead of once per layer. On the GPU
mega-kernel this is the principal perf risk; accepted for v1 (§F-5), optimized later only if
the compile time or warm-render regresses materially.

## D.1 Touch-list (no code yet)

1. **`noise.py` — new sheared-fBm primitive** (the per-octave `S(f_o)` embed + octave sum).
   CPU **source of truth**. Mirrors the `sfbm3` math but owns the octave loop.
2. **`noise._advected_m` / `noise_density_mult`** route the **L0/L1/L2 density** stack
   through the new primitive. **Density layers only** — the modulation envelopes
   (`n_T`, `n_e_in`, `n_e_out`, `n_h` in `noise_modulation_fields`) keep the current uniform
   shear (they are low-frequency; marginal payoff, ~4× surface area — deferred, §F).
3. **`taichi_renderer.py` — GPU twin** of the primitive, ported **verbatim** into the
   `ti.static`-unrolled density octave loop (`_disk_noise_density_mult` / `_disk_noise_m`).
4. **CKS-18 curl-warp ordering** — the curl warp still distorts the **base** `(u, φ′_k)`
   coordinate at the entry; the per-octave shear applies **inside** the loop afterward (§F-3).
   With `S ≡ 1` this is bit-identical to the current order; divergence-free + φ-seam are
   preserved because the warp itself is untouched.
5. **`_setup_disk_noise` param buffer** grows by `shear_cutoff (f_c)`, `shear_falloff (p)`,
   `enabled`; `_NOISE_N` bumps past its current value (69 after CKS-23 — confirm at impl).
   **No CKS-13 resolver change** — all base dials.
6. **No GR / RTE field changes** — `φ′` is a coordinate; emission / `dτ⃗` forms
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
3. **Curl-warp order → curl warp on the base coord, then per-octave shear inside the loop**
   (F-3a). Preserves CKS-18 divergence-free + φ-seam (the warp is untouched); `S≡1` stays
   bit-identical to the current order.
4. **Config home → nest under `disk.noise.shear_cascade`** (F-4a), sibling of
   `disk.noise.curl` — it is a noise-coordinate transform, not a distinct subsystem.
5. **Perf → accept trig-per-octave in v1; defer optimization** (F-5a). Optimize only if the
   mega-kernel cold compile or warm render regresses materially (revisit with angle-addition
   recurrence if needed).

---

# PART G — Tests (TDD)

- **Bit-identity regression** (`tests/test_gpu_regression.py`): `enabled:false` /
  `shear_cutoff` sentinel ⇒ goldens **bit-identical** (the key constraint-6 guard).
- **CPU/GPU twin parity** (`tests/test_noise.py` / `tests/test_noise_gpu.py`): the sheared-fBm
  primitive and the routed density field agree within `_SATOL`.
- **`S≡1` collapse** (`tests/test_noise.py`): with `f_c` at the sentinel, the new sheared-fBm
  equals the existing `sfbm3`-based uniform-shear path **byte-for-byte** (the §C.1/§D hook).
- **Cascade acceptance** (new `tests/test_disk_shear_cascade.py`): at a long `T`, the radial
  power spectrum of the sampled density field **retains high-frequency energy** with the
  cascade ON vs. that band being suppressed OFF (the §A.3 micro-vortex-survival criterion).
- **Monotone transfer** (`tests/test_noise.py`): `S(f)` is monotone-decreasing in `f`,
  `S(0)=1`, `S→0` as `f→∞`; `S≡1` at the sentinel.
- **C0-at-reset** (`tests/test_disk_noise.py`): the blended density field is continuous across
  a reset boundary with the cascade ON (the CKS-12 §2 / CKS-18 §2 guard, now per-octave).
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
| Med | Low–Med | Med | Lifting the octave loop out of `sfbm3` (a structural noise-core refactor across CPU+GPU twins) while keeping `S≡1` bit-identical; trig-per-octave perf on the GPU mega-kernel (§D). |
