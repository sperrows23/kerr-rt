# Accretion-disk procedural turbulence — design spec (D2)

**Date:** 2026-06-13 · **Status:** owner-approved design, **pre-code** (no renderer
change yet) · **Branch:** `feat/accretion-disk`
**Math of record:** `skills/kerr-physics/SKILL.md` Formula **CKS-12** (noise
coordinates, Keplerian shear advection, modulation bookkeeping). This spec carries
the non-GR texturing details (noise primitives, layer stack, config, tests, build
order).

---

## 1. Goal

The disk today is purely radial — `T(r)` (simple or Page-Thorne CKS-11 LUT) × a
Gaussian vertical profile — so it reads as smooth concentric rings. A real fluid
simulation is out of scope; instead we mimic the look with **layered procedural
noise advected by Keplerian shear**.

**Look target (owner, 2026-06-13):** the *Interstellar/Gargantua* base — smooth,
bright, fine filaments stretched along the orbital direction — plus a sparser
high-contrast accent layer (ridged multifractal + Voronoi billow) reading as gas
**clumping and tearing from magneto-rotational (MRI) instability**.

## 2. Decisions of record (owner interview, 2026-06-13)

| Decision | Choice |
|---|---|
| Look | Interstellar base + MRI clump/tear accent layer |
| Time behavior | **Full Keplerian shear advection** (differential rotation, pattern evolves across frames) |
| Modulated quantities | density/emission **and** temperature **and** edges/scale height |
| Evaluation architecture | **Fully procedural in-kernel** (`@ti.func` hash-lattice noise; no baked textures — zero VRAM beside the 16k starmaps, no tiling, advection exact at any zoom) |
| Noise space | Disk-natural `(u = ln r/r_inner, φ, ζ)` computed per sample from CKS `(x,y,z)` — renderer stays CKS everywhere else |
| Formula policy | Owner granted a one-time exception (2026-06-13) to *add* the new advection/bookkeeping math → SKILL.md **CKS-12**. Ω is Formula 3 verbatim; no GR is re-derived. |
| Default | `disk.noise.enabled: false` ⇒ **bit-identical** legacy kernel branch (same pattern as `doppler_strength == 1.0`); golden frames untouched |

“Alligator noise” (requested in the original brief) is SideFX/Houdini-proprietary
sparse-convolution cellular noise; the **Voronoi-billow** primitive below is the
standard open approximation and serves the same visual role. If the owner supplies
a reference implementation later, it can replace the L1 cellular primitive without
touching the architecture.

## 3. Noise primitives (`src/renderer/noise.py` — CPU reference + `@ti.func` twins)

Project pattern (same as `disk.py` ↔ `_disk_emit_cks`): **NumPy CPU implementation
is the source of truth**, the Taichi twins must match it, agreement enforced by
`tests/test_noise.py`. All primitives are deterministic integer-hash lattice
functions (PCG-style hash; **no `ti.random`**) so renders are reproducible from
`disk.noise.seed`.

These are texturing functions, not physics — they live here, not in SKILL.md.
Sources: Perlin, *Improving Noise* (SIGGRAPH 2002) — quintic fade, hashed
gradients; Worley, *A cellular texture basis function* (SIGGRAPH 1996); Musgrave
(*Texturing & Modeling: A Procedural Approach*) for the ridged construction;
McEwan, Sheets, Gustavson & Richardson, *Efficient computational noise in GLSL*
(JGT 16(2), 2012) — verbatim-verified claim that purely computational noise
(“neither textures nor lookup tables”) is GPU-practical, which underwrites the
in-kernel architecture choice. See §10 for corpus provenance.

### 3.1 Lattice gradient noise `n(p) ∈ [0,1]`, periodic in φ

- 2D/3D gradient (Perlin-style) noise with the 2002 quintic fade
  `s(t) = 6t⁵ − 15t⁴ + 10t³` and hashed lattice gradients.
- **The φ lattice dimension wraps with an integer period** (`freq_phi ∈ ℤ`,
  lattice index taken mod `freq_phi`) so `n` is exactly 2π-periodic in azimuth —
  no seam at φ = 0 (CKS-12 constraint N3).

### 3.2 fBm
```
fbm(p; O, L, G) = ( Σ_{o=0}^{O−1} G^o · n(L^o · p) ) / Σ_{o=0}^{O−1} G^o
```
octaves `O`, lacunarity `L` (default 2), gain `G` (default 0.5).

### 3.3 Billow / turbulence
Same sum with `|2·n − 1|` per octave (Perlin turbulence — cusped, cloud-like).

### 3.4 Ridged multifractal (Musgrave-style, simplified)
```
r_o     = (offset − |2·n(L^o·p) − 1|)²          # offset ≈ 1.0
w_o     = clamp(r_{o−1} · feedback, 0, 1)        # spectral-weight feedback, w_0 = 1
ridged  = ( Σ_o  w_o · r_o · G^o ) / (norm)
```
Sharp connected ridge lines — the “tearing” filament edges of L1.

### 3.5 Worley / Voronoi cellular, F1 & F2
Jittered-grid nearest-feature distances (9-cell search in 2D, 27 in 3D), φ-periodic
grid as in §3.1.
```
voronoi_billow = exp(−k · F1)        # bright clump cores  (the “alligator” stand-in)
cell_wall      = F2 − F1             # optional membrane/tearing variant
```

### 3.6 Deferred: hybrid/heterogeneous multifractal
The owner’s brief listed hybrid multifractal. The exact Musgrave transcription is
not on hand, and per project culture we do not write formulas from memory. **L2
ships as plain 2-octave fBm**; a verbatim hybrid-multifractal transcription (from
*Texturing & Modeling*, 2nd ed.) can be added later as an optional L2 upgrade.

## 4. Layer stack and combination

All layers are sampled at the **shear-advected** coordinates of CKS-12 (dual-phase
reset blend). Frequencies are per-axis `(freq_u, freq_phi, freq_z)` — anisotropy is
expressed as frequency ratios.

| Layer | Primitive | Dim | Anisotropy | Role |
|---|---|---|---|---|
| **L0 base streaks** | fBm (§3.2), 4–5 octaves | 2D `(u, φ′)` | `freq_phi ≪ freq_u` (features long along orbit) | Interstellar filaments; low contrast |
| **L1 clump/tear** | ridged MF (§3.4) × voronoi_billow (§3.5), 2–3 octaves | 3D `(u, φ′, ζ)` | mild | MRI clumps + torn filament edges; **gated by a slow low-frequency coverage mask** `M ∈ [0,1]` so clumps appear in patches, not uniformly |
| **L2 patchiness** | fBm, 2 octaves | 2D `(u, φ′)` | none | breaks large-scale ring symmetry; subtle |

Combination on density (CKS-12 §3 — multiplicative/exponential, keeps ρ > 0):
```
m = a₀·(L0 − ½) + a₁·M·(L1 − b₁) + a₂·(L2 − ½)
density_mult = exp( clamp(m, −m_max, +m_max) )
```
`density_mult` multiplies the existing Gaussian vertical density and feeds **both**
emission and absorption (clumps self-shadow).

Secondary modulations (all CKS-12 §3, bookkeeping constraints apply):
- **Temperature:** `T_emit ← T_emit · (1 + τ_amp·(L1_masked − ½))` — clumps run
  hotter (whiter knots), lanes redder. Applied **before** the `g_eff` shift;
  amplitude small (≤ ~0.3) so it shifts chromaticity only.
- **Edges:** hard `r_inner`/`r_outer` cutoffs become smoothstep windows whose
  positions wobble with a 1-octave φ-periodic advected noise; **`r_in_eff` is
  clamped ≥ r_isco** (zero-torque BC, CKS-11) — the rim can only recede outward.
- **Scale height:** `σ_θ ← σ_θ·(1 + h_amp·(n_h − ½))`; the `max_step_vfrac` step
  cap must use the **worst-case (smallest) modulated σ_z**, i.e.
  `σ_z·(1 − h_amp/2)`, or the moiré that the step cap fixed comes back
  (cf. `tests/test_disk_step_convergence.py`).

## 5. Config draft (`configs/render.yaml` — applied in D2.2, **not yet**)

```yaml
disk:
  noise:
    enabled: false        # master switch; false = bit-identical legacy branch
    seed: 1234
    # time_scale / shear_period are NOT stored here — superseded 2026-06-13 (D3,
    # Formula CKS-13): the resolver derives disk.dynamics.time_scale =
    # T_orb(r_inner)/inner_lap_seconds and disk.dynamics.shear_period_M =
    # shear_wrap_budget·2π/(Ω_in−Ω_out) from the BASE look targets
    # disk.dynamics.{inner_lap_seconds: 10.0, shear_wrap_budget: 3.0} (already in
    # render.yaml). They rescale automatically with spin and disk extent.
    variance_preserve: true   # divide blend by sqrt(w0²+w1²) (CKS-12 §2)
    dynamism: 1.0        # NON-PHYSICAL viz gain on the shear amount φ′=φ−dynamism·Ω·a·T
                         # (1.0 = formula, bit-identical; >1 emphasises per-frame swirl,
                         # leaves reset cadence/continuity untouched — like doppler_strength)
    m_max: 2.5            # clamp on the log-density sum
    layers:
      base:  {enabled: true, amp: 0.6, octaves: 5, lacunarity: 2.0, gain: 0.5,
              freq_u: 6.0, freq_phi: 24, freq_z: 0.0, speed: 1.0}
      clump: {enabled: true, amp: 1.2, bias: 0.35, octaves: 3, lacunarity: 2.0,
              gain: 0.5, freq_u: 3.0, freq_phi: 12, freq_z: 1.0, speed: 1.0,
              coverage: 0.45, mask_freq_u: 1.0, mask_freq_phi: 3, ridge_offset: 1.0,
              voronoi_k: 4.0}
      patch: {enabled: true, amp: 0.35, octaves: 2, lacunarity: 2.0, gain: 0.5,
              freq_u: 1.5, freq_phi: 4, freq_z: 0.0, speed: 0.5}
    temperature: {amp: 0.15}            # ΔT/T from the masked clump field
    edges:  {inner_amp: 0.06, outer_amp: 0.12, width_in: 0.08, width_out: 0.15,
             freq_phi: 5, speed: 1.0}
    height: {amp: 0.5, freq_u: 1.0, freq_phi: 3, speed: 0.7}
```

All values above are **placeholders to be tuned in D2.2 look-dev** (`[look-dev]`).
The time mapping is now physical (D3 / CKS-13): with the shipped defaults
(`inner_lap_seconds: 10`, a = 0.999 → `T_orb(r_isco) ≈ 14.35 M`) the resolver
gives `time_scale ≈ 1.435 M/s` and `shear_period_M ≈ 3·14.6 ≈ 43.8 M` (≈ 30 s of
footage between reseeds); tuning rotation speed = editing `inner_lap_seconds`
only. Frequencies `freq_phi` are integers (φ-periodicity, §3.1).

## 6. Renderer integration points (D2.2–D2.5)

| Touch point | Change |
|---|---|
| `src/renderer/noise.py` (new) | CPU reference primitives (§3) |
| `taichi_renderer.py` `@ti.func` noise twins | mirror of `noise.py` |
| `_disk_emit_cks` (`taichi_renderer.py:748`) | compute `(u, φ, ζ)` from the already-available `r`, `x,y,z`, `dz_ang/σ_θ`; apply CKS-12 advection + §4 stack behind `noise_enabled` |
| `render_beauty_frame` (`taichi_renderer.py:1503`) | new `t_disk` parameter (computed by callers as `frame_index / render.fps × disk.dynamics.time_scale` — the CKS-13-derived value) |
| `render_beauty_frame_mb` (`taichi_renderer.py:1641`) | jitter `t_disk` across the shutter **alongside** the camera `dphi` jitter — the rotate-the-camera motion-blur trick alone is only valid for an axisymmetric disk |
| `_setup_disk_noise` (new, beside `_setup_disk_flux`) | upload per-layer params into a small `ti.field` **param buffer** (not baked module constants) so look-dev tuning does **not** re-JIT; `t_disk`, `enabled`, `seed` are kernel args. Follow the D1 `ti.init`-re-setup gotcha: setup must run after `ti.init` on every renderer (re)initialization. |
| `scripts/thumb.py` | accept `--t-disk` (or `--frame`) for advection look-dev |

**Step-cap interaction (required, D2.4 — ✅ done):** the Pipe-B vertical step cap
(`disk.max_step_vfrac`, CKS-5 note) used `σ_z = r·θ_half·σ_frac`. With height
modulation it now uses the worst-case `σ_z·(1 − height_amp/2)` in
`render_beauty_physics` (only when `_NI_MOD_EN > 0.5`), guarded by
`test_disk_step_convergence.py::test_disk_emission_resolves_lumpy_slab`.

## 7. Performance

Budget estimate per disk sample with the §5 defaults: ~2 phases × (5 + 3 + 2
octaves + 1 Voronoi 9-cell + mask/edge/height 1-octave samples) ≈ 25–30 lattice
evaluations — non-trivial but confined to in-slab samples (a small fraction of each
ray’s march). Mitigations, in order, **only if measured slow** (D2.2 gate:
≤ 2× thumb render time at 256², ≤ ~1.5× at 4K):
1. octave dials (config-only),
2. evaluate L2/mask at lower frequency and reuse across phases,
3. skip the second advection phase for L2 (slow layer, blend pop invisible),
4. fall back to the hybrid plan: bake the Voronoi layer into a small 3D tile
   (the architecture decision explicitly kept this as the fallback, not the start).

## 8. Test plan

| Test | Asserts |
|---|---|
| `tests/test_noise.py` CPU↔GPU agreement | Taichi twins match `noise.py` to ~1e-6 on a sample grid |
| φ-periodicity | `n(u, 0, ζ) == n(u, 2π, ζ)` exactly, every primitive |
| Determinism | same seed + same `t_disk` ⇒ bit-identical thumb render twice |
| Bit-identical off-branch | `noise.enabled: false` render == current golden frames (existing pinned GPU regression must keep passing untouched) |
| Temporal continuity | thumb at `t` vs `t + dt` (dt = 1 frame): mean abs diff below a calibrated bound; catches advection-cycle pops at phase resets |
| Step-cap convergence (extended) | `test_disk_step_convergence.py` re-run with noise + height modulation on |
| New golden | one pinned noise-on frame (fixed seed, fixed `t_disk`) added to the GPU regression |
| Conservation suite | untouched — noise never enters the geodesic path |

~~Known pre-existing failure (memory, 2026-06-12): `doppler_strength=0.1` regression
fails before this work starts — do not attribute it to D2.~~ **RESOLVED 2026-06-13:**
root-caused to D3/CKS-13 re-keying the simple-model peak temperature
(`T_0`→`target_peak_temperature`, peak T_eff 18,600→5,500 K), which moved the disk
peak 6.17→14.45 and Doppler ratio 4.32→5.15 — NOT just `doppler_strength`. The
`test_gpu_regression.py` guards were re-anchored + made dynamic in `doppler_strength`
(monotone g^s beaming sweep + re-measured s=1.0 goldens; SKILL.md v1.16).

## 9. Build order (phases gate on each other; each lands with its tests + docs sync)

- **D2.0 — Docs (this commit).** SKILL.md CKS-12, this spec, PROJECT.md §7/§10.
- **D2.1 — Noise primitive library.** `noise.py` CPU reference + `@ti.func` twins
  + `tests/test_noise.py` (agreement, periodicity, determinism). No renderer change.
- **D2.2 — Static structure. ✅ done 2026-06-13.** `(u, φ, ζ)` mapping + L0/L1/L2 on
  **density only**, `t_disk = 0`; `disk.noise` config block; `_setup_disk_noise`
  param buffer; GPU `_disk_noise_density_mult` + CPU `noise.noise_density_mult` twin;
  `thumb.py` look-dev; tests `test_disk_noise.py` (off-branch bit-identity,
  enabled-changes-disk, determinism, seed, GPU↔CPU stack agreement) + 4 CPU stack
  tests. Perf: noise-on ≈ 2.66× off at 960×540 (sub-100 ms; above the ≤2× target but
  fine for offline — tunable via config-only octave dials, defaults are placeholders).
- **D2.3 — Shear advection. ✅ done 2026-06-13.** `t_disk` plumbing
  (`frame/fps·time_scale`, CKS-13) through `render_beauty_frame{,_mb}` →
  `_disk_emit_cks`, `export_exr.py`, and `thumb.py --frame/--t-disk`; the §2
  dual-phase reset blend + per-cycle integer reseed wraps the `m`-stack in both
  `noise.noise_density_mult` (CPU) and `_disk_noise_density_mult` (GPU twin); Ω is
  Formula 3 per disk sample. `disk.noise.variance_preserve` (default true) ÷√(Σw²).
  `shear_period ≤ 0` (no `disk.dynamics`) ⇒ static D2.2 path, **bit-identical** —
  goldens + GPU stack-agreement untouched. Tests: `test_noise.py §2` (static
  fallback, evolution, determinism, reset-continuity over [0,2T], variance-preserve)
  + `test_disk_noise.py::test_advected_stack_matches_cpu_reference` (GPU↔CPU).
  **Follow-up 2026-06-13:** added the non-physical `disk.noise.dynamism` viz gain
  (`φ′=φ−dynamism·Ω·a·T`, default 1.0 = bit-identical) after the swirl read too weak
  in a look-dev render — in the first reset cycle the visible winding is `Ω·t_disk`
  (T cancels), so per-frame swirl was previously only tunable via the *physical*
  `inner_lap_seconds`. The gain emphasises it without touching reset cadence/continuity.
  Tests: `test_noise.py` (unit-gain bit-identity + winding-emphasis) +
  `test_disk_noise.py::test_dynamism_gain_matches_cpu_and_changes_shear`.
- **D2.4 — Temperature + edges + scale height. ✅ done 2026-06-13.** Four advected
  [0,1] fBm envelopes (`n_T, n_e_in, n_e_out, n_h`) co-moving with the §2 dual-phase
  reset blend + `dynamism` gain: CPU `noise.noise_modulation_fields` + GPU twin
  `_disk_noise_mod_fields` (vec4), decorrelated by seed offsets `NSEED_MOD_T/EIN/EOUT/H
  = 503/601/701/809`, **no** variance-preserve divide (convex triangle weights keep a
  [0,1] fBm in range). Applied per CKS-12 §3: `T_emit·(1+τ·(n_T−½))` **pre-g**
  (constraint 2); `r_in_eff = max(r_inner·(1+e_in·(n_e_in−½)), r_isco)` (constraint 3);
  `r_out_eff = r_outer·(1+e_out·(n_e_out−½))`; `_smoothstep_ti` edge windows replace
  hard cutoffs; `σ_θ·(1+h·(n_h−½))` lumpy scale height with the Pipe-B step cap on the
  worst-case `σ_z·(1−h/2)` (constraint 4). σ-circularity avoided by sampling at the
  unmodulated σ then re-evaluating the Gaussian at σ_m. `_setup_disk_noise` buffer grew
  32→43 (`_NI_MOD_*` 32-42); `_disk_emit_cks`/`render_beauty_physics` gained `r_isco`,
  trace band widened to `[r_isco, r_outer·(1+½e_out)+soft]` when modulation is on.
  `thumb.py` look-dev twin. Tests: `test_noise.py §3` (disabled-is-½, unit-range,
  decorrelation, advect+determinism), `test_disk_noise.py::test_mod_fields_match_cpu_reference`
  (GPU↔CPU), `test_disk_step_convergence.py::test_disk_emission_resolves_lumpy_slab`
  (constraint-4 worst-case-σ_z cap, ≤0.06). **Applied globally:** `configs/render.yaml`
  now ships `disk.noise.enabled: true` + `modulation.enabled: true`; the GR/calibration
  guards force noise OFF so the global enable doesn't shift the pinned goldens.
  `enabled:false`/`modulation.enabled:false` stay bit-identical to D2.3 (constraint 6).
- **D2.5 — Finish.** Motion-blur `t_disk` jitter, perf pass, noise-on golden
  frame, owner sign-off. (PROJECT.md/SKILL.md/memory synced through D2.4;
  `disk.noise.enabled` was flipped to `true` for production at D2.4 per owner
  request — "apply it globally".)

## 10. Research provenance

Corpus: `research/accretion-disk/sources-procedural-noise.md` (SB1–SB15), validated
by `research/accretion-disk/validation-procedural-noise.md` (sonnet, 2026-06-08).
**Caution from the validator:** ~7 of 15 sources carry scout-paraphrases presented
as verbatim quotes — only VERIFIED-rated claims were used here:

- **SB11 (SpaceEngine, VERIFIED):** production precedent — “procedurally-generated
  and animated noise to simulate rotating plasma cloud formations around black
  holes”; animated noise also reduces banding.
- **SB8 (McEwan et al. 2012, VERIFIED):** purely computational GPU noise
  (“neither textures nor lookup tables”) is practical — underwrites the in-kernel
  architecture. (Validator note: authors were misattributed as “Unknown” in the
  corpus; correct attribution used here.)
- **SB1 (Bridson et al. 2007, VERIFIED via fallback):** curl noise gives
  divergence-free procedural *velocity* fields. **Not used now** — our advection
  field is the analytic Keplerian Ω(r), which needs no noise — but noted as the
  principled future option for adding local eddy swirl (domain-warp by a curl
  field) if straight shear reads too laminar.
- **SB10 (Bruneton shader):** validator flagged it as **Schwarzschild-only** —
  do not import its disk machinery into this Kerr (a = 0.999) project.
