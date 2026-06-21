# V1 — Volumetric Self-Shadowing + Source Function (2.5D slab)

**Status:** SPEC — implementation-ready, no code yet (owner approved 2026-06-13).
**Parent plan:** `2026-06-13-volumetric-disk-and-gas-flow.md` (V1–V5 roadmap).
**Scope (owner decisions, 2026-06-13):** V1 first, on the **current 2.5D Gaussian
slab** — prove the radiative-transfer + self-shadow math before touching 3D density
(V2) or curl flow (V3). Self-shadow method = **radial deep-shadow-map** (D-V1.a).
Glow stays **hybrid** (Phase-3 Blender production + the `showcase_disk.py --bloom`
stand-in). **Simplex noise is V1.5** (after V1 verifies, before V2/V3).

**Goal of V1:** turn the flat emission+extinction haze into **glowing gas with deep
black voids** — the defining feature of `whatiwant1.png` / `whatiwant2.png` — using
two flag-gated changes that are *bit-identical to today's goldens when off*.

---

## 0. The two missing terms (why config can't reach the look)

Today's Pipe-B accumulation (`taichi_renderer.py` march loop ~L1529-1533, kernel
`_disk_emit_cks` ~L1111):

```
ev          = _disk_emit_cks(...)              # vec4: (emission_rgb, dτ)
emission    = emis_c · ρ · [f_PT] · g_eff⁴ · chroma · ds     # ev[0:3]
dτ          = absb_c · ρ · ds                                 # ev[3]
disk_col   += transm · emission                # color += T·emission
transm     *= exp(−dτ)                          # T *= exp(−dτ)
```

This is **pure emission + extinction** — no source function, no shadow:

1. **No source function.** Thick gas (`dτ` large) never converges to a bright
   emitting *surface*; it just keeps adding `emission` while `T` collapses → high
   opacity reads **black**, not bright. So the high-`absorption` regime is unusable
   (empirically: max radiance ≈ 6e-4 at `absorption=5.0`).
2. **No self-shadow.** Every sample emits as if lit from everywhere; dense gas never
   *occludes* the gas behind it → no dark lanes / voids, only additive haze.

V1 adds exactly these two terms — **CKS-14** (source function) and **CKS-15**
(radial self-shadow). Both are **flagged off by default**; both compose.

---

## 1. CKS-14 — Radiative-transfer source-function march

### 1.1 The algebra (no new GR — assembled from existing terms)

The radiative transfer equation along the ray, in optical-depth form:

```
dI = (S − I) · dτ            S = j/κ  (source function = emissivity / absorption)
```

Our existing per-step quantities already give `j·ds` (the emission RGB) and
`κ·ds = dτ`. Therefore the **source function is just their ratio** — and the density
and `ds` *cancel exactly*:

```
S = (j·ds)/(κ·ds) = emission / dτ
  = (emis_c · ρ · [f_PT] · g_eff⁴ · chroma · ds) / (absb_c · ρ · ds)
  = (emis_c / absb_c) · [f_PT] · g_eff⁴ · chroma          ← ρ and ds gone
```

`S` is density-independent: it is the colour/brightness a *fully opaque* parcel of
gas would show. The front-to-back integration of `dI=(S−I)dτ` over one step is the
standard analytic update:

```
w           = 1 − exp(−dτ)
disk_col   += transm · w · S
transm     *= exp(−dτ)
```

### 1.2 Back-compatibility (proves it reduces to today's path)

Optically thin (`dτ → 0`): `w = 1 − e^{−dτ} → dτ`, so

```
transm · w · S → transm · dτ · S = transm · (κ·ds) · (j/κ) = transm · j·ds = transm · emission
```

— **exactly today's `disk_col += transm·emission`**, to first order in `dτ`. The two
paths differ only at O(dτ²) (the curvature of `1−e^{−dτ}` vs `dτ`). This is the basis
of the *optically-thin equivalence test* (§4). It is **not** bit-identical (second
order differs), so it lives behind a flag; goldens stay on the legacy branch.

Optically thick (`dτ → ∞`): `w → 1`, each opaque step contributes `transm·S` and the
march converges to the analytic surface value `I → S`.

> **⚠ Correction (verified 2026-06-13, V1.1 implementation).** The legacy emission
> march and this RTE march integrate the **same** continuum quantity
> `I = ∫ S e^{−τ} dτ` (since `transm·j·ds = transm·S·dτ`); they differ **only in
> quadrature**. So CKS-14 does **not** by itself turn a black thick disk bright — in
> the thick regime it is actually *dimmer and more accurate* (legacy's left-endpoint
> rectangle rule **over-counts** opaque steps by `dτ/(1−e^{−dτ})`; measured ≈10% on
> the edge-on disk at `absorption=8`). The earlier "thick gas reads black, source
> function makes it a bright surface" framing (and §0 item 1 below) was imprecise: the
> black high-opacity render is the genuinely-dim *front* surface (the cold outer edge
> occluding the hot inner gas), not a missing source term. **CKS-14's standalone value
> is (i) removing the thick-step over-count and (ii) materialising `S`** — the object
> CKS-15 self-shadow dims (`S·e^{−τ_shadow}`) to carve the voids. **The voids need
> CKS-14 + CKS-15 together.** Guard: `test_disk_source_function.py::
> test_source_function_changes_thick_disk`.

### 1.3 Implementation — march loop only, kernel unchanged

`_disk_emit_cks` is **untouched** (still returns `vec4(emission_rgb, dτ)`); CKS-14 is
a ~6-line reinterpretation in the trace loop, gated by `source_function`:

```python
ev = _disk_emit_cks(...)                       # unchanged
dtau = ev[3]
if source_function == 1 and dtau > _RTE_TAU_EPS:
    inv = 1.0 / dtau
    S = vec3(ev[0]*inv, ev[1]*inv, ev[2]*inv)  # = emission/dτ = j/κ (ρ,ds cancel)
    w = 1.0 - ti.exp(-dtau)
    disk_col += transm * w * S
else:
    disk_col += transm * vec3(ev[0], ev[1], ev[2])   # legacy / dτ≈0 thin fallback
transm *= ti.exp(-dtau)
```

- `_RTE_TAU_EPS` (≈1e-6) guards the divide; below it the thin fallback *is* the legacy
  term, so there is no discontinuity.
- `depth`/`total_emission` bookkeeping (L1530-1532) keys off the *contributed*
  radiance; recompute `contribution` from the term actually added (the `w·S` or the
  legacy emission), so the transmittance-weighted Z stays consistent.
- **g-bookkeeping unchanged:** `S` carries `g_eff⁴·chroma` once — same single
  application as today, no g⁸ risk (Formula 9 / CKS-11 Piece 3 intact).

### 1.4 SKILL.md note (CKS-14)

Add **Formula CKS-14 — Volumetric RTE source-function march (CKS)**: states
`dI=(S−I)dτ`, `S = emission/dτ = (emis_c/absb_c)·[f_PT]·g_eff⁴·chroma`, the thin-limit
reduction to Formula 9, and the explicit note that this introduces **no new GR** — it
reuses CKS-9 `g_eff`, CKS-11 `f_PT`, Formula-9 chroma·g⁴. Mark it the standard
emission/absorption RTE, not a metric change. Add a revision row.

---

## 2. CKS-15 — Radial deep-shadow-map self-shadow

The dominant illuminator is the **hot inner edge** (peak `T_eff = T_0·f_PT^¼` near
`r_inner`, strongest `g⁴` beaming). Gas at larger `r` is shadowed by all the gas
between it and the inner edge at the same `(φ, z)`. V1 captures this **in-plane
(radial) self-shadowing** — clumps casting dark wakes *outward* — which is the
dominant void mechanism for the slab. (Full *vertical* self-shadowing needs the V2
3D bulk; explicitly out of V1 scope, §6.)

### 2.1 The deep-shadow-map (baked once per frame)

A 3-D cumulative optical-depth field on the **noise coordinates** (so the grid is
dense where the gas is):

```
u   = ln(r / r_inner)         ∈ [0, ln(r_outer/r_inner)]     (NR bins)
φ   = atan2(y, x)             ∈ [−π, π)  (periodic)           (NPHI bins)
ζ   = dz_ang / σ_θ            ∈ [−ζ_max, +ζ_max]              (NZ bins)
```

For each `(φ, ζ)` column, march `u` **outward from u=0** accumulating the same
absorption the emission march uses (straight, non-geodesic — see 2.3):

```
τ_shadow[i_u, i_φ, i_ζ] = Σ_{u'=0..u}  absb_c · ρ(u',φ,ζ; t_disk) · (r' · Δu)
```

`dr = r·du` (since `u=ln(r/r_in)`), and `ρ` is the **identical** density used in
emission: `Gaussian(ζ) · dmult(noise) · win(edges)` at the current `t_disk`. To keep
the two density evaluations from drifting, factor the density out of `_disk_emit_cks`
into a shared `@ti.func _disk_density_cks(...)` that **both** the bake kernel and the
emit kernel call (see 3.1) — the extraction must be bit-identical (golden-guarded).

`τ_shadow` is clamped to `max_tau` (overflow/caustic safety).

### 2.2 The lookup (per primary sample, in `_disk_emit_cks`)

Trilinear-sample `τ_shadow` at the sample's `(u, φ, ζ)` (φ wraps) and dim the
emissivity **before** it becomes `S` — so a shadowed thick clump goes *dark*, not just
dim, and then also blocks what is behind it (the deep void):

```python
if self_shadow == 1:
    tau_s = _sample_shadow_tau(u_n, phi_n, zeta_n)     # trilinear, φ periodic
    emission *= ti.exp(-shadow_strength * tau_s)        # j → j·e^{−τ_s}; dτ (κ) untouched
```

Only the **emissivity `j`** is attenuated; the **absorption `κ`/`dτ` is NOT** (the gas
still occludes regardless of how lit it is). Composes correctly with CKS-14: `S =
emission/dτ` inherits the `e^{−τ_s}` factor → shadowed surfaces read dark.

`self_shadow` works with the legacy march too (just dims `emission`); the *void* look
(dark surface, not black hole) needs both flags — they are independent but designed to
compose.

### 2.3 Why this is a VISUALIZATION approximation (governance)

- The shadow ray is a **straight radial line in CKS**, not a geodesic. Over the short
  inner-to-sample span at close-up scale the gravitational bending is small; we accept
  it exactly as `doppler_strength` accepts a non-physical shift dial.
- Single illuminator direction (radially inward), single-scattering, no emission from
  the shadow march — it is occlusion bookkeeping, not a transport solve.
- It **never** touches `p_μ`, `u^μ`, `g`, `g⁴`, or `f_PT` — only multiplies the
  emission *amplitude*, exactly the CKS-12 constraint-1 discipline.
- New SKILL.md **Formula CKS-15 — Radial deep-shadow-map self-shadow (VISUALIZATION,
  not a metric)**, flagged like CKS-12/`doppler_strength`. If a *physical* shadow
  transport is ever wanted, STOP and extend SKILL.md first (CLAUDE.md policy).

---

## 3. Code touch-points

### 3.1 `src/renderer/taichi_renderer.py`
- **Refactor (V1.0, bit-identical):** extract the density block of `_disk_emit_cks`
  (L1167-1212: Gaussian × `_disk_noise_density_mult` × edge `win`, incl. the §3
  modulation) into `@ti.func _disk_density_cks(x,y,z,a,r_inner,r_outer,r_isco,
  theta_half,sigma_frac,noise_enabled,noise_seed,t_disk) → density`. `_disk_emit_cks`
  calls it; result must equal the current inline value (golden-guarded).
- **CKS-14:** march-loop reinterpretation (§1.3), gated by a new `source_function`
  kernel arg (int).
- **CKS-15:** new module field `disk_shadow_tau: ti.field` (shape NR×NPHI×NZ); new
  `@ti.kernel bake_disk_shadow(...)` filling it from `_disk_density_cks` along radial
  columns; new `@ti.func _sample_shadow_tau(u,φ,ζ)` (trilinear, φ-periodic); lookup +
  `emission *= exp(−strength·τ_s)` in `_disk_emit_cks` behind a `self_shadow` arg.
- **`setup_renderer` / `render_beauty_frame`:** add `_setup_disk_shadow(cfg)` (allocate
  the field from the grid-res config, like `_setup_disk_noise`); call `bake_disk_shadow`
  before the trace kernel **only when** `self_shadow.enabled` (and re-bake per frame
  because it depends on `t_disk`); thread the new kernel args (`source_function`,
  `self_shadow`, `shadow_strength`) alongside the existing disk params (~L2055-2071).
- **LOD:** the bake + lookup are close-up cost; gate both off for the wide/mid
  thin-slab path (reuse the existing `lod_enabled` / distance gate). Wide Gargantua
  stays on the cheap legacy march — *zero* added cost there.

### 3.2 `configs/render.yaml` — new `disk.volumetric` block (base knobs, no CKS-13 derive)
```yaml
  volumetric:
    source_function: false   # CKS-14 RTE march. false ⇒ legacy color+=T·emission (goldens bit-identical).
                             # true ⇒ thick gas reads as a bright surface (brightness ~ emission_coeff/absorption_coeff).
    self_shadow:
      enabled: false         # CKS-15 radial deep-shadow-map. false ⇒ no lookup (bit-identical).
      strength: 1.0          # emission *= exp(−strength·τ_shadow). 0 = off, higher = deeper voids.
      grid_nu: 96            # deep-shadow-map log-radial (u=ln r/r_in) bins
      grid_nphi: 256         # azimuthal bins (φ, periodic)
      grid_nz: 16            # vertical bins across the slab (ζ = dz/σ_θ)
      zeta_max: 3.0          # ±ζ extent of the grid (slab is ~Gaussian; 3σ covers it)
      max_tau: 8.0           # clamp on accumulated shadow τ (overflow/caustic safety)
```
Both flags default **off** ⇒ the entire V1 path is dead code until opted in.

### 3.3 `scripts/showcase_disk.py` — look-dev flags
`--source-function` (sets `volumetric.source_function=true`), `--self-shadow`,
`--shadow-strength`, `--shadow-grid` (NR,NPHI,NZ override). Hybrid glow unchanged
(`--bloom` stays the in-renderer stand-in; production glow = Phase-3 Blender).

---

## 4. Tests (ship with the code, same task)
1. **Bit-identity (regression guard):** `source_function:false` **and**
   `self_shadow.enabled:false` ⇒ identical to current goldens (whole V1 path dead).
   Extend `test_gpu_regression.py` to assert the flags-off frame is unchanged.
2. **`_disk_density_cks` extraction parity:** the refactored density equals the
   pre-refactor inline density (a few sample points, ~1e-6).
3. **Optically-thin equivalence (CKS-14):** with `source_function:true` and small
   `absb_c` (or small `ds`), the RTE frame matches the legacy emission frame to a tight
   tol (proves the `1−e^{−dτ}→dτ` reduction). New `test_disk_source_function.py`.
4. **Self-shadow regression (CKS-15):** a single injected dense clump must make a
   *measurably darker* region radially outward of it (mean luminance in the wake <
   unshadowed control by a threshold). Plus a CPU reference of one shadow column vs the
   GPU bake (~1e-5). New `test_disk_self_shadow.py`.
5. **New volumetric golden:** one `source_function+self_shadow` frame as its own golden
   (separate from the thin-slab goldens, behind the flag).

## 5. Docs sync (mandatory, same task — Docs Sync Policy)
- **SKILL.md:** Formula **CKS-14** (RTE source function — "no new GR") + Formula
  **CKS-15** (radial deep-shadow-map self-shadow — VISUALIZATION); revision rows;
  update the "REUSED unchanged" / disk-march section to point at CKS-14.
- **PROJECT.md** §6/§7 (disk march description) + §10 roadmap rows (V1 shipped).
- **render.yaml** comments (the block above).
- **This spec:** check the boxes; **memory:** new `project_volumetric_v1.md` +
  MEMORY.md pointer.

## 6. Out of V1 scope (named so it isn't silently assumed)
- **Vertical** self-shadowing (top gas shadowing the midplane from a high camera) —
  needs the V2 3D bulk; V1's radial map is in-plane only.
- 3D flared density (V2), curl/flow advection + domain warp (V3), Simplex (V1.5),
  free camera (V4). One variable at a time per the owner's sequencing.

## 7. Build order (PR-sized, each green before the next)
- **V1.0** ✅ extract `_disk_density_cks` (bit-identical; test 2 + golden guard).
- **V1.1** ✅ CKS-14 source-function march behind `source_function` (test 1, 3).
- **V1.2** ✅ CKS-15 deep-shadow-map bake + lookup behind `self_shadow` (test 1, 4).
      Shipped 2026-06-13: `disk_shadow_tau` field + `bake_disk_shadow` +
      `_sample_shadow_tau` + `_setup_disk_shadow`; `self_shadow`/`shadow_strength`
      kernel args. `tests/test_disk_self_shadow.py` (4: flag-off bit-identity, GPU
      bake vs analytic Gaussian column, outward-steepening dim, noise-on contrast
      rise) green; regression + source-function suites still bit-identical.
- **V1.3** ✅ `showcase_disk.py` flags + LOD gate. Shipped 2026-06-13:
      `--source-function`/`--self-shadow`/`--shadow-strength`/`--shadow-grid`;
      `self_shadow.lod_max_camera_radius` (0 = off) skips the bake for wide/mid
      shots — `test_disk_self_shadow.py::test_lod_gate_drops_self_shadow_for_distant_camera`.
      (Acceptance still = a manual `showcase_disk.py --source-function --self-shadow`
      close-up render; the void mechanism is regression-guarded by the contrast +
      combined-golden tests below.)
- **V1.4** ✅ docs sync (SKILL CKS-14/15, PROJECT.md §6 + new §11 V-epoch build-order,
      render.yaml, memory) + combined golden
      `test_combined_source_function_and_self_shadow_golden` (relational: composition
      active, NaN-free, dimmer than source-fn-only). **Phase V1 complete.**

## 8. Acceptance
A close-up `showcase_disk.py --source-function --self-shadow` still on the **current
2.5D slab** that shows bright turbulent gas with **dark shadow lanes/voids** trailing
the dense clumps — i.e. reads as gas, not haze — while every flags-off render and all
existing goldens are byte-for-byte unchanged.
