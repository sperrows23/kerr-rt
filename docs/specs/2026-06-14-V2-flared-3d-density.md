# V2 — Flared 3D Volumetric Density — Design Spec

**Status:** APPROVED 2026-06-14 (owner). Implementation pending.
**Epoch:** V (volumetric). Builds on V1 (CKS-14 source function + CKS-15 radial
self-shadow, shipped fabae31) and the D2 noise stack (CKS-12).
**Parent plan:** `2026-06-13-volumetric-disk-and-gas-flow.md` §3 "V2 — 3D
volumetric density (thick, flared disk)".

---

## 0. Goal & increment boundary

Give the disk **real vertical bulk** so the procedural noise (which already takes
`ζ` as its third coordinate) stops being squashed flat by a constant thin slab.
This is the envelope half of the "volumetric" look; the lighting half (CKS-14/15)
already shipped in V1.

**This increment is FLARED 3D DENSITY ONLY** (owner decision, one-variable-at-a-time):

- IN: radius-flared angular scale height `σ_θ(r)`; the two knock-on fixes (θ-band
  coverage + step-cap verification); config block; CKS-16; tests; docs.
- OUT (later increments): vertical self-shadow (extend CKS-15 so top/inner gas
  shadows the midplane — needs this 3D bulk but is its own commit); a dedicated V2
  volumetric golden frame (deferred to V5 sign-off); V3 curl-flow.

---

## 1. Core change — flared angular scale height

Everywhere the vertical envelope is evaluated, replace the constant
`σ_θ = theta_half · sigma_frac` with a radius-flared one:

```
σ_θ(r) = (theta_half · sigma_frac) · (r / r_inner)^β
```

- `β = 0` ⇒ `σ_θ(r) ≡ theta_half · sigma_frac` ⇒ **exactly today, bit-for-bit.**
- `σ0 ≡ theta_half · sigma_frac` is the `r = r_inner` value (the inner edge keeps
  today's thickness); `β > 0` thickens the disk **outward** (astrophysical flare,
  H/r increasing with radius).
- The angular envelope is the right object to flare: near the equator
  `z = r cosθ ≈ −r · dz_ang`, so a *constant* angular `σ_θ` is already a
  *constant H/r* (linearly flaring physical height) disk; `β > 0` makes H/r itself
  grow outward.

**Composition with CKS-12 §3 modulation.** The lumpy-scale-height term applies on
top of the flared base, order preserved:

```
σ_θ ← σ_θ(r) · (1 + h_amp·(n_h − ½))          # §3 constraint 4, unchanged form
```

**Genuine 3D falls out for free.** The noise stack
(`_disk_noise_density_mult` → `ridged3`/`fbm3`) already consumes
`ζ = dz_ang / σ_θ`. Once `σ_θ(r)` gives the slab real, radius-varying thickness,
that `ζ`-variation is no longer squashed — **no new noise primitive is added.**
The V1.5 simplex basis stays unwired (reserved for V3 curl).

**Single source of truth.** The flare is implemented once in the shared
`@ti.func _disk_density_cks` (and its CPU twin), so the emission march
(`_disk_emit_cks`) and the CKS-15 shadow bake (`bake_disk_shadow`) inherit the
same `σ_θ(r)` automatically — the V1.0 "extract shared density" refactor is what
makes this a one-point change.

---

## 2. Knock-on fix A — θ bounding band (auto-widened in CKS-13)

Photons only sample the disk inside the angular band `|θ − π/2| < theta_half_width`
(bound `sin(theta_half_width)`). A flared envelope is **thicker at `r_outer`**; if
`theta_half_width` is left at today's value it hard-clips the outer Gaussian and
leaves a visible truncated edge.

**Fix:** the `kerr_params.resolve_config` resolver (CKS-13) **derives**
`theta_half_width` to cover the flared envelope when flare is enabled:

```
theta_half_width ≥ max(theta_half_width_base, K_SIGMA · σ_θ(r_outer))
               = max(base, K_SIGMA · sigma_frac · theta_half_base · (r_outer/r_inner)^β)
```

with `K_SIGMA ≈ 3` (cover ±3σ; a config knob `disk.volumetric.flare.band_sigma`,
default 3.0). Flare disabled / `β = 0` ⇒ the resolver leaves `theta_half_width`
exactly as configured. **No derived literal in the YAML** (config policy); base
value only, widened at load. Patch BOTH loaders (`taichi_renderer` + `thumb.py`)
per the CKS-13 dual-loader rule.

> Note the self-reference: `σ_θ` uses `theta_half · sigma_frac` as `σ0`, and the
> band must cover `3·σ_θ(r_outer)`. The resolver computes `σ0` from the **base**
> `theta_half_width` (the configured slab half-width), then sets the *bounding*
> `theta_half_width` to the widened value. `σ0` is anchored to the base, NOT the
> widened band — otherwise it would feed back on itself. Spelled out so the
> implementer doesn't double-apply.

---

## 3. Knock-on fix B — `max_step_vfrac` cap (verify, don't change)

The Pipe-B vertical step cap protects against stepping *over* a thin slab. Flare
only **thickens outward**, so the thinnest slab is the inner edge `σ0` — today's
worst case, which the existing cap already sizes for. Expectation: **no cap change
needed.** This is verified by a convergence test (a flared slab must resolve at the
current step budget), not assumed.

---

## 4. Config & gating

New block in `configs/render.yaml`:

```yaml
disk:
  volumetric:
    flare:
      enabled: false      # ships OFF — bit-identical to today (owner decision)
      beta: 0.0           # radial flare exponent; 0 = constant-H/r slab (today)
      band_sigma: 3.0     # θ-band coverage factor (×σ_θ(r_outer)) for the resolver
```

- **Default OFF**, matching V1 `source_function` / `self_shadow`: every existing
  golden and the showcase stills stay bit-identical; the owner flips it on during
  look-dev. (The D-noise "on by default" was a separate explicit call; the
  silhouette-changing flare is not presumed on.)
- `enabled:false` takes a branch bit-identical to the pre-flare kernel
  (CKS-12 §3 constraint-6 pattern). `enabled:true, beta:0.0` is also bit-identical
  (the exponent is a no-op) — both paths pinned by tests.

---

## 5. SKILL.md — Formula CKS-16 (geometry/texture, NOT GR)

New formula **CKS-16 — Flared disk scale height** documenting `σ_θ(r)` as a
radial extension of the CKS-12 §3 vertical envelope. Governance identical to
CKS-12: it multiplies an **amplitude/geometry** quantity (the density envelope),
touches no `p_μ`, `u^μ`, `g`, `g⁴`, or `f_PT`. The §3 σ_θ line gains its radial
dependence; the resolver gains the θ-band derivation (a CKS-13 addendum). Add the
SKILL.md revision row. **No new GR derivation** (CLAUDE.md policy) — if review
finds it implies one, STOP and ask the human.

---

## 6. Tests

1. **Bit-identity:** `flare.enabled:false` AND (`enabled:true, beta:0.0`) both
   reproduce the current density field exactly (CPU + GPU).
2. **Flare monotonicity:** for `β > 0`, `σ_θ(r_outer) > σ_θ(r_inner) = σ0`,
   monotonic in `r`; the resolver widens `theta_half_width` to `≥ 3·σ_θ(r_outer)`.
3. **CPU↔GPU twin parity** for the flared density (~1e-6), φ-periodicity preserved.
4. **Step-cap convergence:** a flared slab resolves at the current `max_step_vfrac`
   budget (fix B verification).
5. Existing goldens (`test_gpu_regression.py`, `test_disk_step_convergence.py`) stay
   green with the default-off config.

---

## 7. Docs sync (mandatory, same task as code)

- `skills/kerr-physics/SKILL.md`: CKS-16 + CKS-13 θ-band addendum + revision row +
  capability list (top of file).
- `PROJECT.md` §6/§7 (+ §10 roadmap row if present): V2 flared density entry.
- `2026-06-13-volumetric-disk-and-gas-flow.md`: tick the V2 checkboxes.
- This spec: mark IMPLEMENTED on completion.
- Memory: update `project_volumetric_v1.md` (or a V2 sibling) with the increment.

---

## 8. Out of scope (explicitly)

- Vertical self-shadow (CKS-15 extension) — needs this bulk, but its own commit.
- V2 volumetric golden frame — V5 sign-off.
- V3 curl-flow / domain warp — consumes the unwired V1.5 simplex basis.
- Any change to the wide-Gargantua thin-disk path (shipped & good).
