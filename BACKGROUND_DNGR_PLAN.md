# Background Rearchitecture Plan — DNGR-style Star Field

**Status:** proposal (no code yet) · **Author:** pairing session 2026-06-04
**Motivation:** the baked 16K equirect EXR background smears and dims stars under
strong lensing, and forces a coordinate-singular UV lookup that produces the
spin-axis "static" seam. We want the *Interstellar* DNGR treatment: **point
stars stay sharp; gravitational lensing changes their brightness, not their
size.**

This is a design proposal to review and approve before implementation. It does
**not** change any physics formula. Every new GR/lensing formula it introduces
(magnification, ray-bundle ellipse) is routed through
`skills/kerr-physics/SKILL.md` for human approval **first**, per the project
physics-formula policy. See the already-committed fidelity note (commit
`5700d36`) for the gap analysis this plan acts on.

---

## 1. Why the texture approach fails (and what the seam fix did/didn't do)

| Symptom | Root cause | Status |
|---|---|---|
| Stars smear into streaks under lensing | A baked texture has fixed angular resolution; magnification `µ>1` stretches a few texels across many pixels → blur | **architectural — this plan** |
| Stars dim where they should brighten | Texture energy is per-solid-angle radiance, mip-averaged; magnification is *not* converted to brightness | **architectural — this plan** |
| Center "static" seam at the spin-axis meridian | BL φ folds by π across the meridian caustic; a scalar-LOD trilinear fetch lands on unrelated coarse texels | **mitigated now**: `j_fold` saturates the fold to the coarsest mip (smooth grey). The DNGR rebuild removes the cause entirely. |

The just-landed seam fix (`render.j_fold`, the per-step φ wrap, and the
shortest-arc exit interpolation) is a **stopgap that restores the original
"smooth faint line" behaviour**. It does not make the lensed star field correct
— that is this plan's job.

---

## 2. Target model — two decoupled sky layers

DNGR separates the sky into (a) **point stars** and (b) a **diffuse galaxy/
nebula** map, and filters them differently (paper §2.1–2.3, App. A.2–A.3).

### Layer A — point-star catalog (the sharp, lensed-brightness layer)

- **Data:** a star list `{(θ′, φ′), flux_rgb}` in Boyer-Lindquist celestial
  coordinates — the same `{θ′, φ′}` our integrator already produces at ray exit
  (paper step (iv–v)). Source catalog: Yale Bright Star / Hipparcos / Tycho-2
  or Gaia subset. Convert RA/Dec → our celestial frame; apparent magnitude →
  linear flux; B−V color index → RGB via blackbody (**reuse** `_blackbody_rgb`
  already in `taichi_renderer.py`, no new color code).
- **Rendering = energy gather, not texture fetch.** A star contributes **total
  flux**, scaled by the lensing **magnification** of the pixel's ray bundle:
  - magnified beam (small celestial footprint, `µ>1`) → star is **brighter**;
  - demagnified beam → dimmer;
  - the star's *image* stays a sub-pixel point (sharp), antialiased by the
    pixel/PSF, never by the sky resolution.
- **Multi-imaging falls out for free:** each pixel is already one image sheet
  (one backward ray), so the primary image (outside the Einstein ring) and the
  secondary image (inside) are produced by different pixels naturally — exactly
  the two rays in paper Figure 3.

### Layer B — diffuse galaxy/nebula (low-frequency, anisotropically filtered)

- Keep an equirect texture **only** for the smooth Milky-Way band / dust — it
  has no high-frequency point energy to smear.
- Replace the isotropic scalar-LOD fetch with an **anisotropic (EWA-style)**
  filter driven by the ray-bundle **ellipse** `(µ, δ⁺, δ⁻)` (paper §2.2),
  i.e. the in-architecture analog already named in the SKILL fidelity note.
  This removes the directional smear and the pole-caustic aliasing.

---

## 3. The lensing magnification — the one new piece of physics

We already compute the **scalar** footprint `J = max(Jx, Jy)` from neighbour
exit deltas (Formula 10). DNGR needs the **full 2×2 beam Jacobian**:

```
        ∂(θ′, φ′)
  Jac = ---------          (camera-sky pixel → celestial sphere)
        ∂(x_pix, y_pix)
```

from which:
- the **ellipse** axes `δ⁺, δ⁻` and orientation `µ` are the singular values /
  vectors of `Jac` (paper §2.2 / Fig. 1);
- the **magnification** is the solid-angle ratio
  `mag = Ω_pixel / Ω_beam = 1 / |det Jac · sinθ′|` (Layer-A brightness scale);
- `δ⁻ → 0` marks a **caustic / critical curve** (paper §3.2) — the principled
  replacement for the `j_fold` heuristic.

> **Physics-policy gate.** `mag = Ω_pixel/Ω_beam` and the ellipse extraction are
> **new formulas**. They must be added to `skills/kerr-physics/SKILL.md` and
> approved **before** coding, exactly as Formula 10 was. We can obtain `Jac`
> two ways, to be decided at approval time:
> 1. **Finite-difference** the existing per-pixel exit map (cheap; reuses the
>    two neighbours we already read in `_screen_jacobian_lod`, plus we already
>    store `exit_buf`). Accuracy limited by pixel spacing near caustics.
> 2. **Geodesic deviation** integrated alongside the central ray (paper §2.2,
>    App. A.1–A.2) — the exact DNGR method; needs the deviation ODE added to
>    SKILL.md. Higher fidelity, ~1 extra small linear system per step.
>
> Recommendation: ship FD first (no integrator change, no new ODE), then offer
> geodesic-deviation as a fidelity upgrade behind a config flag.

---

## 4. GPU architecture (fits the existing Phase-2.4 split)

The current split is **physics kernel → shade kernel**. The new flow keeps that
shape:

```
render_beauty_physics   (unchanged):  trace primary ray → exit_buf{θ′,φ′,out}, disk_buf, depth
        │
        ├─ build beam Jacobian Jac(py,px) from exit_buf neighbours  (FD, in-kernel)
        │
render_beauty_shade  (rewritten background half):
        ├─ Layer B: anisotropic EWA fetch of the diffuse map using (δ⁺,δ⁻,µ)
        └─ Layer A: gather stars in the beam ellipse, add flux·mag·PSF
        → frame = disk + transm·(diffuse + stars)
```

**Star gather acceleration (Layer A).** Per pixel we must find stars whose
celestial position lies within (a small dilation of) the beam ellipse:
- Bin the catalog into an **equirect cell grid** (or HEALPix) on `{θ′,φ′}`,
  uploaded as Taichi fields (sorted star array + per-cell ranges).
- Per pixel, query the few cells the ellipse overlaps; for each star inside,
  add `flux_rgb · mag · psf(Δ_pixel)`.
- Bright-star catalogs are ~10⁴–10⁵ stars → grid is tiny; per-pixel candidate
  counts stay O(1–10) away from the galactic plane.

**Fallback / A-B.** Gate the whole new path behind `starfield.mode:
texture | dngr` in `render.yaml`. `texture` keeps today's pipeline (and the
`j_fold` stopgap) so we can regression-compare frame-by-frame.

---

## 5. Config additions (all parameters in `render.yaml`, per CLAUDE.md)

```yaml
starfield:
  mode: dngr                 # texture | dngr
  catalog_path: assets/stars_bsc.npy     # {θ',φ',flux_rgb}
  diffuse_map: assets/milkyway_diffuse.exr  # Layer B only (low-freq)
  jacobian: finite_diff      # finite_diff | geodesic_deviation
  star_psf_px: 1.3           # gaussian splat radius (camera-sky antialias)
  mag_clip: 50.0             # cap on lensing brightness gain (caustic safety)
  caustic_delta_min: 1.0e-3  # δ⁻ below this ⇒ treat as on a caustic
```

`j_fold` stays for the `texture` fallback; it is unused in `dngr` mode (the
ellipse/`δ⁻` handles folds correctly).

---

## 6. Validation (extends the existing GPU regression harness)

1. **Flux conservation:** with lensing off (flat space limit / `a→0`, camera
   far), total integrated star flux in == out, per channel.
2. **Einstein-ring brightness:** a single star placed at the antipode (paper
   Fig. 3 caustic point) must image as a ring of the **correct radius** with
   brightness rising toward `δ⁻→0`, not a blurred smear.
3. **Magnification spot-check:** compare FD `mag` vs geodesic-deviation `mag`
   on a mid-field pixel (should agree to FD truncation error).
4. **Seam gone, not hidden:** the spin-axis meridian shows sharp split star
   images (primary/secondary), no static, **no `j_fold` needed**.
5. **No regression elsewhere:** `mode: texture` reproduces today's golden
   frames bit-for-bit (the new code is gated off).
6. Keep `pytest tests/` green; add `tests/test_starfield_dngr.py` for 1–3.

---

## 7. Phasing / estimated effort (CC-assisted)

| Phase | Deliverable | Risk |
|---|---|---|
| 0 | SKILL.md: add `mag` + ellipse formulas, get approval | none (gate) |
| 1 | Catalog ingest → `{θ′,φ′,flux_rgb}.npy`; B−V→RGB reuse | low |
| 2 | FD beam Jacobian + `mag` in shade kernel; ellipse `(δ⁺,δ⁻,µ)` | med |
| 3 | Layer A star gather (equirect-cell grid) + PSF splat | med |
| 4 | Layer B anisotropic EWA diffuse fetch | med |
| 5 | Config gate, A/B harness, validation suite | low |
| 6 (opt) | Geodesic-deviation Jacobian upgrade | higher (new ODE) |

Phases 1–5 are the minimum to retire the texture star field. Phase 6 is the
"true DNGR" fidelity option.

---

## 8. Open decisions for the human (before Phase 0)

1. **Catalog scope:** bright-star only (sharp foreground) vs. include a faint
   Gaia layer (denser, heavier gather)?
2. **Jacobian method:** FD-first (recommended) vs. go straight to geodesic
   deviation?
3. **Diffuse map:** keep a (smaller) equirect for the Milky-Way band, or drop
   diffuse entirely and render *only* point stars first?
4. **Color fidelity:** blackbody-from-(B−V) (reuse our model) vs. catalog RGB
   if available.

Nothing here is implemented until these are settled and the Phase-0 SKILL.md
formulas are approved.
