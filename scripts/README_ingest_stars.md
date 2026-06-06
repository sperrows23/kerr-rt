# `ingest_stars.py` — Offline Point-Star Catalog Ingest

Pre-processes a bright-star catalogue into the compact, render-ready point-star
flux table that the black-hole renderer's DNGR background path consumes.

> **Scope:** This is **Part A (offline ingest) only** — Phase 1 of the
> PROJECT.md §8 / Formula 13 DNGR background rearchitecture. It reads a star
> catalogue, transforms it, and writes a single `.npy` data file. **It touches
> no renderer code.** The GPU star-gather path (Phases 2–5) is not wired up yet.

---

## Table of Contents

- [What it does](#what-it-does)
- [Where it fits in the pipeline](#where-it-fits-in-the-pipeline)
- [Output format](#output-format)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Configuration](#configuration)
- [Supported input formats](#supported-input-formats)
- [The per-star transforms (the physics)](#the-per-star-transforms-the-physics)
- [Public API](#public-api)
- [Testing](#testing)
- [Related files](#related-files)

---

## What it does

A star catalogue lists, for each star, its **position** on the sky (Right
Ascension / Declination) and its **photometry** (apparent magnitude + colour
index). This script converts every usable star into one row of:

```
catalog[i] = (theta', phi', flux_r, flux_g, flux_b)      # float32, shape [N, 5]
```

- `theta', phi'` — **Boyer–Lindquist celestial-sphere exit angles**, the exact
  frame the geodesic integrator already produces and the equirect starmap is
  sampled in (`theta' = acos(u_exit) ∈ [0, π]`, `phi' = phi_exit`; equirect
  `u = phi'/2π`, `v = theta'/π`).
- `flux_r/g/b` — linear RGB **energy** (not a texture sample). Storing energy is
  what lets the renderer scale each star by its per-pixel lensing magnification
  μ (Formula 13) — a gravitationally-lensed star gets *brighter*, which a baked
  texture lookup cannot reproduce.

Stars fainter than a configurable magnitude limit are dropped, and the Sun is
skipped (it is the observer's own star, not background sky).

## Where it fits in the pipeline

```
  bright-star catalogue (.csv / .dat)         ← downloaded once, external
            │
            ▼
   scripts/ingest_stars.py   ── THIS TOOL ──  (offline, run once per catalogue/config)
            │
            ▼
       assets/stars.npy   (float32 [N, 5])
            │
            ▼
   Taichi GPU star-gather  (Formula 13, Phases 2–5 — not yet wired up)
```

Run it once after you download a catalogue or change a relevant config value;
the renderer then loads the resulting `.npy` at render time.

## Output format

A single NumPy `.npy` file, `float32`, shape `[N, 5]`:

| Column | Name     | Units        | Meaning                                                   |
|:------:|----------|--------------|-----------------------------------------------------------|
| 0      | `theta'` | radians      | Colatitude, BL exit frame. `0` = +z pole, `π` = −z pole.  |
| 1      | `phi'`   | radians      | Azimuth, wrapped into `[0, 2π)`.                          |
| 2      | `flux_r` | linear (a.u.)| Red-channel flux (Pogson brightness × blackbody colour).  |
| 3      | `flux_g` | linear (a.u.)| Green-channel flux.                                       |
| 4      | `flux_b` | linear (a.u.)| Blue-channel flux.                                        |

Flux is in arbitrary linear units; `mag_zero_point` sets the overall scale and
the renderer applies exposure/gain downstream.

## Prerequisites

- Dependencies are managed with `uv` (see the repo root). `uv sync` installs
  `numpy`, `pyyaml`, etc.
- A **star catalogue file** is *not* shipped in the repo — download one:
  - **HYG / ATHYG v3.2** (default): a HYG-like CSV derived from AT-HYG v3.2
    (Hipparcos + Tycho-2 + Gaia). Place it at the path in
    `configs/render.yaml → starfield.source_catalog`.
  - **Yale Bright Star Catalogue, 5th ed.** (`bsc5.dat`): from VizieR
    [`V/50`](https://cdsarc.cds.unistra.fr/viz-bin/cat/V/50) or
    [tdc-www.harvard.edu](http://tdc-www.harvard.edu/catalogs/bsc5.html).

If the source file is missing, the script prints a download hint and exits `1`.

## Usage

```bash
# Use everything from configs/render.yaml (the normal path):
uv run python scripts/ingest_stars.py

# Point at a specific catalogue + format, overriding the config:
uv run python scripts/ingest_stars.py --src assets/bsc5.dat --format bsc5

# Tighten the magnitude limit and print a summary of what was ingested:
uv run python scripts/ingest_stars.py --mag-limit 5.0 -v
```

### CLI options

| Flag           | Overrides config key        | Default                  |
|----------------|-----------------------------|--------------------------|
| `--config`     | —                           | `configs/render.yaml`    |
| `--src`        | `starfield.source_catalog`  | from config              |
| `--out`        | `starfield.catalog_path`    | from config              |
| `--mag-limit`  | `starfield.mag_limit`       | from config              |
| `--format`     | `starfield.format`          | from config (`auto`)     |
| `-v/--verbose` | —                           | off (prints range stats) |

Relative paths are resolved against the repository root. Exit code is `0` on
success, `1` if the source catalogue is missing.

## Configuration

Per the project's config-driven rule, **every numeric parameter comes from
`configs/render.yaml` under `starfield:`** — no hardcoded values in source. The
keys this script reads:

```yaml
starfield:
  format: auto             # auto | hyg | bsc5  (auto -> by source_catalog extension)
  source_catalog: star_image/hyglike_from_athyg_v32.csv
  catalog_path: assets/stars.npy            # ingest output (.npy written here)
  mag_limit: 6.5           # drop stars fainter than this apparent V (~naked-eye)
  mag_zero_point: 0.0      # flux = 10^(-0.4*(Vmag - mag_zero_point)); overall scale
```

(Other `starfield:` keys — PSF size, grid resolution, lensing/caustic caps, the
diffuse Milky Way map — belong to the not-yet-wired GPU gather path, not to this
ingest step.)

## Supported input formats

Dispatched by `starfield.format` (`auto` → by file extension):

| Format | Extension | Reader              | Notes |
|--------|-----------|---------------------|-------|
| `hyg`  | `.csv`    | `parse_hyg_row`     | Header-row CSV. Columns: `ra` (hours), `dec` (deg), `mag` (apparent V), `ci` (B−V). Skips the Sun (`id` 0 / `proper` "Sol"). |
| `bsc5` | `.dat`    | `parse_bsc5_record` | Yale BSC5 fixed-column ASCII (V/50 byte offsets). Skips records with no position or no Vmag. |

Both feed the **same** per-star transforms, so the catalogue choice only changes
the reader. A missing B−V falls back to `_DEFAULT_BV = 0.0` (white-ish ≈ A0).

## The per-star transforms (the physics)

Each surviving star goes through three conversions. The two stellar-astrophysics
relations are *not* GR/Kerr formulas, so they are cited inline here rather than
routed through `skills/kerr-physics/SKILL.md`:

1. **Position — RA/Dec → BL exit angles** (`radec_to_theta_phi`). The black-hole
   spin axis is aligned with the celestial north pole:
   ```
   theta' = π/2 − Dec        (Dec=+90° → theta'=0, the +z / v=0 pole)
   phi'   = RA               (wrapped into [0, 2π))
   ```

2. **Brightness — Pogson (1856) magnitude → flux** (`vmag_to_flux`):
   ```
   flux = 10^(-0.4 * (Vmag - mag_zero_point))
   ```
   5 magnitudes fainter = 100× less flux (definition of the magnitude scale).

3. **Colour — Ballesteros (2012, EPL 97 34008) B−V → temperature → RGB**
   (`bv_to_temperature`, then `blackbody_rgb`):
   ```
   T = 4600 * ( 1/(0.92·BV + 1.7) + 1/(0.92·BV + 0.62) )   [K]
   ```
   `blackbody_rgb` is **reused verbatim** from `renderer.disk` (the host twin of
   the kernel's `_blackbody_rgb`; the Formula 9 *chromaticity-only* helper — no
   `T^4` amplitude). So step 3 supplies the *colour* and step 2 supplies the
   *brightness*:
   ```
   flux_rgb = vmag_to_flux(Vmag) * blackbody_rgb(bv_to_temperature(B−V))
   ```

## Public API

The pure transforms are importable and unit-tested without any catalogue file:

| Function | Purpose |
|----------|---------|
| `vmag_to_flux(vmag, mag_zero_point=0.0)` | Pogson magnitude → linear flux. |
| `bv_to_temperature(bv)`                  | Ballesteros B−V → effective temperature (K). |
| `radec_to_theta_phi(ra_rad, dec_rad)`    | J2000 (RA, Dec) → BL exit angles `(theta', phi')`. |
| `star_flux_rgb(vmag, bv, mag_zero_point=0.0)` | Combined linear RGB flux triple. |
| `parse_bsc5_record(line)` / `parse_hyg_row(row)` | Per-record readers → `(theta', phi', Vmag, B−V)` or `None`. |
| `resolve_format(fmt, src_path)`          | `auto` → `hyg`/`bsc5` by extension. |
| `build_catalog(src_path, mag_limit, mag_zero_point=0.0, fmt="auto")` | Full parse → `float32 [N, 5]`. |
| `load_config(path)` / `main(argv)`       | YAML loader / CLI entry point. |

## Testing

```bash
pytest tests/test_ingest_stars.py -v
```

The tests cover the per-star transforms and both parsers using hand-built
catalogue records (e.g. Sirius), so they run **without** the large,
externally-downloaded catalogue file. They assert the magnitude scale (5 mag =
100×), B−V monotonicity and a Sun-like ~5800 K sanity point, pole/equator
coordinate mapping, blue-vs-red chromaticity, the BSC5 byte offsets, Sun
skipping, fallbacks for missing B−V/Vmag, format dispatch, and an end-to-end
`build_catalog` CSV round-trip.

## Related files

- `scripts/ingest_stars.py` — this tool.
- `tests/test_ingest_stars.py` — its unit tests.
- `configs/render.yaml` (`starfield:`) — all parameters.
- `src/renderer/disk.py` — source of the reused `blackbody_rgb` helper.
- `src/renderer/taichi_renderer.py` — defines the BL exit-angle / equirect frame
  this catalogue targets, and the future consumer of `stars.npy`.
- `PROJECT.md` §8 — the DNGR background rearchitecture this is Phase 1 of.
- `skills/kerr-physics/SKILL.md` — Formula 9 (`blackbody_rgb`) and Formula 13.
