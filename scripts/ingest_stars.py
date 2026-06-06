"""Offline point-star catalog ingest (Formula 13 / PROJECT.md §8 Layer A, Part A).

Pre-processes a bright-star catalogue into the compact, render-ready point-star
table the DNGR background path consumes:

    catalog[i] = (theta', phi', flux_r, flux_g, flux_b)          # float32, shape [N, 5]

where ``(theta', phi')`` are the **Boyer-Lindquist celestial-sphere exit angles**
— *the same frame the integrator already produces and the equirect starmap is
sampled in* (``renderer.taichi_renderer``: ``theta' = acos(u_exit) in [0, pi]``,
``phi' = phi_exit``; equirect ``u = phi'/2pi``, ``v = theta'/pi``). A real
catalogue's J2000 equatorial coordinates are mapped into that frame by aligning
the black-hole spin axis with the celestial north pole:

    theta' = pi/2 - Dec        (colatitude; Dec=+90deg -> theta'=0, the +z / v=0 pole)
    phi'   = RA                (azimuth, wrapped into [0, 2pi))

Per-star RGB **flux** is built exactly as PROJECT.md §8 describes — *energy*, not
a texture sample — so the renderer can scale it by the per-pixel lensing
magnification (Formula 13):

    flux_rgb = vmag_to_flux(Vmag) * blackbody_rgb(bv_to_temperature(B-V))

`blackbody_rgb` is **reused verbatim** from ``renderer.disk`` (the host twin of
the kernel's ``_blackbody_rgb``; Formula 9 chromaticity-only helper — no T^4
amplitude, so it is a *colour*, and the Pogson flux above supplies the
brightness). The two non-GR empirical relations used here are stellar
astrophysics, not Kerr/GR formulas, so they are cited inline rather than routed
through SKILL.md:

  * Pogson (1856) magnitude->flux:  flux = 10^(-0.4 * (Vmag - mag_zero_point))
  * Ballesteros (2012, EPL 97 34008) B-V -> effective temperature:
        T = 4600 * ( 1/(0.92*BV + 1.7) + 1/(0.92*BV + 0.62) )   [K]

Two input formats are supported, dispatched by ``starfield.format``
(``auto`` → by file extension):

  * **HYG / ATHYG** (``.csv``) — the HYG-like database derived from AT-HYG v3.2
    (Hipparcos + Tycho-2 + Gaia), a header-row CSV. Columns used: ``ra`` (hours),
    ``dec`` (degrees), ``mag`` (apparent V), ``ci`` (B−V). The Sun (``id`` 0 /
    ``proper`` "Sol") is skipped — it is the observer's star, not background sky.
  * **Yale Bright Star Catalogue**, 5th ed. (``.dat``; Hoffleit & Warren 1991;
    ADC/VizieR ``V/50``, the fixed-column ``bsc5.dat`` ascii file).

Both feed the *same* per-star transforms below, so the catalogue choice changes only
the reader. Every numeric parameter (paths, magnitude limit, flux zero-point) comes
from ``configs/render.yaml`` under ``starfield:`` — no hardcoded values (project rule).

This is **Part A (offline ingest) only** — Phase 1 of the §8 DNGR rearchitecture.
It writes a data file and touches no renderer code; the GPU star-gather path
(Phases 2-5) is not wired up yet.

Usage
-----
    uv run python scripts/ingest_stars.py                       # uses render.yaml
    uv run python scripts/ingest_stars.py --src assets/bsc5.dat --format bsc5
    uv run python scripts/ingest_stars.py --mag-limit 5.0 -v
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

# Make ``renderer`` importable when run directly (src layout) — mirrors thumb.py.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from renderer.disk import blackbody_rgb  # noqa: E402  (Formula 9 chromaticity, reused)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "configs" / "render.yaml"

_TWO_PI = 2.0 * np.pi


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:  # utf-8: Windows cp949 box
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Pure per-star transforms (importable / unit-tested without any catalogue file)
# --------------------------------------------------------------------------- #
def vmag_to_flux(vmag: float, mag_zero_point: float = 0.0) -> float:
    """Apparent visual magnitude -> linear flux (Pogson relation).

    ``flux = 10^(-0.4 * (Vmag - mag_zero_point))``. The zero-point only sets an
    overall scale (the renderer applies exposure/gain downstream), so brighter
    stars (smaller Vmag) get exponentially more flux, as required.
    """
    return float(10.0 ** (-0.4 * (vmag - mag_zero_point)))


def bv_to_temperature(bv: float) -> float:
    """B-V colour index -> blackbody effective temperature in Kelvin.

    Ballesteros (2012, EPL 97 34008), the standard single-relation fit:
        T = 4600 * ( 1/(0.92*BV + 1.7) + 1/(0.92*BV + 0.62) )
    """
    s = 0.92 * bv
    return float(4600.0 * (1.0 / (s + 1.7) + 1.0 / (s + 0.62)))


def radec_to_theta_phi(ra_rad: float, dec_rad: float) -> tuple[float, float]:
    """J2000 equatorial (RA, Dec) [rad] -> BL exit angles (theta', phi') [rad].

    Spin axis aligned with the celestial north pole: theta' = pi/2 - Dec is the
    colatitude (matches ``v = theta'/pi``), phi' = RA wrapped into [0, 2pi)
    (matches ``u = phi'/2pi``).
    """
    theta = 0.5 * np.pi - dec_rad
    phi = ra_rad % _TWO_PI
    return float(theta), float(phi)


def star_flux_rgb(vmag: float, bv: float, mag_zero_point: float = 0.0) -> np.ndarray:
    """(Vmag, B-V) -> linear RGB flux (Pogson brightness x blackbody chromaticity)."""
    flux = vmag_to_flux(vmag, mag_zero_point)
    chroma = blackbody_rgb(bv_to_temperature(bv))  # Formula 9 helper, reused
    return (flux * chroma).astype(np.float32)


# --------------------------------------------------------------------------- #
# Yale Bright Star Catalogue (V/50, bsc5.dat) fixed-column parser
# --------------------------------------------------------------------------- #
# 1-indexed byte ranges from the V/50 ReadMe (RA/Dec J2000, Vmag, B-V):
#   76-77 RAh  78-79 RAm  80-83 RAs   84 DE-  85-86 DEd  87-88 DEm  89-90 DEs
#   103-107 Vmag                       110-114 B-V
_DEFAULT_BV = 0.0  # white-ish (~A0) fallback when a record has no B-V measurement


def _f(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bsc5_record(line: str):
    """Parse one BSC5 line -> (theta', phi', Vmag, B-V), or None to skip.

    Records with no position (e.g. nova/cluster placeholders) or no Vmag are
    skipped; a missing B-V falls back to ``_DEFAULT_BV``.
    """
    if len(line) < 90:  # need at least through the Dec field
        return None

    rah, ram, ras = _f(line[75:77]), _f(line[77:79]), _f(line[79:83])
    de_sign = line[83:84]
    ded, dem, des = _f(line[84:86]), _f(line[86:88]), _f(line[88:90])
    if (rah is None or ram is None or ras is None
            or ded is None or dem is None or des is None):
        return None  # no usable J2000 position

    vmag = _f(line[102:107]) if len(line) >= 107 else None
    if vmag is None:
        return None  # no photometry -> nothing to render
    bv = (_f(line[109:114]) if len(line) >= 114 else None)
    if bv is None:
        bv = _DEFAULT_BV

    ra_hours = rah + ram / 60.0 + ras / 3600.0
    ra_rad = np.deg2rad(ra_hours * 15.0)
    dec_deg = ded + dem / 60.0 + des / 3600.0
    if de_sign.strip() == "-":
        dec_deg = -dec_deg
    dec_rad = np.deg2rad(dec_deg)

    theta, phi = radec_to_theta_phi(ra_rad, dec_rad)
    return theta, phi, vmag, bv


# --------------------------------------------------------------------------- #
# HYG / ATHYG v3.2 CSV parser (header row; RA in hours, Dec in degrees)
# --------------------------------------------------------------------------- #
def parse_hyg_row(row: dict):
    """Parse one HYG/ATHYG CSV record -> (theta', phi', Vmag, B-V), or None to skip.

    Uses the named columns ``ra`` (hours), ``dec`` (deg), ``mag`` (apparent V) and
    ``ci`` (B-V). The Sun (``id`` 0 / ``proper`` "Sol") is skipped — it is the
    observer's own star, not part of the background sky. Rows with no ``mag`` are
    skipped; a blank ``ci`` falls back to ``_DEFAULT_BV``.
    """
    if row.get("proper", "").strip() == "Sol" or row.get("id", "").strip() == "0":
        return None

    vmag = _f(row.get("mag", ""))
    if vmag is None:
        return None  # no photometry -> nothing to render

    ra_hours = _f(row.get("ra", ""))
    dec_deg = _f(row.get("dec", ""))
    if ra_hours is None or dec_deg is None:
        return None  # no usable position

    bv = _f(row.get("ci", ""))
    if bv is None:
        bv = _DEFAULT_BV

    ra_rad = np.deg2rad(ra_hours * 15.0)   # hours -> degrees -> radians
    dec_rad = np.deg2rad(dec_deg)
    theta, phi = radec_to_theta_phi(ra_rad, dec_rad)
    return theta, phi, vmag, bv


# --------------------------------------------------------------------------- #
# Format dispatch + catalog builder
# --------------------------------------------------------------------------- #
def resolve_format(fmt: str, src_path: Path) -> str:
    """Resolve ``auto`` to ``hyg`` (.csv) or ``bsc5`` (anything else, e.g. .dat)."""
    fmt = (fmt or "auto").lower()
    if fmt != "auto":
        return fmt
    return "hyg" if Path(src_path).suffix.lower() == ".csv" else "bsc5"


def _iter_records(src_path: Path, fmt: str):
    """Yield ``(theta', phi', Vmag, B-V)`` for every usable star in ``src_path``."""
    if fmt == "hyg":
        with open(src_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                rec = parse_hyg_row(row)
                if rec is not None:
                    yield rec
    elif fmt == "bsc5":
        with open(src_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                rec = parse_bsc5_record(line)
                if rec is not None:
                    yield rec
    else:
        raise ValueError(f"Unknown catalogue format {fmt!r} (expected hyg|bsc5).")


def build_catalog(
    src_path: Path,
    mag_limit: float,
    mag_zero_point: float = 0.0,
    fmt: str = "auto",
) -> np.ndarray:
    """Parse a star catalogue -> float32 array ``[N, 5]`` = (theta', phi', r, g, b).

    ``fmt`` selects the reader (``auto`` → by extension; see ``resolve_format``).
    Stars fainter than ``mag_limit`` (apparent V) are dropped.
    """
    fmt = resolve_format(fmt, src_path)
    rows: list[list[float]] = []
    for theta, phi, vmag, bv in _iter_records(src_path, fmt):
        if vmag > mag_limit:
            continue
        flux = star_flux_rgb(vmag, bv, mag_zero_point)
        rows.append([theta, phi, float(flux[0]), float(flux[1]), float(flux[2])])

    if not rows:
        raise ValueError(
            f"No usable stars parsed from {src_path} (format={fmt}, "
            f"mag_limit={mag_limit}). Expected a HYG/ATHYG csv or a Yale Bright "
            "Star Catalogue (V/50) bsc5.dat file."
        )
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a bright-star catalogue into a point-star flux table "
        "(Formula 13 / PROJECT.md §8 Layer A, Part A)."
    )
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH,
                        help="render.yaml path (default: configs/render.yaml)")
    parser.add_argument("--src", type=Path, default=None,
                        help="override starfield.source_catalog (raw bsc5.dat)")
    parser.add_argument("--out", type=Path, default=None,
                        help="override starfield.catalog_path (output .npy)")
    parser.add_argument("--mag-limit", type=float, default=None,
                        help="override starfield.mag_limit (drop fainter stars)")
    parser.add_argument("--format", choices=("auto", "hyg", "bsc5"), default=None,
                        help="override starfield.format (auto = by file extension)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print a summary of the ingested catalog")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    sf = cfg.get("starfield", {})

    def _resolve(p: Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (_ROOT / p)

    src = _resolve(args.src if args.src is not None else sf["source_catalog"])
    out = _resolve(args.out if args.out is not None else sf["catalog_path"])
    mag_limit = args.mag_limit if args.mag_limit is not None else float(sf["mag_limit"])
    mag_zero_point = float(sf.get("mag_zero_point", 0.0))
    fmt = args.format if args.format is not None else str(sf.get("format", "auto"))

    if not src.exists():
        print(
            f"ERROR: source catalogue not found: {src}\n"
            "  Download the Yale Bright Star Catalogue (5th ed.) bsc5.dat from\n"
            "  VizieR V/50 (https://cdsarc.cds.unistra.fr/viz-bin/cat/V/50) or\n"
            "  http://tdc-www.harvard.edu/catalogs/bsc5.html and place it at the\n"
            "  path above (configs/render.yaml: starfield.source_catalog).",
            file=sys.stderr,
        )
        return 1

    catalog = build_catalog(src, mag_limit, mag_zero_point, fmt)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, catalog)

    print(f"Ingested {catalog.shape[0]} stars (V <= {mag_limit}) -> {out}")
    if args.verbose:
        theta, phi = catalog[:, 0], catalog[:, 1]
        lum = catalog[:, 2:5].sum(axis=1)
        print(f"  theta' [rad]: {theta.min():.4f} .. {theta.max():.4f}")
        print(f"  phi'   [rad]: {phi.min():.4f} .. {phi.max():.4f}")
        print(f"  flux (r+g+b): {lum.min():.3e} .. {lum.max():.3e}")
        print(f"  dtype={catalog.dtype}, shape={catalog.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
