"""Showcase still of the turbulent accretion disk (D2 procedural noise).

Renders one beauty frame on the Taichi CUDA backend through the production
``render_beauty_frame`` path (Pipe A + Pipe B), but with a synthetic *inclined*
camera looking down onto the disk face — the framing that makes the CKS-12 §3
turbulence (filaments, clumps, ragged edges, lumpy scale height) clearly visible.
The shipped ``camera_matrix.json`` frames are nearly edge-on (~5° inclination),
which is precisely why the disk structure has been hard to see.

All physics follows ``skills/kerr-physics/SKILL.md``; every numeric parameter
comes from ``configs/render.yaml`` except the synthetic camera (a viewing choice).

Usage
-----
    uv run python scripts/showcase_disk.py                       # 3840x2160 default
    uv run python scripts/showcase_disk.py --width 1280 --height 720 --elevation 25
    uv run python scripts/showcase_disk.py --no-noise            # A/B comparison
    # V1 glowing-gas-with-voids close-up (needs BOTH flags — CKS-14 + CKS-15):
    uv run python scripts/showcase_disk.py --source-function --self-shadow \
        --shadow-strength 1.5 --bloom 0.5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from renderer import taichi_renderer as tr  # noqa: E402


def _bloom(hdr: np.ndarray, strength: float, threshold: float) -> np.ndarray:
    """Soft multi-scale glow on the linear HDR (FFT Gaussian, dependency-free).

    Stand-in for the Phase-3 Blender compositor glow that gives Gargantua its
    luminous halo. A smooth bright-pass above ``threshold`` is blurred at three
    scales and added back; the geometry/physics of the trace are untouched."""
    h, w, _ = hdr.shape
    lum = hdr.max(axis=2, keepdims=True)
    bright = hdr * np.clip((lum - threshold) / (threshold + 1e-3), 0.0, 1.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r2 = fy * fy + fx * fx
    # one combined frequency-domain kernel = Σ wᵢ·exp(−2π²σᵢ²f²) (separable, single ifft/channel)
    gsum = np.zeros_like(r2)
    for sigma, weight in ((6.0, 1.0), (20.0, 0.6), (60.0, 0.3)):
        gsum += weight * np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * r2)
    gsum /= (1.0 + 0.6 + 0.3)
    acc = np.empty_like(bright)
    for c in range(3):
        acc[..., c] = np.real(np.fft.ifft2(np.fft.fft2(bright[..., c]) * gsum))
    return hdr + strength * np.clip(acc, 0.0, None)


def _orbit_camera(radius: float, elevation_deg: float, azimuth_deg: float, fov_rad: float) -> dict:
    """An inclined camera at spherical (radius, elevation, azimuth) looking at the origin.

    ``elevation`` is the angle above the disk (equatorial) plane; 0 = edge-on,
    90 = top-down. The CKS world frame IS the coordinate frame (spin axis = +z),
    so the basis is used directly by ``render_beauty_frame``."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    pos = np.array(
        [radius * math.cos(el) * math.cos(az),
         radius * math.cos(el) * math.sin(az),
         radius * math.sin(el)],
        dtype=float,
    )
    fwd = -pos / np.linalg.norm(pos)               # look at origin
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)                       # true up (orthonormal)
    return {
        "frame": 0,
        "pos": pos.tolist(),
        "fwd": fwd.tolist(),
        "up": up.tolist(),
        "right": right.tolist(),
        "fov": float(fov_rad),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Inclined showcase still of the turbulent disk.")
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--radius", type=float, default=46.0, help="camera distance (M)")
    p.add_argument("--elevation", type=float, default=28.0, help="degrees above the disk plane")
    p.add_argument("--azimuth", type=float, default=0.0, help="degrees around the spin axis")
    p.add_argument("--fov-deg", type=float, default=48.0, help="vertical field of view")
    p.add_argument("--frame", type=int, default=24, help="frame index → t_disk swirl phase")
    p.add_argument("--exposure", type=float, default=None, help="override render.exposure")
    p.add_argument("--gamma", type=float, default=None, help="override render.gamma")
    p.add_argument("--no-noise", action="store_true", help="force disk.noise.enabled=false (A/B)")
    p.add_argument(
        "--noise-boost", type=float, default=1.0,
        help="showcase-only contrast multiplier on the noise AMPLITUDE dials (layer amps, "
        "m_max, temp/edge/height amps) so the sec.3 structure reads strongly. 1.0 = the "
        "committed production look; the geometry/g/g4 path is untouched.",
    )
    p.add_argument("--dynamism", type=float, default=None, help="override disk.noise.dynamism")
    p.add_argument("--doppler", type=float, default=None, help="override disk.doppler_strength (0=symmetric Interstellar look)")
    p.add_argument(
        "--peak-temp", type=float, default=None,
        help="override disk.target_peak_temperature (K). LOWER -> the blackbody chromaticity "
        "(Formula 9) sits in the saturated-amber band (~2500-3200) instead of pale cream "
        "(5500). Re-derives T_0 through the CKS-13 resolver. Look-dev hue dial; pure chroma.",
    )
    p.add_argument(
        "--bloom", type=float, default=0.0,
        help="post-process glow strength (stand-in for the Phase-3 Blender compositor glow). "
        "0 = off. ~0.3-0.8 gives the soft luminous Gargantua halo. Operates on the linear HDR.",
    )
    p.add_argument("--bloom-threshold", type=float, default=2.0, help="HDR-linear bright-pass cutoff feeding the bloom")
    # --- volumetric / close-up look-dev overrides (disk geometry + opacity) ---
    p.add_argument("--thickness", type=float, default=None, help="override disk.theta_half_width (puffier slab)")
    p.add_argument("--sigma-frac", type=float, default=None, help="override disk.vertical_sigma_frac")
    p.add_argument("--absorption", type=float, default=None, help="override disk.absorption_coeff (optical depth -> dark voids)")
    p.add_argument("--emission", type=float, default=None, help="override disk.emission_coeff")
    p.add_argument("--m-max", type=float, default=None, help="override disk.noise.m_max (density swing -> deeper holes)")
    # --- Ergonomic fast↔volumetric switch + self-shadow cost/accuracy preset ---
    p.add_argument(
        "--quality", choices=("fast", "balanced", "hero"), default="fast",
        help="render path preset. fast = noise+bloom only (~12s/4K, the video path). "
        "balanced/hero = volumetric void-carving (CKS-14 source fn + CKS-17 3D self-shadow) "
        "at a moderate / high deep-shadow-map grid (the cost knob is the grid resolution: "
        "bake is O(NU^2*NPHI*NZ)). Explicit --source-function/--self-shadow/--shadow-grid "
        "override the preset. Use this to switch paths and balance render time per shot.",
    )
    # --- V1 volumetric: CKS-14 source function + CKS-15 self-shadow (the voids) ---
    p.add_argument(
        "--source-function", action="store_true",
        help="enable disk.volumetric.source_function (CKS-14 RTE march). Pair with "
        "--self-shadow for the glowing-gas-with-voids look (materialises S so the "
        "shadow can carve it).",
    )
    p.add_argument(
        "--self-shadow", action="store_true",
        help="enable disk.volumetric.self_shadow (CKS-15 radial deep-shadow-map): dims "
        "emissivity by exp(-strength*tau_shadow) so dense clumps cast dark wakes outward.",
    )
    p.add_argument(
        "--shadow-strength", type=float, default=None,
        help="override disk.volumetric.self_shadow.strength (void depth; 0=off, higher=darker).",
    )
    p.add_argument(
        "--shadow-grid", default=None, metavar="NU,NPHI,NZ",
        help="override the deep-shadow-map resolution, e.g. --shadow-grid 128,512,24.",
    )
    # --- V3.0 volumetric: CKS-18 static curl-flow domain warp (the eddies) ---
    p.add_argument(
        "--curl", action="store_true",
        help="enable disk.noise.curl (CKS-18 in-plane divergence-free domain warp): "
        "warps the noise coords by the 2-D curl of an sfbm3 potential so the laminar "
        "sec.2 shear winds turbulent eddies into filaments. If --curl-amp is not given, a "
        "visible default amp (0.12) is applied.",
    )
    p.add_argument(
        "--curl-amp", type=float, default=None,
        help="override disk.noise.curl.amp (displacement amplitude; 0 -> identity). "
        "Try ~0.05-0.2. Implies --curl.",
    )
    p.add_argument(
        "--flow-period", type=float, default=None, metavar="T_C",
        help="override disk.noise.curl.flow_period_M (CKS-18 sec.2 curl-flow advection): "
        "the eddies BOIL in time with clock T_c (geometric M). >0 animates the warp "
        "across frames; <=0 (default) is the static V3.0 warp. Implies --curl. Try ~3-12.",
    )
    # --- P2 multi-phase media: CKS-19 emission rho_hot vs absorption rho_cold split (the silhouettes) ---
    p.add_argument(
        "--multiphase", action="store_true",
        help="enable disk.multiphase (CKS-19): decouple absorption density rho_cold from "
        "emission rho_hot so a cold dust phase carves dark SILHOUETTES into the glow. "
        "Pair with --absorption ~2 so the cold dust has optical depth. Toggling this "
        "forces a one-time kernel recompile (the _MP_COMPILE ti.static gate).",
    )
    p.add_argument(
        "--dust-correlation", type=float, default=None, metavar="CHI",
        help="override disk.multiphase.dust_correlation chi in [-1,1] (implies --multiphase). "
        "chi<0 anti-correlates: cold dust fills the hot voids -> darkest lanes. chi=-1 is "
        "full anti-correlation; chi=+1 makes rho_cold==rho_hot (pure dimming). Default -0.6.",
    )
    p.add_argument(
        "--dust-amp", type=float, default=None, metavar="A",
        help="override disk.multiphase.dust_amp (implies --multiphase): density swing of "
        "the cold modulator -> deeper silhouettes. Try ~1-2.5.",
    )
    p.add_argument(
        "--dust-sigma-frac", type=float, default=None, metavar="F",
        help="override disk.multiphase.dust_sigma_frac (implies --multiphase): cold-slab "
        "scale height as a fraction of the hot slab (thinner dust -> sharper lanes). Try ~0.5-1.",
    )
    p.add_argument(
        "--extinction", default=None, metavar="R,G,B",
        help="override disk.extinction_rgb (CKS-19 Task 7): per-channel kappa multiplier on "
        "absorption_coeff as 'r,g,b'. Grey '1,1,1' = neutral darkening. kB>kR (e.g. "
        "'0.6,1.0,1.6') reddens cold-dust lanes (astrophysical reddening). Pair with "
        "--absorption / --multiphase so there is optical depth to redden.",
    )
    # --- NON-PHYSICAL color grade (VISUALIZATION, like doppler_strength) — the warm-amber
    #     reference look. Overrides render.color_grade; identity defaults ⇒ ungraded. ---
    p.add_argument("--saturation", type=float, default=None,
                   help="override render.color_grade.saturation (1=unchanged, >1 richer amber)")
    p.add_argument("--tint", default=None,
                   help="override render.color_grade.tint as 'r,g,b' linear gain (warm amber ~ '1.15,1.0,0.8')")
    p.add_argument("--amber", action="store_true",
                   help="convenience preset: saturation 1.6 + warm tint 1.18,1.0,0.74 (the Interstellar amber grade)")
    p.add_argument("--save-hdr", default=None, help="also dump the raw HDR buffer to this .npy")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = tr.load_config()
    if args.peak_temp is not None:
        # Re-anchor the disk hue: target_peak_temperature is a BASE param; T_0 is
        # CKS-13-derived from it, so drop the stale derived T_0 and re-resolve.
        cfg["disk"]["target_peak_temperature"] = float(args.peak_temp)
        cfg["disk"].pop("T_0", None)
        cfg = tr.kerr_params.resolve_config(cfg)
    nz = cfg["disk"].setdefault("noise", {})
    if args.no_noise:
        nz["enabled"] = False
    if args.dynamism is not None:
        nz["dynamism"] = float(args.dynamism)
    if args.doppler is not None:
        cfg["disk"]["doppler_strength"] = float(args.doppler)
    if args.thickness is not None:
        cfg["disk"]["theta_half_width"] = float(args.thickness)
    if args.sigma_frac is not None:
        cfg["disk"]["vertical_sigma_frac"] = float(args.sigma_frac)
    if args.absorption is not None:
        cfg["disk"]["absorption_coeff"] = float(args.absorption)
    if args.emission is not None:
        cfg["disk"]["emission_coeff"] = float(args.emission)
    if args.m_max is not None:
        nz["m_max"] = float(args.m_max)
    # --quality preset: the ergonomic fast↔volumetric switch. Applied BEFORE the explicit
    # volumetric flags below so --source-function/--self-shadow/--shadow-grid still win.
    # The cost/accuracy knob is the deep-shadow-map grid (bake is O(NU²·NPHI·NZ)).
    vol = cfg["disk"].setdefault("volumetric", {})
    if args.quality in ("balanced", "hero"):
        vol["source_function"] = True
        ss = vol.setdefault("self_shadow", {})
        ss["enabled"] = True
        gnu, gnphi, gnz = (64, 192, 12) if args.quality == "balanced" else (128, 512, 24)
        ss["grid_nu"], ss["grid_nphi"], ss["grid_nz"] = gnu, gnphi, gnz
    # V1 volumetric flags (CKS-14 / CKS-15). Both default off in the YAML; these turn
    # them on for look-dev. The voids need BOTH (CKS-14 materialises the source
    # function S, CKS-15's self-shadow dims it → dark wakes), so --self-shadow without
    # --source-function only dims the emission, not the deep voids.
    if args.source_function:
        vol["source_function"] = True
    if args.self_shadow:
        vol.setdefault("self_shadow", {})["enabled"] = True
    if args.shadow_strength is not None:
        vol.setdefault("self_shadow", {})["strength"] = float(args.shadow_strength)
    if args.shadow_grid is not None:
        try:
            gnu, gnphi, gnz = (int(v) for v in args.shadow_grid.split(","))
        except ValueError:
            p.error("--shadow-grid must be three comma-separated ints, e.g. 128,512,24")
        sscfg = vol.setdefault("self_shadow", {})
        sscfg["grid_nu"], sscfg["grid_nphi"], sscfg["grid_nz"] = gnu, gnphi, gnz
    # V3.0 curl warp (CKS-18). Default OFF in the YAML; --curl / --curl-amp turn it on
    # for look-dev. enabled:false (or amp:0) is bit-identical to V2, so we set both.
    if args.curl or args.curl_amp is not None or args.flow_period is not None:
        curl = nz.setdefault("curl", {})
        curl["enabled"] = True
        curl["amp"] = float(args.curl_amp) if args.curl_amp is not None else 0.12
        # V3.1 (CKS-18 sec.2): >0 ⇒ eddies boil; <=0 ⇒ static V3.0 warp (bit-for-bit).
        if args.flow_period is not None:
            curl["flow_period_M"] = float(args.flow_period)
    # P2 multi-phase (CKS-19). Default OFF in the YAML; --multiphase / any --dust-* turn it
    # on for look-dev. enabled:false ⇒ ρ_cold≡ρ_hot ⇒ bit-identical to single-phase.
    if args.multiphase or args.dust_correlation is not None or args.dust_amp is not None \
            or args.dust_sigma_frac is not None:
        mp = cfg["disk"].setdefault("multiphase", {})
        mp["enabled"] = True
        if args.dust_correlation is not None:
            mp["dust_correlation"] = float(args.dust_correlation)
        if args.dust_amp is not None:
            mp["dust_amp"] = float(args.dust_amp)
        if args.dust_sigma_frac is not None:
            mp["dust_sigma_frac"] = float(args.dust_sigma_frac)
    # CKS-19 Task 7 chromatic extinction. Applies to the disk dtau regardless of
    # multiphase; grey '1,1,1' is bit-identical to the scalar march.
    if args.extinction is not None:
        try:
            ext = [float(v) for v in args.extinction.split(",")]
            if len(ext) != 3:
                raise ValueError
        except ValueError:
            p.error("--extinction must be three comma-separated floats, e.g. 0.6,1.0,1.6")
        cfg["disk"]["extinction_rgb"] = ext
    if args.noise_boost != 1.0:
        # Scale only the AMPLITUDE dials (how far the [0,1] envelopes swing), never the
        # frequencies/seed/geometry — pure look-dev re-upload through _setup_disk_noise.
        b = float(args.noise_boost)
        nz["m_max"] = float(nz.get("m_max", 2.5)) * b
        for layer in nz.get("layers", {}).values():
            if "amp" in layer:
                layer["amp"] = float(layer["amp"]) * b
        mod = nz.get("modulation", {})
        for k in ("temp_amp", "edge_in_amp", "edge_out_amp", "height_amp"):
            if k in mod:
                mod[k] = min(float(mod[k]) * b, 0.95)  # keep edge/height swings < 1 (r_in_eff≥r_isco floor still holds)

    rcfg = cfg["render"]
    exposure = args.exposure if args.exposure is not None else float(rcfg.get("exposure", 1.0))
    gamma = args.gamma if args.gamma is not None else float(rcfg.get("gamma", 2.2))

    # NON-PHYSICAL color grade (VISUALIZATION): config render.color_grade, overridable.
    grade = rcfg.get("color_grade", {}) or {}
    saturation = float(grade.get("saturation", 1.0))
    tint = [float(v) for v in grade.get("tint", (1.0, 1.0, 1.0))]
    if args.amber:  # convenience preset (overridden by explicit --saturation/--tint below)
        saturation, tint = 1.6, [1.18, 1.0, 0.74]
    if args.saturation is not None:
        saturation = float(args.saturation)
    if args.tint is not None:
        try:
            tint = [float(v) for v in args.tint.split(",")]
            if len(tint) != 3:
                raise ValueError
        except ValueError:
            p.error("--tint must be three comma-separated floats, e.g. 1.15,1.0,0.8")

    cam = _orbit_camera(args.radius, args.elevation, args.azimuth, math.radians(args.fov_deg))

    # D2.3 disk clock: t_disk = frame/fps · time_scale (geometric M, CKS-13-derived).
    dyn = cfg["disk"].get("dynamics") or {}
    t_disk = args.frame / float(rcfg["fps"]) * float(dyn.get("time_scale", 0.0))

    noise_on = bool(cfg["disk"].get("noise", {}).get("enabled", False))
    curl_cfg = cfg["disk"].get("noise", {}).get("curl", {})
    curl_on = bool(curl_cfg.get("enabled", False)) and float(curl_cfg.get("amp", 0.0)) != 0.0
    sf_on = bool(vol.get("source_function", False))
    ssh_on = bool(vol.get("self_shadow", {}).get("enabled", False))
    mp_cfg = cfg["disk"].get("multiphase", {})
    mp_on = bool(mp_cfg.get("enabled", False))
    mp_status = f"on(chi={mp_cfg.get('dust_correlation', -0.6)})" if mp_on else "off"
    ext_rgb = cfg["disk"].get("extinction_rgb", [1.0, 1.0, 1.0])
    ext_status = "grey" if list(ext_rgb) == [1.0, 1.0, 1.0] else str(list(ext_rgb))
    curl_flow = float(curl_cfg.get("flow_period_M", 0.0))
    curl_status = (
        f"on(amp={curl_cfg.get('amp')}"
        + (f",flow_T={curl_flow}" if curl_on and curl_flow > 0.0 else "")
        + ")"
    ) if curl_on else "off"
    if ssh_on and not sf_on:
        print("note: --self-shadow without --source-function only dims emission; the "
              "deep voids need both (CKS-14 materialises S, CKS-15 carves it).")
    print("initialising Taichi (ti.cuda) + uploading starmap mip pyramid ...")
    tr.setup_renderer(cfg)
    print(
        f"rendering {args.width}x{args.height}  incl={args.elevation}°  r={args.radius}M  "
        f"fov={args.fov_deg}°  noise={'on' if noise_on else 'off'}  curl={curl_status}  "
        f"source_fn={'on' if sf_on else 'off'}  self_shadow={'on' if ssh_on else 'off'}  "
        f"multiphase={mp_status}  extinction={ext_status}  t_disk={t_disk:.3f} ..."
    )
    t0 = time.time()
    beauty = tr.render_beauty_frame(
        cfg, cam, args.width, args.height, with_disk=True, lod_enabled=True, t_disk=t_disk,
    )
    beauty = np.nan_to_num(beauty)
    print(f"done in {time.time() - t0:.1f}s  beauty[max={beauty.max():.4g}  mean={beauty.mean():.4g}]")

    if args.save_hdr:
        np.save(args.save_hdr, beauty)
        print(f"Saved HDR buffer {args.save_hdr}")

    if args.bloom > 0.0:
        print(f"applying post bloom (strength={args.bloom}, threshold={args.bloom_threshold}) ...")
        beauty = _bloom(beauty, args.bloom, args.bloom_threshold)

    pixels = tr.tonemap(beauty, exposure, gamma, saturation=saturation, tint=tint)
    from PIL import Image

    out_path = Path(args.out) if args.out else _ROOT / "showcase_disk.png"
    Image.fromarray(pixels, mode="RGB").save(str(out_path))
    print(f"Saved {out_path}  ({args.width}x{args.height}, exposure={exposure}, gamma={gamma}, "
          f"saturation={saturation}, tint={tint})")


if __name__ == "__main__":
    main()
