"""One-off 8K hero still — the Gargantua arc with EVERY shipped technology on.

This is not part of the video pipeline; it is a single highest-quality beauty
frame that exercises the full feature stack through the production
``render_beauty_frame`` path:

  CKS-11 Page-Thorne flux profile      (disk.temperature_model = page_thorne)
  CKS-14 source-function RTE march     (disk.volumetric.source_function)
  CKS-15/17 3D self-shadow voids       (disk.volumetric.self_shadow, hero grid)
  CKS-16 flared 3D density             (disk.volumetric.flare)
  CKS-12 §2/§3 shear + modulation      (disk.noise + modulation, on by default)
  CKS-18 curl-flow domain warp         (disk.noise.curl)
  CKS-19 multiphase hot/cold dust      (disk.multiphase) + chromatic reddening (extinction_rgb)
  CKS-20 single-scatter HG rim-light   (disk.scatter)
  CKS-21 scale-dependent shear cascade (disk.noise.shear_cascade)   ← P1, newest
  CKS-22 Kelvin-Helmholtz edge erosion (disk.edge_erosion)          ← P4
  CKS-23 fractal LOD octave cascade    (disk.lod)                   ← P5
  Formula 13 DNGR hybrid starfield     (starfield.mode = dngr, on by default)

Composition follows the reference frames (Interstellar Gargantua): a near
edge-on camera so the far side of the disk gravitationally lenses up-and-over
the shadow into the iconic double arc, a bright core, a funnel-flared turbulent
disk, and a soft bloom halo. Colour is PHYSICAL — relativistic Doppler beaming
plus natural reddening from blue-heavy dust extinction — rather than an
artificial colour grade (the grade is identity by default here).

Physics follows skills/kerr-physics/SKILL.md; every numeric comes from
configs/render.yaml except the synthetic camera + look-dev dials below (all
VISUALIZATION-class, like doppler_strength / color_grade).

NOTE: starfield.mag_zero_point is baked into assets/stars.npy at INGEST, not
read at render — lower it in configs/render.yaml and re-run scripts/ingest_stars.py.

``--fast-compile`` is ON by default here: the full-feature mega-kernel (RK4
geodesics + CKS-14 march + CKS-19 split + CKS-20 scatter) is the heaviest kernel
in the project and a cold compile with Taichi's super-linear IR/CFG passes can
exceed 80 min. Disabling those passes trades a little single-frame RUNTIME for a
~minutes cold compile and produces the same image (the offline cache keys on the
flags, so re-runs with the same feature set skip the recompile entirely).

Usage
-----
    uv run python scripts/hero_4k.py                 # 7680x4320 (8K) full hero + .exr
    uv run python scripts/hero_4k.py --exposure 4.5 --bloom 0.6   # quick re-grade (cached kernel)
    uv run python scripts/hero_4k.py --width 1280 --height 720    # fast look-dev proxy
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
    """Soft multi-scale FFT-Gaussian glow on the linear HDR (the Gargantua halo;
    stand-in for the Phase-3 Blender compositor glow). Geometry/physics untouched."""
    h, w, _ = hdr.shape
    lum = hdr.max(axis=2, keepdims=True)
    bright = hdr * np.clip((lum - threshold) / (threshold + 1e-3), 0.0, 1.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r2 = fy * fy + fx * fx
    gsum = np.zeros_like(r2)
    for sigma, weight in ((6.0, 1.0), (20.0, 0.6), (60.0, 0.3)):
        gsum += weight * np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * r2)
    gsum /= (1.0 + 0.6 + 0.3)
    acc = np.empty_like(bright)
    for c in range(3):
        acc[..., c] = np.real(np.fft.ifft2(np.fft.fft2(bright[..., c]) * gsum))
    return hdr + strength * np.clip(acc, 0.0, None)


def write_rgb_exr(filename: str, rgb: np.ndarray) -> None:
    """Write the linear-HDR RGB buffer as a 3-channel float OpenEXR (OpenImageIO),
    the same writer the Phase-5 pipeline uses (scripts/export_exr.py)."""
    import OpenImageIO as oiio

    h, w = rgb.shape[:2]
    pixels = np.ascontiguousarray(rgb, dtype=np.float32)
    out = oiio.ImageOutput.create(filename)
    if out is None:  # pragma: no cover - environment guard
        raise RuntimeError(f"OpenImageIO could not create writer for {filename}: {oiio.geterror()}")
    spec = oiio.ImageSpec(w, h, 3, oiio.FLOAT)
    spec.channelnames = ("R", "G", "B")
    out.open(filename, spec)
    out.write_image(pixels)
    out.close()


def _orbit_camera(radius: float, elevation_deg: float, azimuth_deg: float, fov_rad: float) -> dict:
    """Inclined camera at spherical (radius, elevation, azimuth) looking at the origin.
    elevation = angle above the equatorial (disk) plane; 0 = edge-on, 90 = top-down.
    The CKS world frame IS the coordinate frame (spin axis = +z)."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    pos = np.array(
        [radius * math.cos(el) * math.cos(az),
         radius * math.cos(el) * math.sin(az),
         radius * math.sin(el)],
        dtype=float,
    )
    fwd = -pos / np.linalg.norm(pos)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return {"frame": 0, "pos": pos.tolist(), "fwd": fwd.tolist(),
            "up": up.tolist(), "right": right.tolist(), "fov": float(fov_rad)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="8K Gargantua hero still — all features on.")
    # Resolution (8K UHD)
    p.add_argument("--width", type=int, default=7680)
    p.add_argument("--height", type=int, default=4320)
    # Camera (Gargantua arc: near edge-on so the far side lenses over/under the shadow)
    p.add_argument("--radius", type=float, default=36.0, help="camera distance (M)")
    p.add_argument("--elevation", type=float, default=13.0, help="degrees above the disk plane (low = edge-on arc)")
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--fov-deg", type=float, default=38.0, help="vertical field of view")
    p.add_argument("--frame", type=int, default=30, help="frame index → t_disk swirl phase")
    # Look-dev / grade (VISUALIZATION dials; cheap to re-run — kernel is cached)
    p.add_argument("--exposure", type=float, default=5.0)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--bloom", type=float, default=0.55, help="post glow strength (Gargantua halo)")
    p.add_argument("--bloom-threshold", type=float, default=1.8)
    # Warm gold grade: low disk temperature + a moderate saturation/warm tint give the rich
    # amber-gold look (Interstellar-adjacent). The (desaturated) blackbody can't get there alone.
    p.add_argument("--saturation", type=float, default=1.6, help="colour-grade saturation (>1 = more vivid)")
    p.add_argument("--tint", default="1.22,0.8,0.55", help="linear gain 'r,g,b' (warm gold: boosts red, cuts blue)")
    p.add_argument("--doppler", type=float, default=0.5,
                   help="disk.doppler_strength: 0.5 = visible relativistic beaming asymmetry; "
                        "1.0 = full physics; ~0.1 = symmetric film look.")
    # Disk brightness / opacity balance (the darkening features must not crush the bright core)
    p.add_argument("--emission", type=float, default=14.0, help="disk.emission_coeff (brighter disk)")
    p.add_argument("--absorption", type=float, default=0.8,
                   help="disk.absorption_coeff — raised so blue-heavy extinction reddens naturally")
    p.add_argument("--shadow-strength", type=float, default=0.5, help="self-shadow void depth")
    p.add_argument("--thickness", type=float, default=0.07, help="disk.theta_half_width (thin ⇒ crisp edge-on Gargantua)")
    p.add_argument("--flare-beta", type=float, default=0.4,
                   help="CKS-16 flare exponent — the 'trumpet/funnel' flare (σ_θ∝(r/r_in)^β; >0 = funnels outward)")
    p.add_argument("--extinction", default="0.6,1.0,2.5",
                   help="disk.extinction_rgb κ⃗ — blue-heavy (B≫R) ⇒ natural dust reddening of transmitted light")
    p.add_argument("--dynamism", type=float, default=12.0, help="disk.noise.dynamism — shear swirl gain (Interstellar look)")
    p.add_argument("--m-max", type=float, default=3.5, help="disk.noise.m_max — log-density clamp (higher = more contrast)")
    p.add_argument("--noise-boost", type=float, default=1.5,
                   help="multiplier on the L0/L1/L2 noise-layer amplitudes (overall randomness level)")
    p.add_argument("--max-steps", type=int, default=8000,
                   help="render.max_steps_pipe_a — cranked high to kill march under-sampling artifacts (runtime arg, no recompile)")
    p.add_argument("--peak-temp", type=float, default=3000.0,
                   help="disk.target_peak_temperature (K) — lower ⇒ redder/fierier blackbody (resolve_config derives T_0)")
    p.add_argument("--shadow-nphi", type=int, default=2048,
                   help="self-shadow deep-shadow-map azimuthal bins — raise to kill the radial 'blade' banding")
    p.add_argument("--max-step-vfrac", type=float, default=0.35,
                   help="disk.max_step_vfrac — vertical step cap (lower = finer disk AA; needs higher --max-steps)")
    p.add_argument("--flow-period", type=float, default=6.0,
                   help="disk.noise.curl.flow_period_M (CKS-18 §2): >0 ⇒ eddies BOIL (form/stretch/merge); 0 = static")
    p.add_argument("--shutter-arc", type=float, default=0.0,
                   help="motion-blur azimuthal sweep (rad) about the spin axis ≈ disk rotational smear; 0 = off "
                        "(off by default — 4-sample MB ghosts the sharp point-stars into dotted trails)")
    p.add_argument("--no-fast-compile", action="store_true",
                   help="keep Taichi IR/CFG opt passes ON (production runtime, but the full-feature "
                        "kernel cold-compile can exceed 80 min). Default OFF = fast compile, same image.")
    p.add_argument("--out", default=None)
    p.add_argument("--save-exr", default=None,
                   help="path for the linear-HDR RGB OpenEXR dump (default <out stem>.exr; '' to skip)")
    args = p.parse_args(argv)

    cfg = tr.load_config()

    # --- compile strategy: fast compile by default (see module docstring) ---
    if not args.no_fast_compile:
        cfg["render"]["advanced_optimization"] = False
        cfg["render"]["cfg_optimization"] = False

    # --- integration quality (anti-aliasing): more steps + a tighter vertical step cap kill
    #     the disk-crossing march under-sampling moiré. Both are runtime args (no re-JIT). ---
    cfg["render"]["max_steps_pipe_a"] = int(args.max_steps)
    cfg["disk"]["max_step_vfrac"] = float(args.max_step_vfrac)

    # --- disk body / opacity ---
    cfg["disk"]["temperature_model"] = "page_thorne"   # CKS-11 physical flux profile
    cfg["disk"]["target_peak_temperature"] = float(args.peak_temp)  # lower ⇒ fierier red (resolve_config → T_0)
    cfg["disk"]["doppler_strength"] = float(args.doppler)
    cfg["disk"]["emission_coeff"] = float(args.emission)    # brighter disk core (dominates the frame)
    cfg["disk"]["absorption_coeff"] = float(args.absorption)  # cold dust optical depth (lanes), not opaque
    # CKS-19 Task 7: blue-heavy κ⃗ ⇒ light transmitted through the cold dust reddens NATURALLY
    # (blue absorbed more than red) — replaces the artificial amber colour grade.
    cfg["disk"]["extinction_rgb"] = [float(v) for v in args.extinction.split(",")]
    cfg["disk"]["theta_half_width"] = float(args.thickness)  # thin slab ⇒ crisp edge-on ring (not fog)

    # --- noise turbulence stack: cranked up overall for a much more dynamic, Interstellar-like disk ---
    nz = cfg["disk"].setdefault("noise", {})
    nz["enabled"] = True
    nz["dynamism"] = float(args.dynamism)              # stronger per-frame shear winding (CKS-12 §2)
    nz["m_max"] = float(args.m_max)                    # wider log-density clamp ⇒ higher-contrast filaments
    # Scale every density-layer amplitude (L0 base streaks / L1 clump-tear / L2 patchiness) by noise_boost.
    layers = nz.setdefault("layers", {})
    for lname, ldefault in (("base", 0.6), ("clump", 1.2), ("patch", 0.35)):
        layer = layers.setdefault(lname, {})
        layer["amp"] = float(layer.get("amp", ldefault)) * float(args.noise_boost)
    nz.setdefault("modulation", {})["enabled"] = True  # CKS-12 §3 (also required by edge erosion)
    # CKS-18 curl-flow domain warp (eddies wound into filaments by the §2 shear)
    curl = nz.setdefault("curl", {})
    curl["enabled"] = True
    curl["amp"] = 0.12
    curl["flow_period_M"] = float(args.flow_period)    # CKS-18 §2: boil the eddies over t_disk
    # CKS-21 scale-dependent shear cascade — coarse filaments wind, fine micro-vortices protected
    nz["shear_cascade"] = {"enabled": True, "shear_cutoff": 2.0, "shear_falloff": 2.0}

    # --- CKS-22 Kelvin-Helmholtz edge erosion (the rim TEARS into vacuum) ---
    cfg["disk"]["edge_erosion"] = {"enabled": True, "strength": 0.35,
                                   "freq_u": 4.0, "freq_phi": 12, "freq_z": 1.0, "octaves": 3,
                                   "soft_width": 0.0}

    # --- CKS-23 fractal LOD octave cascade (anti-aliased noise; full detail at this close range) ---
    cfg["disk"]["lod"] = {"enabled": True, "n_max": 6.0, "n_min": 2.0, "j0": 0.02}

    # --- CKS-14/15/16/17 volumetric (glowing-gas-with-voids + flared bulk + 3D self-shadow) ---
    vol = cfg["disk"].setdefault("volumetric", {})
    vol["source_function"] = True                      # CKS-14 RTE march (materialises S)
    ss = vol.setdefault("self_shadow", {})
    ss["enabled"] = True                               # CKS-15/17 3D deep-shadow-map → dark wakes
    ss["strength"] = float(args.shadow_strength)
    ss["grid_nu"], ss["grid_nphi"], ss["grid_nz"] = 128, int(args.shadow_nphi), 24   # hero grid (high nφ kills blades)
    flare = vol.setdefault("flare", {})
    flare["enabled"] = True                            # CKS-16 flared 3D density
    flare["beta"] = float(args.flare_beta)             # subtle outward flare (keep the thin Gargantua band)

    # --- CKS-19 multiphase hot/cold media (cold dust carves dark silhouette lanes) ---
    mp = cfg["disk"].setdefault("multiphase", {})
    mp["enabled"] = True
    mp["dust_correlation"] = -0.6                      # anti-correlated: dust in the cool gaps
    mp["dust_amp"] = 1.2
    mp["dust_sigma_frac"] = 1.0

    # --- CKS-20 single-scatter + Henyey-Greenstein rim-light (forward "silver lining") ---
    sc = cfg["disk"].setdefault("scatter", {})
    sc["enabled"] = True
    sc["albedo"] = 0.35                                # modest scatter (avoid net-darkening the edge-on disk)
    sc["hg_g"] = 0.6                                   # forward lobe
    sc["inner_glow"] = 2.0                             # compensate single-scatter net-darkening edge-on

    # Re-resolve derived params (T_0 etc.) in case base look params shifted.
    cfg = tr.kerr_params.resolve_config(cfg)

    rcfg = cfg["render"]
    tint = [float(v) for v in args.tint.split(",")]

    cam = _orbit_camera(args.radius, args.elevation, args.azimuth, math.radians(args.fov_deg))
    dyn = cfg["disk"].get("dynamics") or {}
    t_disk = args.frame / float(rcfg["fps"]) * float(dyn.get("time_scale", 0.0))

    print("initialising Taichi (ti.cuda) + uploading starmap mip pyramid + baking 3D self-shadow ...")
    print(f"  fast_compile={'OFF (production)' if args.no_fast_compile else 'ON'}  "
          f"(first cold compile of the full-feature kernel may take a few minutes)")
    tr.setup_renderer(cfg)
    mb_samples = int(rcfg["motion_blur_samples"])
    mb = args.shutter_arc > 0.0 and mb_samples > 1
    print(f"rendering {args.width}x{args.height}  incl={args.elevation}°  r={args.radius}M  "
          f"fov={args.fov_deg}°  doppler={args.doppler}  t_disk={t_disk:.3f}  "
          f"flow_period={args.flow_period}M  "
          f"motion_blur={'%d samples, arc %.3frad' % (mb_samples, args.shutter_arc) if mb else 'off'}")
    print("  features: page_thorne + source_fn + self_shadow(hero) + flare + curl-boil + "
          "shear_cascade + edge_erosion + lod + multiphase + extinction + scatter + dngr"
          + (" + motion_blur" if mb else ""))
    t0 = time.time()
    if mb:
        # render.motion_blur_samples sub-frames, camera swept ±arc/2 about +z, averaged
        # (≈ the disk's rotational smear). 4× the trace + self-shadow-bake cost.
        beauty = tr.render_beauty_frame_mb(
            cfg, cam, args.width, args.height, shutter_arc=float(args.shutter_arc),
            with_disk=True, lod_enabled=True, t_disk=t_disk,
        )
    else:
        beauty = tr.render_beauty_frame(
            cfg, cam, args.width, args.height, with_disk=True, lod_enabled=True, t_disk=t_disk,
        )
    beauty = np.nan_to_num(beauty)
    print(f"done in {time.time() - t0:.1f}s  beauty[max={beauty.max():.4g}  mean={beauty.mean():.4g}]")

    out_path = Path(args.out) if args.out else _ROOT / "hero_blackhole_8k.png"

    # Linear-HDR EXR dump (pre-bloom, pre-tonemap radiance) — replaces the old .npy.
    exr_path = args.save_exr if args.save_exr is not None else str(out_path.with_suffix(".exr"))
    if exr_path:
        write_rgb_exr(exr_path, beauty)
        print(f"saved linear-HDR EXR {exr_path}")

    if args.bloom > 0.0:
        print(f"applying post bloom (strength={args.bloom}, threshold={args.bloom_threshold}) ...")
        beauty = _bloom(beauty, args.bloom, args.bloom_threshold)

    pixels = tr.tonemap(beauty, args.exposure, args.gamma, saturation=args.saturation, tint=tint)
    from PIL import Image

    Image.fromarray(pixels, mode="RGB").save(str(out_path))
    print(f"saved {out_path}  ({args.width}x{args.height}, exposure={args.exposure}, "
          f"gamma={args.gamma}, saturation={args.saturation}, tint={tint})")


if __name__ == "__main__":
    main()
