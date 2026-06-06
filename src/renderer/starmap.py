"""16K equirectangular starmap: loader + in-memory mip pyramid.

Pipe A traces a photon backward from the camera; rays that escape the Kerr
potential hit the background sky. The sky is a 16K equirectangular HDRI
(``starmap_2020_16k.exr``, 16384x8192, half-float RGB). Near the photon ring a
single pixel's ray bundle smears across a large solid angle on the sky, so
point-sampling the full-res map aliases badly (flicker). Formula 10 fixes this
with a differential-ray mip LOD: estimate the on-sky footprint of the pixel from
an offset ray's Jacobian and sample a pre-filtered mip level.

This module owns the *host-side* work:
  * load the base level (f32 working copy from the f16 file),
  * build a box-filtered mip pyramid down to 1x1, stored f16 to fit VRAM,
  * a reference host sampler (trilinear over mips, bilinear within a level) used
    by tests and for cross-checking the Taichi sampler.

The Taichi renderer uploads these levels into GPU fields; the math here is the
single source of truth for that GPU sampler so the two stay in agreement.

Equirectangular convention (matches the geodesic's CKS-10 exit direction):
    u = phi' / (2*pi)   in [0, 1)   -> column
    v = theta' / pi     in [0, 1]   -> row   (theta'=0 north pole at v=0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


def load_equirect(path: str) -> np.ndarray:
    """Load an equirectangular EXR as an ``(H, W, 3)`` float32 array.

    Uses OpenImageIO. Text/EXR I/O on this box must never rely on the platform
    default encoding (it is cp949 here); OIIO reads bytes directly so that is a
    non-issue, but we keep all *config* reads explicitly utf-8 elsewhere.
    """
    import OpenImageIO as oiio  # imported lazily so non-render tooling needn't have it

    inp = oiio.ImageInput.open(path)
    if inp is None:  # pragma: no cover - asset-presence guard
        raise FileNotFoundError(f"could not open starmap EXR: {path}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        h, w, c = spec.height, spec.width, spec.nchannels
        pixels = inp.read_image(format=oiio.FLOAT)
    finally:
        inp.close()

    arr = np.asarray(pixels, dtype=np.float32).reshape(h, w, c)
    return np.ascontiguousarray(arr[:, :, :3])


def build_mip_pyramid(base_rgb: np.ndarray) -> list[np.ndarray]:
    """Box-filter mip pyramid from a base ``(H, W, 3)`` image, down to 1x1.

    Level 0 is the full-resolution image. Each subsequent level halves both
    dimensions with a 2x2 box average. Levels are stored as float16 to keep the
    whole pyramid near ~1.07x the base size (base 16K RGB f16 ~= 0.8 GB), which
    fits the 6 GB VRAM budget. The box average is computed in float32 then cast,
    so repeated halving does not accumulate f16 rounding.
    """
    base32 = np.ascontiguousarray(base_rgb, dtype=np.float32)
    levels = [base32.astype(np.float16)]

    cur = base32
    while min(cur.shape[0], cur.shape[1]) > 1:
        h, w = cur.shape[:2]
        h2, w2 = h // 2, w // 2
        # Trim any odd row/col before the 2x2 fold (equirect dims here are powers of 2).
        c = cur[: 2 * h2, : 2 * w2, :]
        folded = 0.25 * (
            c[0::2, 0::2, :] + c[1::2, 0::2, :] + c[0::2, 1::2, :] + c[1::2, 1::2, :]
        )
        levels.append(folded.astype(np.float16))
        cur = folded

    return levels


@dataclass
class Starmap:
    """Loaded starmap with its mip pyramid and derived constants."""

    levels: list[np.ndarray]   # level 0 = full res, each (Hk, Wk, 3) float16
    width: int                 # base width (== levels[0].shape[1])
    height: int                # base height
    max_lod: float             # log2(width) — the LOD clamp ceiling (Formula 10)

    @classmethod
    def load(cls, path: str) -> "Starmap":
        base = load_equirect(path)
        levels = build_mip_pyramid(base)
        h, w = levels[0].shape[:2]
        return cls(levels=levels, width=w, height=h, max_lod=math.log2(w))

    # --- reference host sampler (ground truth for the Taichi GPU sampler) -----

    def _sample_level_bilinear(self, level: int, u: float, v: float) -> np.ndarray:
        """Bilinear sample of one mip level. u wraps (phi periodic); v clamps."""
        img = self.levels[level].astype(np.float32)
        h, w = img.shape[:2]

        # Pixel-center sampling: map [0,1) -> [-0.5, n-0.5].
        fu = (u % 1.0) * w - 0.5
        fv = min(max(v, 0.0), 1.0) * h - 0.5

        u0 = math.floor(fu)
        v0 = math.floor(fv)
        du = fu - u0
        dv = fv - v0

        x0 = int(u0) % w
        x1 = (x0 + 1) % w          # wrap in phi
        y0 = min(max(int(v0), 0), h - 1)
        y1 = min(max(int(v0) + 1, 0), h - 1)  # clamp in theta

        c00 = img[y0, x0]
        c10 = img[y0, x1]
        c01 = img[y1, x0]
        c11 = img[y1, x1]
        top = c00 * (1 - du) + c10 * du
        bot = c01 * (1 - du) + c11 * du
        return top * (1 - dv) + bot * dv

    def sample(self, u: float, v: float, lod: float) -> np.ndarray:
        """Trilinear sample: bilinear within two bracketing mip levels, lerp by lod.

        ``lod`` is clamped to ``[0, max_lod]``; levels beyond the pyramid tail are
        clamped to the smallest available level.
        """
        lod = min(max(lod, 0.0), self.max_lod)
        n = len(self.levels)
        l0 = int(math.floor(lod))
        l1 = min(l0 + 1, n - 1)
        l0 = min(l0, n - 1)
        f = lod - math.floor(lod)
        c0 = self._sample_level_bilinear(l0, u, v)
        c1 = self._sample_level_bilinear(l1, u, v)
        return c0 * (1 - f) + c1 * f


def celestial_to_uv(dx: float, dy: float, dz: float) -> tuple[float, float]:
    """Map an escaped ray's CKS celestial direction to equirect (u, v) — CKS-10.

    Under Cartesian Kerr-Schild an escaped photon (``rho >= r_max``) lives in the
    asymptotically-flat region, where its contravariant spatial momentum
    direction ``d = (dx, dy, dz)`` IS the incoming sky direction. The equirect
    lookup is then a plain spherical projection:

        theta' = acos(clamp(d_z / |d|, -1, 1))
        phi'   = atan2(d_y, d_x)
        u = wrap(phi' / 2pi),   v = clamp(theta' / pi, 0, 1)

    ``d`` is a genuine Cartesian unit vector for EVERY ray, so the BL spin-axis
    seam, the phi-accumulation blow-up, and the old ``normalize_sphere_angles``
    punch-through fold are all gone (SKILL.md CKS-10). The only residual pole
    effect is the ordinary equirect-texture coordinate at theta'=0,pi, handled by
    the phi-wrap below.
    """
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:                       # degenerate guard (never happens for a ray)
        return 0.0, 0.0
    theta = math.acos(min(max(dz / norm, -1.0), 1.0))
    phi = math.atan2(dy, dx)
    u = (phi / TWO_PI) % 1.0              # wrap azimuth into [0, 1)
    v = min(max(theta / math.pi, 0.0), 1.0)
    return u, v
