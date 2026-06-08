# Sources — Sub-domain B (Procedural / Noise-Based Disk Generation)

## Overview
This file documents primary and secondary sources for procedural and noise-based accretion disk rendering, including FBM, curl/flow noise, flow maps, real-time shader techniques, and game approximations.

---

### SB1 — Curl-noise for procedural fluid flow
- **Authors:** Robert Bridson, Jim Houriham, Marcus Nordenstam
- **Year:** 2007
- **Venue:** SIGGRAPH 2007 (34th Annual Conference on Computer Graphics and Interactive Techniques)
- **Type:** [peer-reviewed]
- **URL:** https://dl.acm.org/doi/10.1145/1275808.1276435
- **Sub-domain:** B (procedural fluid flow, curl noise foundations)
- **Relevance:** Foundational work on curl noise for turbulent fluid procedural generation; incompressible, boundary-respecting velocity fields without full simulation.
- **Key excerpts (verbatim):**
  > "exactly incompressible (necessary for the characteristic look of everyday fluids)"
  > "exactly respects solid boundaries (not allowing fluid to flow through arbitrarily-specified surfaces)"
  > "offers an extremely simple approach to efficiently generating turbulent velocity fields based on Perlin noise"
- **Notes:** This is the canonical curl noise paper; highly relevant for procedural disk animation and flow field generation. Authors from UBC, Tweak Films, and Double Negative (relevant to Interstellar pipeline).

---

### SB2 — Improving noise
- **Authors:** Ken Perlin
- **Year:** 2002
- **Venue:** SIGGRAPH 2002 (29th Annual Conference on Computer Graphics and Interactive Techniques)
- **Type:** [peer-reviewed]
- **URL:** https://history.siggraph.org/learning/improving-noise/
- **Sub-domain:** B (foundational noise algorithm improvements)
- **Relevance:** Improved Perlin noise addressing interpolation discontinuity and gradient computation; fundamental to all modern procedural texture and disk generation.
- **Key excerpts (verbatim):**
  > "addresses two specific problems in the original Noise algorithm: second order interpolation discontinuity and unoptimal gradient computation"
  > "results in noise that both looks better and runs faster"
  > "enables a uniform mathematical reference standard for the algorithm"
- **Notes:** Essential foundation for all FBM and noise-based procedural techniques. Builds on Perlin's original 1985 work.

---

### SB3 — The Book of Shaders: Fractal Brownian Motion
- **Authors:** Unknown (community resource)
- **Year:** unknown
- **Venue:** Online educational resource (https://thebookofshaders.com/)
- **Type:** [official-docs]
- **URL:** https://thebookofshaders.com/13/
- **Sub-domain:** B (FBM implementation and concepts)
- **Relevance:** Comprehensive tutorial on FBM techniques including octaves, lacunarity, and gain parameters for procedural texture generation.
- **Key excerpts (verbatim):**
  > "By adding different iterations of noise (octaves), where we successively increment the frequencies in regular steps (lacunarity) and decrease the amplitude (gain) of the noise we can obtain a finer granularity."
  > "Octaves: Progressive layers of noise that add detail to the output. Increasing octaves creates increasingly complex patterns with visible self-similarity."
  > "Gain: The amplitude reduction applied at each octave, typically around 0.5 to control how much each layer contributes."
- **Notes:** Excellent educational resource with code examples; practical guide for shader implementation of FBM-based disk generation.

---

### SB4 — Inigo Quilez: Fractional Brownian Motion
- **Authors:** Iñigo Quilez
- **Year:** unknown (active resource)
- **Venue:** iquilezles.org (personal technical blog)
- **Type:** [blog]
- **URL:** https://iquilezles.org/articles/fbm/
- **Sub-domain:** B (FBM theory, implementation, spectral analysis)
- **Relevance:** Detailed technical article on FBM with spectral characteristics, implementation variants, and applications to procedural terrain and natural textures.
- **Key excerpts (verbatim):**
  > "A Fractional Brownian Motion is a similar process in which the increments are not completely independent from each other but feature correlated memory characteristics."
  > "The Hurst Exponent (H) ranges from 0-1, controlling the statistical self-similarity and smoothness."
  > "Most graphics programmers use G=0.5 (equivalent to H=1) for natural-looking terrain, as this produces isotropic scaling in all directions."
  > "natural mountain profiles exhibit -9dB/octave frequency decay (yellow noise), matching fBM with H=1, validating this choice for realistic landscape generation."
- **Notes:** Authoritative resource from Shadertoy creator; includes spectral analysis relevant to disk density variations.

---

### SB5 — Inigo Quilez: Domain Warping
- **Authors:** Iñigo Quilez
- **Year:** unknown (active resource)
- **Venue:** iquilezles.org (personal technical blog)
- **Type:** [blog]
- **URL:** https://iquilezles.org/articles/warp/
- **Sub-domain:** B (domain warping for procedural animation)
- **Relevance:** Comprehensive guide to domain warping technique for creating organic, flowing distortions by modifying input coordinates to noise functions.
- **Key excerpts (verbatim):**
  > "Domain warping uses noise to distort the coordinate before sampling noise again."
  > "Each pixel's sampling location gets pushed by the noise field itself, producing curling, spiral-like shapes."
  > "Animating the warp field transforms the pattern, and warped animation produces motion that looks like the texture is alive, folding, stretching, and breathing."
- **Notes:** Highly relevant for animated disk structures; demonstrates recursive nesting for organic deformation effects.

---

### SB6 — Inigo Quilez: Noise (gradient and value noise)
- **Authors:** Iñigo Quilez
- **Year:** unknown (active resource)
- **Venue:** iquilezles.org (personal technical blog)
- **Type:** [blog]
- **URL:** https://iquilezles.org/articles/morenoise/
- **Sub-domain:** B (noise algorithm variants, voronoise)
- **Relevance:** Technical comparison of value noise vs. gradient noise; introduces voronoise (blend between Perlin and Voronoi) for procedural pattern generation.
- **Key excerpts (verbatim):**
  > "examples demonstrate the differences between value noise and gradient noise"
  > "using two-dimensional noise to rotate the space where straight lines are rendered can produce swirly effects that look like wood"
  > "Voronoise function allows a gradual blend between regular noise and voronoi"
  > "analytical derivatives computation is much faster and more accurate than the central differences method"
- **Notes:** Noise derivatives section particularly relevant for disk shading variations.

---

### SB7 — A cellular texture basis function
- **Authors:** Steven Worley
- **Year:** 1996
- **Venue:** SIGGRAPH 1996
- **Type:** [peer-reviewed]
- **URL:** https://dl.acm.org/doi/pdf/10.1145/237170.237267
- **Sub-domain:** B (Worley/cellular noise, procedural texturing)
- **Relevance:** Original paper introducing Worley (cellular/Voronoi) noise for procedural texture basis; foundational for organic cracked/crusty surface patterns.
- **Key excerpts (verbatim):**
  > "A cellular texture basis function based on a partitioning of space into a random array of cells"
  > "has been used to produce textured surfaces resembling flagstone-like tiled areas, organic crusty skin, crumpled paper, ice, rock, mountain ranges, and craters"
  > "outputs a real value at a given coordinate that corresponds to the distance of the nth nearest seed (usually n=1)"
- **Notes:** Canonical source for cellular noise; applicable to disk surface detail and roughness patterns.

---

### SB8 — Efficient computational noise in GLSL
- **Authors:** Unknown
- **Year:** 2012
- **Venue:** arXiv / GLSL technical reference
- **Type:** [unknown]
- **URL:** https://arxiv.org/pdf/1204.1461
- **Sub-domain:** B (GPU shader noise implementation)
- **Relevance:** Technical paper on efficient Perlin and simplex noise implementations for GPU shaders (GLSL 1.20+).
- **Key excerpts (verbatim):**
  > "Perlin noise, developed by Ken Perlin in the 1980s, is one of the most widely used noise functions in computer graphics"
  > "Simplex noise is generally more efficient than Perlin noise when you need noise in three or more dimensions"
  > "GLSL implementations of Perlin noise and simplex noise can run fast enough for practical use on current generation GPU hardware"
  > "purely computational—using neither textures nor lookup tables"
- **Notes:** Critical for GPU-based disk rendering; specifies GLSL 1.20 compatibility with modern platforms.

---

### SB9 — The Book of Shaders: Cellular Noise
- **Authors:** Unknown (community resource)
- **Year:** unknown
- **Venue:** Online educational resource (https://thebookofshaders.com/)
- **Type:** [official-docs]
- **URL:** https://thebookofshaders.com/12/
- **Sub-domain:** B (Worley/cellular noise tutorial)
- **Relevance:** Educational guide to Worley/cellular noise implementation and applications in GLSL shaders.
- **Key excerpts (verbatim):**
  > "Worley noise, also called Voronoi noise and cellular noise"
  > "noise interpolates/averages random values (as in value noise) or gradients (as in gradient noise), while Voronoi computes the distance to the closest feature point"
- **Notes:** Practical shader code examples for procedural texture generation; complements Worley 1996 paper.

---

### SB10 — A Real-time High-quality Black Hole Shader
- **Authors:** Eric Bruneton
- **Year:** 2020
- **Venue:** Personal technical portfolio
- **Type:** [blog]
- **URL:** https://ebruneton.github.io/black_hole_shader/
- **Sub-domain:** B (real-time shader implementation for accretion disks)
- **Relevance:** Real-time black hole shader with accretion disk rendering, demonstrating beam tracing with precomputed tables for interactive performance.
- **Key excerpts (verbatim):**
  > "beam tracing with a distinctive optimization: precomputed tables to find the intersections of each curved light beam with the scene in constant time per pixel"
  > "custom shading model for the accretion disk that integrates relativistic effects"
  > "accounts for Doppler color shifts (blue-shift ahead, red-shift behind objects in motion)"
  > "uses a specific texture filtering scheme to integrate the contribution of the light sources to each beam"
- **Notes:** Directly relevant to real-time disk approximation; demonstrates integration of relativistic effects with shader-based rendering.

---

### SB11 — SpaceEngine: General Relativity 3: Volumetric Accretion Disks
- **Authors:** Unknown (SpaceEngine development team)
- **Year:** 2022
- **Venue:** SpaceEngine official blog/news
- **Type:** [blog]
- **URL:** https://spaceengine.org/news/blog220830/
- **Sub-domain:** B (volumetric procedural disk generation, upscaling, animation)
- **Relevance:** Commercial real-time implementation of volumetric accretion disk rendering with procedural noise and performance optimization techniques.
- **Key excerpts (verbatim):**
  > "the shaders have to integrate the emission and opacity of multiple points of a volume along a ray"
  > "performs geodesic ray-tracing through warped spacetime, integrates disk brightness and opacity, stores results in two low-resolution textures: deflection vectors and disk brightness/opacity"
  > "procedurally-generated and animated noise to simulate rotating plasma cloud formations around black holes"
  > "animated noise reduces banding artifacts during the upscaling phase"
  > "applies multiple upscaling filters (linear, bicubic, Lanczos) with AMD FidelityFX Contrast Adaptive Sharpening (CAS)"
- **Notes:** Demonstrates two-pass volumetric rendering strategy; procedural noise combined with upscaling for real-time performance.

---

### SB12 — Flow Map Shader Techniques
- **Authors:** Multiple (game dev community)
- **Year:** 2010+ (Valve SIGGRAPH 2010 presentation referenced)
- **Venue:** Game engine documentation and tutorials (Medium, Unity, GDC)
- **Type:** [blog]
- **URL:** https://louisgamedev.medium.com/shader-tutorial-flow-map-4410af832a8d
- **Sub-domain:** B (flow maps for animated procedural effects)
- **Relevance:** Tutorial on flow-map shader technique for animating procedural textures by encoding velocity information; applicable to disk rotation and swirl patterns.
- **Key excerpts (verbatim):**
  > "A flow-map shader animates UV mapping by using a specially crafted texture encoded with velocity information"
  > "The flowmap simply adds a number to the UV coordinate so it's accessing other pixels"
  > "usually used with two normal maps to simulate water"
  > "first presented at SIGGRAPH 2010 by Valve"
- **Notes:** Historical reference to Valve's flow map work; applicable to procedural disk animation without full simulation.

---

### SB13 — Real-Time Volumetric Clouds with Ray Marching
- **Authors:** Sushil Gangaraju
- **Year:** 2023
- **Venue:** Medium technical blog
- **Type:** [blog]
- **URL:** https://medium.com/@sushilgangaraju/real-time-volumetric-clouds-with-ray-marching-c6b46d1edeb0
- **Sub-domain:** B (volumetric rendering, ray marching with procedural noise)
- **Relevance:** Detailed explanation of ray marching volumetric rendering using procedural 3D noise patterns; techniques applicable to disk rendering.
- **Key excerpts (verbatim):**
  > "Volumetric fog in ray marching is achieved by stepping through the scene and accumulating fog density based on distance"
  > "Light Scattering – Simulates god rays by accumulating light along the ray path"
  > "Procedural Noise Fog – Uses random noise to create a more natural, rolling mist effect"
  > "common method of rendering volumetrics with ray marching is through the use of 3D volume textures"
  > "blue noise pattern, which has fewer patterns or clumps than other noises and is less visible to the human eye"
- **Notes:** Practical ray marching techniques directly applicable to volumetric disk generation.

---

### SB14 — Ray Marching and Signed Distance Functions
- **Authors:** Jamie Wong
- **Year:** 2016
- **Venue:** Personal technical blog
- **Type:** [blog]
- **URL:** https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
- **Sub-domain:** B (ray marching fundamentals, sphere tracing)
- **Relevance:** Comprehensive tutorial on ray marching using signed distance fields (SDF) and sphere tracing for efficient volumetric rendering.
- **Key excerpts (verbatim):**
  > "A distance field is a function that gives an estimate (a lower bound) of the distance to the closest surface at any point in space"
  > "Sphere tracing is a sophisticated optimization of basic ray marching that uses signed distance functions to take larger, safer steps along the ray"
  > "Instead of using fixed small steps, sphere tracing queries the distance field at each point and uses that distance as the maximum safe step size"
- **Notes:** Foundational resource for sphere tracing optimization in volumetric disk rendering.

---

### SB15 — Gravitational Lensing by Spinning Black Holes in Astrophysics, and in the Movie Interstellar
- **Authors:** Oliver James, Eugenie von Tunzelmann, Paul Franklin, Kip S. Thorne
- **Year:** 2015
- **Venue:** Classical and Quantum Gravity, Volume 32, Article 065001
- **Type:** [peer-reviewed]
- **URL:** https://arxiv.org/abs/1502.03808
- **Sub-domain:** B (DNGR renderer disk visualization context, ray-bundle techniques)
- **Relevance:** Interstellar DNGR (Double Negative Gravitational Renderer) paper; while focused on gravitational lensing physics, includes disk rendering approach context.
- **Key excerpts (verbatim):**
  > "develop DNGR (Double Negative Gravitational Renderer), specialized software for rendering black hole imagery"
  > "solve the equations for ray-bundle (light-beam) propagation through the curved spacetime of a spinning (Kerr) black hole"
  > "ray-bundle techniques were crucial for achieving IMAX-quality smoothness without flickering"
  > "When a ray originates on the surface of an accretion disk, the code integrates the null geodesic equation backward from the camera until it hits the disk's surface, deducing the map from a point on the disk's surface to one on the camera's sky"
- **Notes:** While this paper is primarily about gravitational lensing physics rather than procedural techniques, it documents the DNGR disk rendering integration context (see SEED.md for potential specific disk procedural/visualization details).

---

## Coverage Summary

**Procedural Noise Foundations:** Perlin noise improvements (SB2), FBM theory (SB3, SB4), domain warping (SB5), noise variants (SB6), Worley cellular noise (SB7, SB9).

**GPU Implementation:** Efficient GLSL noise (SB8), real-time black hole shader (SB10), volumetric rendering (SB11, SB13), ray marching (SB14).

**Procedural Animation:** Curl noise (SB1), flow maps (SB12), animated procedural noise (SB11).

**VFX Context:** DNGR Interstellar disk rendering (SB15).

**Notable Gaps:** Limited coverage of specific game engine implementations (Unreal, Godot), detailed shader source code for disk-specific techniques, spectral/temperature color models for procedural disk emission.
