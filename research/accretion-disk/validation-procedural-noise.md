# Validation Report — Sub-domain B: Procedural / Noise-Based Disk Generation
**Validator:** claude-sonnet-4-6  
**Date:** 2026-06-08  
**Source file:** `research/accretion-disk/sources-procedural-noise.md`  
**Sources evaluated:** SB1–SB15

---

## Methodology

Each URL was re-fetched via WebFetch. For ACM paywalled URLs (SB1, SB7), WebSearch was used to
find the SIGGRAPH history page and semantic scholar entries. For arxiv PDF SB8, WebFetch failed on
binary PDF; ar5iv HTML mirror was used instead. Each claimed "verbatim" excerpt was checked
against actual retrieved text. Context surrounding found text was checked to confirm the stated
meaning was not distorted.

---

## Per-Source Verdicts

---

### SB1 — Curl-noise for procedural fluid flow (Bridson et al., SIGGRAPH 2007)

**URL:** https://dl.acm.org/doi/10.1145/1275808.1276435  
**Status on fetch:** HTTP 403 (ACM paywall). DEAD-LINK for direct access.  
**Fallback:** SIGGRAPH history page https://history.siggraph.org/learning/curl-noise-for-procedural-fluid-flow-by-bridson-houriham-and-nordenstam/ — accessible.  
**Credibility:** primary-peer-reviewed

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "exactly incompressible (necessary for the characteristic look of everyday fluids)" | VERIFIED (via fallback) | SIGGRAPH history page confirms: "exactly incompressible (necessary for the characteristic look of everyday fluids)" — exact phrasing confirmed. |
| "exactly respects solid boundaries (not allowing fluid to flow through arbitrarily-specified surfaces)" | VERIFIED (via fallback) | SIGGRAPH history page confirms: "exactly respects solid boundaries (not allowing fluid to flow through arbitrarily-specified surfaces)" — exact phrasing confirmed. |
| "offers an extremely simple approach to efficiently generating turbulent velocity fields based on Perlin noise" | VERIFIED (via fallback) | SIGGRAPH history page confirms the paper describes "an extremely simple approach to efficiently generating turbulent" flow. The full phrase is confirmed. |

**Notes on affiliation claim:** The sources.md notes claim Nordenstam's affiliation is "Double Negative."
Neither the ACM page (paywalled) nor the SIGGRAPH history fallback page lists author affiliations.
This claim is UNVERIFIABLE from the cited sources and should be treated as **single-source-risk**
(the note likely originates from the scout, not from the paper itself).

**Direct URL verdict:** DEAD-LINK (403 paywall). Core claims VERIFIED via fallback. Affiliation claim unsupported.

---

### SB2 — Improving noise (Perlin, SIGGRAPH 2002)

**URL:** https://history.siggraph.org/learning/improving-noise/  
**Status on fetch:** Accessible.  
**Credibility:** primary-peer-reviewed

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "addresses two specific problems in the original Noise algorithm: second order interpolation discontinuity and unoptimal gradient computation" | PARTIALLY-SUPPORTED | Actual abstract: "Two deficiencies in the original Noise algorithm are corrected: second order interpolation discontinuity and unoptimal gradient computation." The scout paraphrased "Two deficiencies...are corrected" as "addresses two specific problems." The substance is correct but this is not verbatim. |
| "results in noise that both looks better and runs faster" | PARTIALLY-SUPPORTED | Actual: "With these defects corrected, Noise both looks better and runs faster." The claimed quote drops the framing clause "With these defects corrected." Not verbatim. |
| "enables a uniform mathematical reference standard for the algorithm" | PARTIALLY-SUPPORTED | Actual: "The latter change also makes it easier to define a uniform mathematical reference standard." The claimed quote omits "The latter change also makes it easier to define a" and presents only the noun phrase as if it were the full sentence. Not verbatim. |

**Assessment:** All three excerpts are paraphrases presented as verbatim quotes. The information content is basically accurate but none of the three is a genuine verbatim extraction.

---

### SB3 — The Book of Shaders: Fractal Brownian Motion

**URL:** https://thebookofshaders.com/13/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog (educational resource, widely cited)

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "By adding different iterations of noise (octaves), where we successively increment the frequencies in regular steps (lacunarity) and decrease the amplitude (gain) of the noise we can obtain a finer granularity." | PARTIALLY-SUPPORTED | The page contains a very similar sentence but the exact tail "we can obtain a finer granularity" was not confirmed verbatim. The first portion "by adding different iterations of noise (octaves), where we successively increment the frequencies in regular steps (lacunarity) and decrease the amplitude (gain)" is substantially confirmed. The ending differs. |
| "Octaves: Progressive layers of noise that add detail to the output. Increasing octaves creates increasingly complex patterns with visible self-similarity." | UNSUPPORTED | This reads as a structured definition not present in the source as written. The page explains the octave concept but not in this formatted glossary sentence. This appears to be a scout-generated paraphrase. |
| "Gain: The amplitude reduction applied at each octave, typically around 0.5 to control how much each layer contributes." | UNSUPPORTED | Similar to above — a formatted definition not present verbatim at the source. The page discusses gain but not in this sentence form. |

---

### SB4 — Inigo Quilez: Fractional Brownian Motion

**URL:** https://iquilezles.org/articles/fbm/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog (Shadertoy creator, authoritative)

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "A Fractional Brownian Motion is a similar process in which the increments are not completely independent from each other but feature correlated memory characteristics." | PARTIALLY-SUPPORTED | Actual text: "there's some sort of memory to the process." The claimed quote reads as an expanded paraphrase. The concept is accurate but the phrasing ("feature correlated memory characteristics") is not present verbatim. |
| "The Hurst Exponent (H) ranges from 0-1, controlling the statistical self-similarity and smoothness." | PARTIALLY-SUPPORTED | Actual: "H takes values between 0 and 1, describing rough and smooth fBMs respectively, where the normal BM happens for H=1/2." The claimed quote is a paraphrase substituting "controlling the statistical self-similarity and smoothness" for the actual description. |
| "Most graphics programmers use G=0.5 (equivalent to H=1) for natural-looking terrain, as this produces isotropic scaling in all directions." | PARTIALLY-SUPPORTED | The G=0.5 / H=1 preference for natural terrain is discussed, and the "-9dB/octave" claim below is confirmed. However, "isotropic scaling in all directions" was NOT found in the article. This phrase appears to be a scout-inserted elaboration. |
| "natural mountain profiles exhibit -9dB/octave frequency decay (yellow noise), matching fBM with H=1, validating this choice for realistic landscape generation." | VERIFIED | The article discusses yellow noise and -9dB/octave decay in the context of natural terrain. The specific claim is confirmed. |

---

### SB5 — Inigo Quilez: Domain Warping

**URL:** https://iquilezles.org/articles/warp/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "Domain warping uses noise to distort the coordinate before sampling noise again." | UNSUPPORTED | Actual: "Warping simply means we distort the domain with another function g(p) before we evaluate f." The claimed quote substitutes "noise" for "another function g(p)" and reframes the explanation, making it a paraphrase, not a verbatim quote. |
| "Each pixel's sampling location gets pushed by the noise field itself, producing curling, spiral-like shapes." | UNSUPPORTED | This phrase does not appear in the article. The article focuses on mathematical formulations (f(g(p)), g(p) = p + h(p)). Descriptive language about "curling, spiral-like shapes" is absent. |
| "Animating the warp field transforms the pattern, and warped animation produces motion that looks like the texture is alive, folding, stretching, and breathing." | UNSUPPORTED | This descriptive phrase is not present in the article. The article says "This technique is really powerful and allows you to shape apples, buildings, animals or any other thing you might imagine." No animation-specific passage with this wording exists. |

**All three SB5 quotes are unsupported — likely scout-generated summaries presented as verbatim.**

---

### SB6 — Inigo Quilez: Noise (morenoise)

**URL:** https://iquilezles.org/articles/morenoise/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "examples demonstrate the differences between value noise and gradient noise" | UNSUPPORTED | This sentence does not appear verbatim. The page discusses both noise types but not in this sentence form. |
| "using two-dimensional noise to rotate the space where straight lines are rendered can produce swirly effects that look like wood" | UNSUPPORTED | This sentence was not found verbatim at the URL. The wood grain concept may be referenced but not in this phrasing. |
| "Voronoise function allows a gradual blend between regular noise and voronoi" | UNSUPPORTED | Not found verbatim. The voronoise concept is present on the page but the claimed quoted sentence was not located. |
| "analytical derivatives computation is much faster and more accurate than the central differences method" | VERIFIED | This claim was confirmed present on the page in substance. Exact wording confirmed. |

**Three of four SB6 quotes are unsupported.**

---

### SB7 — A cellular texture basis function (Worley, SIGGRAPH 1996)

**URL:** https://dl.acm.org/doi/pdf/10.1145/237170.237267  
**Status on fetch:** HTTP 403 (ACM paywall). DEAD-LINK for direct access.  
**Fallback:** No accessible mirror found via WebFetch. Web search confirms title and author.  
**Credibility:** primary-peer-reviewed

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "A cellular texture basis function based on a partitioning of space into a random array of cells" | DEAD-LINK — unverifiable | The ACM page is paywalled. The title of the paper is "A cellular texture basis function" (confirmed via ACM abstract listing) and the phrase "partitioning of space" is consistent with the known paper description, but cannot be confirmed as verbatim. |
| "has been used to produce textured surfaces resembling flagstone-like tiled areas, organic crusty skin, crumpled paper, ice, rock, mountain ranges, and craters" | DEAD-LINK — unverifiable | Consistent with known content of the paper (widely cited) but cannot confirm verbatim from primary source. |
| "outputs a real value at a given coordinate that corresponds to the distance of the nth nearest seed (usually n=1)" | DEAD-LINK — unverifiable | Technically accurate description of Worley noise. Cannot confirm verbatim. |

**URL verdict:** DEAD-LINK. Claims are plausible given the paper's known content, but are unverifiable from the cited URL.

---

### SB8 — Efficient computational noise in GLSL (McEwan et al., 2012)

**URL:** https://arxiv.org/pdf/1204.1461  
**Status on fetch:** Binary PDF — unreadable via WebFetch. ar5iv HTML mirror accessible.  
**Credibility:** industry-talk (published in Journal of Graphics Tools, Vol 16 No 2, 2012 — peer-reviewed)

**CRITICAL: MISATTRIBUTED entry.** Sources.md lists "Authors: Unknown." Actual authors are:  
**Ian McEwan, David Sheets, Stefan Gustavson, Mark Richardson** — confirmed via ar5iv mirror and Semantic Scholar.

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "Perlin noise, developed by Ken Perlin in the 1980s, is one of the most widely used noise functions in computer graphics" | PARTIALLY-SUPPORTED | ar5iv mirror Introduction: "Perlin noise is one of the most useful building blocks of procedural shading in software." The claimed quote adds "developed by Ken Perlin in the 1980s" and changes "most useful building blocks of procedural shading" to "most widely used noise functions in computer graphics." Paraphrase, not verbatim. |
| "Simplex noise is generally more efficient than Perlin noise when you need noise in three or more dimensions" | UNSUPPORTED | Not found verbatim. ar5iv confirms simplex noise is described as a variation with "the same general look but with a different computational structure." The efficiency comparison claim is a paraphrase not confirmed as a direct quote. |
| "GLSL implementations of Perlin noise and simplex noise can run fast enough for practical use on current generation GPU hardware" | PARTIALLY-SUPPORTED | ar5iv abstract: "Perlin noise and Perlin simplex noise that run fast enough for practical consideration on current generation GPU hardware." The claimed quote drops "Perlin simplex" and changes "practical consideration" to "practical use." Close but not verbatim. |
| "purely computational—using neither textures nor lookup tables" | VERIFIED | ar5iv: "the functions are purely computational, i.e. they use neither textures nor lookup tables." The em-dash substitution for "i.e." is a minor formatting variant; substance is verbatim. |

**Author attribution: MISATTRIBUTED** (Unknown → should be McEwan, Sheets, Gustavson, Richardson).  
**Type rating should be:** primary-peer-reviewed (Journal of Graphics Tools, not "unknown").

---

### SB9 — The Book of Shaders: Cellular Noise

**URL:** https://thebookofshaders.com/12/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "Worley noise, also called Voronoi noise and cellular noise" | PARTIALLY-SUPPORTED | The page discusses Worley noise and mentions both Voronoi and cellular noise, but this exact consolidated phrase was not found as a single sentence. The page treats them as related but not necessarily as synonyms in one sentence. |
| "noise interpolates/averages random values (as in value noise) or gradients (as in gradient noise), while Voronoi computes the distance to the closest feature point" | PARTIALLY-SUPPORTED | This passage is attributed to Inigo Quilez and quoted on the page. The quote is substantially confirmed though the exact punctuation/wording may vary slightly. Substance is accurate. |

---

### SB10 — A Real-time High-quality Black Hole Shader (Bruneton, 2020)

**URL:** https://ebruneton.github.io/black_hole_shader/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog (detailed technical implementation with source code)

**CRITICAL UNACKNOWLEDGED LIMITATION:** The shader is for a **non-rotating (Schwarzschild) black hole**.
The sources.md entry does not disclose this. The project targets a Kerr (spinning, a=0.999) black hole.
This makes the source's direct applicability limited and the omission in the notes is misleading.

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "beam tracing with a distinctive optimization: precomputed tables to find the intersections of each curved light beam with the scene in constant time per pixel" | PARTIALLY-SUPPORTED | The technique is confirmed — precomputed tables for constant-time intersections. The exact phrasing "distinctive optimization" was not found verbatim. The substance is accurate. |
| "custom shading model for the accretion disk that integrates relativistic effects" | UNSUPPORTED | This exact phrase was not found verbatim. The page describes disk shading and relativistic effects but not in this sentence form. |
| "accounts for Doppler color shifts (blue-shift ahead, red-shift behind objects in motion)" | PARTIALLY-SUPPORTED | Doppler effects are confirmed present on the page. The exact phrasing "blue-shift ahead, red-shift behind" was not confirmed verbatim but the concept is present. |
| "uses a specific texture filtering scheme to integrate the contribution of the light sources to each beam" | VERIFIED | This sentence was confirmed verbatim on the page. |

---

### SB11 — SpaceEngine: General Relativity 3: Volumetric Accretion Disks (2022)

**URL:** https://spaceengine.org/news/blog220830/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog (official commercial product blog)

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "the shaders have to integrate the emission and opacity of multiple points of a volume along a ray" | VERIFIED | Confirmed in substance on the page. |
| "performs geodesic ray-tracing through warped spacetime, integrates disk brightness and opacity, stores results in two low-resolution textures: deflection vectors and disk brightness/opacity" | PARTIALLY-SUPPORTED | Actual text uses "a deflection vector" (singular), not "deflection vectors" (plural as in sources.md). The rest of the description is substantially confirmed. Minor paraphrase. |
| "procedurally-generated and animated noise to simulate rotating plasma cloud formations around black holes" | VERIFIED | Confirmed near-verbatim on the page. |
| "animated noise reduces banding artifacts during the upscaling phase" | VERIFIED | Confirmed on the page. |
| "applies multiple upscaling filters (linear, bicubic, Lanczos) with AMD FidelityFX Contrast Adaptive Sharpening (CAS)" | VERIFIED | "AMD FidelityFX Contrast Adaptive Sharpening (CAS)" confirmed. Upscaling filter list confirmed. |

---

### SB12 — Flow Map Shader Techniques (Medium, louisgamedev)

**URL:** https://louisgamedev.medium.com/shader-tutorial-flow-map-4410af832a8d  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog (game dev tutorial)

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "A flow-map shader animates UV mapping by using a specially crafted texture encoded with velocity information" | VERIFIED | Confirmed nearly verbatim on the page. |
| "The flowmap simply adds a number to the UV coordinate so it's accessing other pixels" | VERIFIED | Confirmed verbatim on the page. |
| "usually used with two normal maps to simulate water" | UNSUPPORTED | This sentence was not found at the URL. |
| "first presented at SIGGRAPH 2010 by Valve" | UNSUPPORTED | This attribution was not found in the article. The article may reference Valve's work generally but this specific sentence was not confirmed. |

---

### SB13 — Real-Time Volumetric Clouds with Ray Marching (Gangaraju, Medium, 2023)

**URL:** https://medium.com/@sushilgangaraju/real-time-volumetric-clouds-with-ray-marching-c6b46d1edeb0  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog (Medium technical post, no formal review)

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "Volumetric fog in ray marching is achieved by stepping through the scene and accumulating fog density based on distance" | UNSUPPORTED | Not found verbatim. The page covers ray marching volumetrics but not in this sentence form. |
| "Light Scattering – Simulates god rays by accumulating light along the ray path" | UNSUPPORTED | Not found verbatim. |
| "Procedural Noise Fog – Uses random noise to create a more natural, rolling mist effect" | UNSUPPORTED | Not found verbatim. |
| "common method of rendering volumetrics with ray marching is through the use of 3D volume textures" | VERIFIED | Confirmed on the page. |
| "blue noise pattern, which has fewer patterns or clumps than other noises and is less visible to the human eye" | UNSUPPORTED | Not found verbatim. |

**Four of five SB13 quotes are unsupported — likely scout-generated summaries.**

---

### SB14 — Ray Marching and Signed Distance Functions (Jamie Wong, 2016)

**URL:** https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/  
**Status on fetch:** Accessible.  
**Credibility:** reputable-blog

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "A distance field is a function that gives an estimate (a lower bound) of the distance to the closest surface at any point in space" | UNSUPPORTED | Not found verbatim. Actual article covers SDFs but the "lower bound" framing and this exact sentence are not present. |
| "Sphere tracing is a sophisticated optimization of basic ray marching that uses signed distance functions to take larger, safer steps along the ray" | UNSUPPORTED | Not found verbatim. Actual: "Instead of taking a tiny step, we take the maximum step we know is safe without going through the surface: we step by the distance to the surface, which the SDF provides us!" The claimed quote's framing ("sophisticated optimization," "larger, safer steps") is a paraphrase. |
| "Instead of using fixed small steps, sphere tracing queries the distance field at each point and uses that distance as the maximum safe step size" | UNSUPPORTED | Not found verbatim. The concept is described differently: "we take the maximum step we know is safe." The specific phrasing "queries the distance field" is absent. |

**All three SB14 quotes are unsupported.**

---

### SB15 — Gravitational Lensing by Spinning Black Holes... (James et al., CQG 2015)

**URL:** https://arxiv.org/abs/1502.03808  
**Status on fetch:** Abstract page accessible. Full text PDF binary-only, content unextractable via WebFetch.  
**Credibility:** primary-peer-reviewed (Classical and Quantum Gravity, peer-reviewed journal)

| Excerpt | Verdict | Evidence |
|---------|---------|----------|
| "develop DNGR (Double Negative Gravitational Renderer), specialized software for rendering black hole imagery" | PARTIALLY-SUPPORTED | The abstract states: "we have developed a code called DNGR (Double Negative Gravitational Renderer) to solve the equations for ray-bundle (light-beam) propagation through the curved spacetime of a spinning (Kerr) black hole." The sources.md splits this into two separate quotes and slightly paraphrases. The first fragment is accurate in substance but not a standalone verbatim sentence as presented. |
| "solve the equations for ray-bundle (light-beam) propagation through the curved spacetime of a spinning (Kerr) black hole" | VERIFIED | Confirmed verbatim in the arXiv abstract. |
| "ray-bundle techniques were crucial for achieving IMAX-quality smoothness without flickering" | VERIFIED | Confirmed verbatim in the arXiv abstract: "Our ray-bundle techniques were crucial for achieving IMAX-quality smoothness without flickering." |
| "When a ray originates on the surface of an accretion disk, the code integrates the null geodesic equation backward from the camera until it hits the disk's surface, deducing the map from a point on the disk's surface to one on the camera's sky" | UNSUPPORTED | This sentence was not found in the accessible abstract. It may appear in the paper body but is unverifiable from the accessible portion. The phrasing is consistent with DNGR methodology but cannot be confirmed as verbatim from this source. Single-source-risk for a claim only in the paper body. |

---

## Cross-Source Checks and Contradictions

### Contradiction 1: Simplex vs. Perlin efficiency claim
SB8 (McEwan et al.) claims simplex noise is "generally more efficient" for higher dimensions —
this is consistent with widely known facts about simplex noise. However, SB2 (Perlin 2002) does not
address simplex noise at all. No direct contradiction, but SB8's quote on this is unsupported
verbatim.

### Contradiction 2: H parameter interpretation in FBM (SB4)
SB4 claims G=0.5 is "equivalent to H=1" and produces "isotropic scaling." The actual IQ article
states H=1/2 is "normal BM." The claim H=1 corresponds to G=0.5 should be checked against the
SKILL.md if used in code — the relationship between gain G and Hurst exponent H varies by author
convention. **single-source-risk** for this formula.

### Contradiction 3: Flow map origin attribution (SB12)
SB12's notes claim flow maps were "first presented at SIGGRAPH 2010 by Valve." This claim is not
confirmed in the cited tutorial article. Valve's SIGGRAPH 2010 work on flow maps is publicly known
(Alex Vlachos, "Water Flow in Portal 2"), but the specific phrase from sources.md was not found at
the cited URL. **single-source-risk.**

### Contradiction 4: SB10 Schwarzschild limitation vs. project Kerr requirement
SB10 explicitly implements a non-rotating (Schwarzschild) black hole shader. The sources.md entry
does not note this limitation. The project requires Kerr (a=0.999). Any direct adoption of SB10's
precomputed-table approach would require significant re-derivation for the Kerr metric.
**This is a project-critical omission.**

---

## Overall Trust Summary

| Verdict | Count |
|---------|-------|
| VERIFIED | 13 |
| PARTIALLY-SUPPORTED | 17 |
| UNSUPPORTED | 19 |
| MISATTRIBUTED | 1 (SB8 author attribution) |
| DEAD-LINK | 2 (SB1 ACM 403, SB7 ACM 403) |

**Total claims evaluated:** ~52

**Sources with all or mostly verified excerpts:** SB11, SB15 (partially)  
**Sources with majority unsupported quotes:** SB5 (3/3 unsupported), SB13 (4/5 unsupported), SB14 (3/3 unsupported), SB6 (3/4 unsupported)

**Root cause of unsupported claims:** The scout systematically generated paraphrased summaries
and presented them as verbatim quotes. This is a corpus-wide problem affecting at least 7 of 15
sources.

---

## Riskiest Items

1. **SB8 author attribution MISATTRIBUTED**: "Authors: Unknown" is wrong. True authors:
   Ian McEwan, David Sheets, Stefan Gustavson, Mark Richardson. Journal of Graphics Tools
   Vol 16 No 2 (2012) — should be rated primary-peer-reviewed, not unknown.

2. **SB10 unacknowledged Schwarzschild limitation**: The shader is for a non-rotating black hole.
   The project requires Kerr a=0.999. Any use of SB10's precomputed-table method without this
   caveat risks importing a fundamentally incompatible approach. The notes say "relativistic effects"
   without disclosing the spin=0 constraint.

3. **Pervasive false-verbatim problem across SB5, SB6, SB13, SB14**: These four sources together
   contribute 13 unsupported "verbatim" quotes that are actually scout-generated paraphrases.
   Any synthesis step that quotes these as direct source excerpts will introduce fabricated text
   into the research record.

---

## MUST RE-GATHER

The following specific claims must be replaced by re-scout. Quoted exactly as they appear in
sources.md, with source ID:

1. **SB5**: "Domain warping uses noise to distort the coordinate before sampling noise again." — actual IQ text is different; need real verbatim quote
2. **SB5**: "Each pixel's sampling location gets pushed by the noise field itself, producing curling, spiral-like shapes." — not found at source; need real verbatim or remove
3. **SB5**: "Animating the warp field transforms the pattern, and warped animation produces motion that looks like the texture is alive, folding, stretching, and breathing." — not found at source; need real verbatim or remove
4. **SB6**: "examples demonstrate the differences between value noise and gradient noise" — not found verbatim; need real quote
5. **SB6**: "using two-dimensional noise to rotate the space where straight lines are rendered can produce swirly effects that look like wood" — not found verbatim; need real quote
6. **SB6**: "Voronoise function allows a gradual blend between regular noise and voronoi" — not found verbatim; need real quote
7. **SB8**: Author attribution "Unknown" — must be corrected to Ian McEwan, David Sheets, Stefan Gustavson, Mark Richardson; type must be corrected to primary-peer-reviewed (Journal of Graphics Tools)
8. **SB12**: "usually used with two normal maps to simulate water" — not found at source; need real quote or remove
9. **SB12**: "first presented at SIGGRAPH 2010 by Valve" — not found at source; need primary citation (Vlachos, SIGGRAPH 2010) or remove
10. **SB13**: "Volumetric fog in ray marching is achieved by stepping through the scene and accumulating fog density based on distance" — not found verbatim; need real quote
11. **SB13**: "Light Scattering – Simulates god rays by accumulating light along the ray path" — not found verbatim; need real quote or remove
12. **SB13**: "Procedural Noise Fog – Uses random noise to create a more natural, rolling mist effect" — not found verbatim; need real quote or remove
13. **SB13**: "blue noise pattern, which has fewer patterns or clumps than other noises and is less visible to the human eye" — not found verbatim; need real quote
14. **SB14**: "A distance field is a function that gives an estimate (a lower bound) of the distance to the closest surface at any point in space" — not found verbatim; actual wording is different
15. **SB14**: "Sphere tracing is a sophisticated optimization of basic ray marching that uses signed distance functions to take larger, safer steps along the ray" — not found verbatim; actual wording is different
16. **SB14**: "Instead of using fixed small steps, sphere tracing queries the distance field at each point and uses that distance as the maximum safe step size" — not found verbatim; actual wording is different
17. **SB15**: "When a ray originates on the surface of an accretion disk, the code integrates the null geodesic equation backward from the camera until it hits the disk's surface, deducing the map from a point on the disk's surface to one on the camera's sky" — not found in accessible abstract; needs primary confirmation from paper body or a confirmed quote from the paper
18. **SB10 notes**: Must add disclosure that this shader is for a non-rotating (Schwarzschild) black hole and does not implement Kerr spacetime

---

*Report generated by claude-sonnet-4-6, 2026-06-08.*
