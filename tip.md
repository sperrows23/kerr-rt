# ARCHITECTURE SPECIFICATION: CINEMATIC VOLUMETRIC KERR-SCHILD RENDERER
## OVERVIEW
This document outlines the required paradigm shifts to transition a physically-based, single-phase, emission-absorption black hole accretion disk renderer (operating in Cartesian Kerr-Schild coordinates) into a cinematic, multi-phase, scattering-aware volumetric engine. The specification is formatted for automated ingestion and semantic parsing, detailing all mathematical, thermodynamic, and optical modifications without implementation-specific code.

---

## PILLAR 1: FLUID DYNAMICS (TURBULENT CASCADES VS. LAMINAR SHEARING)

### 1.1 The Scientific Problem
In a differentially rotating spacetime, the accretion disk follows a Keplerian shear profile where angular velocity $\Omega(r)$ is inversely proportional to the radius to the power of 1.5:
$$ \Omega(r) \propto r^{-1.5} $$
When applying a Neyret-style advected texture model, continuous mapping of this shear vector to a static procedural noise field causes localized structures to wind infinitely tightly. This mathematical "laminarization" smears turbulent details into perfectly concentric rings, failing to represent physical fluid dynamics.

### 1.2 The Paradigm Shift: Dynamic Turbulent Cascades
Physical accretion disks are governed by Magneto-Rotational Instability (MRI) and hydrodynamic turbulence, which inject localized eddies that resist bulk laminarization (Kolmogorov energy cascade).
*   **Time Decoupling:** The time variable used to advect the fluid coordinates ($t_{shear}$) must be mathematically decoupled from the time variable used to evolve the procedural noise ($t_{boil}$). 
*   **Scale-Dependent Shear Application:** The shear operator must function as a cascade. Large-scale noise frequencies are heavily subjected to the rotational shear tensor, whereas small-scale, high-frequency procedural noise (driven by a divergence-free curl-flow potential, e.g., CKS-18) maintains structural integrity, regenerating micro-vortices before they can be completely smeared by the bulk flow.

---

## PILLAR 2: THERMODYNAMICS (DECOUPLED MULTI-PHASE MEDIA)

### 2.1 The Scientific Problem
The baseline engine assumes a tightly coupled single-phase medium where total density $\rho$ dictates both the emissive source function ($j_\nu$) and the absorption coefficient ($\kappa_\nu$).
$$ j_\nu \propto \rho, \quad \kappa_\nu \propto \rho $$
This linear coupling prohibits the rendering of optically thick, non-luminous matter. Regions of high density mathematically enforce high luminosity, making dark, obscuring dust lanes impossible.

### 2.2 The Paradigm Shift: Multi-Phase Radiative Transfer
Astrophysical disks are multi-phase environments. The thermodynamic mapping must be bifurcated into two statistically independent or anti-correlated scalar fields:
1.  **Ionized Plasma Phase ($\rho_{hot}$):** Low-density, high-temperature. Governs the emission profile ($j_\nu$) representing the glowing, diffuse background.
2.  **Dust/Neutral Gas Phase ($\rho_{cold}$):** High-density, low-temperature, high-frequency clump distribution. Governs the absorption coefficient ($\kappa_\nu$).

By solving the Radiative Transfer Equation (RTE) through these decoupled fields, the $\rho_{cold}$ field acts strictly as an obscuring medium, carving mathematically sharp, high-contrast black silhouettes across the $\rho_{hot}$ emission field.

---

## PILLAR 3: OPTICS (SCATTERING-PHASE-AWARE MEDIA)

### 3.1 The Scientific Problem
The current RTE calculates line-of-sight attenuation using purely emissive and absorptive components:
$$ \frac{dI}{ds} = j_\nu - \kappa_\nu I $$
This is sufficient for distant observations but fails for close-proximity volumetric rendering. Foreground clouds illuminated by the inner disk cannot exhibit edge-glows or "silver-lining" because photons are merely absorbed rather than redirected into the camera path.

### 3.2 The Paradigm Shift: In-Scattering and Anisotropic Phase Functions
The RTE must be upgraded to include an in-scattering coefficient ($\sigma_s$), forming the complete transport equation:
$$ \frac{dI}{ds} = j_\nu - (\kappa_\nu + \sigma_s) I + \sigma_s \int_{4\pi} I(\hat{s}') P(\hat{s}', \hat{s}) \, d\Omega' $$
To model the highly directional Mie scattering typical of dense astrophysical dust, the engine must integrate the **Henyey-Greenstein Phase Function**:
$$ P(\cos \theta) = \frac{1 - g^2}{4\pi (1 + g^2 - 2g \cos \theta)^{3/2}} $$
*   **Parameters:** $\theta$ is the scattering angle between the incoming light vector $\hat{s}'$ and the viewing vector $\hat{s}$.
*   **Anisotropy Factor ($g$):** Defined in the domain $g \in (-1, 1)$. By setting $g > 0.5$ (strongly forward-scattering), light originating from the inner event horizon is preferentially deflected through the optically thin edges of foreground clouds toward the observer, yielding cinematic volumetric depth and localized rim-lighting.

---

## PILLAR 4: BOUNDARY PHYSICS (TURBULENT EROSION)

### 4.1 The Scientific Problem
Applying a 1D or 2D radial smoothstep function to truncate the accretion disk boundary yields a geometrically perfect, synthetic edge. It fails to simulate the interface interactions between a high-velocity shear flow and a vacuum (or infalling envelope).

### 4.2 The Paradigm Shift: Kelvin-Helmholtz Erosion
The boundary must be defined by volumetric fluid instabilities rather than geometric masking. The outer boundary radius $R_{outer}$ becomes a function of 3D spatial coordinates and time, eroded by a high-amplitude, high-frequency noise scalar:
$$ R_{outer}(\theta, \phi, t) = R_0 + \delta R(\theta, \phi, t) $$
Instead of multiplying the base density by an attenuation window, the baseline density field is aggressively thresholded (clipped) against a 3D volumetric noise field at the boundary limits. This mathematically mimics Kelvin-Helmholtz instabilities, resulting in shredding, tearing, and fraying of the gas into the vacuum.

---

## PILLAR 5: SCALE INVARIANCE (MULTI-SCALE LEVEL OF DETAIL)

### 5.1 The Scientific Problem
Procedural noise architectures utilizing a fixed number of octaves and fixed spatial frequencies suffer from the Nyquist limit. 
*   **Macro-scale view:** High frequencies fall below the pixel footprint, causing extreme spatial aliasing, moiré patterns, and shimmering.
*   **Micro-scale view (Close-up):** The lack of sub-scale octaves results in a smooth, blurry, low-resolution appearance.

### 5.2 The Paradigm Shift: Fractal LOD Cascades
A mathematically rigorous Level of Detail (LOD) system must govern the volumetric step evaluation.
*   **Screen-Space Jacobian ($J$):** The renderer must calculate the derivative of the ray trajectory relative to screen-space pixels (or directly map the camera's physical distance to the local integration volume).
*   **Dynamic Octave Modulation:** The maximum number of noise octaves ($N_{octaves}$), the frequency multipliers, and the integration step size ($d\lambda$) must scale dynamically as a function of $J$.
$$ N_{octaves} \propto -\log_2(J), \quad d\lambda \propto J $$
This continuous modulation ensures a constant visual detail density at all scales. High-frequency octaves are mathematically culled at a distance to prevent aliasing, and dynamically injected upon camera proximity to resolve micro-gaseous wisps.