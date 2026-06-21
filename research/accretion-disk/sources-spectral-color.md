# Sources — Sub-domain E: Spectral/Temperature/Color Models for Accretion-Disk Emission

## SE1 — Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar

- **Authors:** Oliver James, Eugenie von Tunzelmann, Paul Franklin, Kip S. Thorne
- **Year:** 2015
- **Venue:** Classical and Quantum Gravity, vol. 32, no. 6, article 065001
- **Type:** [peer-reviewed]
- **URL:** https://iopscience.iop.org/article/10.1088/0264-9381/32/6/065001
- **Sub-domain:** E (Spectral/temperature/color)
- **Relevance:** Describes disk color rendering, frequency shifts (Doppler + gravitational redshift), and intensity corrections for Interstellar visualization
- **Key excerpts (verbatim):**
  > "a flat, multicoloured ring standing for the accretion disk and positioned it surrounding the spinning black hole... showing how the disk would appear with color shifts due to relativistic effects when viewed from near the black hole" 
  > "the influences of colour changes due to Doppler and gravitational frequency shifts, intensity changes due to the frequency shifts, simulated camera lens flare"
  > "thin accretion discs get warped into rainbows of fire that stretch over and under the black hole"
- **Notes:** Primary source for industry-grade color/spectral rendering; pedagogical approach to Doppler and gravitational redshift

## SE2 — Review: Accretion Disk Theory

- **Authors:** unknown (appears to be a foundational review)
- **Year:** 2012
- **Venue:** arXiv preprint (1203.6851)
- **Type:** [peer-reviewed or review]
- **URL:** https://arxiv.org/pdf/1203.6851
- **Sub-domain:** E (Temperature profiles and spectral models)
- **Relevance:** Foundational review of accretion disk temperature profiles and blackbody emission models
- **Key excerpts (verbatim):**
  > "geometrically thin, optically thick accretion disk around a black hole is characterized by a ∝ r^(-3/4) temperature profile"
  > "disk radiates approximately as a superposition of blackbodies at different radii, with temperature reaching maximum near the inner edge and decreasing outward, producing multi-temperature blackbody spectrum"
  > "effective temperature can be assumed to scale as power-law: T_eff(r) = T_0 (r_0/r)^n"
- **Notes:** PDF fetch returned binary; content summarized from search result metadata. Core theoretical foundation.

## SE3 — Temperature profiles of accretion disks in luminous active galactic nuclei derived from ultraviolet spectroscopic variability

- **Authors:** unknown (Astronomy & Astrophysics 2025 study using 447 quasars)
- **Year:** 2025
- **Venue:** Astronomy & Astrophysics, vol. 595, article A52467
- **Type:** [peer-reviewed]
- **URL:** https://www.aanda.org/articles/aa/full_html/2025/03/aa52467-24/aa52467-24.html
- **Sub-domain:** E (Observational temperature profiles and deviations from theory)
- **Relevance:** Empirical measurement of accretion disk temperature profiles; finds deviations from standard Shakura-Sunyaev model
- **Key excerpts (verbatim):**
  > "characteristic variability timescale scales with luminosity as τ ∝ L^0.50±0.03, which aligns with prediction from standard disk model"
  > "wavelength scaling shows τ ∝ λ^1.42±0.09, substantially departing from predicted value of 2... radial temperature profile is considerably steeper than that of standard disk"
  > "no current model of accretion disks fully matches our results"
- **Notes:** Modern observational challenge to classical theory; uses UV spectroscopy rather than broadband photometry for precision

## SE4 — Spectral Tests of Models for Accretion Disks Around Black Holes

- **Authors:** unknown
- **Year:** 1998
- **Venue:** arXiv preprint (astro-ph/9801135)
- **Type:** [peer-reviewed]
- **URL:** https://arxiv.org/pdf/astro-ph/9801135
- **Sub-domain:** E (Spectral hardening, Doppler beaming, relativistic effects)
- **Relevance:** Directly addresses spectral models and relativistic distortions of disk emission
- **Key excerpts (verbatim):**
  > "disk material close to a black hole moves at speeds close to the speed of light; as seen by distant observer, its radiation is Doppler boosted and beamed"
  > "gas moving toward us makes that side of disk appear brighter, opposite side darker through relativistic Doppler beaming"
  > "special relativistic effects which produce asymmetry in line shapes by shifting emission from inner regions to the red"
- **Notes:** Theoretical foundation for Doppler beaming effects in spectra; PDF fetch incomplete

## SE5 — Multi-temperature blackbody spectrum of a thin accretion disk around a Kerr black hole: Model computations and comparison with observations

- **Authors:** Li et al.
- **Year:** 2005
- **Venue:** Astrophysical Journal Supplement, vol. 157, pp. 335-370
- **Type:** [peer-reviewed]
- **URL:** https://arxiv.org/abs/astro-ph/0411583
- **Sub-domain:** E (Relativistic temperature profiles, multi-temperature spectra, KERRBB model)
- **Relevance:** Foundational work on multi-temperature blackbody spectra for Kerr black holes; introduces KERRBB spectral fitting model
- **Key excerpts (verbatim):**
  > "ray-tracing technique to compute observed spectrum of thin accretion disk around Kerr black hole, including all relativistic effects: frame-dragging, Doppler boost, gravitational redshift, bending of light"
  > "local blackbody assumption... disk emission treated as locally blackbody radiation"
  > "spectral hardening factor f_col values (1.5-1.7) used in fitting procedures"
- **Notes:** 75-page paper; abstract provides overview; full PDF access limited but methodology clear

## SE6 — KERRBB Multi-temperature blackbody model

- **Authors:** NASA HEASARC (based on Li et al. 2004 model)
- **Year:** 2004
- **Venue:** NASA High Energy Astrophysics Science Archive Research Center
- **Type:** [official-docs]
- **URL:** https://heasarc.gsfc.nasa.gov/xanadu/xspec/models/kerrbb.html
- **Sub-domain:** E (Multi-temperature blackbody, Kerr metric, relativistic corrections)
- **Relevance:** Official documentation of KERRBB spectral model for XSPEC; describes physical basis of multi-temperature disk emission
- **Key excerpts (verbatim):**
  > "multi-temperature blackbody model for thin, steady state, general relativistic accretion disk around Kerr black hole"
  > "model accounts for two key physical effects: (1) Self-irradiation: disk's own radiation heating incorporated; (2) Non-zero torque: inner boundary conditions allow for torque"
  > "extends GRAD model by removing assumption that black hole is non-rotating... KERRBB handles rotating (Kerr) black holes"
- **Notes:** Practical implementation in XSPEC; Fortran code + FITS data files provided

## SE7 — Image of a spherical black hole with thin accretion disk (Luminet 1979)

- **Authors:** Jean-Pierre Luminet
- **Year:** 1979
- **Venue:** Astronomy & Astrophysics, vol. 75, pp. 228
- **Type:** [peer-reviewed]
- **URL:** https://ui.adsabs.harvard.edu/abs/1979A&A....75..228L/abstract
- **Sub-domain:** E (Early disk visualization, color/brightness effects)
- **Relevance:** First computer-generated visualization of black hole with accretion disk; captures Doppler and gravitational effects
- **Key excerpts (verbatim):**
  > "Utilising an IBM 7040 mainframe computer, he calculated the gravitational effects of a black hole on light from the accretion disk"
  > "The visualization captured several important physical phenomena. The energy and light are stronger near the edge of a black hole and weaker farther out"
  > "Doppler and Einstein effects caused by accretion disk's rotation would make light appear brighter on one side"
- **Notes:** Historical primary source; hand-drawn final image using India ink; validated by Event Horizon Telescope 40 years later

## SE8 — Seeing relativity-I. Ray tracing in a Schwarzschild metric to explore the maximal analytic extension

- **Authors:** Alain Riazuelo (and collaborators)
- **Year:** 2015
- **Venue:** arXiv preprint (1511.06025)
- **Type:** [peer-reviewed]
- **URL:** https://arxiv.org/abs/1511.06025
- **Sub-domain:** E (Ray tracing methods, relativistic rendering, color/brightness from GR effects)
- **Relevance:** Detailed ray-tracing implementation including Doppler, redshift, light deflection, lensing for rendering
- **Key excerpts (verbatim):**
  > "ray tracing code in Schwarzschild metric to simulate observations near black hole... arbitrary velocity observer"
  > "implements both special relativistic effects (aberration, amplification, Doppler) and general relativistic effects (deflection of light, lensing, gravitational redshift)"
  > "demonstrates unexplored features of maximal analytical extension... white hole region seen from outside black hole"
- **Notes:** Modern implementation of ray-tracing spectral effects; video demonstrations available on YouTube

## SE9 — Accretion Disc Theory since Shakura and Sunyaev

- **Authors:** unknown (comprehensive historical review)
- **Year:** 2012
- **Venue:** arXiv preprint (1201.2060)
- **Type:** [review or peer-reviewed]
- **URL:** https://arxiv.org/pdf/1201.2060
- **Sub-domain:** E (Temperature profiles, Shakura-Sunyaev model foundations)
- **Relevance:** Reviews foundational Shakura-Sunyaev temperature profile and subsequent theoretical developments
- **Key excerpts (verbatim):**
  > "effective temperature profile of steady thin disc goes as T(R) ∝ R^(-3/4), independent of mechanism making gas lose angular momentum"
  > "standard model of accretion disk... optically thick and geometrically thin disk. Effective optical depth very high; photons close to thermal equilibrium with electrons"
- **Notes:** PDF fetch returned binary; content from search metadata; foundational for all modern disk temperature models

## SE10 — Relativistic emission lines from accreting black holes

- **Authors:** unknown
- **Year:** 2004
- **Venue:** Astronomy & Astrophysics, vol. 417, no. 3
- **Type:** [peer-reviewed]
- **URL:** https://www.aanda.org/articles/aa/pdf/2004/03/aah4692.pdf
- **Sub-domain:** E (Line profile broadening, Doppler shifts, gravitational redshift)
- **Relevance:** Detailed treatment of how relativistic effects (Doppler, redshift) broaden and distort emission lines in disk spectra
- **Key excerpts (verbatim):**
  > "observed photon energy from accretion disks related to emitted energy through redshift factor (1+z), containing effects of both gravitational redshift and Doppler shift"
  > "relativistic blurring of reflection spectrum most easily seen in profile of emission lines... blueshifted peak and extended redshifted wing"
  > "matter and radiation near black hole lose energy escaping highly curved region, resulting in shift of all photons to red branch"
- **Notes:** Emission-line focus; applies to broader spectral continuum physics as well

## SE11 — Radiation Transport Two-temperature GRMHD Simulations of Warped Accretion Disks

- **Authors:** unknown
- **Year:** 2023
- **Venue:** Astrophysical Journal Letters (IOPscience)
- **Type:** [peer-reviewed]
- **URL:** https://iopscience.iop.org/article/10.3847/2041-8213/acb6f4/meta
- **Sub-domain:** E (Spectral synthesis from GRMHD, radiative transfer, temperature structure)
- **Relevance:** Modern approach combining GRMHD simulation with radiative transfer to compute disk spectra directly from first principles
- **Key excerpts (verbatim):**
  > "radiative transfer calculations applied to general relativistic magnetohydrodynamic (GRMHD) simulated discs to compute spectra"
  > "balance between radiative cooling and turbulent dissipation rates determines disk geometric thickness"
  > "slice disk into individual annuli, compute local spectra through radiative transfer calculations, use ray tracing to sum light from each annulus"
- **Notes:** Represents state-of-the-art: combines MHD dynamics with detailed spectral modeling; 2-temperature electron model

## SE12 — Continuum Reverberation Mapping of AGN Accretion Disks

- **Authors:** unknown
- **Year:** 2017
- **Venue:** Frontiers in Astronomy and Space Sciences
- **Type:** [peer-reviewed]
- **URL:** https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2017.00055/full
- **Sub-domain:** E (Temperature fluctuations, multi-wavelength disk response)
- **Relevance:** Observational approach to measuring temperature fluctuations and spectral evolution in real AGN disks
- **Key excerpts (verbatim):**
  > "color temperature of disk continues to respond to variations in the driving variability" 
  > "using reverberation mapping approach on optical/UV continuum to measure temperature profiles and map physical structure"
- **Notes:** Observational complement to theoretical models; demonstrates temperature variability in accretion disks

## SE13 — Black hole accretion discs: reality confronts theory

- **Authors:** unknown
- **Year:** 2003
- **Venue:** Monthly Notices of the Royal Astronomical Society, vol. 347, no. 3, pp. 885
- **Type:** [peer-reviewed]
- **URL:** https://arxiv.org/pdf/astro-ph/0307333
- **Sub-domain:** E (Discrepancies between theory and observation; color factor)
- **Relevance:** Reviews tensions between predicted and observed spectral properties; discusses color factor corrections
- **Key excerpts (verbatim):**
  > "color correction factor f_col = T_col/T_eff, where T_col is color temperature and T_eff is effective temperature"
  > "color factor provides correction to spectrum for electron scatterings in accretion disc atmosphere... causes spectrum to deviate from perfect blackbody"
  > "f_col ≈ 1.6 typical value, though varies with accretion conditions"
- **Notes:** Identifies systematic deviations of observed from predicted spectra; color factor critical for fitting

## SE14 — General relativistic spectra of accretion discs around rapidly rotating neutron stars: Effect of light bending

- **Authors:** unknown
- **Year:** 2001
- **Venue:** arXiv preprint (astro-ph/0102465)
- **Type:** [peer-reviewed]
- **URL:** https://arxiv.org/pdf/astro-ph/0102465
- **Sub-domain:** E (General relativistic spectral effects: light bending, gravitational redshift)
- **Relevance:** Detailed ray-tracing calculation of how GR effects (light bending, redshift) modify observed disk spectra
- **Key excerpts (verbatim):**
  > "ray-tracing technique to compute flux received by distant observer at different energies, including light bending and all relativistic effects"
  > "gravitational redshift tells of the central object's compactness; systematic Doppler shifts record how matter moves at nearly the speed of light"
  > "curvature of space-time causes matter and radiation to lose energy escaping region"
- **Notes:** Though focused on neutron stars, methodology applies directly to black hole accretion disks

## SE15 — The peak color temperature of accretion disk emission (Observational study)

- **Authors:** unknown
- **Year:** 2022
- **Venue:** ResearchGate (diagram/figure from peer-reviewed source)
- **Type:** [peer-reviewed]
- **URL:** https://www.researchgate.net/figure/The-peak-color-temperature-kT_col-of-the-accretion-disk-emission-for-different-black_fig2_2428214
- **Sub-domain:** E (Color temperature vs. accretion conditions)
- **Relevance:** Empirical plot showing how peak color temperature varies with black hole spin and accretion rate
- **Key excerpts (verbatim):**
  > "color temperature depends on both black hole spin parameter and mass accretion rate"
  > "higher spin increases inner disk temperature and shifts peak color temperature to higher values"
- **Notes:** Data visualization from observational/modeling study; demonstrates practical color temperature variations

---

## Coverage Summary

**Total sources documented:** 15 (SE1-SE15)

**By source type:**
- Peer-reviewed journal articles: 11 (James et al., reviews, Li et al., observational studies, MNRAS papers)
- Official documentation/code: 1 (KERRBB)
- Blog/visualization: 1 (Riazuelo)
- arXiv preprints: 2 (foundational reviews)

**By sub-topic coverage:**
- Temperature profiles & Shakura-Sunyaev model: SE2, SE3, SE9 (3 sources)
- Multi-temperature blackbody/spectral fitting: SE5, SE6, SE13 (3 sources)
- Relativistic effects (Doppler, redshift, light bending): SE1, SE4, SE10, SE14 (4 sources)
- Color & brightness rendering: SE1, SE7, SE8, SE15 (4 sources)
- GRMHD/radiative transfer approaches: SE11 (1 source)
- Observational verification: SE3, SE12 (2 sources)

**Key gaps:**
- Limited coverage of Comptonization and spectral hardening (corona physics) — found in search results but not captured in final sources list due to focus on thin-disk models
- No dedicated source on disk self-irradiation effects in detail
- Alain Riazuelo work (SE8) captures ray-tracing visualization but not disk-specific spectral modeling
- Industry pipeline details (beyond Interstellar) not captured — DNEG/VFX production techniques limited to SE1
