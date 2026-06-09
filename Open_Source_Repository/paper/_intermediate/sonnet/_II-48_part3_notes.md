# Notes: Novikov & Thorne (1973) §§4.5–5.13 — Accretion Disk Physics (lines 1901–EOF)

## §4.5–4.9 Spherical Accretion and Validity

- **Eddington limit**: $L_{\rm Edd} = 4\pi G M m_p c / \sigma_T$; accretion rate capped at $\dot{M}_{\rm Edd} \sim 10^{18}(M/M_\odot)\,\text{g\,s}^{-1}$.
- **Hydrodynamic validity** (§4.6): Bremsstrahlung emissivity dominates; mean-free path $\ll$ scale height confirms fluid approximation.
- **Near-horizon gas** (§4.7): $T \sim 10^{12}\,\text{K}$; synchrotron emissivity significant; Comptonization broadens spectrum.
- **Bondi-Hoyle accretion** (§4.8): $\dot{M} \propto \rho_\infty (GM)^2 / (u_\infty^2 + c_s^2)^{3/2}$.
- **Optical appearance** (§4.9): flare timescale lower bound $\Delta t_{\rm flares} \geq 2 b_{\rm capture}/u_\infty$.

## §5.2 Thin-Disk Conservation and Spectrum

- Mass, angular-momentum, energy conservation in Newtonian thin disk; blackbody surface temperature $T(r) \propto [F(r)/\sigma]^{1/4}$.
- Peak spectrum $h\nu_{\rm max} \approx 1\,\text{keV}$ for stellar-mass holes accreting near $\dot{M}_{\rm Edd}$.
- Vertical pressure balance: half-thickness $h \sim c_s/\Omega_K$.
- Opacity regimes: Kramers free-free $\kappa_{\rm ff} \propto \rho T^{-7/2}$ (outer); electron scattering $\kappa_{\rm es} = 0.40\,\text{cm}^2\,\text{g}^{-1}$ (inner).
- Shakura-Sunyaev $\alpha$-viscosity: $t_{r\phi} = \alpha p$; $\alpha < 1$ parameterises unknown turbulence.

## §5.3 Galactic Nuclei / Quasar Disks

- Same thin-disk equations scaled to $M \sim 10^8 M_\odot$; dust-cloud formation in outer disk; far-IR reemission.

## §5.4 Kerr Metric — Equatorial Orbital Quantities

- ISCO radius: $r_{\rm ms} = M\!\left[3 + Z_2 - \sqrt{(3-Z_1)(3+Z_1+2Z_2)}\right]^{1/2}$ (functions of spin $a$; $Z_1, Z_2$ auxiliary).
- Orbital angular velocity, specific angular momentum $\tilde{L}$, and specific energy $\tilde{E}$ for circular equatorial orbits given explicitly.
- Orthonormal ZAMO (orbiting) tetrad defined for local measurements.

## §5.5–5.6 Relativistic Disk Radial Structure

- Assumptions: quasisteady state, geometrically thin ($h \ll r$), equatorial plane, separable vertical/radial structure.
- Rest-mass conservation (5.6.2), angular-momentum conservation (5.6.3–5.6.6), energy conservation (5.6.7–5.6.12).
- Radiative flux from combined conservation: relativistic correction factor $\mathcal{Q}(r) \to 0$ at inner edge $r_{\rm ms}$ (stress-free inner boundary).

## §5.7–5.8 Vertical Structure

- Pressure balance (5.7.4a); radiative transport; equation of state $p = \rho_0 (k/\mu m_p)T + \tfrac{1}{3}aT^4$ (gas + radiation).
- Rosseland mean opacity $\kappa$ evaluated in orbiting frame.
- Explicit values: $\kappa_{\rm es} = 0.40\,\text{cm}^2\,\text{g}^{-1}$; Kramers $\kappa_{\rm ff} \propto \rho T^{-7/2}$.

## §5.9 Three-Zone Explicit Models

- **Outer zone**: free-free opacity, gas-pressure dominated; power-law profiles in $\alpha, M_*, \dot{M}, r$.
- **Middle zone**: electron-scattering opacity, gas-pressure dominated.
- **Inner zone**: electron-scattering opacity, radiation-pressure dominated; may be absent at low $\dot{M}$.
- Transition radii and inflow timescale given explicitly; zones collapse depending on accretion rate.

## §5.10 Disk Spectrum

- Homogeneous atmosphere: specific flux emerging $\propto \int B_\nu \cos\theta\,d\Omega$; modification factor for inhomogeneous atmosphere (5.10.5–5.10.7).
- Surface temperature middle region: $T_s \approx (6\times10^7\,\text{K})\,\alpha^{-2/9} M_{10}^{-1/9} \dot{M}^{17/9} r^{-\ldots}$ (eq. 5.10.8).
- Comptonization: $\Delta h\nu/h\nu \geq 4(T/m_e c^2) \geq (T/2\times10^9\,\text{K})$; hardens spectrum in inner zone.
- Optically thin inner region radiates free-free; $\sim$10\% of reradiated luminosity in optical lines, $\sim$90\% in UV free-bound/free-free.

## §5.12–5.13 Fluctuations and Supercritical Accretion

- Fluctuations: turbulence, magnetic reconnection, plasma instabilities; Doppler asymmetry from hole spin.
- Supercritical accretion: if $\dot{M} > \dot{M}_{\rm Edd}$, radiation pressure drives outflow; total luminosity self-regulates at $L_{\rm Edd}$.

## Rendering-Relevant Summary

- Emitted flux $F(r)$ peaks at $r \sim 2$–$3\,r_{\rm ms}$; vanishes at inner edge (stress-free).
- Local blackbody temperature $T(r) = [F(r)/\sigma]^{1/4}$; spin $a$ controls $r_{\rm ms}$ and thus peak temperature and luminosity.
- Spectrum broadened by Comptonization in inner zone; photon number density and specific intensity transform with gravitational + Doppler redshift.
- $\kappa_{\rm es}$ controls photon diffusion depth in inner disk; number of scatterings $N \sim \tau^2$ (random walk).
