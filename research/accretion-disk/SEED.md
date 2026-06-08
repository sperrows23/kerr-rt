# Seed targets — Interstellar-grade accretion disk

Starting **research targets** for `research-scout`. These are leads to chase and
**verify**, NOT established facts. The scout must locate the primary source,
confirm it exists, and capture verbatim excerpts; the validator confirms every
claim. Do not treat anything here as true until validated.

## Sub-domains to cover
- **A. Physically-based fluid / MHD simulation** of accretion disks (GRMHD, SPH, grid).
- **B. Procedural / noise-based** disk generation (FBM, curl/flow noise, shaders, flow maps).
- **C. GR raytracing & gravitational lensing**, Doppler beaming, gravitational redshift.
- **D. VFX production pipeline** & artistic controllability (how a film actually shipped it).
- **E. Spectral / temperature / color** models for disk emission.

## Candidate primary sources to locate & verify
> Verify author, year, venue, and exact claims. Flag any you cannot find.

- The Double Negative / Interstellar lensing paper — *"Gravitational lensing by
  spinning black holes in astrophysics, and in the movie Interstellar"*
  (James, von Tunzelmann, Franklin, Thorne) — find the journal, year, and the
  actual renderer description ("DNGR").
- Kip Thorne, *The Science of Interstellar* — chapters on the disk/visualization.
- Oliver James et al., follow-up / *Classical and Quantum Gravity* articles on
  the same rendering work.
- Jean-Pierre Luminet's early black-hole-with-disk imaging work.
- Alain Riazuelo's relativistic black-hole visualizations.
- GRMHD accretion-disk simulation literature (e.g. HARM and successors) — find
  canonical method papers.
- SIGGRAPH / FMX talks on volumetric disks, curl-noise flow, or film black holes.
- Real-time / game approximations of accretion disks (shader/noise techniques).

## What to capture per source
Follow the `sources.md` template in `.claude/agents/research-scout.md`:
title, authors, year, venue, URL, type tag, relevance, verbatim excerpts, notes.
