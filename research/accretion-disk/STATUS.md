# Accretion-disk research — STATUS

_Checkpoint saved 2026-06-09. Workflow paused to conserve usage; switched to
frugal, sequential, main-thread work (no parallel agent fan-out)._

## Corpus state
| Sub-domain | Sources file | Gathered | Validated |
|---|---|---|---|
| A. Fluid / MHD            | `sources-fluid-mhd.md`        | yes | no |
| B. Procedural / noise     | `sources-procedural-noise.md` | yes | yes (`validation-procedural-noise.md`) |
| C. GR lensing & raytracing| `sources-gr-lensing.md`       | yes | no |
| D. VFX pipeline           | `sources-vfx-pipeline.md`     | yes | no |
| E. Spectral / color       | `sources-spectral-color.md`   | yes | no |

Search logs: `search-log.md`, `search-log-E.md`.

## Done
- All 5 sub-domains gathered by `research-scout` (haiku). Verbatim excerpts +
  provenance captured on disk.
- Sub-domain B validated by `source-validator` (sonnet).

## Not done (deliberately deferred for cost)
- Validation of A, C, D, E.
- Synthesis / decision brief.

## Cost lesson
The parallel fan-out (5 scouts + sonnet validators re-fetching every URL at
once) spiked the 5-hour limit. Gathering — the costly half — is already paid
for. Do NOT re-run the full workflow. Going forward: one step at a time, in the
main thread, validating only decision-critical claims.
