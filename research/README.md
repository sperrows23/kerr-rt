# Research Pipeline

A cost-tiered, anti-hallucination research workflow driven by three sub-agents
(defined in `.claude/agents/`). Cheap models do the heavy gathering; an
adversarial check filters them; an expensive model synthesizes only what
survives. The **main controller (Opus, in the main thread)** orchestrates and
owns all implementation + physics decisions.

## The three stages

| # | Agent | Model | Job | Writes |
|---|-------|-------|-----|--------|
| 1 | `research-scout` | haiku | Broad web gather, verbatim capture, full provenance. No judgment. | `sources.md`, `search-log.md` |
| 2 | `source-validator` | sonnet | Re-fetch every URL, break the claims, flag hallucinations, rate credibility. | `validation-report.md` |
| 3 | `research-synthesizer` | opus | Organize *validated* findings into domains + decision brief. | `synthesis.md`, `decision-brief.md` |

**Loop rule:** if the validator returns a `MUST RE-GATHER` list, re-run
`research-scout` on just those items before synthesizing. Don't synthesize over
unverified claims.

## Why these model tiers
- **Gathering is high-volume and forgiving** → haiku (cheap, fast).
- **Validation needs reasoning but not genius** → sonnet.
- **Synthesis is where accuracy compounds** → opus.
- **Re-derivation of GR/Kerr formulas and renderer code stays in the main
  thread**, per `CLAUDE.md` (formulas come from `skills/kerr-physics/SKILL.md`,
  never re-derived). Sub-agents research and organize; they never decide physics.

## Output convention
One folder per topic: `research/<topic-slug>/`. First topic seeded at
`research/accretion-disk/`.

## How the controller runs it (example)
```
1. Agent(research-scout):       topic="Interstellar-grade accretion disk",
                                output="research/accretion-disk/",
                                seed="research/accretion-disk/SEED.md"
2. Agent(source-validator):     validate research/accretion-disk/sources.md
3. (if MUST RE-GATHER) → back to research-scout on those items
4. Agent(research-synthesizer): synthesize research/accretion-disk/
5. Main thread: read decision-brief.md → make physics/implementation calls.
```
Run scout/validator in the background for long sweeps; they report back when done.
