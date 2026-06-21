export const meta = {
  name: 'directory-review-parallel',
  description: 'Read-only parallel review of stable parts of the Kerr renderer tree (avoids files in flux), with real-time per-agent logs and a final synthesis',
  phases: [
    { title: 'Review', detail: '5 Sonnet agents review independent domains, logging in real time' },
    { title: 'Synthesize', detail: 'consolidate logs into master file + final summary' },
  ],
}

const ROOT = 'C:\\\\Users\\\\songj\\\\Projects\\\\Black'
const LOGS = ROOT + '\\\\review_2026-06-14\\\\logs'

const OFF_LIMITS = [
  'PROJECT.md', 'configs/render.yaml', 'skills/kerr-physics/SKILL.md',
  'src/renderer/noise.py', 'src/renderer/taichi_renderer.py',
  'scripts/thumb.py', 'scripts/showcase_disk.py',
  'docs/specs/2026-06-13-disk-noise-turbulence.md',
  'docs/specs/2026-06-13-V1-self-shadow-source-function.md',
  'docs/specs/2026-06-13-volumetric-disk-and-gas-flow.md',
  'tests/test_disk_noise.py', 'tests/test_disk_step_convergence.py',
  'tests/test_gpu_regression.py', 'tests/test_noise.py',
  'tests/test_disk_self_shadow.py', 'tests/test_disk_source_function.py',
].join(', ')

const COMMON = `Project root: ${ROOT}. This is an offline Kerr black hole renderer (Python + Taichi GPU + Blender pipeline).

STRICT RULES:
- READ-ONLY. Do NOT modify, edit, or create any file EXCEPT your own log file named below.
- Other agents are editing these files LIVE — do NOT open or review them, skip entirely: ${OFF_LIMITS}.
- Use only Read / Grep / Glob to investigate, and Write ONLY for your own log file.

PROJECT CONVENTIONS to check against:
- GR/Kerr formulas must follow skills/kerr-physics/SKILL.md (you MAY read it to cross-check; never edit it). Do not propose re-derivations — just FLAG mismatches.
- Units geometric G=M=c=1; Cartesian Kerr-Schild coords, spin axis +z; metric signature (-+++); spin a=0.999.
- GPU backend LOCKED to ti.init(arch=ti.cuda) — flag any use of ti.gpu.
- Config-driven: numerical params belong in configs/render.yaml, NOT hardcoded in source.
- Windows/cp949 box: text files should be opened with encoding="utf-8"; non-ASCII printed to console (e.g. argparse help) can crash.

REAL-TIME LOGGING (critical): As you FINISH reviewing EACH file, immediately append your findings for that file to your log via the Write tool. To append, Read your current log then Write the full accumulated content back (do NOT lose earlier sections). Use severity tags [CRITICAL]/[HIGH]/[MEDIUM]/[LOW]/[NIT] and file:line references for every finding. End the log with a "## Summary" listing the top issues.`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['domain', 'files_reviewed', 'findings', 'summary'],
  properties: {
    domain: { type: 'string' },
    files_reviewed: { type: 'array', items: { type: 'string' } },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'location', 'title', 'detail'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NIT'] },
          location: { type: 'string', description: 'file:line' },
          title: { type: 'string' },
          detail: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

const DOMAINS = [
  {
    key: 'agent1_core_physics',
    label: 'review:core-physics',
    title: 'Agent 1 — Core Renderer Physics',
    scope: `YOUR DOMAIN — core renderer physics modules (review each):
- src/renderer/disk.py
- src/renderer/disk_flux.py
- src/renderer/geodesic.py
- src/renderer/kerr_params.py
- src/renderer/metric.py
- src/renderer/starmap.py
- src/renderer/__init__.py

Assess per file: (1) correctness/bugs (sign errors, index mismatch, off-by-one, NaN/div-by-zero, sqrt of negatives unguarded); (2) physics fidelity vs SKILL.md (flag mismatches only); (3) hardcoded numeric constants violating config-driven policy; (4) encoding/cross-platform issues; (5) dead code, duplication, unclear naming, missing error handling; (6) inferable test-coverage gaps.`,
  },
  {
    key: 'agent2_tests',
    label: 'review:tests',
    title: 'Agent 2 — Stable Test Suite',
    scope: `YOUR DOMAIN — the STABLE tests under tests/ (review each; skip the off-limits test files):
- tests/test_disk_flux.py
- tests/test_geodesic.py
- tests/test_kerr_params.py
- tests/test_starmap.py
- tests/test_starfield_artifacts.py
- tests/test_starfield_dngr.py
- tests/test_ingest_stars.py
- tests/test_noise_gpu.py
- tests/cuda_smoke_test.py
- tests/test_geodesic/ (data dir, e.g. regression CSV)

Assess per file: (1) do assertions actually validate the claimed property, or are they tautological/too-loose; (2) coverage gaps; (3) flakiness risks (hardcoded paths, GPU/machine dependence, unfixed random seeds, order/time dependence); (4) skipped/xfail and stale refs to renamed/removed code; (5) fixture/golden/CSV hygiene; (6) ti.gpu vs ti.cuda; encoding issues.`,
  },
  {
    key: 'agent3_scripts_blender',
    label: 'review:scripts-blender',
    title: 'Agent 3 — Scripts & Blender Pipeline',
    scope: `YOUR DOMAIN — utility scripts + Blender pipeline (review each; skip off-limits thumb.py/showcase_disk.py):
- scripts/check_starless_map.py
- scripts/export_exr.py
- scripts/gpu_test.py
- scripts/ingest_stars.py
- scripts/README_ingest_stars.md
- src/blender/export_camera.py
- src/blender/__init__.py

Assess per file: (1) correctness/bugs in CLI arg parsing, file I/O, EXR/JSON handling, Windows path handling; (2) encoding bugs (open() without encoding="utf-8"; non-ASCII in argparse help on cp949 console); (3) hardcoded numeric/path constants violating config-driven policy; (4) ti.gpu misuse, error handling, robustness to missing files/bad args; (5) Blender camera-matrix export math & coordinate conventions, headless-safety; (6) dead code, duplication, naming.`,
  },
  {
    key: 'agent4_docs_meta',
    label: 'review:docs-meta',
    title: 'Agent 4 — Docs, Config & Project Meta',
    scope: `YOUR DOMAIN — docs / build config / project meta (review each; skip off-limits PROJECT.md, render.yaml, SKILL.md and the 2026-06-13 specs):
- README.md
- AGENTS.md
- pyproject.toml
- pyrightconfig.json
- docs/specs/2026-06-06-dngr-artifact-remediation.md
- REFERENCE_dngr_paper.md
- research/ (README.md, accretion-disk/*.md status & source logs)
- .codex/config.toml (if present)

Assess: (1) internal consistency — do docs match the actual code layout/commands (cross-check a few claims against real files via Grep/Read, e.g. build commands, file paths, test names referenced)? (2) stale/contradictory instructions, dead links, references to removed files; (3) pyproject.toml / pyrightconfig.json: dependency pins, python version, tool config sanity, missing/duplicated entries; (4) research/ docs: are cited sources/status internally consistent and not contradicting project conventions; (5) doc-sync gaps (claims a feature exists that code doesn't, or vice versa). Do NOT edit anything.`,
  },
  {
    key: 'agent5_reference_repos',
    label: 'review:reference-repos',
    title: 'Agent 5 — Vendored Reference Repositories',
    scope: `YOUR DOMAIN — Open_Source_Repository/ vendored reference material (this is third-party reference code/papers, NOT this project's source):
- Open_Source_Repository/Blackhole/ (shaders, src, docs, assets)
- Open_Source_Repository/Gargantua/
- Open_Source_Repository/starless/
- Open_Source_Repository/tika/
- Open_Source_Repository/paper/*.md (synthesis + intermediate)

These are nested git repos / vendored references. Use Glob+Grep to map them; do NOT try to build them. Assess: (1) what each reference is and what technique it contributes (lensing, disk, noise, etc.) — give the orchestrator a useful map; (2) licensing presence (LICENSE files) and any attribution concerns for borrowed code/formulas; (3) repo hygiene — nested .git dirs committed, large binary assets, anything that bloats the tree or shouldn't be tracked; (4) whether papers/SYNTHESIS.md are consistent with the project's stated physics conventions. Treat this as an inventory + risk scan, not a line-by-line bug hunt.`,
  },
]

phase('Review')
const reviews = await parallel(DOMAINS.map((d) => () => {
  const logPath = `${LOGS}\\\\${d.key}.md`
  const prompt = `${COMMON}

${d.scope}

YOUR LOG FILE (create + append to this, the ONLY file you may write): ${logPath}
Start the log with a "# ${d.title}" header. Use a "## <filename>" subsection per file as you go. Append in real time after each file.

When fully done, return the structured findings object (your final output must be the StructuredOutput call).`
  return agent(prompt, { label: d.label, phase: 'Review', model: 'sonnet', schema: SCHEMA })
    .then((r) => (r ? { ...r, key: d.key, title: d.title, logPath } : null))
})).then((rs) => rs.filter(Boolean))

log(`Review phase complete: ${reviews.length}/${DOMAINS.length} domains returned findings`)

phase('Synthesize')
const masterPath = `${ROOT}\\\\review_2026-06-14\\\\DIRECTORY_REVIEW.md`
const totals = reviews.map((r) => `- ${r.title}: ${r.findings.length} findings (log: ${r.logPath})`).join('\\n')
const allFindings = reviews.flatMap((r) => r.findings.map((f) => ({ ...f, domain: r.title })))
const critHigh = allFindings.filter((f) => f.severity === 'CRITICAL' || f.severity === 'HIGH')

const synthPrompt = `You are the synthesis lead consolidating a parallel read-only directory review of a Kerr black hole renderer.

There is an existing master file at ${masterPath} that I (the orchestrator) created for this review. It has placeholder sections "## Real-Time Findings" and "## Final Synthesis". Your job:
1. READ the master file ${masterPath}.
2. READ all five per-agent log files in ${LOGS}\\\\ (agent1_core_physics.md, agent2_tests.md, agent3_scripts_blender.md, agent4_docs_meta.md, agent5_reference_repos.md). Some may be missing if an agent failed — note that.
3. Rewrite the master file with Write, preserving everything ABOVE the "## Real-Time Findings" header verbatim, then:
   - Under "## Real-Time Findings": embed each agent's full log content under a "### <agent title>" heading (concatenate the logs verbatim so the real-time record is preserved in one place).
   - Under "## Final Synthesis": write a comprehensive cross-cutting summary: (a) a severity-sorted table of the most important findings across ALL domains with file:line; (b) cross-cutting themes (e.g. recurring encoding bugs, config-policy violations, weak tests, physics-vs-SKILL mismatches); (c) a prioritized recommended action list; (d) explicit note of which areas were EXCLUDED because other agents were editing them live (list the off-limits files) so the reader knows coverage boundaries.

This master file IS the intended deliverable — writing to it is authorized. Do NOT modify any other existing project file.

Here is structured data to help you (the logs are the source of truth for detail):
TOTALS:
${totals}

CRITICAL/HIGH findings across all domains (JSON):
${JSON.stringify(critHigh, null, 2)}

Return a short confirmation plus the count of findings by severity.`

const synth = await agent(synthPrompt, { label: 'synthesize:master', phase: 'Synthesize', model: 'sonnet' })

return {
  domainsReturned: reviews.length,
  totalFindings: allFindings.length,
  criticalHigh: critHigh.length,
  bySeverity: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NIT'].map((s) => `${s}: ${allFindings.filter((f) => f.severity === s).length}`),
  synthConfirmation: synth,
  masterPath,
}
