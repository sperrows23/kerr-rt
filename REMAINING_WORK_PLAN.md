# Kerr Renderer — Remaining Work Plan

> **Planning only — no code changes proposed for execution here.** This document
> enumerates every known unresolved item, the recommended fix, the implementation
> order, and the dependencies/risks for each. Nothing below has been applied.
>
> Source of truth for status: `IMPLEMENTATION_PLAN.md` §Status and the
> `PROJECT_MAP.md` per-file notes. Physics policy unchanged: any formula touch
> must cite a `skills/kerr-physics/SKILL.md` number — never re-derive.

**Baseline (updated 2026-06-04):** `black_hole.r_plus` is restored to `1.0447`
(Tier 0, done). Since the last update three items shipped (working tree, pending
commit): the **center "static" seam fix** (`render.j_fold` LOD fold-saturation +
per-step φ-wrap + shortest-arc exit interpolation in `taichi_renderer.py`), **F2**
(shutter fix, implemented), and **F3** (`tests/test_gpu_regression.py`, committed).
GPU Doppler ≈ 7.77×; CPU pytest green plus the new CUDA-gated GPU regression
(skips without a GPU). The `BACKGROUND_DNGR_PLAN.md` background-rearchitecture
proposal is also new (planning only, unimplemented). Remaining working-tree
leftovers: `pyrightconfig.json` untracked, `.codex/config.toml` pinned.

---

## 1. Inventory of unresolved items

| ID | Item | State | Class |
|----|------|-------|-------|
| **F2** | `export_exr._shutter_arc` ignored the `camera.shutter_fraction` config key (hardcoded a `0.5` factor) | **DONE** (implemented via Option A — added `render.fps`; see below) | Code + config |
| **F4** | `render.max_steps_pipe_b` declared in `render.yaml` but read by no kernel (inert key) | Open (dead config) | Config or code |
| **F5** | Docstring / cross-reference drift after the `[y,u,…]` state migration | Open (docs) | Docs only |
| **F3** | No automated regression for the GPU path (Doppler check was a **manual** `gpu_test.py` run) | **DONE** (`tests/test_gpu_regression.py` landed, CUDA-gated) | Tests |
| **3.1/3.2** | Ship depth occlusion (early ray termination vs. Blender ship Z) | **Blocked** on a Blender ship **Z-depth EXR** asset that does not exist yet | Pipeline asset + code |
| **2.3** | Hardware `ti.Texture` starmap + `sample_lod` | **Deferred** — Taichi 1.7.4 exposes no mip-upload API on the pinned build | External (Taichi) |
| **T3** | Moving-camera observer model (camera with peculiar velocity, not just ZAMO) | **Roadmap, gated** — needs a new `SKILL.md` formula approved first | Physics (gated) |

---

## 2. Per-item detail

### F2 — `shutter_fraction` was silently ignored → **DONE (Option A implemented)**

> **Resolved (2026-06-04, working tree).** Implemented as **Option A** below, not
> the originally-recommended Option B. `_shutter_arc(frames, idx, shutter_fraction,
> fps)` now returns `dphi * fps * shutter_fraction`, and a new
> `render.fps: 24.0` key was added. Behaviour is byte-identical to the old
> `dphi * 0.5` at 24 fps (`24 × 1/48 = 0.5`) but now scales correctly at other
> frame rates, and `camera.shutter_fraction` is finally read. The historical
> analysis below is retained for context. (Option B was *not* taken — see the
> note at the end of this item.)

**What was wrong.** `scripts/export_exr.py:40` defined
`_shutter_arc(frames, idx, shutter_fraction)` but the body only computed
`dphi * 0.5` — the `shutter_fraction` argument was never used. The config key
`camera.shutter_fraction: 0.020833` (documented as "1/48 s") was therefore dead on
this path, and the `0.5` was an undocumented magic number.

**Root cause — a unit mismatch.** `0.020833 s = 1/48 s` is an *absolute shutter
time*. Converting it to a *per-frame azimuthal arc fraction* needs the frame
interval (i.e. an fps), which is **not** in the config. The hardcoded `0.5`
happens to equal `(1/48 s) ÷ (1/24 s) = 180° shutter` **only if** the render is
24 fps — a coincidence that is currently load-bearing and unstated.

**Recommended fix (Option B — dimensionless).** Replace the seconds-based key with
a dimensionless shutter-angle fraction and read it:
- `configs/render.yaml`: `camera.shutter_fraction: 0.020833 # 1/48 s` →
  `camera.shutter_angle_fraction: 0.5  # 180° shutter (open half the frame interval)`.
- `_shutter_arc`: `return dphi * shutter_angle_fraction` (use the argument; delete
  the magic `0.5`).

This makes the call site honest, removes the fps dependency entirely, and the
behaviour is byte-identical to today (still `dphi * 0.5`).

**Alternative (Option A — keep seconds) — THIS IS WHAT SHIPPED.** Add a
`render.fps` key and compute `arc = dphi * fps * shutter_fraction`. More physically
literal; introduces one new key. The original recommendation favored Option B as
lighter, but Option A was chosen at implementation time to keep `shutter_fraction`
meaningful in seconds and make the fps dependency explicit rather than baked into a
dimensionless constant. (No guard on the multiplication is needed — it is a product,
not a division.)

**Risk:** Low. Motion blur is opt-in (`--motion-blur`); the default export path is
untouched. No physics, no SKILL formula. Verified byte-identical to the legacy
`dphi * 0.5` at the assumed 24 fps.

> **Decision note (for the record).** This is the one place the shipped code
> diverges from this plan's original recommendation (Option B). If the dimensionless
> `shutter_angle_fraction` is ever preferred, it would *replace* the `fps ×
> shutter_fraction` product — but that is now a refactor of working code, not an
> open bug.

---

### F4 — `max_steps_pipe_b` is an inert config key (recommended: remove, or wire deliberately)

**What's wrong.** `render.max_steps_pipe_b: 200` exists in `render.yaml` but no
kernel reads it; disk marching (Pipe B) is bound to the Pipe A geodesic loop
(`max_steps_pipe_a`). The key implies an independent disk step budget that does
not exist.

**Recommended fix (Option Remove).** Delete the key and its comment from
`render.yaml`; add a one-line note in `PROJECT_MAP.md` that Pipe B shares the Pipe
A step count by design. Lowest risk, zero behaviour change.

**Alternative (Option Wire).** If an independent disk budget is actually wanted
(e.g. fewer disk samples than geodesic steps for speed), thread the key into
`render_beauty_physics` as a second loop bound. This is a **behaviour change** to
the render and would need a Doppler re-check and a disk-max regression — only do
this if there is a real performance motive. Default recommendation is **Remove**.

**Risk:** Remove = none (dead key). Wire = changes disk sampling → guarded by the
manual Doppler check; do not bundle with anything else.

---

### F5 — Docstring / cross-reference refresh (docs only)

**What's wrong.** After the Phase-1 `[r,θ,φ,t,v_r,v_θ] → [y,u,φ,t,v_y,v_u]`
state migration, a few comments/docstrings still describe the renderer in the old
frame, and some module headers predate the kernel split. The **CPU**
`geodesic.py` docstring (`[r,θ,…]`, `v_r=Δ·p_r`) is *correct and must stay* — it
describes the CPU reference, which did not migrate. The drift is on the GPU side
and in cross-references.

**Recommended fix.** Pass over `taichi_renderer.py` module/function docstrings and
any `# state = [...]` comments to match the shipped `[y,u,…]` reality (most of
this is already captured accurately in `PROJECT_MAP.md`, which can serve as the
copy source). Confirm `geodesic.py` is left untouched.

**Risk:** None (comments only). Do **not** let a docstring edit drift into a code
edit on `disk.py` or the physics funcs.

---

### F3 — GPU regression harness → **DONE**

> **Resolved (committed).** `tests/test_gpu_regression.py` landed. It drives the
> production `render_beauty_frame` (frame 0, 1920×1080, disk on), marked
> `pytest.mark.gpu` and skipped cleanly when CUDA is absent (Taichi init deferred
> into a module-scoped fixture). Asserts implemented:
> 1. `NaN == 0`,
> 2. right/left Doppler luminance ratio in `[7.0, 8.5]` with right > left,
> 3. `disk_max` within 5% of the pinned `_DISK_MAX_REF = 12.7707` (measured on the
>    production path; note this is the **FHD beauty-buffer peak**, not the 14.11
>    Pipe-A disk-march figure quoted in the original plan).
>
> This is the standing safety net for any future GPU physics edit; the manual
> `gpu_test.py` Doppler eyeball is now backed by CI.

---

### 3.1 / 3.2 — Ship depth occlusion (blocked on a missing asset)

**What's blocked.** Early ray termination (`if ray_length > ship_z: break`) and
the ship-depth texture input need a **Blender ship Z-depth EXR** pass per frame.
`camera_matrix.json` exists; the ship Z pass does **not**. The kernel-side hooks
(`ti.Texture(r32f)` arg, the break) are scoped in `IMPLEMENTATION_PLAN.md` but
cannot be verified without the input.

**Hard caveat (units).** `ray_length` is a **Mino-affine** path length, *not*
metric distance and *not* the same units as a Blender camera-space Z. Comparing
them requires a documented mapping; **do not assume they are interchangeable.**
This is the single biggest correctness trap in the whole remaining backlog.

**Recommended sequence (when unblocked):**
1. **(Phase-1 work)** Add a Z-depth output to the Blender ship render
   (`render_spaceship/ship_####.exr` with a `Z` channel, or a sidecar
   `shipz_####.exr`). Document its color space and unit (camera-space metres or
   Blender units).
2. **Derive and document the Mino-affine ↔ camera-Z mapping** as a short note
   (this is bookkeeping, not GR — no SKILL formula — but write it down).
3. Wire the `ti.Texture(r32f)` ship-Z input + the early-out, guarded for a
   missing asset (no-op when absent).
4. Verify with a synthetic depth plane before trusting real ship Z.

**Risk:** Medium. Wrong unit mapping ⇒ ship clips through / floats above the disk.
Gate behind a flag defaulting off until the mapping is validated.

---

### 2.3 — Hardware `ti.Texture` starmap (deferred, external)

**Status.** Probed and deferred: Taichi 1.7.4 (pinned for the RTX 5060 / sm_120)
exposes no mipmap-upload / `from_numpy` path for `ti.Texture`, so `sample_lod`
would sample base level only and reintroduce starmap flicker. The correct manual
f16 mip pyramid is retained and is within the VRAM budget (§D of the plan).

**Recommendation.** **Do nothing now.** Revisit only if/when Taichi is upgraded
past 1.7.4 — and that itself is gated by re-confirming Blackwell/sm_120 JIT on the
new version (CLAUDE.md pins 1.7.4 for a reason). Track as "external dependency,"
not as actionable backlog.

---

### T3 — Moving-camera observer model (roadmap, physics-gated)

**Status.** The camera currently uses the ZAMO tetrad (Formula 7). A camera with
its own peculiar velocity (e.g. the spaceship's frame, with the correct
aberration/Doppler for the *observer's* motion, not just the ZAMO) requires a new
tetrad-boost formula.

**Recommendation.** **Gated — do not write physics yet.** Per policy: draft the
formula as a new `SKILL.md` entry (boost of the ZAMO tetrad by the observer
4-velocity), get **human approval**, *then* implement. Until approved this stays a
roadmap line, not a task.

**Risk:** High if rushed (sign/normalization errors in a boosted tetrad). The
policy gate exists precisely for this.

---

## 3. Recommended implementation order

Ordered so that the safety net comes first, cheap/clean items next, asset- and
physics-blocked items last.

1. ~~**F3 — GPU regression harness.**~~ **DONE** — landed as
   `tests/test_gpu_regression.py` (the verification backbone for everything after).
2. **F5 — docstring refresh.** Docs only, zero risk; do it while context is fresh.
   No dependencies. *(Still open.)*
3. ~~**F2 — honor `shutter_fraction`.**~~ **DONE** — implemented via Option A
   (`render.fps` + `Δφ·fps·shutter_fraction`), not the originally-planned Option B.
4. **F4 — remove `max_steps_pipe_b`.** Dead-key removal; trivial. (Choose *Remove*,
   not *Wire*, unless a perf motive appears.) *(Still open.)*
5. **3.1 / 3.2 — ship occlusion.** *Blocked* until the Blender ship Z-depth pass
   exists. Sequence: produce the asset → write & document the unit mapping → wire
   the kernel input behind an off-by-default flag → validate. Depends on F3 for
   regression safety.
6. **2.3 — `ti.Texture`.** *External.* Only after a Taichi upgrade is independently
   justified and re-validated on sm_120.
7. **T3 — moving camera.** *Gated.* Only after a new `SKILL.md` formula is approved.

```
F3 ─┬─▶ F5
    ├─▶ F2
    ├─▶ F4
    └─▶ 3.1/3.2  (also needs: Blender ship-Z asset + unit-mapping note)

2.3  ◀── (external) Taichi > 1.7.4 re-validated on sm_120
T3   ◀── (gated)   new SKILL.md tetrad-boost formula approved
```

---

## 4. Dependencies & risks at a glance

| Item | Hard dependency | Top risk | Mitigation |
|------|-----------------|----------|------------|
| F3 | none | tolerance band too tight/loose | widen band to survive JIT nondeterminism; pin disk_max ± ε |
| F5 | none | a docs edit creeping into `disk.py`/physics | comments only; leave `geodesic.py` CPU docstring intact |
| F2 | none (F3 helps) | unit confusion (seconds vs. ratio) | Option B removes the unit entirely |
| F4 | none | *Wire* path changes render | choose *Remove*; if *Wire*, Doppler + disk-max re-check |
| 3.1/3.2 | Blender ship Z-depth asset | Mino-affine vs. camera-Z unit mismatch | document mapping; off-by-default flag; synthetic-plane test |
| 2.3 | Taichi > 1.7.4 | breaks sm_120 JIT | keep 1.7.4 pin; manual pyramid already correct |
| T3 | approved SKILL formula | boosted-tetrad sign/normalization errors | physics gate; human review before code |

---

## 5. Out of scope / explicitly not changing

- `src/renderer/disk.py` — frozen for numerical regression; do not touch.
- `src/renderer/geodesic.py`, `metric.py` CPU reference — correct as-is; the
  `[r,θ,…]` docstring is intentional, not drift.
- Any `render.yaml` **physics** value (spin, ISCO, temperature model, etc.).
- GPU backend stays `ti.init(arch=ti.cuda)` — never `ti.gpu`.
- No formula re-derivation; gated items (T3, and any physics edit) wait for an
  approved `SKILL.md` entry.
