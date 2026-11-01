# Task: Fetcher Recording Approach

## Goal
Record ground-truth source file dependencies from daily CI model runs and feed them
into `tests_fetcher.py` as supplemental edges — covering dynamic dispatch patterns
(image/video/feature extraction processors) that static import analysis misses.

## Background
- PR #49826 (merged, `c2d2559517`): heuristic fix wiring `image_processing_<model>.py`
  to `test_image_processing_<model>.py` and `test_processing_<model>.py` by naming convention.
- Still missing: `video_processing_*.py`, `feature_extraction_*.py`, cross-model deps
  (e.g. `got_ocr_2/image_processing` -> `tests/models/internvl/`).
- Heuristic stays as fallback; recording is the robust long-term solution.

## Repos involved
- `transformers` (huggingface/transformers): components 1, 3, 5 + generated JSON
- `transformers-ci` (huggingface/transformers-ci): components 2, 4

## The 5 components

### 1. Recorder -- `testing_utils.py` + `conftest.py` in transformers
NOT minimal -- needs: env var check, repo root resolution, sys.modules scan,
module.__file__ resolution, relative path computation, filtering to src/transformers/,
JSON write, error handling. ~25-30 lines of real logic.

Follows the established pattern exactly:
- `patch_testing_methods_to_collect_info()` -- defined in testing_utils.py, called from conftest line 448
- `pytest_terminal_summary_main()` -- defined in testing_utils.py, called from conftest line 392
- `patch_psutil_cpu_memory()` -- defined in testing_utils.py, called from conftest line 202

**Why NOT a new file**: `conftest.py` only adds `src/` to sys.path, not `utils/`.
Importing from `utils/ci_deps_recorder.py` would require ugly importlib hacks.
testing_utils.py is already imported by conftest -- no new plumbing needed.

`testing_utils.py`: add one function
```python
def record_ci_source_deps() -> None:
    """Record imported transformers source files to CI_RECORD_DEPS_OUTPUT (no-op if unset)."""
    ...  # ~25 lines
```

`conftest.py`: extend existing pytest_sessionfinish (line 395) + add to imports (line 33-42)
```python
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0
    record_ci_source_deps()  # no-op unless CI_RECORD_DEPS_OUTPUT is set
```

### 2. Trigger -- `daily-ci_reusable_model_job.yml` in transformers-ci (~2 lines)
In the "Run all tests on GPU" step env block, add:
```
CI_RECORD_DEPS_OUTPUT: /transformers/reports/${machine_type}_run_models_gpu_${matrix_folders}_test_reports/deps.json
```
The deps.json lands inside the existing report directory and is auto-included in the
artifact upload -- no separate upload step needed.

### 3. Collation script -- `utils/collate_ci_source_deps.py` in transformers (~50 lines)
Takes a directory of downloaded artifacts as input.
- For each dir matching `*_run_models_gpu_models_*_test_reports/` that contains `deps.json`:
  - Parse dir name -> model folder (e.g. `models_clip` -> `tests/models/clip`)
  - Read `deps.json` -> list of imported source files
  - Glob `tests/models/clip/*.py` -> test files to attribute
  - Accumulate: `{source_file -> set(test_files)}`
- Write merged result to `utils/fetcher_source_deps.json`.

### 4. Collation job -- `daily-ci_reusable.yml` in transformers-ci (~25 lines YAML)
Add after `run_models_gpu`, following same pattern as existing `run_extract_warnings` job:
- Checkout transformers
- Install transformers
- `uses: actions/download-artifact` -- download all model GPU artifacts
- Run `python3 utils/collate_ci_source_deps.py --artifacts-dir <path>`
- Commit updated `utils/fetcher_source_deps.json` to transformers main
  (needs `contents: write` permission + PAT or GITHUB_TOKEN with push rights)

### 5. Consumer -- `tests_fetcher.py` in transformers (~10 lines)
In `create_reverse_dependency_map()`, after static analysis builds the reverse map:
- Try to read `utils/fetcher_source_deps.json`; silently skip if missing (file not yet
  generated, fresh clone, etc.) -- heuristic remains as fallback.
- For each `{source_file: [test_files]}` entry, inject as supplemental edges
  into `reverse_map`.

## Data flow
```
Daily CI
  pytest tests/models/clip/  (CI_RECORD_DEPS_OUTPUT set)
    -> sys.modules at session end -> deps.json -> artifact
  pytest tests/models/internvl/ -> deps.json -> artifact
  ...

Collation job (after all model jobs)
  download all artifacts
  collate_ci_source_deps.py
    models_clip    -> [image_processing_clip.py, ...]  -> attribute to tests/models/clip/*.py
    models_internvl -> [got_ocr_2/image_processing_got_ocr_2.py, ...]
                                                       -> attribute to tests/models/internvl/*.py
    -> utils/fetcher_source_deps.json -> commit to main

PR CI (next day+)
  tests_fetcher.py reads fetcher_source_deps.json
  -> supplemental edges merged into reverse_map
  -> image_processing_clip.py -> test_processing_clip.py  (fixed)
  -> got_ocr_2/image_processing_got_ocr_2.py -> test_processing_internvl.py  (fixed)
```

## Design decisions
- Attribution granularity: all test files in a folder attributed to all source files
  imported during that folder's session. Slightly conservative (modeling test might
  get attributed to image processor) but acceptable.
- Staleness: heuristic (PR #49826) stays as fallback for new models not yet recorded.
- Scope: only `run_models_gpu` for now. Trainer/quantization jobs out of scope.
- Update cadence: after every daily CI run. May throttle to weekly if commit noise is an issue.
- File location: `utils/fetcher_source_deps.json` in transformers repo (committed, ~few hundred KB).

## What recording catches that heuristics never can
- All modality types (image, video, feature extractor) -- automatically, zero maintenance
- Cross-model deps (e.g. `got_ocr_2` image processor -> `internvl` processor tests)
- Any future dynamic dispatch pattern -- self-updating

## Status
- [x] Component 1: recorder hook in conftest.py
  - Three functions in `src/transformers/testing_utils.py`:
    - `snapshot_ci_source_deps_baseline()` — called from `pytest_sessionstart`, snapshots sys.modules before collection
    - `record_ci_source_deps()` — called from `pytest_sessionfinish`, diffs against baseline and writes JSON
    - `_ci_source_deps_current()` — shared helper, filters sys.modules to src/transformers/ files
  - Bug fixed: `parents[3]` -> use `.parent.parent.parent.resolve()` to reach repo root from src/transformers/
  - Both functions imported and called in `conftest.py` (pytest_sessionstart + pytest_sessionfinish)
  - Noise analysis — two confirmed sources, both common to ALL model tests:
    1. `exporter_dynamo.py` eagerly imports 37 fixed model packages via `@register_patch` at module load:
       clip, llama, gpt2, t5, qwen2, gemma, cohere_compass, glm4v, minicpmv4_6, rwkv, sam3, ...
       (cohere_asr is NOT in this list)
    2. `test_pipeline_mixin.py` — every model test that uses PipelineTesterMixin imports it, which
       mass-imports ALL pipeline test classes at module level. Those classes materialize auto-mappings
       at class-definition time (e.g. `list(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items())`), which
       triggers the lazy loader to import ALL registered models including cohere_asr, albert, bart, etc.
       Chain: test_modeling_vit.py → test_pipeline_mixin.py → test_pipelines_asr.py
              → MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items() → auto_factory lazy load → cohere_asr
  - Why the static fetcher does NOT see this: auto files only contain string dicts
    (`{"cohere_asr": "CohereAsrForCTC"}`), no actual model imports. The auto module is a
    complete static-analysis firewall — fetcher stops there, never reaches cohere_asr.
    This is correct: changing cohere_asr should NOT trigger vit tests.
  - Noise source 3 (framework): `from transformers import MODEL_MAPPING` in test_modeling_common.py
    pulls in pipelines/__init__.py and ~28 pipeline files + bert/bart/gpt2/llama configs at session start.
    These are genuinely common to all model tests.
  - Import paths into test_export.py (which imports exporter_dynamo) — there are TWO entry points:
    (a) test_modeling_common.py line 127: `from .exporters.test_export import ExportTesterMixin`
    (b) tests/generation/test_utils.py line 125: `from tests.exporters.test_export import ExportGenerateTesterMixin`
    Stubbing only one (as initially tried for ExportTesterMixin) had ZERO effect because (b) remained.
  - EMPIRICALLY VERIFIED noise experiment (old, stubs applied to vit test only in-place):
    - 704 files: baseline (no stubs)
    - 197 files: stub PipelineTesterMixin in vit test (eliminates pipeline mixin chain: ~507 files)
    - 170 files: stub ExportGenerateTesterMixin in generation/test_utils.py (partial: ExportTesterMixin path remains)
    - 79 files: stub all 3 (PipelineTesterMixin + ExportGenerateTesterMixin + ExportTesterMixin)
    - 4 files: genuine vit-specific signal
  - ACTUAL RESULT with committed [dummy] stubs (branch fetcher-recording):
    - Measured via: CI_RECORD_DEPS_OUTPUT=... pytest tests/models/vit/test_modeling_vit.py --collect-only
    - Per-commit breakdown:
      | commit                                          | files recorded | delta  |
      | (recorder only, no stubs)                       |     728        |        |
      | [dummy] 1: stub pipeline mixin (7d0014f785)     |     215        | -513   |
      | [dummy] 2: stub export mixins (e8636eb326)      |      96        | -119   |
      | [dummy] 3: move pipeline to local (c69a31eede)  |      64        |  -32   |
  - FETCHER vs RECORDER per-commit comparison (verified 2026-10-23, using create_reverse_dependency_map):
    | State                              | fetcher (reverse map, src→triggers vit) | recorder (runtime imports) |
    | no stubs (Windows fix only)        |                854                      |           728              |
    | [dummy]1: stub pipeline mixin      |                808                      |           215              |
    | [dummy]2: stub export mixins       |                795                      |            96              |
    | [dummy]3: move pipeline local      |                795                      |            64              |
    - Fetcher 854→795: stubs barely help (−59). Src→src transitive chains (modeling_utils,
      integrations, generation...) live in src files — test file stubs can't touch them.
    - Recorder 728→64: stubs massively help (−664). Runtime noise from test imports eliminated.
    - Fetcher over-approximation (795 src files) is a pre-existing property, not introduced by our work.
    - Note: required fix to create_reverse_dependency_map — line 886 `raise ValueError` →
      `continue` to skip ghost uppercase paths from Windows case-insensitive is_file()
      (e.g. FULL_ATTENTION.py matching full_attention.py in cache_allocators/).
    - Final 64 files: 4 vit-specific + 60 framework noise
    - 4 vit-specific files (all under src/transformers/models/vit/):
        __init__.py, configuration_vit.py, image_processing_pil_vit.py, modeling_vit.py
    - 60 framework-noise files will be filtered by frequency (appear in every model test)
  - Frequency filter validity: CONFIRMED. The 60 noise files appear in every model test;
    the 4 vit files appear only in vit. Threshold >10% correctly separates them.
  - Signal survives: `image_processing_pil_vit.py` CONFIRMED present in vit-specific set.

## Recorder analysis (verified 2026-10-23)

### Baseline subtraction design
- Recorder uses `final - baseline`: baseline taken at `pytest_sessionstart`, final at `pytest_sessionfinish`
- By `pytest_sessionstart`, conftest has already loaded → imports `testing_utils` → pulls in
  151 transformers modules including `modeling_utils.py` and `deepspeed.py`
- VERIFIED: `modeling_utils.py` and `deepspeed.py` absent from recorder output in BOTH
  no-stubs (728) and with-stubs (64) runs — they are in the baseline, subtracted out
- The 728 and 64 numbers are the DIFF counts (not total sys.modules), confirmed by reading
  the actual JSON files

### Why recorder has far fewer files than fetcher (64 vs 854)
- Fetcher parses ALL import statements including:
  1. `TYPE_CHECKING` guards (e.g. `if TYPE_CHECKING: from .trainer import Trainer`) —
     fetcher sees it, runtime never executes it (TYPE_CHECKING=False at runtime)
  2. `if is_X_available():` guards at module level — skipped if package not installed
  3. `try/except ImportError` optional imports
  4. Function-body lazy imports — skipped during --collect-only, may be skipped at runtime
     if the function is never called with that code path
- Recorder captures only what actually ends up in sys.modules in THIS run

### Fetcher false positives (verified)
- `trainer.py` → triggers 944/946 model tests via `TYPE_CHECKING` guard in testing_utils.py
  Chain: test → testing_utils → `if TYPE_CHECKING: from .trainer import Trainer`
  Recorder: never in output (TYPE_CHECKING=False, never in sys.modules)
- `deepspeed.py` → triggers 3879 tests via real top-level import in modeling_utils.py
  Chain: test → modeling_vit → modeling_utils → `from .integrations.deepspeed import ...`
  Recorder: in baseline (imported before sessionstart via conftest→testing_utils), subtracted out
- `modeling_bert.py`, `quantizer_bnb_4bit.py`, `gpt2/modeling_gpt2.py` etc. — all via
  fetcher's transitive src→src closure; some are real runtime imports (in baseline), some
  are TYPE_CHECKING/conditional guards

### Recorder as fetcher replacement (future direction)
To replace fetcher entirely, recorder needs redesign:
1. Remove baseline subtraction (or take baseline BEFORE any transformers import)
2. Full test execution (not --collect-only) to capture function-body imports
3. Collation builds complete runtime reverse map: file X → tests that imported it
4. More precise than fetcher (no TYPE_CHECKING false positives)
5. More complete (captures dynamic dispatch: image/video processors)
6. Gap: new files with no CI history → keep fetcher as fallback for those

## Alternative: coverage.py-based reverse map

### What coverage.py does
- Uses `sys.settrace` internally — fires on every line/function call, not just first imports
- `pytest --cov=src/transformers --cov-context=test` tags each covered line with which test triggered it
- Gives map: `test_modeling_vit::test_model → {modeling_vit.py lines 1-342, modeling_utils.py lines 50-200, ...}`
- Build reverse map: "which tests had line X covered?" → run only those when line X changes
- Example: `deepspeed.py` ~4% coverage for vit test → change to uncovered lines → vit not triggered

### Advantage over import-hook / sys.modules approaches
- `sys.settrace` fires on EVERY function call, even in already-imported modules
- Even though `modeling_utils.py` is in sys.modules from conftest, when vit test calls
  `model.forward()` → coverage.py captures those lines as belonging to the vit test
- Solves the "already in sys.modules" problem for function-body code

### Remaining blind spot
- Top-level module code executed at import time (during conftest/collection) is NOT attributed
  to any specific test — it ran before the test context was established
- Example: `all_model_classes = (ViTModel,)` at class definition time in test file
  → if ViTModel already imported, coverage.py misses this as a vit dep
- Class-level attributes, module-level constants built at import time → same gap

### Fundamental limitation — no perfect solution
- Import hooks: miss already-cached modules
- sys.modules diff (current recorder): same
- OS-level file tracking (strace): same — Python never re-opens already-imported files
- coverage.py: misses top-level/collection-time code
- Fresh process per test: complete but extremely slow
- Static analysis (fetcher): over-approximates via TYPE_CHECKING and transitive chains

### Import-time coverage (fresh process per test file)
Alternative lighter approach: for each test file, run a fresh Python process that just imports it:
  `coverage run --source=src/transformers -c "import tests.models.vit.test_modeling_vit"`
- Fresh process → empty sys.modules → all module-level code runs fresh
- Captures full import chain: test file → modeling_vit → modeling_utils → deepspeed (all at top-level)
- No TYPE_CHECKING false positives (runtime, TYPE_CHECKING=False)
- No GPU needed, fast (just importing, no test execution)
- Correctly excludes trainer.py (TYPE_CHECKING guard never fires)

Import-time vs runtime coverage comparison:
| Approach                        | Import chain deps | Dynamic dispatch (processor, backbone) |
| Static fetcher                  | ✓ (over-approx)  | ✗                                      |
| Import-time coverage (fresh)    | ✓ (precise)      | ✗                                      |
| Runtime coverage (actual tests) | ✓                | ✓                                      |

### Why runtime coverage is still needed
Import-time coverage misses dynamic dispatch patterns — exactly the original motivation
for the recording approach (cross-model processor deps, backbone reuse):
- `test_processing_internvl.py` loads `got_ocr_2/image_processing_got_ocr_2.py` via
  `AutoProcessor.from_pretrained()` during test execution — not visible at import time
- A model using ViT as visual backbone — `modeling_vit.py` only imported when the
  composite model is instantiated inside a test method
- `AutoModel.from_pretrained()` resolves to specific model class at runtime from config
These are precisely what static fetcher misses and what Raushan/user's recording work targets.
Runtime coverage (actual test execution with GPU) is required to capture them.

### Practical recommendation
- coverage.py runtime is the most complete approach but needs GPU + actual test execution
- Import-time coverage (fresh process) is a cheap supplement: precise import chain, no GPU
- Neither alone is complete; together they cover most cases
- Keep fetcher as fallback for: new files with no CI history

### Decision: start with sys.modules diff (current recorder), keep fetcher
Since fetcher is kept as baseline, sys.modules diff is sufficient as supplement:
- Dynamic dispatch deps (cross-model processor, backbone reuse) are NEW imports during
  test execution → not in baseline → captured correctly by sys.modules diff
- Framework files already in baseline (modeling_utils, deepspeed) → handled by fetcher already
- No overlap needed between recorder and fetcher for those files

coverage.py only adds value if replacing fetcher entirely (function-level precision for
already-imported files). With fetcher kept → unnecessary complexity + 2x slowdown.

Path forward:
1. sys.modules diff (already implemented) — cheap, no overhead, captures dynamic dispatch deps
2. Measure real CI results — how many correct new edges does it add?
3. Consider coverage.py only if specific gaps found that sys.modules diff can't cover

Baseline subtraction note: current design (baseline after conftest loads) is correct for the
supplemental purpose — framework files subtracted out are already handled by fetcher.

### Baseline subtraction safety — verified 2026-10-23
Concern: would backbone/processor deps (the key signal) also end up in baseline and get subtracted?
VERIFIED: after importing testing_utils (simulating conftest load), baseline contains 151 files,
of which only 1 is under /models/ (models/__init__.py). No model-specific files at all:
- modeling_idefics3.py → NOT in baseline ✓
- image_processing_got_ocr_2.py → NOT in baseline ✓
- backbone files, processor files → NOT in baseline ✓
Reason: transformers uses lazy loading — `from transformers import logging` (what testing_utils
does) does not trigger any model imports. Model-specific files only import when explicitly
accessed during test collection or execution.
Conclusion: subtraction is safe. Model-specific deps captured correctly in diff.

- [x] Component 2: trigger env var in daily-ci_reusable_model_job.yml (in transformers-ci)
- [x] Component 3: collate_ci_source_deps.py — reads `deps_by_file.json`, merges into `test_file → [src_files]`, writes `--output` path
- [x] Component 4: collation job in daily-ci_reusable.yml (in transformers-ci)
- [ ] Component 5: consumer in tests_fetcher.py
- [ ] Initial empty `utils/fetcher_source_deps.json` committed to repo

## Side PRs / Housekeeping (2026-10-27)

- **PR #50001** (merged): defer `SpeechT5HifiGan` and `BasicTokenizer` imports in pipeline source files to call site — removes 4 spurious model files from collection-phase deps of all ~511 `test_modeling_*.py` files (case 6 layer 2).
- **PR #50000** (merged): lazy-resolve `@register_patch` paths in `apply_patches` instead of at decoration time — removes ~111 spurious model files from collection-phase deps of all ~511 `test_modeling_*.py` files (case 5).
- **PR #49994** (merged): fix 4 missing `.as_posix()` calls in `tests_fetcher.py`
  (`create_reverse_dependency_tree` all_modules, `create_reverse_dependency_map` all_modules,
  `preprocessor_patterns` test_files key, `preprocessor_patterns` src_key).
  Added on `fetcher-recording` branch (commit `91ac42ede4`), identified as standalone fix
  worth landing on main. Both sets reverted from recorder branch (commits `c5df1feee7`, `21d07cb80c`).

- **[DBG2] debug print removed** from `src/transformers/testing_utils.py`: leftover
  `elif "distilbert" in name.lower(): print("[DBG2] ...")` branch added during
  `sys.modules` caching investigation, cleaned up before committing.

## Output file restructuring (2026-10-25)

Each model CI job now writes 3 files (not just `deps.json`):
- `deps_by_file.json` — `{test_file: [src_files]}` — primary output, used by collation
- `deps_by_sys_module.json` — flat list of all imported src files (sys.modules diff)
- `deps_chain.json` — raw per-import log with phase/test_id

`conftest.py` `pytest_sessionfinish` calls all three:
```python
record_ci_source_deps()
record_ci_source_deps_by_file()
record_ci_source_deps_chains()
```

`collate_ci_source_deps.py` reads `deps_by_file.json` (not `deps.json`), merges across artifact dirs.

## Import tracking fixes (2026-10-25)

Two fixes to `_tracking_import` in `testing_utils.py`:

**1. Fromlist submodules**: `from transformers.models.clip import image_processing_clip` calls
`__import__("transformers.models.clip", fromlist=["image_processing_clip"])`. Python loads the
submodule into sys.modules silently. Fix: after the import, check `f"{resolved_name}.{item}" in sys.modules`
for each fromlist item.

**2. Relative imports (level > 0)**: `from ...image_processing_backends import TorchvisionBackend`
calls `__import__("image_processing_backends", level=3)`. The name has no `transformers.` prefix.
Fix: resolve using `globals["__package__"]` and level to get the absolute module name.

## Symlink abspath fix (2026-10-25)

`notification_service.py` aggregate step used `os.symlink(d, dst)` where `d` is a relative path.
This creates a broken symlink (resolves relative to tmpdir, not cwd).
Fix: `os.symlink(os.path.abspath(d), os.path.join(tmpdir, os.path.basename(d)))`.

## CI confirmed working (2026-10-25)

Run `40133147306`: written 284 edges to `fetcher_source_deps.json`. Subdirs currently `['models/bert', 'models/clip']`.
Latest commit on fetcher-recording: `997736acda`.

## pytest_runtest_logfinish / _ci_current_test_id gap — not an issue (2026-10-25)

After `pytest_runtest_teardown` completes for test N, `_ci_current_test_id` remains set to
test N's nodeid until `pytest_runtest_setup` fires for test N+1. Concern: could a stale test_id
cause wrong attribution?

**Not an issue** for two reasons:
1. Nothing imports transformers modules in the inter-test gap — it's just pytest internal bookkeeping.
2. Even if something did: attribution is at **test file** level (first `::` segment of nodeid).
   A stale bert test_id would still correctly attribute to `tests/models/bert/test_modeling_bert.py`.
   Wrong only if per-method precision were needed, which it isn't.

The observed llava_next false positives come from `get_values(MODEL_FOR_PRETRAINING_MAPPING)`
called **during** an actual test method (phase=`"test"`, test_id correctly set) — unrelated to this gap.

## False positives: get_values(MODEL_FOR_PRETRAINING_MAPPING) (2026-10-25)

### Finding
`test_modeling_bert.py` deps include `models/llava_next/configuration_llava_next.py`,
`models/llava_next/modeling_llava_next.py`, `models/vit_mae/modeling_vit_mae.py`, etc.

Root cause: 13 model test files override `_prepare_for_class` with this check:
```python
if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
    inputs_dict["labels"] = ...
    inputs_dict["next_sentence_label"] = ...
```

`get_values()` calls `MODEL_FOR_PRETRAINING_MAPPING.values()` → `_LazyAutoMapping.values()`
which calls `_load_attr_from_module` for ALL 98 models in the mapping → imports ALL their modules.

The 13 affected test files: bert, albert, big_bird, chinese_clip, electra, ernie, fnet, funnel,
lxmert, megatron_bert, mobilebert, modernbert, roc_bert.

### Key distinction from base class
`test_modeling_common.py` base `_prepare_for_class` correctly uses string names:
```python
if model_class.__name__ in get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES):  # just strings, no imports
```
The 13 overrides mistakenly use the lazy mapping instead of the `_NAMES` variant.

### Two approaches to fix

**Option A — Fix the 13 test files (test-side fix)**
Change each of the 13 overrides from:
```python
if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
```
to:
```python
if model_class.__name__ in get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES):
```
And update the import (remove `MODEL_FOR_PRETRAINING_MAPPING`, add `MODEL_FOR_PRETRAINING_MAPPING_NAMES`).
This eliminates the 98-model eager load and removes false positive deps from all 13 model tests.
TODO: implement this fix across the 13 files listed above.

**Option B — Discard imports via call stack filtering (recorder-side fix)**
In `_record_import_chain` / `_tracking_import`, inspect the call stack at import time.
If any frame is `_load_attr_from_module` (in `auto_factory.py`), the import is a registry
side-effect — discard it instead of recording.

```python
import inspect
for frame_info in inspect.stack(context=0):
    if frame_info.function == "_load_attr_from_module":
        return  # registry side-effect, skip
```

Advantage: fixes ALL such patterns across ALL test files without touching test code.
Disadvantage: `inspect.stack()` is expensive — fires on every transformers import.
Mitigation: only inspect when we're in execution phase (test_id is set), not collection.

### Empirical results (2026-10-25) — Option B implemented and verified

Implementation in `_record_import_chain` using `sys._getframe()` (cheap — no source line reads):
```python
frame = sys._getframe(2)  # skip _record_import_chain + _tracking_import
while frame is not None:
    if frame.f_code.co_name == "_load_attr_from_module":
        return
    frame = frame.f_back
```

Results:
- **bert test**: 3 files remaining (was many with llava_next/vit_mae) — `conversion_mapping.py`,
  `modeling_layers.py`, `modeling_utils.py`. No llava_next, no vit_mae. ✓
- **clip processor test**: `image_processing_clip.py` + `tokenization_clip.py` still captured. ✓
  Total 24 files — real signal preserved.

Why `image_processing_clip.py` survives the filter: it's re-detected via our fromlist side-effect
tracking. After `_load_attr_from_module` loads it into sys.modules, a later `from transformers.models.clip import X`
call triggers our `_tracking_import` fromlist check which finds the submodule already in sys.modules
— at that point `_load_attr_from_module` is NOT in the call stack, so it gets recorded.

Conclusion: Option B alone is sufficient. Option A (fix 13 test files) is still a valid cleanup
but not required for recorder correctness. Option B is simpler, self-contained, and handles ALL
such patterns across all test files automatically.

### Performance overhead of the recorder (measured on clip modeling test, 2026-10-25)

| Configuration | Time | Overhead |
|---|---|---|
| No recorder | 55.9s | — |
| Recorder, no frame check | 64.7s | +8.8s |
| Recorder, frame check + test_id guard | 64.5s | +8.6s |
| Recorder, frame check, no guard | 66.5s | +10.6s |

Key findings:
- The `__import__` hook interception itself costs ~8.6-8.8s regardless of what runs inside it
- `sys._getframe()` frame walking is essentially free (adds ~0.2s vs no frame check)
- The test_id guard saves ~2s (skips frame walk during collection) — still worth keeping
- The 8.6s overhead is the unavoidable floor of having `builtins.__import__` hooked
- On CI (GPU tests, much longer runs), this overhead is proportionally negligible

### Why recorder overhead is unavoidable (~8.6s on clip modeling test)

Python's built-in `__import__` is a C function (`Python/ceval.c`) — essentially free (nanoseconds).
When we override `builtins.__import__` with our Python `_tracking_import`, ALL import statement
executions route through a Python-level function instead. Overhead = Python function call setup
× number of calls.

Measured on clip modeling test (308 passed, 461 skipped):
- Total `__import__` calls: **255,383** — same with or without recorder (Python calls this for
  every import statement execution, even cached modules already in sys.modules)
- Overhead: ~8.6s ≈ 255k × ~34µs per Python function call vs C built-in

**Key insight**: the 255k calls happen regardless. The cost is purely from routing them through
Python instead of C. Nothing to optimize inside the hook reduces this floor.

Deduplication (skipping already-seen src_file per test_id) would shrink the chain JSON
dramatically (clip modeling: 27,264 entries → 771 unique) but does NOT reduce timing —
the overhead hits before `_record_import_chain` is even called.

Chain bloat measured (clip modeling test):
- 27,264 total chain entries, 771 unique modules
- `models/auto/__init__.py`: 6,823× repeated
- `modeling_utils.py`: 5,758× repeated
- Dedup is worth doing for chain JSON size, not for performance

### Deduplication + collection exclusion (implemented, commits ead11ec247)

Unified dedup logic in `_record_import_chain` using a single `dedup_key`:
- Execution phase: `dedup_key = test_file` (first `::` segment of test_id)
- Collection phase (opt-in): `dedup_key = "__collection__"` (special key)
- Default: collection entries skipped entirely (not attributed to any test)

Result for clip modeling test:
- Before: 27,264 chain entries
- After dedup (execution only): 3 entries (unique execution-phase imports)
- With `CI_RECORD_DEPS_INCLUDE_COLLECTION=1`: 774 entries (771 collection + 3 execution)

### Correctness verified (2026-10-25)

**Clip processor test** (`test_processing_clip.py`): 24 files, includes:
- `models/clip/image_processing_clip.py` ✓
- `models/clip/tokenization_clip.py` ✓
- `models/clip/configuration_clip.py` ✓

**Bert modeling test** (`test_modeling_bert.py`): 3 files only:
- `conversion_mapping.py`, `modeling_layers.py`, `modeling_utils.py`
- Zero llava_next / vit_mae / clip false positives ✓

### Fallback: more precise filter if current approach is too aggressive

Current filter discards ALL imports where `_load_attr_from_module` is anywhere in the call stack.
If this proves too broad (missing real deps that legitimately go through `_load_attr_from_module`),
narrow it to only discard when `_load_attr_from_module` is called FROM `values()`/`items()`/`keys()`:

```python
if frame.f_code.co_name == "_load_attr_from_module":
    caller = frame.f_back
    if caller and caller.f_code.co_name in ("values", "items", "keys"):
        return
```

This preserves `__getitem__`-based lookups (specific object retrieval) while still discarding
bulk iteration side-effects. One-line change to revert if needed.

### Q: Does discarding _load_attr_from_module lose AutoXXX.from_pretrained() deps?

No — because `AutoXXX.from_pretrained()` uses a DIFFERENT code path for loading the actual
component (e.g. `CLIPImageProcessor`). It reads the class name from config/preprocessor_config.json
then does `getattr(transformers, "CLIPImageProcessor")` → hits the lazy `transformers.__getattr__`
→ `_get_module` → `importlib.import_module(".image_processing_clip")` → our hook fires.
`_load_attr_from_module` is NOT in this call stack → not filtered → captured correctly.

`_load_attr_from_module` is only called by `_LazyAutoMapping.__getitem__/.values()/.items()/.keys()`.
The lazy top-level `transformers` module `__getattr__` uses a separate mechanism (`_get_module`).

Two code paths:
- `get_values(MODEL_FOR_PRETRAINING_MAPPING)` → `_LazyAutoMapping.values()` → `_load_attr_from_module` → **filtered** ✓
- `AutoProcessor.from_pretrained()` loading actual component → `getattr(transformers, class_name)` → lazy `__getattr__` → **captured** ✓

In short: the filter catches "bulk registry iteration" and NOT "actually using a specific class".
Semantically clean — exactly the boundary we want.

### Q: Why is processing_clip.py (CLIPProcessor) not in the clip processor test deps?

`test_processing_clip.py` line 24: `from transformers import CLIPProcessor` — explicit top-level
import. This fires during pytest COLLECTION phase (when the test file is imported), before any
test runs. No `test_id` is set at that point → not recorded in `deps_by_file.json`.

This is correct behavior. The fetcher catches it via static analysis (direct import statement
in the test file). The recorder only records EXECUTION-TIME imports (those with a test_id set).

### Fetcher + recorder split (fundamental design)

| What | Caught by |
|---|---|
| Top-level imports in test files (e.g. `from transformers import CLIPProcessor`) | Fetcher (static) |
| Execution-time dynamic imports (e.g. `image_processing_clip` via `from_pretrained`) | Recorder |
| `_load_attr_from_module` bulk registry iteration | Neither (filtered by recorder) |

The recorder is NOT meant to replace the fetcher — it supplements it by adding edges the
fetcher structurally cannot see (dynamic dispatch, runtime model/processor loading).
They are expected to work together.

---

## Why vit test has many deps — root cause analysis (2026-10-23)

### What triggers vit test (no-stub commit, 91ac42ede4)
Total: 884 src files trigger vit test via `create_module_to_test_map`:
- 553 `__init__.py` files (false positives — see below)
- 42 pipeline files (real, via test_pipeline_mixin import chain)
- 75 non-vit model files (partially real, partially false positive)
- 3 actual vit files (legitimate)
- 211 other src files (core infra: modeling_utils, quantizers, integrations, trainer, deepspeed)

Full breakdown saved to: `vit_test_triggers.json`

### Fetcher over-selection: root causes

**Cause 1 — `modeling_utils.py` → quantizers/integrations chain (real impact)**
`test_modeling_vit.py` → `test_modeling_common.py` → `modeling_utils.py` → `quantizers/auto.py` → all quantizers.
Every quantizer change triggers ALL model tests. Legitimate architecturally (model loading needs
quantizers), but means any integration/quantizer fix runs full model CI.
Example chain found: `quantizer_mxfp4.py` → `integrations/mxfp4.py` → `models/gpt_oss/modeling_gpt_oss.py`
(local import inside function body — fetcher's static regex catches it regardless of indentation).

**Cause 2 — pipeline test file imports (real but less impactful)**
`test_modeling_vit.py` → `test_pipeline_mixin.py` → `test_pipelines_any_to_any.py` →
`test_pipelines_common.py` → `modeling_bert.py` (direct import at top level).
Pipeline test files import specific model classes (BERT, DistilBERT etc.) → those models
trigger vit test via reverse map.

**Cause 3 — `__init__.py` expansion loop bug (false positives, low real impact)**
Every model `__init__.py` uses `_LazyModule` which imports `import_utils.py`.
`import_utils.py` is imported by everything → `reverse_map[import_utils.py]` = all tests.
`__init__.py` expansion loop: `reverse_map[albert/__init__.py]` = `[import_utils.py]` +
`reverse_map[import_utils.py]` = everything. So all 553 `__init__.py` files "trigger" vit.
Low real impact: `__init__.py` changes are rare in isolation (always accompany real model file
changes which correctly only select that model's tests).
NOT a bug in practice — an over-approximation that's tolerated.

### Recorder over-capture: root causes (different mechanism)

**Cause — model mapping materialization at collection time**
`test_modeling_vit.py` → `test_pipeline_mixin.py` (no-stub) imports 20+ pipeline test files.
Several pipeline test files call `.items()` on `_LazyAutoMapping` objects at CLASS BODY level:
- `test_pipelines_mask_generation.py`: `model_mapping = dict(list(MODEL_FOR_MASK_GENERATION_MAPPING.items()))`
- `test_pipelines_automatic_speech_recognition.py`: `MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items()`
- `test_pipelines_image_segmentation.py`: multiple mapping `.items()` calls
- `test_pipelines_text_classification/token_classification/zero_shot.py`: all call `.items()`

Class bodies execute at module import time (collection phase), so all referenced model classes
get loaded into `sys.modules` between baseline snapshot and test execution.
Verified empirically: importing just 3 of those pipeline test files loads 178 model modules
(beit, canary, data2vec, detr, hubert, maskformer, sam, wav2vec2, whisper, and many more).

### Key distinction: fetcher vs recorder
- **Fetcher**: polluted by static imports + transitive closure (quantizers via modeling_utils)
- **Recorder**: polluted by runtime model mapping materialization (`.items()` on LazyAutoMapping at class def time)
- Same trigger (`test_pipeline_mixin.py`), completely different mechanisms
- Stub fixes address recorder pollution (make pipeline test classes lightweight stubs)
- Stub fixes barely help fetcher (it sees static imports regardless)

---

## Recorder design decisions (2026-10-24)

### Baseline: pytest_collection_finish (not pytest_sessionstart)
Changed baseline hook from `pytest_sessionstart` → `pytest_collection_finish` (commit e5a32d7302).
Reasoning: collection-time imports (pipeline mixin → model mapping materialization) are noise.
Taking baseline AFTER collection subtracts all that noise. Only test-execution-time imports remain.
Result for vit test: 0 files (correct — vit has no dynamic deps, fetcher covers it statically).
Result for clip processor test: 6 files (clean, model-specific).

### 3 stub commits reverted (commit 60c000856a)
The [dummy] stub commits were measurement-only. Reverted on fetcher-recording branch.
They cannot go to production as they break actual pipeline test functionality.
With pytest_collection_finish baseline, stubs are no longer needed — collection noise is subtracted.

### Recorder output for test_processing_clip.py (6 files)
```
src/transformers/image_processing_backends.py
src/transformers/models/aimv2/__init__.py           ← false positive (__init__ loop bug)
src/transformers/models/clip/configuration_clip.py  ← real but fetcher misses it (loaded at from_pretrained time)
src/transformers/models/clip/image_processing_clip.py  ← real signal, fetcher already has it via heuristic
src/transformers/models/clip/tokenization_clip.py   ← real signal
src/transformers/tokenization_mistral_common.py     ← false positive (MODALITY_TO_BASE_CLASS_MAPPING tuple resolved unconditionally)
```

### False positives in recorder output
1. `aimv2/__init__.py` — auto mapping resolution side effect (any CLIPImageProcessor model lookup)
2. `tokenization_mistral_common.py` — `MODALITY_TO_BASE_CLASS_MAPPING["tokenizer"]` is a tuple
   `("PreTrainedTokenizerBase", "MistralCommonBackend")`, both resolved unconditionally in
   `check_argument_for_proper_class` → `get_possibly_dynamic_module` → `getattr(transformers_module, "MistralCommonBackend")`
   → triggers lazy module load. Happens for EVERY processor test. Theoretical dep only (syntax error in file
   would break test, but logic changes wouldn't). Fetcher correctly does NOT wire it to clip tests.

### Key insight: recorder adds value for truly dynamic deps
- vit test: 0 new edges (fetcher already covers everything) ✓
- clip processor test: `image_processing_clip.py` + `tokenization_clip.py` real signals
  (fetcher already has image_processing_clip via heuristic; tokenization_clip also via heuristic)
- Recorder most valuable for: backbone files, cross-model processor deps — not yet tested

---

## Import chain tracker — deep analysis tool (2026-10-24)

### Design
Added `deps_chains.json` alongside `deps.json`. Records full call chain for every
`transformers.*` import during the session. Implemented by overriding:
1. `builtins.__import__` — catches all `import X` and `from X import Y` statements
2. `importlib.import_module` — catches dynamic dispatch via `_get_module` (lazy loader)

Both (1) and (2) are needed. `image_processing_clip.py` is loaded via `importlib.import_module`
inside `import_utils._get_module` — it never appears as a `builtins.__import__` call.

### Why not sys.modules dict subclass
First attempt was `sys.modules.__class__ = _TrackedModulesDict`. Python rejects this:
"__class__ assignment only supported for mutable types or ModuleType subclasses".
Replacing `sys.modules = new_dict` breaks C extensions (zoneinfo KeyError: 'zoneinfo._tzpath')
because `_zoneinfo.so` uses a cached C-level pointer to the original dict.
`builtins.__import__` override is clean and has no such issues.

### Phase tracking
Each entry includes `phase` and `test_id` (when in test execution):
- `"startup"` — before collection (conftest loading)
- `"collection"` — during pytest collection (test file importing)
- `"post_collection"` — between collection and first test
- `"setup"` — during setUpClass/setUp (pytest_runtest_setup)
- `"test"` — during test method (pytest_runtest_call)
- `"teardown"` — during teardown (pytest_runtest_teardown)

### For production dep extraction
Filter chains to phase in ("setup", "test", "teardown") → `src_file` values.
For clip processor test, this gives 15 files including `image_processing_clip.py`,
`tokenization_clip.py`, `configuration_clip.py` — exactly the dynamic deps.

### Noise in setup/test phases
Even filtered to setup/test phases, some false positives remain:
- `aimv2/__init__.py` — auto mapping side effect during from_pretrained
- `tokenization_mistral_common.py` — MODALITY_TO_BASE_CLASS_MAPPING unconditional resolution
These appear in setup phase (from_pretrained path), not collection. Will be filtered by
frequency across model tests (appear in every processor test → not model-specific).

### Chain format
```json
{
  "module": "transformers.models.clip.image_processing_clip",
  "src_file": "src/transformers/models/clip/image_processing_clip.py",
  "phase": "setup",
  "test_id": "tests/models/clip/test_processing_clip.py::CLIPProcessorTest::test_apply_chat_template_assistant_mask",
  "chain": [
    {"file": "src/transformers/utils/import_utils.py", "line": 2841, "func": "_get_module"},
    {"file": "src/transformers/models/auto/image_processing_auto.py", "line": 675, "func": "from_pretrained"},
    {"file": "tests/test_processing_common.py", "line": 184, "func": "setUpClass"}
  ]
}
```

### Commit
`360e11a611` — "Add import chain tracker to CI dep recorder"

---

## Per-test-file collection tracking (pytest_collectstart)

### Problem
Collection-phase imports with `CI_RECORD_DEPS_INCLUDE_COLLECTION=1` previously went into
a global `__collection__` pool (not attributed to any specific test file). This made it
impossible to merge collection-time deps into the correct test file's entry in deps_by_file.json.

### Root cause
Collection happens before any test runs — there is no `_ci_current_test_id` set.
But Python does import the test file module during collection, and those imports can capture
real static deps (processors, configs) that aren't captured during test execution.

### Solution
Use `pytest_collectstart(collector)` hook in `conftest.py`:
- Fires just before pytest starts collecting from each collector object
- Filter to `isinstance(collector, pytest.Module)` → only actual test files
- `collector.nodeid` is already repo-relative (e.g. `tests/models/clip/test_processing_clip.py`)
- Set `_ci_currently_collecting_file` → dedup key becomes `(test_file, "collection")`
- Clear in `pytest_collection_finish` (via `set_ci_collecting_file(None)`)

In `record_ci_source_deps_by_file()`, both collection and execution sets merge into
the same test file key — so deps_by_file.json always has one entry per test file
regardless of which phase captured the import.

### Empirical results (clip processor test)
```
deps_by_file.json keys: ['tests/models/clip/test_processing_clip.py']
Phase breakdown in chains: {'collection': 42, 'setup': 21, 'test': 3}
Unique (file, src) pairs — collection: 42, execution: 24
Total in deps_by_file.json: 49 (union — some overlap between phases)
```

No orphan keys. All collection imports correctly attributed to the clip test file.

### Path consistency
- Execution dedup key: `_ci_current_test_id.split("::")[0]` → repo-relative test file path
- Collection dedup key: `collector.nodeid` → also repo-relative
- Both use same format → `merged.setdefault(test_file, set())` merges correctly

### Commit
`ac1d85cc72` — "feat(ci-recorder): track per-test-file collection-phase imports via pytest_collectstart"

---

## deps_by_file.json phase-split format

### New format (committed `b74a50ea41`, fix `097ef4b0f2`)
Each test file entry now has separate `"collection"` and `"execution"` keys:

```json
{
  "tests/models/clip/test_processing_clip.py": {
    "collection": ["src/transformers/activations.py", ...],
    "execution":  ["src/transformers/models/clip/image_processing_clip.py", ...]
  }
}
```

Both keys always present — empty list `[]` if a phase had no imports (consistent schema).

`collate_ci_source_deps.py` unions both phases via `for src_files in phases.values()`.

### "execution" bucket definition
Dedup key is `(test_file, "execution")` whenever `_ci_current_test_id is not None`.
This covers phase values: `"setup"`, `"test"`, `"teardown"` — all have a test_id set.

### setUpClass and test_id
`setUpClass` runs inside `pytest_runtest_setup` of the **first test item in the class**.
So it has a full test_id (e.g. `...::CLIPProcessorTest::test_apply_chat_template_*`)
not just a class-level id. Imports during `setUpClass` → execution bucket ✅.

### Empirical results (clip processor — full run)
- `collection: 42`, `execution: 24` (setup: 21 + test: 3)
- The 3 test-phase files: `models/aimv2/__init__.py`, `convert_slow_tokenizer.py`, `models/clip/tokenization_clip.py`

### Empirical results (bert — single test vs full file)
- Single test (`BertModelTest::test_model`): `collection: 770, execution: 0`
  → 0 because `test_model` triggers no new imports beyond collection
- Full file (144 passed): `collection: 770, execution: 3`
  → 3 execution files: `conversion_mapping.py`, `modeling_layers.py`, `modeling_utils.py`
  → likely imported during `setUpClass` calling `BertModel(config)`

**Lesson**: always run the full test file for accurate coverage, not a single test.
Single-test runs can show 0 execution entries even when other tests in the file
do trigger new imports.

### Output paths (local verification)
- `C:/tmp/ci_deps_new_format/deps_by_file.json` — clip, new format
- `C:/tmp/ci_deps_bert_new2/deps_by_file.json` — bert single test
- `C:/tmp/ci_deps_bert_full/deps_by_file.json` — bert full file

---

## False positives in collection phase (jamba, olmo3, qwen3_moe in bert)

### Problem
Files like `models/jamba/configuration_jamba.py`, `models/olmo3/modeling_olmo3.py`,
`models/qwen3_moe/modeling_qwen3_moe.py` appeared in bert's **collection** phase.
These are completely unrelated to bert.

### Root cause
The `_load_attr_from_module` filter was guarded by `if _ci_current_test_id is not None:`
— so it only ran during test execution. But the same false-positive pattern fires during
collection: `get_values(MODEL_FOR_PRETRAINING_MAPPING)` at class definition time (when
the test class body is executed during import) triggers `_load_attr_from_module` and
eagerly imports all 98 registered models.

### Fix (`7d20b3e211`)
Moved dedup key check BEFORE the frame walk (avoids wasted frame walks when nothing
would be recorded anyway), then apply the `_load_attr_from_module` filter unconditionally
in all phases (collection + execution).

### Results after fix
- bert collection: 770 → 286 files, zero false positives (jamba/olmo3/qwen3_moe gone)
- bert execution: 3 files, zero false positives
- clip: collection 42 (unchanged), execution 24, real deps preserved

### Output paths
- `C:/tmp/ci_deps_bert_fix/deps_by_file.json` — bert full file, post-fix
- `C:/tmp/ci_deps_clip_fix/deps_by_file.json` — clip, post-fix

---

## Recorder vs fetcher correctness comparison

### Method
Run `get_module_dependencies(test_file)` from `tests_fetcher.py` (static AST deps, transitive)
and compare with recorder union of collection + execution.

### Results

| Target | Fetcher | Recorder | Fetcher deps missed by recorder |
|--------|---------|----------|---------------------------------|
| bert   | 6 files | 286 files | 1: `src/transformers/__init__.py` |
| clip   | 3 files |  49 files | 0 ✅ |

Recorder covers everything the fetcher finds for clip.
Recorder captures far more (transitive runtime imports) than the fetcher in both cases.

### Why `__init__.py` is missed for bert

The tracker filter:
```python
if resolved_name.startswith("transformers."):  # dot required
```
`import transformers` resolves to `resolved_name = "transformers"` (no dot) → not recorded.
Clip avoids this because it uses `from transformers.xxx import yyy` → `resolved_name = "transformers.xxx"`.

`src/transformers/__init__.py` is always present (every test imports it), so it's not a
meaningful distinguishing dep. Not worth fixing unless completeness is required.

### Fetcher deps for bert (all 6)
```
src/transformers/__init__.py
src/transformers/models/auto/auto_factory.py
src/transformers/models/auto/modeling_auto.py
src/transformers/models/bert/configuration_bert.py
src/transformers/models/bert/modeling_bert.py
src/transformers/testing_utils.py
```

### Fetcher deps for clip processor (all 3)
```
src/transformers/models/clip/processing_clip.py
src/transformers/testing_utils.py
src/transformers/utils/import_utils.py
```

All 3 clip fetcher deps are present in recorder output ✅.
5 of 6 bert fetcher deps are present in recorder output (only `__init__.py` missing).

---

## Expanding recorder to track tests/ files and bare `transformers` package

### Commit `22c0bd4bc6`
Three changes to `_tracking_import` / `_tracking_import_module` / `_record_import_chain`:

1. **Module name filter** — changed from `startswith("transformers.")` to `split(".")[0] in ("transformers", "tests")`:
   - Catches bare `import transformers` (resolves to `"transformers"`, no dot)
   - Catches `tests.*` modules (test mixins like `test_modeling_common.py`)
   - Safe: `startswith("transformers")` without dot would also match `transformers_modules.*`
     (HuggingFace dynamic module loader) — split approach avoids this

2. **`_record_import_chain`**: accept files under `tests/` in addition to `src/transformers/`
   — tries `relative_to(src_root)` first, falls back to `relative_to(repo_root)` and
   checks `rel_str.startswith("tests/")`

3. **Fromlist filter**: same `split(".")[0]` logic, dropped redundant `startswith` check

### Results after expansion
| Target | Fetcher | Recorder | Missed |
|--------|---------|----------|--------|
| bert   | 10 files | 321 files | 0 ✅ |
| clip   |  3 files |  55 files | 0 ✅ |

Recorder now captures test mixin files per target:
- bert: `tests/test_modeling_common.py`, `tests/generation/test_utils.py`, `tests/test_pipeline_mixin.py`, etc.
- clip: `tests/test_processing_common.py`

---

## Fetcher scope: why only src/transformers/ and tests/

### `all_modules` seed set
```python
all_modules = list(PATH_TO_TRANSFORMERS.glob("**/*.py"))  # src/transformers/
all_modules += list(PATH_TO_TESTS.glob("**/*.py"))        # tests/
```
Only these two dirs are seeded — `utils/` is never included.

### Transitive closure stays bounded
The recursion guard at line 885:
```python
if d not in direct_deps:
    continue  # Windows ghost path
```
Since `direct_deps` only has keys from `all_modules`, any import pointing outside
`src/transformers/` or `tests/` (e.g. third-party libraries, `utils/`) hits this guard
and is silently dropped. No explicit end-filter needed — the boundary is enforced
by the seed set.

### Windows backslash issue in reverse map
`create_reverse_dependency_map()` on Windows produces ~300 keys with backslash paths
(e.g. `src\transformers\models\clip\image_processing_clip.py`) from the heuristic
processor dep injection (PR #49826). These are duplicate entries of their forward-slash
equivalents. Our recorder always uses `.as_posix()` so it's unaffected.

---

## 28 pipeline test files in bert deps — real or false positives?

### Finding
Recorder captures 34 `tests/` files for bert, including 28 `tests/pipelines/test_pipelines_*.py`.
These look alarming but are **real transitive imports**, not false positives.

### Import chain
```
BertModelTest
  → inherits PipelineTesterMixin
  → tests/test_pipeline_mixin.py (explicit top-level imports at lines 59–78+)
  → from .pipelines.test_pipelines_text_generation import TextGenerationPipelineTests
  → from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
  → ... (all 28 pipeline test files)
```
Also: `tests/test_modeling_common.py` imports `ExportTesterMixin` from `tests/exporters/test_export.py`,
and `tests/test_configuration_common.py` imports `config_common_kwargs` from `tests/utils/test_configuration_utils.py`.

### Fetcher confirms: get_module_dependencies() vs full transitive closure
`get_module_dependencies('tests/models/bert/test_modeling_bert.py')` called directly
returns only 10 files — it does single-file static AST analysis, not full transitive closure.

The full `create_reverse_dependency_map()` transitive closure (which recurses across all
modules) gives 30 `tests/` deps for bert — including all 28 pipeline files. This matches
what the recorder captures (34, slightly more due to `__init__.py` files and the test file itself).

**Conclusion**: recorder and fetcher are consistent. All 28 pipeline files are legitimate deps.

## Full model list CI run — deps_comparison.json analysis (2026-10-25)

CI run: 40209337242 (full model list, fetcher-recording branch)
Artifacts dir: `7b4b8de9-c1c7-4fcf-b096-96831e8dfcaa_ci_results_run_models_gpu (67).zip.caa/`
Files: `fetcher_source_deps.json`, `fetcher_reverse_map.json`, `deps_comparison.json`

### Stats
- 1029 test files in recorder (fetcher_source_deps.json)
- 3358 src keys in fetcher reverse_map
- 1140 test files total after merging both sides
- 0 perfect matches (expected — different scope by design)
- 1130 have fetcher_only entries
- 1027 have recorder_only entries

### fetcher_only (recorder MISSES these) — WHY?
These files appear in fetcher's reverse_map for a test, but recorder never captured them.
ALL 1130 files appear in every single test file's fetcher_only set — they are globally
"reachable" via static transitive analysis but NEVER actually imported at runtime.

Root cause: **conditional/optional imports not executed in CI**
Examples (all appear in 1130/1130 test files):
- `src/transformers/integrations/aqlm.py`, `awq.py`, `bitnet.py`, `compressed_tensors.py`,
  `flash_attention.py`, `eetq.py`, `fbgemm_fp8.py` — quantization integrations, guarded by
  `is_X_available()` at runtime but static AST sees the import statement
- `src/transformers/generation/continuous_batching/**` — optional dep, not installed in CI
- `src/transformers/_typing.py`, `distributed/mixin.py` — pulled in via transitive static
  chain but not actually executed at runtime

**This is the fundamental difference**: fetcher = conservative static analysis (includes
unreachable conditional paths); recorder = precise runtime capture (only what was actually
imported).

Implication for fetcher consumer: if we use recorder data as the reverse_map source,
we'd be LESS conservative — won't trigger tests for files that are behind `is_X_available()`
guards when those optional deps aren't installed. This is actually more accurate.

### recorder_only (fetcher MISSES these) — WHY?
Files the recorder captured at runtime but fetcher doesn't link to this test file.
Top patterns (each appearing in ~511-552 test files):

1. `tests/__init__.py`, `tests/models/__init__.py` — recorder captures these because pytest
   imports the test package during collection, which loads `tests/__init__.py`, and the
   recorder records it as a dep of the test being collected.
   Fetcher doesn't have this: no test file has a transitive direct import chain (not passing
   through any `__init__.py`) that leads to `tests/__init__.py` — test files never import
   from it, so it never appears in the reverse map.
   **Technically a real dep** (if `tests/__init__.py` changes, pytest re-imports it for every
   test), but **practically irrelevant** — `tests/__init__.py` is essentially an empty package
   marker and never changes in a way that breaks tests. Safe to ignore.

2. ~79 model config/tokenizer files (e.g. `roberta/tokenization_roberta.py`,
   `cohere_compass/configuration_cohere_compass.py`, `qwen2_vl/configuration_qwen2_vl.py`,
   etc.) each appearing in ~511 test files — at runtime, `import transformers` loads
   `src/transformers/__init__.py` which eagerly imports all model configs. Recorder
   captures this full chain. Fetcher stops at `__init__.py` boundaries when building the
   transitive dep closure (line: `if d.endswith("__init__.py"): continue`), so these model
   files are NOT added as deps of tests that only reach them through `__init__.py`.
   `reverse_map[tokenization_roberta.py]` only contains tests with a transitive direct
   import chain reaching it WITHOUT passing through any `__init__.py`.

   Fetcher's design intent: changing `roberta/tokenization_roberta.py` should only trigger
   tests that directly depend on it — not all 511 tests that happen to pull it in via
   `import transformers` → `__init__.py` → `tokenization_roberta.py`. This avoids massive
   over-triggering.

   Recorder's view: "it was imported at runtime, so it's a dep" — leads to many more files
   per test, which would cause over-triggering if used naively as a reverse map.

### Conclusion: two complementary views
- **Recorder is more precise** about what was actually used at runtime (no false positives
  from conditional imports), but more expansive across init boundaries.
- **Fetcher is more conservative** (includes conditional imports), but stops at inits
  to avoid triggering everything when `__init__.py`-reachable files change.
- For the fetcher consumer, the right approach is likely: use recorder data for the
  "execution" phase (precise runtime deps), and keep fetcher's static analysis for
  collection-phase and __init__.py boundary handling.

### Why recorder is safe from __init__.py false positive explosions

Transformers uses **lazy loading** (`_LazyModule`) in `__init__.py` — model files like
`modeling_bert.py` are only imported when actually accessed, not at `import transformers`
time. So the recorder only captures what's genuinely imported at runtime, not everything
listed in `__init__.py`.

### Why fetcher still needs the __init__.py stop

Fetcher does **static AST analysis** — it reads `__init__.py` source and sees all the
import statements or `_LazyModule` entries listing every model. It has no way to
distinguish lazy vs eager loading at parse time. Following through `__init__.py` statically
would add ALL model files as deps of every test — completely useless over-triggering.

The `__init__.py` stop is the fetcher's pragmatic workaround: "I can't reason about lazy
vs eager statically, so I stop at any `__init__.py` boundary entirely." The recorder
doesn't need this workaround — it observes what actually runs.

### Key insight: recorder is strictly better for CPU CI
In the main PR CPU CI (where fetcher is used), optional deps (aqlm, awq, bitnet,
continuous_batching, etc.) are also NOT installed — same environment as the recording CI.
Therefore:
- Changing those files literally cannot break any test in CPU CI
- Fetcher's static analysis would still trigger those tests (false positive), wasting resources
- Recorder correctly excludes them: "never imported at runtime → not a dep → don't trigger"

The recorder_only entries (tests/__init__.py, model configs through __init__) are the
fetcher's intentional design choice — not a recorder deficiency.

**Bottom line**: recorder data produces a more accurate, less wasteful reverse map for CPU CI.

### recorder_only — the opposite direction concern
recorder_only entries are more nuanced and NOT uniformly good:

- Recorder says ~511 tests depend on `roberta/tokenization_roberta.py`,
  `cohere_compass/configuration_cohere_compass.py`, etc.
- Root cause: at runtime `import transformers` → `src/transformers/__init__.py`
  → eagerly loads all model configs → recorder captures them all in collection phase
- If recorder data used naively as reverse map: any model config change would trigger
  511 tests → massive over-triggering, worse than current fetcher

This is exactly why fetcher has the "stop at init" design — deliberate choice to prevent
one model config change from forcing 511 unrelated tests to run.

### Implication: use execution phase only for fetcher consumer
Don't use raw flattened (collection + execution) recorder data for the reverse map.
Better split:
- **execution phase only** → imports triggered by actual test code running → precise,
  avoids init-transitive explosion from `import transformers` at collection time
- **skip collection phase** for reverse map → collection is where `import transformers`
  happens, pulling all model configs through __init__, causing the 511-test problem

Next step: re-run comparison using execution-only recorder data to verify this produces
cleaner results closer to fetcher's reverse map.

### recorder_only root cause investigation — case 1: roberta/tokenization_roberta.py in bert test

Initial hypothesis ("import transformers eagerly loads all model configs") was WRONG.

Actual chain:
```
test_modeling_bert.py
  → test_pipeline_mixin.py (imports PipelineTesterMixin)
    → tests/pipelines/test_pipelines_text2text_generation.py (or similar pipeline test)
      → from transformers import BartForConditionalGeneration / BartTokenizer
        → src/transformers/models/bart/tokenization_bart.py
          → from ..roberta.tokenization_roberta import RobertaTokenizer  ← explicit static import
            → src/transformers/models/roberta/tokenization_roberta.py  ← recorded here
```

Root cause: `BartTokenizer` inherits from `RobertaTokenizer`, so `tokenization_bart.py`
has a hardcoded direct import of `tokenization_roberta.py`. This is a REAL transitive dep.

Fetcher misses it because at some point in the chain an `__init__.py` is crossed, and
fetcher stops recursing there (by design). The recorder captures the full runtime chain.

This is a LEGITIMATE dep — if tokenization_roberta.py changes, it affects BartTokenizer
which is exercised in bert's pipeline tests. Recorder is correct here; fetcher has a miss.

### recorder_only root cause investigation — case 2: test file in its own recorder_only

1013 out of 1027 recorder_only cases are explained by: **the test file appears in its own
recorder_only set**.

Reason: the hook ordering causes this. `pytest_collectstart` fires first and sets
`_ci_currently_collecting_file = "tests/models/bert/test_modeling_bert.py"`. Then pytest
imports that file, which fires `builtins.__import__` → recorder sees `currently_collecting
= test_modeling_bert.py` and records `test_modeling_bert.py` as a dep of itself.

Fetcher never has this: a test file never imports itself (no object in test_X.py is
imported from test_X.py), so there is no static import path from test_X.py to test_X.py.

**No actual issue**: normal behavior, direct consequence of hook ordering. Not a real dep
miss in either direction.

### recorder_only root cause investigation — case 3: tokenization_mistral_common.py in 135 processor tests

`src/transformers/tokenization_mistral_common.py` in recorder_only for 135 tests.
Phase: **execution** (not collection).
Fetcher maps it to only 6 tests (voxtral/mistral-specific).

Root cause: purely runtime — NOT a static import. Traced via debug statement directly in
`tokenization_mistral_common.py`. The actual call chain for a surprise case
(`test_processing_align.py`, which has nothing to do with Mistral):

```
setUpClass (test_processing_common.py:208)
  → _setup_from_components → AlignProcessor.__init__
  → processing_utils.py:649, __init__: self.check_argument_for_proper_class(...)
  → processing_utils.py:1007, check_argument_for_proper_class:
      proper_class = tuple(self.get_possibly_dynamic_module(n) for n in class_name ...)
  → processing_utils.py:1925, get_possibly_dynamic_module:
      if hasattr(transformers_module, module_name):   ← iterates ALL transformers attrs
  → import_utils.py:2612, _LazyModule.__getattr__("MistralCommonBackend")
  → import_utils.py:2841, _get_module:
      importlib.import_module(".tokenization_mistral_common", "transformers")
```

`processing_utils.get_possibly_dynamic_module` calls `hasattr(transformers_module, name)`
for each expected class name. `hasattr` on the `_LazyModule` triggers `__getattr__` for
any name in `_class_to_module` — including `"MistralCommonBackend"` — which eagerly loads
`tokenization_mistral_common.py`. This happens in `setUpClass` for EVERY processor test,
not just Mistral ones, because `check_argument_for_proper_class` probes the full
`transformers` namespace.

This goes through `_LazyModule._get_module` → `importlib.import_module` directly (NOT
through `_load_attr_from_module`), so our recorder's frame-walk filter does NOT block it
— it IS recorded. But fetcher's static AST analysis cannot follow this runtime `hasattr`
probe → genuine fetcher miss.

Fetcher maps `tokenization_mistral_common.py` to only 6 tests (voxtral/mistral-specific
with direct static imports). Recorder correctly captures 135+ tests at runtime.

### Why this bypasses the recorder's _load_attr_from_module filter

The recorder's frame-walk filter discards imports triggered by `_load_attr_from_module`,
which is used by `_LazyAutoMapping` (the model registry) for bulk iteration side effects.
But `_LazyModule` (the `transformers` top-level namespace) uses a DIFFERENT code path:

```python
# _LazyModule.__getattr__:
module = self._get_module(self._class_to_module[name])   # ← _get_module, not _load_attr_from_module
# _get_module:
return importlib.import_module("." + module_name, self.__name__)  # ← direct call
```

Two separate lazy mechanisms:
- `_LazyAutoMapping` → `_load_attr_from_module` → **filtered** by recorder (bulk registry iteration)
- `_LazyModule` → `_get_module` → `importlib.import_module` → **NOT filtered** → recorded

Could we unify and filter `_get_module` too? No — `_get_module` is used for both "just
probing" (`hasattr` check) and genuine accesses (`from transformers import BertModel`).
Adding a filter would suppress real deps too. The `_load_attr_from_module` filter was safe
only because `_LazyAutoMapping` bulk iteration is exclusively a registry side effect.

### Is mistral_common a real dep for align?

Theoretically yes: `check_argument_for_proper_class` calls `get_possibly_dynamic_module`
which probes `hasattr(transformers, "MistralCommonBackend")` to validate processor
component types. If `tokenization_mistral_common.py` changes (e.g. `MistralCommonBackend`
class is renamed or removed), `check_argument_for_proper_class` could behave differently,
potentially affecting all 135 processor tests. In practice, such a change would also
directly break Mistral tests, so the align dep is mostly theoretical. But the recorder
is correct to capture it — it IS imported at runtime during setUpClass for every
processor test.

**TODO**: Investigate whether the recorder could detect `_LazyModule._get_module` probing
(e.g. via `hasattr`) vs genuine attribute access, and filter the probe-only cases to avoid
recording false deps like mistral_common for align. Non-trivial — both paths use the same
`importlib.import_module` call.

### recorder_only root cause investigation — case 4: Windows backslash artifact in local reverse_map

When `fetcher_reverse_map.json` was generated locally on Windows, 381 keys were stored
with backslash paths (`src\transformers\...`) instead of posix paths (`src/transformers/...`).

Root cause: `tests_fetcher.py` preprocessor_patterns wiring uses:
  `src_key = str(src_file.relative_to(PATH_TO_REPO))`  ← backslash on Windows
instead of:
  `src_key = src_file.relative_to(PATH_TO_REPO).as_posix()`  ← always forward slash

The rest of `create_reverse_dependency_map()` correctly uses `.as_posix()` for all_modules.
This inconsistency means preprocessor_patterns entries land under wrong keys on Windows.

Impact on comparison: any `recorder_only` entry that is actually wired via
preprocessor_patterns (tokenizers, image/video/audio processors) was incorrectly classified
as recorder_only in the local comparison. 381 such keys × ~1.5 tests each = ~549 false
recorder_only entries.

NOTE: This is a **local Windows artifact only** — in CI (Linux), `str(PosixPath(...))` uses
forward slashes, so the wiring is correct. The fetcher itself is not buggy in practice.

Action: **Fixed** — PR #49994 (merged) replaces `str()` with `.as_posix()` in all 4 affected
places in `tests_fetcher.py`. Local comparison now also correct (regenerated with posix fix).

## Fixed comparison results (deps_comparison_fixed.json, 2026-10-25)

After fixing `.as_posix()` bug in tests_fetcher.py (commit 890dd96447), re-generated
all intermediate dicts with posix-clean reverse_map. Files in same artifact dir:
- `fetcher_reverse_map_fixed.json` — raw reverse_map, 3094 src keys, 0 backslash keys
- `fetcher_by_test.json` — inverted: {test_file: [src_files]}, 1130 test files
- `recorder_by_test.json` — flattened collection+execution: {test_file: [src_files]}, 1029 test files
- `deps_comparison_fixed.json` — per-test fetcher_only / recorder_only, 1140 test files

Summary (unchanged totals, but chmv2 now correct):
- Perfect match: 0
- Has fetcher_only: 1130 (all 1130 from optional/conditional imports — global false positives)
- Has recorder_only: 1027
  - 1013: test file appears in its own recorder_only (trivial artifact)
  - 764: have at least one non-self recorder_only file
  - Top non-self entries: tests/__init__.py (552), tests/models/__init__.py (536),
    model configs/tokenizers (~511 each) — all due to fetcher's "stop at init" design

chmv2 fix confirmed: `image_processing_chmv2.py` no longer in recorder_only after fix.
Remaining recorder_only for chmv2 is only the test file itself (trivial).

## fetcher_only analysis — final verdict

All 1203 unique fetcher_only files fall into legitimate "false positive" categories:

| Category | Count | Why recorder is correct |
|---|---|---|
| integrations/ (aqlm, awq, bitnet...) | 43 | optional deps not installed in CI |
| quantizers/ | 30 | same |
| generation/continuous_batching/ | 20 | optional dep |
| loss/ (ctc, deformable_detr...) | 13 | specialized losses, not used in general tests |
| models/deprecated/ + unused models | 496 | never imported at runtime |
| _typing.py, distributed/mixin.py | ~5 | TYPE_CHECKING only, never runtime import |
| trainer.py, training_args.py, optimization.py | ~6 | training code; model tests don't import it |
| test files outside tests/models/ | 420 | not yet covered by recording run |

**Conclusion**: recorder results make MORE sense than fetcher for CPU CI.
- Fetcher over-triggers (optional deps, TYPE_CHECKING, training code that model tests don't use)
- Recorder is precise: only what actually ran at runtime → no wasted CI on unexercisable code paths
- The only gap: test files outside tests/models/ have no recorder coverage yet (generation,
  trainer, pipelines, etc.) — needs a broader recording run to complete coverage.

## Targeted fetcher_only investigation: wav2vec2 vs blip_text, distilbert vs data2vec_text

These are cases where wav2vec2/distilbert files appear in fetcher_only for specific other
model tests — meaning fetcher thinks those models are deps, but recorder never captured them.

### wav2vec2 in fetcher_only for tests/models/blip/test_modeling_blip_text.py
- Fetcher maps 813 src files to blip_text (via static transitive closure)
- Recorder only captured 11 files for blip_text — all collection-phase, no
  modeling_blip_text.py, no execution-phase imports at all
- Diagnosis: blip_text tests were collected but almost entirely SKIPPED at runtime
  (no execution-phase imports → all tests skipped by decorators e.g. @require_torch)
- The wav2vec2 link in fetcher's static analysis comes via a multi-hop chain through
  common test infrastructure: blip_text → test_modeling_common.py → (export/pipeline
  test chain) → ASR pipeline test → Wav2Vec2ForCTC import → wav2vec2 files
- Recorder is CORRECT: blip_text never imports wav2vec2 at runtime
- Conclusion: fetcher false positive via multi-hop static transitive chain

### distilbert in fetcher_only for tests/models/data2vec/test_modeling_data2vec_text.py

**Question**: data2vec_text DOES import PipelineTesterMixin (line 25):
```python
from ...test_pipeline_mixin import PipelineTesterMixin
```
And `test_pipeline_mixin.py` has top-level imports of all pipeline test files including
`from .pipelines.test_pipelines_any_to_any import AnyToAnyPipelineTests`, which imports
`from .test_pipelines_common import ANY`, and `test_pipelines_common.py` has at line 33:
```python
from transformers import DistilBertForSequenceClassification  # top-level!
```
So statically the chain exists. Why didn't the recorder capture distilbert for data2vec_text's
collection phase?

**Root cause — `sys.modules` caching (empirically verified)**:

Verified by adding debug statements to testing_utils.py and running pytest with `-s`:
- The recorder hook (`builtins.__import__` override) **does** fire and **does record**
  distilbert when `test_pipelines_common.py` is imported fresh (verified: local
  `--collect-only` on data2vec_text alone → `configuration_distilbert.py` IS recorded)
- The `_load_attr_from_module` filter does NOT block it — distilbert gets through fine

**The real mechanism**: The CI data2vec job collects ALL data2vec test files together.
Alphabetically, `test_modeling_data2vec_audio.py` comes before `test_modeling_data2vec_text.py`.
When `data2vec_audio` is collected, it imports `PipelineTesterMixin` → `test_pipeline_mixin.py`
body runs for the first time → `test_pipelines_common.py` body runs → `from transformers import
DistilBertForSequenceClassification` executes → distilbert IS recorded for `data2vec_audio`
(317 files total).

When `data2vec_text` is collected next, `from ...test_pipeline_mixin import PipelineTesterMixin`
fires the hook (so `test_pipeline_mixin.py` is recorded for data2vec_text). But Python finds
`test_pipeline_mixin` in `sys.modules` (cached) → does NOT re-execute its body → the imports
inside it (`test_pipelines_common`, etc.) never fire again → distilbert is NOT imported again →
NOT recorded for data2vec_text.

Confirmed in recorder data:
- `data2vec_audio`: distilbert=True, total=317 files
- `data2vec_text`: distilbert=False, total=15 files
- `data2vec_vision`: distilbert=False, total=26 files

**Verdict**: `sys.modules` caching is the definitive reason. Distilbert IS correctly recorded
— just attributed to the first test in the CI job to import `test_pipeline_mixin` fresh.
For data2vec_text specifically, that first test was data2vec_audio.

**CORRECTION**: This is a **recorder limitation**, not a fetcher false positive. Fetcher
correctly identifies distilbert as a dep of data2vec_text (it IS imported during the CI job,
just attributed to data2vec_audio). The recorder misses it for data2vec_text due to
`sys.modules` caching. So distilbert appearing in `fetcher_only` for data2vec_text reflects
a gap in the recorder, not an error in the fetcher.

### Summary
The two `fetcher_only` cases have different root causes:
- **wav2vec2 / blip_text**: genuine fetcher false positive — backbone map logic flaw
- **distilbert / data2vec_text**: recorder limitation — sys.modules caching means distilbert
  is attributed to data2vec_audio (first in job) instead of data2vec_text

For the global fetcher_only analysis, fetcher's
actually traversed at runtime in the specific test's context. Recorder is correct in both.

Key insight: recorder only captures **direct full-path submodule imports** like
`from transformers.models.X.modeling_X import Y`. It filters out any import that goes
through `_load_attr_from_module` — which covers all lazy top-level `from transformers
import X` accesses. So the recorder is inherently narrower than full static analysis,
but more precise for actual runtime deps.

### Correction: wav2vec2/blip_text chain not fully verified

The specific chain explanation ("test_modeling_common → pipeline → ASR → wav2vec2") was
speculative and NOT verified. What IS confirmed:
- wav2vec2/config IS in fetcher's reverse_map for blip_text (fact)
- Single-hop get_module_dependencies from blip_text does NOT include wav2vec2 (fact)
- Full BFS via get_module_dependencies also did NOT find wav2vec2 (fact)
- Therefore it must come from the TRANSITIVE CLOSURE in create_reverse_dependency_map,
  where a multi-hop chain (e.g. blip_text → test_modeling_common → export tests → ...)
  eventually reaches wav2vec2 after many expansions
- Exact chain not traced (would require rerunning the full closure with path tracking)

Core conclusion stands: fetcher false positive via transitive closure; recorder correct.

### Verified: exact chain for wav2vec2 in fetcher_only for blip_text

Confirmed via `create_reverse_dependency_tree()` BFS + backbone map:

```
src/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py
  → modeling_speech_encoder_decoder.py       (imports config)
    → tests/generation/test_utils.py          (imports SpeechEncoderDecoder)
      → tests/test_modeling_common.py         (imports generation test_utils)
        → tests/models/blip/test_modeling_blip_text.py  (imports test_modeling_common)
```

Then backbone map adds:
  `reverse_map[wav2vec2/config] += reverse_map[speech_encoder_decoder/config]`

because wav2vec2 is a child component of speech_encoder_decoder.

Result: blip_text lands in reverse_map[wav2vec2/config] even though blip_text has zero
connection to wav2vec2 at runtime.

This is a clear fetcher false positive from the backbone map propagation:
wav2vec2 is a child of speech_encoder_decoder → speech_encoder_decoder's reverse_map
(which includes blip_text via test_modeling_common → generation test_utils) gets copied
into wav2vec2's reverse_map. Recorder correctly never captured wav2vec2 for blip_text.

## Logic flaw in backbone map (added by Raushan)

The backbone map section in `create_reverse_dependency_map()` does:
```python
reverse_map[child/config] += reverse_map[parent/config]
```
e.g. `reverse_map[wav2vec2/config] += reverse_map[speech_encoder_decoder/config]`

`reverse_map[speech_encoder_decoder/config]` = 518 test files + 2 src files.
These 518 tests depend on speech_encoder_decoder via ANY import path — including
test infrastructure (test_modeling_common → generation/test_utils → SpeechEncoderDecoder).

Result: ALL 518 tests (including blip_text, which has zero connection to wav2vec2)
get added to `reverse_map[wav2vec2/config]` and `reverse_map[bert/config]`.

**The logic flaw**: the intended meaning was:
  "if bert/wav2vec2 (child) changes → speech_encoder_decoder (parent) might break → run its tests"

The correct propagation should only add speech_encoder_decoder's OWN test files
(e.g. `tests/models/speech_encoder_decoder/test_modeling_speech_encoder_decoder.py`),
NOT all 518 transitive dependents of speech_encoder_decoder.

The recorder correctly avoids this: it only records what was actually imported at runtime,
so wav2vec2 never appears in blip_text's deps.

## Fundamental recorder limitation: transitive imports of cached modules

### The precise rule (empirically verified with debug statements + `-s` pytest)

`builtins.__import__` fires for every `import` statement **in code that actually executes**.
Two cases:

1. **Direct import in a test file** (e.g. `from ...test_pipeline_mixin import PipelineTesterMixin`
   in data2vec_text.py): hook fires even if test_pipeline_mixin is already cached →
   `test_pipeline_mixin.py` IS recorded for data2vec_text.

2. **Imports inside an already-cached module's body** (e.g. `from .pipelines.test_pipelines_any_to_any
   import ...` inside test_pipeline_mixin.py): these are bytecode instructions in test_pipeline_mixin.py's
   body, which only ran the FIRST time it was imported. After that, Python returns the cached
   module without re-executing the body → those `import` statements never fire again →
   hook is NOT called → NOT recorded for subsequent tests.

### Consequence: collection-order dependency

Transitive deps of shared test infrastructure are only attributed to the **first test in the
CI job** whose collection causes that infrastructure to be imported fresh.

Verified in artifact data for the data2vec CI job:
- `test_modeling_data2vec_audio.py`: distilbert=True, total=317 files — collected first,
  test_pipeline_mixin imported fresh → test_pipelines_common ran → distilbert recorded
- `test_modeling_data2vec_text.py`: distilbert=False, total=15 files — collected second,
  test_pipeline_mixin already cached → transitive imports don't re-fire
- `test_modeling_data2vec_vision.py`: distilbert=False, total=26 files — same

The same pattern holds across all 434 test files that have distilbert in the recorder: it
always appears for the alphabetically first test in each CI job that uses PipelineTesterMixin.

### Why this is a problem

The recorder is supposed to tell us "which src files does THIS test depend on?". But for
shared test infrastructure (test_pipeline_mixin, test_pipelines_common, etc.), the transitive
deps are attributed only to the first test in the job — NOT to all tests that actually use
that infrastructure. So for data2vec_text, the recorder says 15 files, but the true runtime
dep set (if it were collected alone) would include distilbert and hundreds more.

This means: **recorder_only** entries (files recorder has but fetcher doesn't) are inflated
for the first test in each job, and **fetcher_only** entries (files fetcher has but recorder
doesn't) are inflated for all subsequent tests in each job — even for legitimate deps.

The recorder is still useful for its primary goal (capturing processor/feature-extractor deps
that fetcher misses entirely), but it cannot be treated as a complete ground-truth dep set
for any individual test file in a multi-file CI job.

### Empirical verification (three independent checks)

**Check 1**: `builtins.__import__` hook in `_tracking_import` — print when `pipelines_any_to_any`
appears in name:
```
audio+text session: fires ONCE, currently_collecting='tests/models/data2vec/test_modeling_data2vec_audio.py'
                    never fires for data2vec_text
```

**Check 2**: distilbert recorded/not-recorded via `_record_import_chain` debug print:
```
audio+text session: RECORDED configuration_distilbert.py for data2vec_audio  (3 distilbert files)
                    zero distilbert entries for data2vec_text
text alone session: RECORDED configuration_distilbert.py for data2vec_text   (3 distilbert files)
```

**Check 3**: print statement inserted directly in `test_pipeline_mixin.py` at line 59
(the body, before `from .pipelines.test_pipelines_any_to_any import AnyToAnyPipelineTests`):
```
audio+text session: [DBG_MIXIN] fires ONCE — stack shows _find_and_load_unlocked → exec_module
                    (Python actually loading the module body, not a cache hit)
                    never fires again for data2vec_text
```

All three checks confirm the same mechanism: `test_pipeline_mixin.py`'s body executes
exactly once per pytest session (for the first test that needs it), never re-runs for
subsequent tests even though they also `import` it.

---

## recorder_only root cause investigation — cases 5–8

Analyzed from `deps_comparison_fixed.json` (CI run 40209337242).
Tool: count each src file in recorder_only across all test files, excluding known cases
(self-reference, tests/__init__.py, tokenization_mistral_common.py).

### Case 5: Collection-time @register_patch noise (~511–514 count)

**Which files**: `cohere_compass`, `glm4v`, `glm4v_moe`, `sam3`, `sam3_video`, `rwkv`,
`qwen2_vl`, `deepseek_v2`, `phi3`, `phimoe`, `minicpmv4_6`, `ernie4_5_vl_moe`, `paddleocr_vl`,
`qwen3_5_moe`, `video_llama_3`, `exaone4_5`, `kimi_k25`, `muse_spark`, `glm_ocr`,
`wavlm`, `data2vec_audio`, `blt`, `nllb_moe`, `mra`, `yoso`, etc.
(~37 models across all 4 exporter files; config + modeling + `__init__.py` for each → ~3 files per model)

**Exact import chain** (verified):
```
test_modeling_bert.py  line 26:
    from ...test_modeling_common import ModelTesterMixin
      → test_modeling_common.py  line 128:
          from .exporters.test_export import ExportTesterMixin
          (triggered because ModelTesterMixin inherits ExportTesterMixin at line 749)
            → test_export.py  lines 27–30:
                from transformers.exporters.exporter_dynamo import DynamoConfig
                from transformers.exporters.exporter_executorch import ExecutorchConfig
                from transformers.exporters.exporter_onnx import OnnxConfig
                from transformers.exporters.exporter_openvino import OpenVINOConfig
                  → each exporter module body executes → @register_patch decorators fire
                    → _resolve_dotted_path() → importlib.import_module("...cohere_compass...")
                    → importlib.import_module("...phi3...")  etc. (~37 models)
```
`test_modeling_bert.py` never mentions `ExportTesterMixin` directly — it imports `ModelTesterMixin`,
which inherits from `ExportTesterMixin`, causing `test_export.py` (and all 4 exporters) to load.

**Why ~511 and not all ~1029**: Exactly 505 are `test_modeling_*.py` files (all import
`ModelTesterMixin`). The remaining ~6 are non-modeling tests that also reach `test_export.py`
(e.g. `tests/exporters/test_export.py` itself, `tests/generation/test_logits_process.py`).
Processor/tokenization/feature-extraction tests never import `ModelTesterMixin` → untouched.

**NOT filtered by `_load_attr_from_module` guard**: `@register_patch` calls `importlib.import_module`
directly — `_load_attr_from_module` is never in the call stack.

**Real dep?** No. Pure collection-time noise from export infrastructure.
Potential fix: make exporters load models lazily (inside functions) instead of at decoration time.

---

### Case 6: Collection-time pipeline source file imports via PipelineTesterMixin (~510–511 count)

**CORRECTED** — original hypothesis (pipeline mapping `.items()` materialization) was wrong.
The `_load_attr_from_module` filter in `7d20b3e211` IS active in this CI run. Verified: models
that appear via bulk `.items()` iteration (albert, lxmert, fnet, funnel, megatron_bert,
roc_bert...) have counts of 1–3, NOT ~511.

**What actually causes ~510 count**:

**Exact import chain** (verified):
```
test_modeling_bert.py  line 27:
    from ...test_pipeline_mixin import PipelineTesterMixin
      → test_pipeline_mixin.py  lines 59–78+:
          from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
          from .pipelines.test_pipelines_text_to_audio import TextToAudioPipelineTests
          ... (all pipeline test classes imported at top level)
            → test_pipelines_fill_mask.py  line 18:
                from transformers import FillMaskPipeline
                  → _LazyModule._get_module("fill_mask") → pipelines/fill_mask.py  ← recorded
            → test_pipelines_text_to_audio.py  line 23:
                from transformers import TextToAudioPipeline
                  → _LazyModule._get_module("text_to_audio") → pipelines/text_to_audio.py
                      line 30: from ..models.speecht5.modeling_speecht5 import SpeechT5HifiGan
                        → speecht5/configuration_speecht5.py + modeling_speecht5.py  ← recorded
```
`PipelineTesterMixin` is imported directly by `test_modeling_bert.py` (line 27), NOT via
`test_modeling_common.py` — it is a separate sibling import in the same test file.

**speecht5 specifically**: `text_to_audio.py` has a direct top-level import inside
`if is_torch_available():` — no `_load_attr_from_module` in the call stack → not filtered.

**Real dep?** No. speecht5/fill_mask in bert's collection-phase deps is pipeline infrastructure
noise — not a real dep for bert tests.

**Two concrete import chains (2026-10-26)**:

Example 1 — pipeline file only, no extra models:
```
test_modeling_bert.py line 27:
  from ...test_pipeline_mixin import PipelineTesterMixin
    → Python executes all of test_pipeline_mixin.py top-level, including:
        from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
          → Python executes all of test_pipelines_fill_mask.py top-level:
              from transformers import FillMaskPipeline
                → _LazyModule.__getattr__ → _get_module
                  → pipelines/fill_mask.py  ← recorded (no model imports → stops here)
```

Example 2 — pipeline file + model dragged in:
```
test_modeling_bert.py line 27:
  from ...test_pipeline_mixin import PipelineTesterMixin
    → Python executes all of test_pipeline_mixin.py top-level, including:
        from .pipelines.test_pipelines_text_to_audio import TextToAudioPipelineTests
          → Python executes all of test_pipelines_text_to_audio.py top-level:
              from transformers import TextToAudioPipeline
                → _LazyModule.__getattr__ → _get_module
                  → pipelines/text_to_audio.py  ← recorded
                    → text_to_audio.py line 30 (top-level, inside if is_torch_available()):
                        from ..models.speecht5.modeling_speecht5 import SpeechT5HifiGan
                          → speecht5/modeling_speecht5.py  ← recorded (one extra level)
```

Example 1 is "pipeline infrastructure noise". Example 2 is worse — `text_to_audio.py` eagerly
imports `SpeechT5HifiGan` at module load time, which it only needs when the pipeline actually
runs. Moving that import inside the method that uses it would eliminate the extra level.

**Root cause note — Python "execute on import" design**: `from module import X` executes the
entire module file, not just X. There is no way to selectively import one name without running
everything else. Languages like Java/C# lazy-load classes; Python does not. `_LazyModule` in
transformers is itself a workaround for this fundamental limitation. The fix in `text_to_audio.py`
would be to move the `SpeechT5HifiGan` import inside `__init__` where it is actually used.

**Two layers of spurious deps (verified on test_modeling_vit.py after case 5 fix, 2026-10-26)**:

- **Layer 1 — 24 test pipeline files + ~30 model files they drag in**: `test_pipeline_mixin.py`
  imported all 24 pipeline test classes at top level → Python executed each test file body at
  collection time → those test files imported specific model classes (distilbert, donut, maskformer,
  wav2vec2, whisper, ...) at their top level → those model source files loaded during collection.
  Fix: replace top-level imports with lazy string-based mapping; resolve via `importlib` only at
  test execution time (inside `_run_pipeline_test`). Result: 24 test files + ~30 model files no
  longer appear in collection phase; they only appear in execution phase of model tests that
  actually run pipeline tests (correct attribution, not noise).
  **CORRECTION (2026-10-29, empirically verified)**: The ~28 `src/transformers/pipelines/` source
  files were originally attributed to this layer, but recorder comparison shows they are present
  BOTH before and after the fix (collection-phase only) — they are NOT caused by the 24 test class
  imports. They come from TWO separate triggers (both independently load all 28 via eager `pipelines/__init__.py`):
  (a) `test_pipeline_mixin.py` line 40: `from transformers.pipelines import (8 pipeline classes)`;
  (b) `tests/generation/test_utils.py` line 40: `from transformers import pipeline`.
  See "Remaining collection-phase pipeline noise" section below for full root cause and Fix A.
  The actual layer 1 payload is: 24 test files + ~30 model files = ~54 files moved out of collection.
  **Done — 2 commits on fetcher-recording branch (2026-10-29)**.

- **Layer 2 — 4 model files**: 2 pipeline source files had direct non-lazy model imports at
  module load time (`text_to_audio.py` → `SpeechT5HifiGan` → speecht5 ×3 files;
  `token_classification.py` → `BasicTokenizer` → bert_legacy). The "59 files" in early
  estimates was inflated — those were mostly the layer 1 test pipeline files, not extra
  model files from layer 2.
  Fix: move those imports inside the methods that actually use them. **Done — PR #50001 (merged)**.

**Layer 2 DONE (2026-10-26)** — only 2 pipeline source files had direct model imports:
- `pipelines/text_to_audio.py`: moved `SpeechT5HifiGan` import inside `__init__` (was at top-level under `if is_torch_available()`)
- `pipelines/token_classification.py`: moved `BasicTokenizer` import inside `__init__` (was top-level)

Effect: removes 4 files from collection-phase deps of every model test (speecht5 x3 + bert/tokenization_bert_legacy).
Verified on 4 test files: vit, bert, gpt2, distilbert — all 0 speecht5/bert_legacy deps after fix.
**PR #50001 (merged)** (`fix-pipeline-eager-model-imports` branch, 1 commit, clean from remote main).

**Layer 1 DONE (2026-10-29)** — see "Layer 1 fix — DONE" section below.

### Layer 1 fix — DONE (2026-10-29, 2 commits on fetcher-recording branch)

**Step 1 — delete dead code** (`eb59341457`)
Lines 123–127: module-level loop built `task_info["mapping"] = {"pt": test.model_mapping}`.
The `"mapping"` key was NEVER READ anywhere (confirmed: grep found no use sites outside the loop itself,
and `pipeline_test_mapping` is never imported outside the file). Deleted the loop entirely.

**Step 2 — lazy string-based mapping** (`ac5c36ed20`)
Replaced the 24 top-level class imports with dotted-path strings in `pipeline_test_mapping`:
```python
# Before (24 top-level imports + class references):
from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
pipeline_test_mapping = {"fill-mask": {"test": FillMaskPipelineTests}, ...}

# After (no top-level imports, string paths):
pipeline_test_mapping = {"fill-mask": {"test": "tests.pipelines.test_pipelines_fill_mask.FillMaskPipelineTests"}, ...}
```
- Lines 271, 420 (class name only needed): `test.__name__` → `test.rsplit(".", 1)[-1]`
- Line 440 (instantiation, only at test execution time): resolve via `importlib.import_module` + `getattr`

**Measured effect (recorder, vit test file)**:
- Before: 321 collection-phase files
- After:  267 collection-phase files  (−54)
  - 24 `tests/pipelines/test_pipelines_*.py` test files: removed ✓
  - 30 model source files pulled in transitively by those test files: also removed ✓
    (detr, distilbert, donut, maskformer, parakeet, speech_to_text, videomae, wav2vec2, whisper, ...)
  - `src/transformers/pipelines/` source files: **unchanged** — same 28 files present both before
    and after. Empirically confirmed: these are NOT caused by the test class imports. They have
    two independent triggers (see "Remaining collection-phase pipeline noise" below): `test_pipeline_mixin.py`
    line 40 `from transformers.pipelines import (8 classes)` AND `tests/generation/test_utils.py`
    line 40 `from transformers import pipeline`. Fixed separately by Fix A (commits `e8ea677c8c` + `1840b9096e`).

**Verification**: confirmed that `tests/pipelines/test_pipelines_*.py` (all 24) were present before
and are absent after. The 30 model-source removals are a side-effect of eliminating those test files.

**Remaining collection-phase pipeline noise — root cause analysis (2026-10-30)**:

28 `src/transformers/pipelines/` source files still appear in collection after layer 1 fix.

**Two triggers, both verified empirically**:

Trigger 1 — `tests/generation/test_utils.py` line 40:
```
test_modeling_vit.py → test_modeling_common.py (line 129: GenerationTesterMixin)
  → tests/generation/test_utils.py
    → line 40: from transformers import pipeline
      → transformers.__init__ lazy → _get_module("pipeline")
        → pipelines/__init__.py  ← eager → all 28 files
```

Trigger 2 — `tests/test_pipeline_mixin.py` line 40:
```
test_modeling_vit.py → test_pipeline_mixin.py (PipelineTesterMixin)
  → line 40: from transformers.pipelines import (AudioClassificationPipeline, ...)
    → pipelines/__init__.py  ← eager → all 28 files
```

**Import order masking**: for model tests (test_modeling_*.py), Trigger 1 fires first
(`test_modeling_common.py` imports `generation/test_utils.py` at line 129), so all 28 files
are already in `sys.modules` by the time Trigger 2 runs — making Trigger 2 a no-op for vit.
This caused us to initially miss Trigger 2 by only recording the vit test, and initially
over-attribute the cause to Trigger 2 alone when not considering Trigger 1.

**Verified independently**:
- `importlib.import_module("tests.test_pipeline_mixin")` alone → 27 pipeline modules loaded ✓
- `importlib.import_module("tests.generation.test_utils")` alone → 27 pipeline modules loaded ✓
- Both are real triggers; which fires "first" depends on import order in a given test session.

**Key finding**: `from transformers.pipelines import X` and
`importlib.import_module("transformers.pipelines.audio_classification")` both load all 28 files.
Python always executes parent `__init__.py` when importing any submodule. Empirically verified
(both forms: 28 new pipeline modules loaded in fresh process). No import form can bypass this
as long as `pipelines/__init__.py` is eager.

**The 8 class imports in `test_pipeline_mixin.py` line 40 are only used in
`task_to_pipeline_and_spec_mapping`** (lines 92–99), passed to `compare_pipeline_args_to_hub_spec`
inside `run_task_tests`. No `isinstance` checks — the class is needed for
`inspect.getdoc(pipeline_class.__call__)`.

**Two independent future fixes**:

**Fix A — Move to execution time** (test-only changes, safe):
Two files to fix:
1. `tests/test_pipeline_mixin.py`: replace `from transformers.pipelines import (8 classes)` with
   dotted-path strings in `task_to_pipeline_and_spec_mapping`; resolve via `importlib` at the
   use site inside `run_task_tests`.
2. `tests/generation/test_utils.py`: move `from transformers import pipeline` (line 40) inside
   the methods that use it.

Both must be fixed to eliminate the 28 files from collection for all model tests. Fixing only
one leaves the other as a trigger.

Benefits by environment:
- **No pipeline tests run** (e.g. `--collect-only`, model tests without `RUN_PIPELINE_TESTS`):
  28 files never loaded at all → 0 pipeline source files in any phase ✓
- **Pipeline tests run**: 28 files in execution phase only, correctly attributed to the specific
  model test that ran pipeline tests — not noise across all 511 model tests ✓

Note: even in execution, all 28 files still load when first pipeline class is imported
(due to eager `__init__.py`) — but in the right phase with correct attribution.

**Fix A — DONE — PR #50158 merged 2026-10-30**:

Extracted to a clean branch (`fix-pipeline-collection-imports`, 6 commits) and merged as PR #50158:
1. `e6e92ee37c` — remove dead loop in `pipeline_test_mapping` (output was never read)
2. `aad55e9e43` — lazy-load 24 pipeline test classes (dotted-path strings + `importlib`)
3. `f684e32c70` — lazy-load 8 pipeline source classes in `task_to_pipeline_and_spec_mapping`
4. `1963149f52` — move `pipeline` import to call site in `test_stop_sequence_stopping_criteria`
5. `62b8c3171d` — ruff fix + format
6. `08fcb43a14` — TODO note in `utils/add_pipeline_model_mapping_to_test.py` (broken by string refactor, fix next time someone runs it)

Original prototype was 2 commits on `fetcher-recording` (`e8ea677c8c`, `1840b9096e`).
`fetcher-recording` still needs a rebase on main (conflict in `test_pipeline_mixin.py` — dummy stubs vs merged lazy strings).

**Measured effect (recorder, vit test, --collect-only)**:
- Before Fix A (after layer 1 step 2): 267 collection files, 28 pipeline source files
- After Fix A (both parts): **235 collection files, 0 pipeline source files**
  - 28 pipeline source files eliminated from collection ✓
  - 4 additional files also removed (collateral from generation/test_utils.py change) ✓
- Execution phase (single test run): 0 pipeline source files (correct — pipeline not exercised) ✓

**Fix B — Make `pipelines/__init__.py` lazy** (src change, larger):
Convert `pipelines/__init__.py` to use `_LazyModule` (same pattern as `transformers/__init__.py`).
Then importing any single pipeline class loads only that 1 file instead of all 28.
Combined with Fix A already done: collection = 0 pipeline files (already achieved),
execution = 1 file per task tested (instead of all 28).
Out of scope for current work — separate future PR.

---

### Case 7: Auto processor/tokenizer class files (~22–36 count, mixed phases)

**Which files**: `models/auto/tokenization_auto.py` (36), `models/auto/processing_auto.py` (33),
`models/auto/video_processing_auto.py` (31), `models/auto/feature_extraction_auto.py` (29),
`models/auto/image_processing_auto.py` (22)

**Root cause**: Tests that directly use `AutoTokenizer`, `AutoProcessor`, `AutoVideoProcessor`
etc. (e.g., `tests/exporters/test_export.py`, `tests/generation/test_candidate_generator.py`,
some specific model tests) trigger lazy loading of these auto files. The `transformers._LazyModule`
resolves `AutoTokenizer` → `_get_module` → `importlib.import_module(".auto.tokenization_auto")`
→ our hook records it. This goes through `_LazyModule._get_module` (not `_load_attr_from_module`),
so it is NOT filtered by the frame-walk guard.

**Why recorder_only**: Fetcher misses these because the chain goes through lazy attribute lookup
on the `transformers` namespace (`__init__.py` boundary). Fetcher stops at `__init__.py`
and never follows into `tokenization_auto.py`.

**Real dep?** Yes. If `tokenization_auto.py` changes, `test_export.py` (which calls
`AutoTokenizer.from_pretrained`) should be re-run. These are genuine missing edges in the
fetcher's reverse map for the 33–36 affected tests.

---

### Case 8: Execution-time cross-model component deps (5–15 count) — the intended signal

**Which files** (execution-only, recorder_only):
- `qwen2/tokenization_qwen2.py` — 15 tests
- `clip/image_processing_clip.py` — 6 tests
- `vit/configuration_vit.py` — 6 tests
- `models/auto/tokenization_auto.py` — 6 tests (execution phase subset)
- `wav2vec2/tokenization_wav2vec2.py` — 5 tests
- `vision_encoder_decoder/configuration_vision_encoder_decoder.py` — 7 tests
- `llava/configuration_llava.py`, `siglip/configuration_siglip.py`,
  `pixtral/configuration_pixtral.py`, `dac/configuration_dac.py`,
  `whisper/feature_extraction_whisper.py`, etc. — 3–4 tests each

**Concrete example — `qwen2/tokenization_qwen2.py` in 15 processor tests**:
```
audioflamingo3/test_processing_audioflamingo3.py  ← audioflamingo3 uses qwen2 tokenizer
colqwen2/test_processing_colqwen2.py
qwen2_audio/test_processing_qwen2_audio.py
qwen2_vl/test_processing_qwen2_vl.py
qwen2_5_vl/test_processing_qwen2_5_vl.py
llava_onevision/test_processing_llava_onevision.py
musicflamingo/test_processing_musicflamingo.py
vibevoice_asr/test_processing_vibevoice_asr.py
video_llama_3/test_processing_video_llama_3.py
... (15 total)
```

**Root cause**: Processor tests call `AutoProcessor.from_pretrained()` at test execution time,
which reads the model's config and dynamically loads the component files (e.g., audioflamingo3's
processor has a qwen2 tokenizer as a component → `qwen2/tokenization_qwen2.py` is imported).
This goes through `_LazyModule._get_module` → `importlib.import_module` → our hook records it.
It's in the **execution phase** (test_id is set), not collection.

**Why recorder_only**: Fetcher completely misses these cross-model runtime deps. There is no
static import chain from `audioflamingo3/test_processing_audioflamingo3.py` to
`qwen2/tokenization_qwen2.py` — the link only exists at runtime via `AutoProcessor`.

**Real dep?** Yes, absolutely. This is exactly the cross-model dynamic dep the recording
system was designed to capture. If `tokenization_qwen2.py` changes, all 15 processor tests
that use qwen2 as a component tokenizer should be re-run.

### Summary table (all 8 cases)

| Case | Count range | Phase | Real dep? | Notes |
|------|------------|-------|-----------|-------|
| 1: tests/__init__.py | 536–552 | execution | Technically, not practically | Empty package marker, never meaningful changes |
| 2: self-reference | 1013/1027 tests | collection | No | Hook ordering artifact |
| 3: tokenization_mistral_common | ~135 | execution | Theoretically | `hasattr` probe via `_LazyModule._get_module` |
| 4: Windows backslash | — | — | No | Fixed via PR #49994 (merged) |
| 5: exporter_dynamo @register_patch | ~511–514 | collection | No | 37 models eagerly loaded at exporter import; **fix: PR #50000 (merged)** — lazy resolve in `apply_patches` |
| 6: pipeline eager model imports | ~510–511 | collection | No | 24 test pipeline files + ~30 model files they drag in (NOT the 28 pipeline src files — those are from Fix A below); **layer 2: PR #50001 (merged)** — 4 model files removed; **layer 1: 2 commits on fetcher-recording** — −54 collection files; **Fix A (2 more commits)**: also removes 28 pipeline src files → 267→235 collection files, 0 pipeline src in collection |
| 7: auto processor class files | ~22–36 | mixed | **Yes** | Real fetcher miss for tests using Auto classes |
| 8: cross-model component deps | ~5–15 | **execution** | **Yes** | Core value of recorder; processor/tokenizer components |

---

## Root cause of case 5 eager import — and a potential fix (2026-10-26)

### Why the class object is stored instead of the string path

`register_patch` resolves the dotted path to the actual class object at decoration time
and stores `(class_object, attribute_name, patch_factory)` in `_PATCHES[backend]`.

```python
obj = _resolve_dotted_path(obj_path)          # imports the model module
_PATCHES.setdefault(backend, []).append((obj, attribute, fn))
```

The class object reference is needed so that `apply_patches()` can do:
```python
setattr(obj, attribute, patch_factory(getattr(obj, attribute)))
```
i.e. it needs the actual class to call `setattr` on it.

### Why this causes eager imports

`_resolve_dotted_path` calls `importlib.import_module` segment by segment to walk
`"transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router"` — loading
`modeling_nllb_moe.py` as a side effect just to get the class object.
This fires at decoration time (module load), not at export time.

### The fix: store the string path, resolve lazily in apply_patches()

Instead of resolving at decoration time, store the raw string and resolve in `apply_patches()`:

```python
# In register_patch decorator — store string, not resolved object:
_PATCHES.setdefault(backend, []).append((obj_path, attribute, fn))

# In apply_patches() — resolve lazily, model already loaded by then:
for obj_path, attribute, fn in _PATCHES.get(backend, []):
    obj = _resolve_dotted_path(obj_path)
    if obj is None:
        continue
    original = getattr(obj, attribute)
    setattr(obj, attribute, fn(original))
```

### Why this is safe and minimal

- `apply_patches()` is only called during actual export, when the model being exported
  is already loaded. `_resolve_dotted_path` at that point is cheap (module already in
  sys.modules → just `getattr` walks, no actual imports).
- Early validation ("fail fast if path is wrong") is lost — but that's acceptable:
  a bad path would just silently skip (same as today's `if obj is None: continue`).
- No functional change to the patch mechanism — only when resolution happens.
- Change is entirely within `exporters/utils.py` (~5 lines changed).
- Eliminates ~111 src files from recorder_only for all 511 modeling tests.

### Why this is NOT our job to fix right now

This is an existing design issue in the exporter infrastructure, not related to the
recorder work. The recorder correctly identifies it as noise.

**PR #50000 (merged)** fixes this by storing the dotted-path string in `_PATCHES`
at decoration time and resolving lazily inside `apply_patches` — eliminating the eager
model imports entirely. Verified: 105 spurious model files removed from VIT collection-phase
deps (164 → 59). Real export tests (`test_torch_export_*`, `test_onnx_export_*`) confirmed
passing on GPU CI through `apply_patches` directly.

---

## Next: make `pipelines/__init__.py` lazy (Fix B)

### Verified behavior (2026-10-30)

```python
# transformers.models — lazy ✅
import transformers.models       # → 0 model submodules loaded
from transformers import BertModel  # → 8 files (bert + auto essentials only)

# transformers.pipelines — eager ❌
import transformers.pipelines    # → 27 pipeline submodules loaded immediately
from transformers import AudioClassificationPipeline  # → still 27 (pipelines/__init__.py runs first)
```

Even via the lazy `transformers` namespace, accessing any single pipeline class loads all 27
submodules — Python must execute `pipelines/__init__.py` (as parent package) before the submodule,
and that file is eager.

### Planned fix
Convert `pipelines/__init__.py` to use `_LazyModule`, same pattern as `transformers/__init__.py`
and `transformers/models/__init__.py`. After fix: `from transformers import AudioClassificationPipeline`
loads only `audio_classification.py`.

### Slack message draft (to share with team before starting)

> **Proposal: make `transformers.pipelines` lazy (like `transformers.models`)**
>
> Currently, any import touching `transformers.pipelines` loads all 27 pipeline source files
> at once, because `pipelines/__init__.py` is eager. This adds noise to pytest collection
> and CI dependency tracking.
>
> The fix: convert `pipelines/__init__.py` to `_LazyModule`, same pattern already used by
> `transformers/__init__.py` and `transformers/models/__init__.py`.
>
> Planning to work on this — any objections or concerns?

### How `_LazyModule` works — complete flow for `from transformers import BertModel`

```
from transformers import BertModel
│
├─ Python: lookup sys.modules["transformers"]
│         → _LazyModule("transformers", ...)   ← replaced at module load time
│
├─ _LazyModule.__getattr__("BertModel")
│   ├─ not in self._objects (cache miss)
│   ├─ not in self._object_missing_backend
│   ├─ name in self._class_to_module
│   │         → "models.bert.modeling_bert"
│   │
│   ├─ self._get_module("models.bert.modeling_bert")
│   │   └─ importlib.import_module(".models.bert.modeling_bert", "transformers")
│   │       │
│   │       ├─ Python: needs parent "transformers.models"
│   │       │   └─ sys.modules["transformers.models"] already exists
│   │       │      → _LazyModule("transformers.models", ...)   ← no re-execution
│   │       │
│   │       ├─ Python: needs parent "transformers.models.bert"
│   │       │   └─ not in sys.modules → execute models/bert/__init__.py
│   │       │       └─ else branch: sys.modules["transformers.models.bert"]
│   │       │              = _LazyModule("transformers.models.bert", ...)
│   │       │
│   │       └─ Python: import "transformers.models.bert.modeling_bert"
│   │           └─ execute modeling_bert.py → BertModel class defined
│   │              sys.modules["transformers.models.bert.modeling_bert"] = <module>
│   │              returns the module object
│   │
│   ├─ value = getattr(module, "BertModel")   ← the real class
│   └─ setattr(self, "BertModel", value)      ← cached: next access skips __getattr__
│
└─ returns BertModel class
```

Key points:
- `sys.modules["transformers"]` and `sys.modules["transformers.models"]` always hold `_LazyModule` instances — never the real module
- Parent packages are NOT re-executed on second access (already in sys.modules)
- `setattr(self, "BertModel", value)` means the second access skips `__getattr__` entirely — goes straight to the instance attribute
- Only `modeling_bert.py` gets loaded — no other model files touched

### How `models/__init__.py` installs the lazy module

```python
# models/__init__.py — entire runtime content:
from typing import TYPE_CHECKING
from ..utils import _LazyModule
from ..utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from .bert import *   # for IDEs/type checkers only, never executed at runtime
    from .clip import *
    # ...
else:
    import sys
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
```

`define_import_structure(_file)` scans all `from .X import *` in the file, reads `__all__` from each
submodule, and builds the `{frozenset(backends): {submodule: {ClassName, ...}}}` dict that
`_LazyModule` uses to build `_class_to_module`.

### Why `pipelines/__init__.py` can't use this pattern as-is

`models/__init__.py` is pure re-exports. `pipelines/__init__.py` is 1159 lines of real logic:
`pipeline()` function, `PIPELINE_REGISTRY`, `PipelineRegistry` class, `check_task()`, etc.
The `_LazyModule` would replace all of that.

Approach: keep the logic, but move the 27 pipeline class imports to lazy resolution.
Options:
- **Split**: extract class re-exports to a thin `_lazy_imports.py` using `_LazyModule`
- **`__getattr__`**: add a `__getattr__` to `pipelines/__init__.py` that lazily resolves
  pipeline class names on demand (~10 lines, no restructuring)

### Implementation (2026-10-30) — minimal __getattr__ approach

**Chosen approach**: minimal — module-level `__getattr__` + dotted-path strings in `SUPPORTED_TASKS["impl"]`.
No new files, no restructuring. For long term, a structural split (`_init_logic.py`) would be cleaner (noted in PR description).

**Changes to `src/transformers/pipelines/__init__.py`**:
1. Add `import importlib`
2. Remove the 27 eager pipeline class imports
3. Add `_PIPELINE_CLASS_TO_MODULE` dict (name → submodule path)
4. Add `_resolve_impl(impl)` helper — resolves string→class, passes through class objects unchanged
5. Change all `SUPPORTED_TASKS["impl"]` values from class objects to dotted-path strings
6. Update the 2 `targeted_task["impl"]` read sites in `pipeline()` to use `_resolve_impl()`
7. Add module-level `__getattr__` — loads submodule on demand, caches in `globals()`

**Backward compat**: none broken. If later migrated to structural split, just remove `__getattr__` + move logic out.

**Verified locally**: 8 passed, 4 skipped on bert + vit pipeline tests (4 skips = pre-existing `use_safetensors` issue).

### Dependency impact analysis — VERIFIED with real recorder (2026-10-30)

Measurement method: fetcher-recording recorder (`CI_RECORD_DEPS_OUTPUT` + `CI_RECORD_DEPS_INCLUDE_COLLECTION=1`),
each test file run in its own subprocess for clean isolation.
States: S1=Before Fix A (`0460c6d977`), S2=After Fix A (`4fed66c283`), S3=Fix A+B.
Envs: pipeline_OFF=`RUN_PIPELINE_TESTS=0`, pipeline_ON=`RUN_PIPELINE_TESTS=1`.
use_safetensors=True patched out in all runs (pre-existing unrelated bug).

**ALL27** = all 27 pipeline submodules + __init__ + base + pt_utils + audio_utils

#### test_modeling_bert.py (bert_model: test_attention_outputs + test_pipeline_text_classification + test_pipeline_fill_mask)

| State | Env | Collection | Execution |
|---|---|---|---|
| S1 Before Fix A | pipeline_OFF | ALL27 | (none) |
| S1 Before Fix A | pipeline_ON  | ALL27 | (none) |
| S2 After Fix A  | pipeline_OFF | (none) | (none) |
| S2 After Fix A  | pipeline_ON  | (none) | ALL27 |
| S3 Fix A+B      | pipeline_OFF | (none) | (none) |
| S3 Fix A+B      | pipeline_ON  | (none) | __init__, base, pt_utils, audio_utils, automatic_speech_recognition, fill_mask, image_text_to_text, mask_generation, text_classification, text_generation |

Note S1: ALL27 at collection means everything is pre-loaded; execution adds nothing new (already in sys.modules).
Note S2 pipeline_ON: ALL27 at execution — pipelines/__init__.py still eager, loads all 27 when first pipeline() call happens.
Note S3 pipeline_ON: only ~10 files — lazy init loads only what bert actually needs.

#### test_modeling_vit.py (vit_model: test_attention_outputs + test_pipeline_image_classification)

| State | Env | Collection | Execution |
|---|---|---|---|
| S1 Before Fix A | pipeline_OFF | ALL27 | (none) |
| S1 Before Fix A | pipeline_ON  | ALL27 | (none) |
| S2 After Fix A  | pipeline_OFF | (none) | (none) |
| S2 After Fix A  | pipeline_ON  | (none) | ALL27 |
| S3 Fix A+B      | pipeline_OFF | (none) | (none) |
| S3 Fix A+B      | pipeline_ON  | (none) | __init__, base, pt_utils, audio_utils, automatic_speech_recognition, image_classification, image_text_to_text, mask_generation, text_classification, text_generation |

#### test_pipelines_text_classification.py (non-slow tests)

| State | Env | Collection | Execution |
|---|---|---|---|
| S1 Before Fix A | pipeline_OFF | ALL27 | (none) |
| S1 Before Fix A | pipeline_ON  | ALL27 | (none) |
| S2 After Fix A  | pipeline_OFF | ALL27 | (none) |
| S2 After Fix A  | pipeline_ON  | ALL27 | (none) |
| S3 Fix A+B      | pipeline_OFF | __init__, base, pt_utils, audio_utils, automatic_speech_recognition, image_text_to_text, mask_generation, text_classification, text_generation | (none) |
| S3 Fix A+B      | pipeline_ON  | __init__, base, pt_utils, audio_utils, automatic_speech_recognition, image_text_to_text, mask_generation, text_classification, text_generation | text_classification |

Note S1: all 27 loaded at collection because of eager test_pipeline_mixin.py imports — regardless of env.
Note S2: Fix A cleaned model tests but not pipeline test files (their own eager import still loads all 27 at collection).
Note S3: collection reduced to ~9 files; execution adds only text_classification (was already in collection).

#### test_pipelines_image_classification.py (non-slow tests)

| State | Env | Collection | Execution |
|---|---|---|---|
| S1 Before Fix A | pipeline_OFF | ALL27 | (none) |
| S1 Before Fix A | pipeline_ON  | ALL27 | (none) |
| S2 After Fix A  | pipeline_OFF | ALL27 | (none) |
| S2 After Fix A  | pipeline_ON  | ALL27 | (none) |
| S3 Fix A+B      | pipeline_OFF | __init__, base, pt_utils, audio_utils, automatic_speech_recognition, image_classification, image_text_to_text, mask_generation, text_classification, text_generation | (none) |
| S3 Fix A+B      | pipeline_ON  | __init__, base, pt_utils, audio_utils, automatic_speech_recognition, image_classification, image_text_to_text, mask_generation, text_classification, text_generation | image_classification |

#### Key takeaways from real recorder data

1. **S1 pipeline_OFF**: even without running pipeline tests, collection of bert/vit model tests loads ALL27 — because test_pipeline_mixin.py eagerly imports all pipeline test classes which pull in pipelines/__init__.py.
2. **S2 (Fix A only)**: model test collection is fully clean (0). But execution with pipeline_ON still loads ALL27 — eager init is the bottleneck.
3. **S3 (Fix A+B)**: model test execution drops from ALL27 to ~10 task-specific files. Pipeline test collection drops from ALL27 to ~9.
4. **S1 execution=(none)**: not because pipeline tests don't run — it's because all 27 were already in sys.modules from collection; the recorder only tracks NEW imports at execution.
5. **Fix B does NOT help model tests in pipeline_OFF env** (expected — no pipeline() call = no pipeline imports at all).
6. **Unresolved extra files in S3** (audio_utils, automatic_speech_recognition, image_text_to_text, mask_generation, text_generation in bert execution): these come from cross-imports between pipeline submodules or bert's broader pipeline_model_mapping; not a regression, still far fewer than ALL27.

---

### Clean summary table (consolidated view)

Legend:
- S1 = Before Fix A  (test_pipeline_mixin does eager import of pipelines)
- S2 = After Fix A   (test_pipeline_mixin uses lazy strings — PR #50158)
- S3 = Fix A + Fix B (pipelines/__init__.py is lazy too)
- init  = `src/transformers/pipelines/__init__.py` loaded (YES / no)
- base  = `src/transformers/pipelines/base.py` loaded (YES / no)
- #task = # per-task submodule files loaded (text_classification.py, audio_classification.py, ...)
- Collection is identical for pipe_OFF and pipe_ON (verified — RUN_PIPELINE_TESTS has no effect on collection)
- NOTE: classify() requires `src/transformers/` prefix — to avoid false-matching `tests/pipelines/__init__.py`

**Model tests (bert + vit — identical results)**

```
+-------+-----------------------------+--------------------------+-----------------------------+
| State | Collection                  | Exec  pipe_OFF           | Exec  pipe_ON               |
+-------+-----------------------------+--------------------------+-----------------------------+
| S1    | init:YES  base:YES  task:24 | init:no  base:no  task:0 | init:no  base:no  task:0    |
| S2    | init:no  base:no  task:0    | init:no  base:no  task:0 | init:YES  base:YES  task:24 |
| S3    | init:no  base:no  task:0    | init:no  base:no  task:0 | init:YES  base:YES  task:6  |
+-------+-----------------------------+--------------------------+-----------------------------+
```

**Pipeline test: test_pipelines_text_classification.py**

```
+-------+-----------------------------+--------------------------+--------------------------+
| State | Collection                  | Exec  pipe_OFF           | Exec  pipe_ON            |
+-------+-----------------------------+--------------------------+--------------------------+
| S1    | init:YES  base:YES  task:24 | init:no  base:no  task:0 | init:no  base:no  task:0 |
| S2    | init:YES  base:YES  task:24 | init:no  base:no  task:0 | init:no  base:no  task:0 |
| S3    | init:YES  base:YES  task:5  | init:no  base:no  task:0 | init:no  base:no  task:1 |
+-------+-----------------------------+--------------------------+--------------------------+
```

**Pipeline test: test_pipelines_image_classification.py**

```
+-------+-----------------------------+--------------------------+--------------------------+
| State | Collection                  | Exec  pipe_OFF           | Exec  pipe_ON            |
+-------+-----------------------------+--------------------------+--------------------------+
| S1    | init:YES  base:YES  task:24 | init:no  base:no  task:0 | init:no  base:no  task:0 |
| S2    | init:YES  base:YES  task:24 | init:no  base:no  task:0 | init:no  base:no  task:0 |
| S3    | init:YES  base:YES  task:6  | init:no  base:no  task:0 | init:no  base:no  task:1 |
+-------+-----------------------------+--------------------------+--------------------------+
```

### Analysis

**Why collection is identical for pipe_OFF and pipe_ON:**
`RUN_PIPELINE_TESTS` only controls whether `@is_pipeline_test`-decorated methods run at
execution time. It has zero effect on what Python imports when pytest first loads the test
file. Verified: all 12 combinations (3 states x 4 test files) have identical collection
sets across pipe_OFF and pipe_ON.

**Why `init:YES` at S1 model test collection:**
In S1, `test_pipeline_mixin.py` eagerly imports all pipeline test classes at module level.
Those test classes import `from transformers.pipelines import ...`, which pulls in
`pipelines/__init__.py` and all 24 submodules. This happens during pytest collection of any
model test file that inherits `PipelineTesterMixin` — even if `RUN_PIPELINE_TESTS=0`.

**Why `init:YES` always for pipeline test files at collection (unavoidable):**
Pipeline test files (`test_pipelines_*.py`) do top-level imports like
`from transformers import TextClassificationPipeline`. Python must import
`pipelines/__init__.py` to resolve this — unavoidable without refactoring all pipeline test
files to use lazy imports inside test methods. Fix B reduces task:24->task:5-6 at collection
(big win) but cannot eliminate `init:YES` at collection.

**Why `init:no` at pipeline test execution (both pipe_OFF and pipe_ON):**
`src/transformers/pipelines/__init__.py` is NOT re-encountered at execution for pipeline
test files. The recorder hooks `builtins.__import__` for every import statement. During
test execution, no code path issues a new `import transformers.pipelines` statement.
The `init:YES` that previously appeared here was a **bug in classify()**: the pattern
`"pipelines/__init__"` also matched `tests/pipelines/__init__.py` (the test package's own
`__init__`, which IS encountered during execution as pytest accesses the test module).
Fixed by requiring `"src/transformers/pipelines/__init__"` prefix.

**`init:YES` in src/transformers only affects model tests when pipe_ON:**
For model test files, `src/transformers/pipelines/__init__.py` is never imported at
collection (clean in S2/S3). It only appears at execution, and only when `pipe_ON`
(pipeline test methods call `pipeline()` which triggers `_resolve_impl()` which imports
the needed task submodule, and `__init__` is encountered as parent package). With
`pipe_OFF`, model tests have zero dependency on any `src/transformers/pipelines/` file.

**Fetcher graph implications:**

| Who loads `src/transformers/pipelines/__init__.py`? | When? |
|---|---|
| Pipeline test files (`test_pipelines_*.py`) | Always at collection, unavoidably |
| Model test files (`test_modeling_*.py`) | Only at execution, only when `pipe_ON` |

For the fetcher dependency graph recorded with `pipe_ON` (the default in CI):
- S1/S2: a change to ANY of 24 task submodules triggers ALL model tests (too broad — e.g.
  `audio_classification.py` change would trigger `test_modeling_bert.py` even though bert
  has no audio pipeline)
- S3: a change to a task submodule only triggers model tests that actually use that pipeline
  (e.g. `text_classification.py` -> bert, `image_classification.py` -> vit). Precise and correct.

Fix B's win is therefore dual:
- **Speed**: fewer imports per test run
- **Fetcher precision**: narrower, accurate dependency edges (24 -> 6 per model test)

**`init`/`base` impact pipeline test files at collection only — semantically correct:**
A change to `pipelines/__init__.py` or `pipelines/base.py` correctly triggers pipeline
test files (they import both at collection) and model test files (they load both at
execution with pipe_ON). These edges are meaningful: `base.py` defines the `Pipeline`
base class; `__init__.py` is the entry point. Re-running tests on changes to either is correct.

**pipe_OFF harmlessness:**
Even if a change to `__init__`/`base` schedules a model test file with `pipe_OFF`, all
`@is_pipeline_test` methods are skipped instantly — near-zero cost. The model's own
non-pipeline tests still run, which is correct regardless. So the dependency edge is
conservative but never harmful.

**Correction: static fetcher precision claim (verified Oct 2026):**
The "24 → 6" fetcher precision claim above applies to the *recorded* graph (pipe_ON). For
the *static* fetcher (`create_reverse_dependency_map`), the picture is different due to
the `__init__.py` stop rule (line 883 of `tests_fetcher.py`):

```python
if d.endswith("__init__.py"):
    continue  # don't propagate __init__'s own deps during transitive closure
```

Two barriers prevent `fill_mask.py` from ever being reached — verified 100% via debug prints
added directly to `tests_fetcher.py` on Oct 15 commit + `create_reverse_dependency_map` run:

**Barrier 1 — `get_module_dependencies` fails to resolve `FillMaskPipeline` (fetcher bug):**

`get_module_dependencies` does NOT stop at `__init__.py` — it actively looks through it.
For `from transformers import FillMaskPipeline`, it finds `transformers/__init__.py` and
calls `define_import_structure(transformers/__init__.py)` to trace where `FillMaskPipeline` comes from.

On Oct 15, `transformers/__init__.py` has:
```python
_import_structure = {
    "pipelines": ["FillMaskPipeline", "AudioClassificationPipeline", ...],  # all pipeline classes under ONE package key
    ...
}
```

`define_import_structure` returns `{"pipelines": {"FillMaskPipeline", ...}}`. The path conversion:
```python
_module = "pipelines".replace(".", "/")          # → "pipelines"  (no dots)
_module = "src/transformers/__init__.py".replace("__init__.py", "pipelines.py")
#        → "src/transformers/pipelines.py"       # ← DEAD PATH — file does not exist
```

`src/transformers/pipelines/` is a package (directory with `__init__.py`), not a `.py` file.
The path `"src/transformers/pipelines.py"` is added to the submodule map but never matches
anything useful. `FillMaskPipeline` stays unresolved → `transformers/__init__.py` kept as
fallback dep (line 772). `fill_mask.py` is never produced.

This is a **fetcher bug** — purely a static analysis limitation. Real Python usage is unaffected:
`from transformers import FillMaskPipeline` works perfectly at runtime via `_LazyModule.__getattr__`.

The bug would be fixed if `_import_structure` used per-file keys like `"pipelines.fill_mask": ["FillMaskPipeline"]`
instead of the package-level `"pipelines"` key — then the path conversion would produce
`"src/transformers/pipelines/fill_mask.py"` (real file), and the fetcher would resolve correctly.

**TODO (low priority)**: fix `define_import_structure` path conversion to handle package-level keys
correctly (i.e. `"pipelines"` → `"src/transformers/pipelines/__init__.py"` instead of the dead
`"src/transformers/pipelines.py"`).

**Barrier 2 — transitive closure stop rule:**

The transitive closure loop stops at `__init__.py` deps:
```python
if d.endswith("__init__.py"):
    continue  # line 883 — don't expand __init__'s deps
```

Note: Barrier 2 does NOT block `get_module_dependencies` itself — that function actively looks
THROUGH `__init__` files (that's the purpose of lines 729-772). Barrier 2 only blocks the
*transitive closure loop* from expanding `__init__` deps of OTHER files.

**Key insight — fixing Barrier 1 alone WOULD help `test_pipelines_fill_mask.py`:**

If Barrier 1 were fixed, `get_module_dependencies('tests/pipelines/test_pipelines_fill_mask.py')`
would trace:
```
from transformers import FillMaskPipeline
  → transformers/__init__.py  (look through — no stop in get_module_dependencies)
    → pipelines/__init__.py   (FillMaskPipeline found — look through again)
      → fill_mask.py          (from .fill_mask import FillMaskPipeline — concrete file ✓)
```
`fill_mask.py` is non-init → added directly to `direct_deps[test_pipelines_fill_mask.py]`.
Barrier 2 doesn't apply here — `fill_mask.py` is already a direct dep, not something reached
via transitive expansion of an `__init__`.

Full impact if Barrier 1 fixed:

| State | `fill_mask.py` change selects |
|---|---|
| Oct 15, Barrier 1 still broken (current) | 0 tests |
| Oct 15, Barrier 1 fixed (hypothetical) | `test_pipelines_fill_mask.py` + 434 model tests (via test_pipeline_mixin real imports) |
| After PR #50158, Barrier 1 fixed (hypothetical) | `test_pipelines_fill_mask.py` only (string refs broke model test chain) |

So fixing Barrier 1 alone is actually meaningful — it would correctly select at least
`test_pipelines_fill_mask.py`. Still low priority since the recorder handles these deps at runtime.

The "After PR #50158" row is the current state of main. Even with Barrier 1 fixed, model tests
that exercise `FillMaskPipeline` at runtime (e.g. bert's pipeline tests) would NOT be selected
— the static fetcher has no way to know about that runtime connection. This is exactly the gap
the recorder fills: by capturing actual CI imports, it provides the `fill_mask.py → test_modeling_bert.py`
edge that static analysis structurally cannot see.

**Verified with `create_reverse_dependency_map` on Oct 15 baseline** (all Windows path
issues fixed; result saved to `reverse_map_oct15_original_real.json`):
- `fill_mask.py` with original import (`from transformers import X`): **1 total dependent** (`src/transformers/pipelines/__init__.py` only)
  - 0 test file dependents — `test_pipelines_fill_mask.py` is NOT triggered even though it imports `FillMaskPipeline`
  - `test_pipelines_fill_mask.py` IS a dependent of `pipelines/__init__.py` (3880 total), but NOT of `fill_mask.py`
  - Pre-existing behavior — unrelated to any of our work
- `fill_mask.py` with direct import (`from transformers.pipelines.fill_mask import X`): **434 model test dependents**
  (temporary modification to `test_pipelines_fill_mask.py`; result saved to `reverse_map_oct15_real.json`)

So for the static fetcher, changing a pipeline submodule selects **0 model tests** regardless
of whether `pipelines/__init__.py` is lazy or eager. The static fetcher precision
benefit of the lazy PR does **not** apply.

### Status
- [x] Slack message — deferred (team can review via PR)
- [x] Implement lazy `pipelines/__init__.py` — done (uncommitted on `fetcher-recording`)
- [x] Verify with real recorder at 3 commits x 2 envs — DONE (see table above)
- [x] Verify: bert + vit pipeline tests pass (8 passed, 4 skipped)

---

## Is the lazy pipelines PR over-engineering?

A valid question: is the added complexity worth it, ignoring the CI fetcher angle?

**Apparent counter-arguments:**
- "27 small Python files load fast anyway"
- "whoever calls `pipeline(...)` triggers the submodule load immediately anyway"
- "if you care about startup time, don't import transformers.pipelines"

**Why the last point is wrong — and matters most:**
The third argument is invalid: `import transformers` itself loads `transformers/pipelines/__init__.py`
via the top-level `transformers/__init__.py`. Every user of `import transformers` pays the cost of
loading all 27 pipeline submodules, with no opt-out.

**On complexity:**
The added code is actually idiomatic Python:
- `_PIPELINE_CLASS_TO_MODULE` — a plain dict, self-documenting
- `_resolve_impl()` — 4 lines, obvious
- `__getattr__` — standard lazy module pattern, well-known idiom
- `TYPE_CHECKING` imports — boilerplate, no logic

The `check_pipeline_typing.py` fix was one line (`rsplit(".", 1)[-1]`). Not complex.

**Verdict:**
The PR is justified on one solid ground:
1. **Startup time**: every `import transformers` user was paying to load 27 submodules they may not use

The CI fetcher precision claim (point 2 from earlier) does **not hold** for the static fetcher —
see correction above. The `__init__.py` stop rule means 0 model tests are selected for any
pipeline submodule change regardless of lazy/eager. The recorded fetcher (pipe_ON) does benefit,
but that's independent of the lazy PR itself.

---

## Which PRs changed static fetcher behavior (Oct 2026)

Three related PRs were merged around the same time. Only one changed static fetcher behavior:

| PR | Description | Static fetcher impact |
|---|---|---|
| #50000 | `@register_patch` lazy resolve — stores string path instead of `importlib.import_module` at decoration time | **None** — `importlib.import_module(...)` calls are function calls, invisible to AST parsing |
| #50001 | Defer eager model imports in `token_classification.py` / `text_to_audio.py` to method bodies | **None** — fetcher parses imports inside methods too; same deps before and after |
| #50158 | `test_pipeline_mixin.py` — switch from real imports to string references for pipeline test classes | **Significant** — broke the static chain `model tests → test_pipeline_mixin.py → pipeline test files → pipeline src files` |

PR #50158 (`4fed66c283`) is the only one that actually changed static fetcher dependency graph. Before it, changing any pipeline test file would select all 434 model tests (via the real import chain). After it, that chain is broken — changing a pipeline test file selects only that file itself.

**Latent risk introduced by PR #50158**: if pipeline test files were switched to direct submodule imports
(`from transformers.pipelines.fill_mask import FillMaskPipeline` instead of `from transformers import FillMaskPipeline`),
then changing `fill_mask.py` would correctly select `test_pipelines_fill_mask.py` (no `__init__` hop)
but would NOT select model tests that exercise `FillMaskPipeline` at runtime (e.g. bert) —
because the real-import chain that used to connect model tests → pipeline test files was severed by PR #50158.
Before PR #50158, those model tests were selected (accidental side effect of real imports); after, they'd silently disappear.

In the current state (indirect import via `from transformers import X`), this risk is dormant — 0 tests
were selected both before and after PR #50158 (pre-existing fetcher miss). No regression introduced.
The recorder is the correct long-term solution for pipeline src → model test runtime edges.

---

## Static fetcher analysis — pipeline src files (verified Oct 2026, PR #50180)

### Verified with real CI (PR #50180: trivial change to `fill_mask.py`)

**Result**: changing `fill_mask.py` → **0 tests selected** on current main.

The static fetcher misses even `test_pipelines_fill_mask.py` because:
- `test_pipelines_fill_mask.py` imports `from transformers import FillMaskPipeline`
- Fetcher resolves this to `transformers/__init__.py` → **STOP** (`__init__` stop rule)
- `fill_mask.py` never appears as a dep of `test_pipelines_fill_mask.py`
- So `reverse_map[fill_mask.py]` = only `[pipelines/__init__.py]` — no test files

### The two failure modes (on Oct 15 baseline, real imports in `test_pipeline_mixin.py`)

| Import style in `test_pipelines_fill_mask.py` | `fill_mask.py` change selects |
|---|---|
| `from transformers import FillMaskPipeline` | **0 tests** — misses even the pipeline test |
| `from transformers.pipelines.fill_mask import FillMaskPipeline` | **434 model tests** — far too many |

There is no middle ground with the static fetcher on Oct 15: either miss the relevant test,
or select all model tests (because `test_pipeline_mixin.py` had real imports connecting all
model tests → all pipeline test files → pipeline src files).

### Why the string-ref refactor changed things

A recent merged PR refactored `test_pipeline_mixin.py` from real imports to string references:
```python
# Before (Oct 15):
from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests

# After (current main):
"fill-mask": {"test": "tests.pipelines.test_pipelines_fill_mask.FillMaskPipelineTests"},
```

The static fetcher can't see string references → the chain model tests → pipeline test files
is broken. This fixes the false-positive (434 model tests), but the miss remains.

With direct import + string-ref refactor (current main):
- `fill_mask.py` change → selects `test_pipelines_fill_mask.py` only ✓ (ideal)
- BUT model tests that exercise `FillMaskPipeline` at runtime (e.g. bert's pipeline tests)
  are NOT selected ✗ — this is a runtime dependency, invisible to static analysis

### Conclusion: recorder is needed

The static fetcher cannot bridge runtime pipeline→model dependencies. When `fill_mask.py`
changes, `test_modeling_bert.py`'s pipeline tests (which load `FillMaskPipeline` at runtime)
won't be selected — the static fetcher has no way to know this connection exists.

The recorder captures these edges from actual CI runs (`pipe_ON` mode), making it the only
robust solution for correct pipeline src → model test dependency tracking.

---

## FUNDAMENTAL FLAW IN deps_by_file.json APPROACH (2026-10-31)

### What was built (fetcher-recording branch)
`deps_by_file.json` records per-test-file deps split into `collection` and `execution` phases.
The import hook (`builtins.__import__`) fires for every import statement Python encounters.
`pytest_collectstart` sets `_ci_currently_collecting_file` so hook can attribute imports to
the right test file.

### The xdist collection ordering problem
In xdist workers, **test modules are imported BEFORE `pytest_collectstart` fires**.
xdist workers receive test node IDs from the controller and import test modules directly
to run them. Only then does pytest fire `pytest_collectstart` for bookkeeping. By the time
`set_ci_collecting_file("tests/models/bert/test_modeling_bert.py")` is called, Python has
already executed test_modeling_bert.py's module body — all `import` statements have run
with `_ci_currently_collecting_file = None` → attributed to `__collection__` → **filtered out**.

### Empirical evidence
Verified from CI run 41206149905 (tests_torch, 8 shards, n=8 xdist):
- `tests/models/bert/test_modeling_bert.py` → `collection: ["tests/models/bert/test_modeling_bert.py"]`
  Only 1 dep: the test file itself. `src/transformers/models/bert/modeling_bert.py` — MISSING.
- Only 8 files (one per shard = one per fresh worker process) get correct collection deps (~233-241).
  These are the first test file imported in each fresh worker process.

Local verification:
- bert alone (no xdist): 233 collection deps ✓ (full transitive chain captured)
- afmoe then bert (no xdist, sequential): afmoe=240, bert=29 (fewer — some modules cached)
- bert in CI (xdist, ~8th test in worker): bert=1 (effectively all modules cached before hook active)

### Execution phase also mostly useless
For bert execution: `conversion_mapping.py`, `modeling_layers.py`, `modeling_utils.py` — 3 files.
Missing: `modeling_bert.py` itself (already in sys.modules before test ran, module body won't re-run).
Only genuinely NEW imports during test execution get captured; most framework files are pre-cached.

### Root cause summary
The import hook correctly fires for every `import X` Python ENCOUNTERS. But:
1. Python only executes a module body ONCE (module bodies don't re-run for cached modules).
2. In xdist, the test module body runs BEFORE `_ci_currently_collecting_file` is set.
   → All those imports are unattributed (go to `__collection__`, filtered).
3. By execution phase, almost everything is in sys.modules already → only dynamic dispatch
   deps (AutoProcessor.from_pretrained etc.) are genuinely new.

### What actually works
The original `deps_by_sys_module.json` (session-level sys.modules diff) correctly captures
dynamic dispatch deps that are NEW during the session. But it gives no per-test-file attribution.

### Approaches to fix per-test-file attribution
1. **sys.modules snapshot diff per test file**: Before/after each test file's collection,
   diff `sys.modules`. Works regardless of xdist ordering. But needs pytest hooks that fire
   around the actual module import, not `pytest_collectstart` (which fires after).
   Hook to use: `pytest_pycollect_makemodule` (fires when pytest is about to import a module).

2. **Fresh subprocess per test file**: `subprocess.run(["python", "-c", "import tests.models.bert.test_modeling_bert"])`
   → empty sys.modules → full transitive chain captured via coverage or sys.modules diff.
   Slow but precise for collection-time deps. No GPU needed.

3. **Execution-only (abandon collection)**: Give up on collection-phase per-file attribution.
   Use only execution deps (which are correctly attributed via `_ci_current_test_id`).
   Supplement with the static fetcher for collection-time deps. More practical but less complete.

4. **coverage.py per test**: Most complete but requires coverage overhead in CI.

### Current status
`deps_by_file.json` in CI gives mostly useless collection data for most test files.
Execution data is correct but limited to NEW imports during test execution only.
The `collate_deps_by_file.py` script aggregates per-shard → per-job correctly (verified,
no duplicate keys within a job's shards). But the underlying data quality is poor.

**Next step needed**: Decide on approach before investing more in the pipeline.

### Verification: xdist-only flaw confirmed (2026-10-31)

Tested whether the hook correctly captures `modeling_bert.py` under bert's collection key
when `modeling_bert.py` was pre-loaded by an earlier test in the same process (no xdist):

1. Added `from transformers.models.bert.modeling_bert import BertModel` to afmoe's test file
2. Ran `pytest tests/models/afmoe/test_modeling_afmoe.py tests/models/bert/test_modeling_bert.py`
   (no xdist, sequential, afmoe collected first)
3. Result:
   - `afmoe`: coll=45, `modeling_bert.py=True` (afmoe pre-loaded it)
   - `bert`: coll=233, `modeling_bert.py=True` ✓ (still captured despite being pre-cached)

**Conclusion**: The hook fires for every `import` statement Python encounters, even for
already-cached modules. Without xdist, `pytest_collectstart` fires BEFORE the test module
is imported, so all imports in the test file body are correctly attributed to that file.
The flaw is **xdist-specific only** — caused by xdist importing test modules before
`pytest_collectstart` fires, not by module caching itself.

Full collection breakdown for bert (collected second, after afmoe pre-loaded modeling_bert.py):
- 226 src files, 7 test files — complete and accurate
- `modeling_bert.py`, `configuration_bert.py`, `models/bert/__init__.py` all present ✓
- Also picks up ~200 unrelated models (bart, clip, blt, llama, etc.) — expected noise from
  `test_pipeline_mixin.py` and `test_modeling_common.py` pulling in the full transformers
  ecosystem at collection time. This is the same over-approximation the static fetcher has,
  but from runtime rather than static analysis.
- The reverse map (src→test) from these deps will have many-to-many edges; fetcher already
  handles this with its existing union/intersection logic.

## Collection phase order-dependency root cause (2026-10-31)

### Background: how transformers lazy-loading works

`transformers/__init__.py` does NOT define `BertModel` directly. Instead it defines
`__getattr__` which, on first access of any name, calls `importlib.import_module(...)` to
load the real module and stores the result in `transformers.__dict__`. After first access,
the name lives in `__dict__` and `__getattr__` is never called again for it.

### Background: how our hook works

We override `builtins.__import__`. Every `import` statement calls `builtins.__import__`
with `name` = the module being imported and `fromlist` = the names imported from it.
Our hook calls the original import, then records the module if it is under `transformers/`.

`_record_import_chain(module_name)` does NOT trigger any new import. It looks up
`sys.modules.get(module_name)` to get the already-loaded module object, reads its `__file__`,
and records the path. If the module is not in `sys.modules`, it silently returns.

### Issue 1 (severe) — CI "1 item" for bert

**Normal case** (BertModel NOT yet in `transformers.__dict__`):

```
from transformers import BertModel
  -> builtins.__import__("transformers", fromlist=("BertModel",))
  -> our _tracking_import hook fires
  -> _record_import_chain("transformers") -> records __init__.py
  -> fromlist loop: item="BertModel"
       checks sys.modules.get("transformers.BertModel") -> None  (old code: gives up here)
  -> Python calls transformers.__getattr__("BertModel")
       -> importlib.import_module("transformers.models.bert.modeling_bert")
       -> our _tracking_import_module hook fires
       -> _record_import_chain("transformers.models.bert.modeling_bert") -> records modeling_bert.py
       -> BertModel stored in transformers.__dict__
```

Result: `__init__.py` + `modeling_bert.py` both recorded for bert.

**Problem case** (BertModel ALREADY in `transformers.__dict__`, put there by an earlier test):

```
from transformers import BertModel
  -> builtins.__import__("transformers", fromlist=("BertModel",))
  -> our _tracking_import hook fires
  -> _record_import_chain("transformers") -> records __init__.py
  -> fromlist loop: item="BertModel"
       checks sys.modules.get("transformers.BertModel") -> None  (old code: gives up here)
  -> Python finds BertModel in transformers.__dict__ -> returns directly, NO __getattr__ call
  -> _tracking_import_module hook never fires
```

Result: only `__init__.py` recorded for bert. `modeling_bert.py` is completely missed.

**Why the old fromlist check always failed for classes:**

The old code checked `sys.modules.get("transformers.BertModel")`. This key NEVER exists in
`sys.modules`. Python stores the module under `"transformers.models.bert.modeling_bert"`, not
`"transformers.BertModel"`. The check was designed for real subpackages (e.g.
`from transformers import models` -> checks `"transformers.models"`, which does exist). For
classes/functions coming from a nested module, this check always fell through, and in the
normal case that was fine because `__getattr__` fired and `_tracking_import_module` saved us.
In the problem case `__getattr__` is never called, so nothing else saves us.

**Why the problem case occurs in CI but not locally:**

In CI each shard runs many test files (albert, bert_generation, camembert ... bert in order).
Some file before bert imports a bert symbol (e.g. `from transformers import BertModel`),
which fires `__getattr__` and populates `transformers.__dict__` with that bert class. By the
time bert's own collection runs, `__getattr__` is bypassed for any symbol already in `__dict__`.
Locally, running only bert's test, no earlier test populates `__dict__` first.
This is why CI shard0 gave 1 item for bert while local runs gave 13 items.

### Fix for Issue 1 (committed e903201198)

In `_tracking_import`, after the fromlist submodule check falls through, instead of giving
up, we read `__module__` from the class object to find its real defining module:

```python
else:
    # item is a class/function -- find its defining module via __module__
    parent_mod = sys.modules.get(resolved_name)
    if parent_mod is not None:
        attr = parent_mod.__dict__.get(item)  # __dict__ avoids triggering __getattr__
        if attr is not None:
            mod_name = getattr(attr, "__module__", None)
            if isinstance(mod_name, str) and mod_name.split(".")[0] in ("transformers", "tests"):
                _record_import_chain(mod_name)
```

**Step by step for the problem case:**

```
fromlist item = "BertModel"
"transformers.BertModel" not in sys.modules -> else branch (fix)
parent_mod = sys.modules["transformers"]           # the transformers package object
attr = parent_mod.__dict__.get("BertModel")        # <class BertModel> -- no __getattr__ triggered
mod_name = attr.__module__                         # "transformers.models.bert.modeling_bert"
_record_import_chain("transformers.models.bert.modeling_bert")
  -> sys.modules.get("transformers.models.bert.modeling_bert") -> found
     (loaded when the earlier test triggered __getattr__; stays in sys.modules for entire process)
  -> reads __file__ -> records modeling_bert.py
```

**Why `__dict__` instead of `getattr`:**
`getattr(parent_mod, "BertModel")` would call `__getattr__` if the name was not in `__dict__`,
triggering our hook recursively. `__dict__.get(item)` is a direct dict lookup -- no `__getattr__`.

**Why `attr.__module__` is always correct:**
`__module__` is set when the class body executes inside `modeling_bert.py`. It is always
`"transformers.models.bert.modeling_bert"` regardless of how the class was later imported or
cached. It is not affected by lazy loading or `transformers.__dict__` caching.

**Why `_record_import_chain` can find the module in `sys.modules`:**
`"transformers.models.bert.modeling_bert"` was put in `sys.modules` by the earlier test that
triggered `__getattr__`. It stays there for the entire process lifetime. Even though bert's own
`from transformers import BertModel` did not trigger a new load, the module object is available
for our recorder to read its `__file__`.

### Empirical verification (script: `debug_verify_fix.py`)

Exact CI scenario simulated in sequence:
1. Install tracker (as conftest does at session start)
2. Set collecting_file = afmoe's test file
3. `from transformers import BertModel` -> fires `__getattr__` -> BertModel enters `__dict__`
4. Set collecting_file = bert's test file
5. `from transformers import BertModel` -> BertModel already in `__dict__` -> fix fires

Result: `modeling_bert.py` appears in bert's recorded deps. Fix confirmed.

### Issue 2 (minor, not fixed) -- 13 vs 29 items

Even in the normal case (no pre-loading), bert gets 13 items instead of 29. The 16 missing
items are transitive imports inside `modeling_bert.py`'s body (`import activations`,
`import cache_utils`, etc.). These only fire when `modeling_bert.py`'s body executes for
the first time. If any earlier test already caused `modeling_bert` to be loaded (even via a
different path), the body does not re-execute and those 16 imports do not fire again.

The 16 missing files are common framework files (`activations.py`, `cache_utils.py`,
`modeling_utils.py`, etc.) that appear in every model test and are filtered out by the
fetcher anyway. They carry no bert-specific signal. Not worth fixing.

### Debug instrumentation (2026-10-31)
All debug code (`_ci_dbg*`, `_CI_DBG_TARGETS`, HOOK_FIRE* events, LOST/SKIP_DUP/
FILTERED_LOAD_ATTR/RECORD events, SESSIONFINISH dump) was extracted to
`ci_deps_debug_patch.py` (untracked) before committing the fix.

To re-enable:
1. See `ci_deps_debug_patch.py` -- 8 numbered patches with exact insertion points.
2. Run with `CI_RECORD_DEPS_DEBUG=1 CI_RECORD_DEPS_OUTPUT=/tmp/out/deps_by_file.json pytest ...`
3. Logs land in `<CI_RECORD_DEPS_OUTPUT-parent>/debug_<worker>_pid<pid>.log`

### Issue 3 (severe) -- bert still 1 item in CI after Issue 1 fix

After deploying commit e903201198 (attr.__module__ fix), CI run 41241063696 still showed
bert=1 item in all 8 workers. The attr.__module__ fix was correct but insufficient on its own.

**Root cause**: `test_modeling_auto.py` line 38 has a module-level import:
```python
from ..bert.test_modeling_bert import BertModelTester
```
This import runs at collection time, while `_ci_currently_collecting_file` is set to
`test_modeling_auto.py`. By the time pytest fires `pytest_collectstart` for
`test_modeling_bert.py`, the module is already in `sys.modules` — its body does not
re-execute — so our `builtins.__import__` hook never fires for its imports under the
correct collecting-file context.

**Key distinction**: The attr.__module__ fix (Issue 1) handles the case where a
transformers CLASS is already in `transformers.__dict__`. Issue 3 is different: here the
entire TEST MODULE body already ran under a wrong context. These are two orthogonal problems:
- Issue 1: `BertModel` already in `transformers.__dict__` → attr.__module__ fix
- Issue 3: `test_modeling_bert.py` already in `sys.modules` → scan_preloaded fix

### Fix for Issue 3 (committed 8cd4b3dec3)

New function `scan_preloaded_test_module_deps(nodeid)` in `testing_utils.py`:

```python
def scan_preloaded_test_module_deps(nodeid: str) -> None:
    """Record deps for a test module that was pre-loaded before its pytest_collectstart.
    ...
    """
    if not os.environ.get("CI_RECORD_DEPS_OUTPUT"):
        return
    if _ci_currently_collecting_file is None:
        return

    # Convert nodeid path to Python module name
    module_name = nodeid.replace("\\", "/").removesuffix(".py").replace("/", ".")
    module = sys.modules.get(module_name)
    if module is None:
        return  # Not pre-loaded; the normal import hook will handle it.

    # Module already in sys.modules -- its body already ran under a different context.
    # Scan its __dict__ for every object whose __module__ points into transformers/ or tests/.
    for obj in module.__dict__.values():
        mod_name = getattr(obj, "__module__", None)
        if isinstance(mod_name, str) and mod_name.split(".")[0] in ("transformers", "tests"):
            _record_import_chain(mod_name)
```

Called from `conftest.py`'s `pytest_collectstart`, AFTER `set_ci_collecting_file`:

```python
def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        set_ci_collecting_file(collector.nodeid)
        scan_preloaded_test_module_deps(collector.nodeid)  # handles pre-loaded modules
```

**How it works**: When `test_modeling_bert.py` is already in `sys.modules`, this function
walks its `__dict__` and calls `_record_import_chain` for every object whose `__module__`
points into `transformers/` or `tests/`. By the time this runs, `set_ci_collecting_file`
has already set the correct context, so all recorded deps are attributed to bert.

### CI confirmation of both fixes (run 41243330513)

After both fixes (e903201198 + 8cd4b3dec3), CI run 41243330513:
- All 8 shard0 workers: bert=**10 items** (up from 1)
- `src/transformers/models/bert/modeling_bert.py` present ✓
- Other bert-specific files present (configuration_bert.py, models/bert/__init__.py, etc.)

### bert+vit local experiment (final verification)

Ran with no xdist, bert collected first:
```
CI_RECORD_DEPS_OUTPUT=... pytest tests/models/bert/test_modeling_bert.py \
    tests/models/vit/test_modeling_vit.py --collect-only -p no:xdist -q
```

Results:
- `test_modeling_bert.py`: **234 items** — first collected, module body runs fresh,
  full transitive framework deps captured (the "framework import tax")
- `test_modeling_vit.py`: **29 items** — second collected, framework already in sys.modules,
  only vit-specific modules captured via `scan_preloaded_test_module_deps()`

**Item count pattern**:
- First test file in any worker/session: 200+ items (entire import chain from scratch)
- Subsequent test files: 10-60 model-specific items (only freshly-imported deps)
- In CI shard0: afmoe is always first → 241 items; all others get 10-60 items

**The 200+ "framework tax" on the first test is expected and harmless**: The reverse map
(src→test) built from these deps over-approximates which tests to run for each changed file,
but the static fetcher already does the same. Extra edges = more tests run = safe.

### Recorder final status

The recorder is **working correctly** on CI as of commits e903201198 + 8cd4b3dec3:
- Every test file gets its model-specific deps correctly attributed
- The first-collected test in each worker gets extra framework deps (noise, not a problem)
- Dynamic dispatch deps (AutoProcessor.from_pretrained, etc.) are captured in execution phase
- xdist-specific pre-loading is handled by `scan_preloaded_test_module_deps`
- `attr.__module__` pre-population case handled by the attr.__module__ fix

**Next**: Component 5 — consumer in `tests_fetcher.py`

---

## Module dependency graph — design and implementation (2026-11-01)

### Motivation
The per-test-file recorder correctly captures imports for the FIRST test file in a session
that imports a given module (the module body executes, hook fires, recorded). For the
SECOND or later test file that imports the same module, Python uses the sys.modules cache —
the module body does NOT re-execute, the hook does NOT fire for its transitive imports.

Result: bert (2nd collected) got 32 collection deps vs afmoe (1st) got 241 — missing
~200 transitive imports (modeling_utils, configuration_bert, pipeline infra, etc.).

### Solution: compact directed graph X→Y, Y→Z
Instead of recording the full transitive closure at load time, record ONLY direct edges:
- When module X's body runs and imports Y: record edge X → Y
- When Y's body runs and imports Z: record edge Y → Z
- At session end: BFS from each test file's seed set through the graph → full transitive set

This is compact (one entry per unique (parent, child) pair across the whole session) and
handles second-encounter correctly: X's graph edges were built when X first loaded;
BFS from test B's seed through X finds the same transitive set as test A.

### Implementation: loading stack

Key mechanism: `_ci_module_loading_stack: list[str]` — stack of module names whose
bodies are currently executing.

```python
# In _tracking_import, BEFORE calling _original_import:
is_new = resolved_name not in sys.modules
if is_new:
    _ci_module_loading_stack.append(resolved_name)
try:
    result = _original_import(name, ...)   # body runs here if is_new; recursive calls fire
finally:
    if is_new:
        _ci_module_loading_stack.pop()

# AFTER _original_import:
_record_module_graph_edge(resolved_name)   # records stack[-1] → resolved_name
_record_import_chain(resolved_name)        # existing per-test-file attribution (unchanged)
```

`_record_module_graph_edge(module_name)`:
- If stack is empty: return (seed import from test file code — no graph edge needed)
- If stack non-empty: record `_ci_module_graph[stack[-1]].add(module_name)`
- Fast path: skip frame walk if edge already recorded
- Filter: discard `_load_attr_from_module` bulk-iteration imports (same rule as before)

### Second-encounter walkthrough (bert collected after afmoe)

When bert imports `from transformers import BertModel` (transformers cached, BertModel in __dict__):
- `_tracking_import("transformers", fromlist=["BertModel"])` fires
- stack = [] (nothing loading) → is_new=False → don't push
- fromlist __module__ fix: `BertModel.__module__ = "transformers.models.bert.modeling_bert"`
  → `_record_import_chain("transformers.models.bert.modeling_bert")` → records `modeling_bert.py` ✓
  → `_record_module_graph_edge("transformers.models.bert.modeling_bert")` → stack empty → return

At session end: BFS from bert's seeds (includes `modeling_bert.py`) through `file_graph`:
- `modeling_bert.py` → {`configuration_bert.py`, `activations.py`, `cache_utils.py`, ...} ✓
- Each of those → their own deps (graph edges built when they first loaded during afmoe)
- Full transitive set recovered with zero re-imports

### Session-end transitive closure

In `record_ci_source_deps_by_file`:
1. Convert module-name graph → file-path graph (via `sys.modules[mod].__file__`)
2. For each test file: gather seeds (collection + execution deps), BFS through file_graph
3. Store newly-discovered files under `"graph_transitive"` key in `deps_by_file.json`

### Empirical results (2026-11-01, afmoe→bert→vit, no xdist, --collect-only)

| Test file        | collection | graph_transitive | total |
|------------------|-----------|-----------------|-------|
| afmoe (1st)      | 241       | 0               | 241   |
| bert  (2nd)      | 32        | 202             | 234   |
| vit   (3rd)      | 29        | 207             | 236   |

Module graph: 183 parent modules, 1932 total direct edges.

Key signal files confirmed present in bert and vit totals:
- `modeling_bert.py`, `configuration_bert.py`, `modeling_utils.py` — in bert's graph_transitive ✓
- `modeling_vit.py`, `configuration_vit.py`, `image_processing_pil_vit.py` — in vit's graph_transitive ✓

`fill_mask.py` absent from bert: correct — PR #50158 uses string refs, breaking the static
chain from test_pipeline_mixin to pipeline test files.

Total coverage parity: afmoe=241, bert=234, vit=236. The ~7 difference from afmoe are
modules imported during afmoe's collection but not reachable from bert/vit's seed sets
via graph edges (framework modules loaded before any test file was set as current context).

### Output files
- `deps_by_file.json`: now has `"graph_transitive"` key per test file (in addition to
  `"collection"` and `"execution"`)
- `module_graph.json`: raw graph `{module_name: [direct_children]}`, written at session end
  xdist-aware: workers write fragments, controller merges with set union

### New functions / globals in testing_utils.py
- `_ci_module_graph: dict[str, set[str]]` — module-name → direct child module names
- `_ci_module_loading_stack: list[str]` — currently-loading module stack
- `_record_module_graph_edge(module_name)` — records edge from stack[-1] to module_name
- `write_ci_module_graph()` — writes module_graph.json at session end (xdist-aware)
- Modified `_tracking_import`: name resolution moved before `_original_import` call,
  push/pop loading stack added, calls `_record_module_graph_edge` in addition to existing
- Modified `_tracking_import_module`: same pattern
- Modified `record_ci_source_deps_by_file`: augments output with `graph_transitive`

### Bug fix: collate_ci_source_deps.py (2026-11-01)

`deps_by_file.json` now has three phase keys: `collection`, `execution`, `graph_transitive`.

The original `defaultdict` initializer hardcoded only the two original phases:
```python
# BEFORE (broken — KeyError on "graph_transitive"):
test_to_phases = defaultdict(lambda: {"collection": set(), "execution": set()})
test_to_phases[test_file][phase].update(src_files)  # KeyError if phase = "graph_transitive"
```

Fixed to accept any phase dynamically:
```python
# AFTER:
test_to_phases = defaultdict(dict)
test_to_phases[test_file].setdefault(phase, set()).update(src_files)
```

Docstring updated to show all three phases in example output.

### Empirical verification re-run (2026-11-01, auto→bert→vit)

Re-run with `CI_RECORD_DEPS_INCLUDE_COLLECTION=1`, no xdist, `--collect-only`:

| Test file        | collection | graph_transitive | total |
|------------------|-----------|-----------------|-------|
| auto (1st)       | 248       | —               | 248   |
| bert (2nd)       | 10        | 224             | 234   |
| vit  (3rd)       | 29        | 207             | 236   |

Module graph: 187 parent modules, 1985 total direct edges.

Output files land at `Path(CI_RECORD_DEPS_OUTPUT).parent / "deps_by_file.json"` and
`module_graph.json` — so the env var should point to a FILE inside the desired dir,
e.g. `CI_RECORD_DEPS_OUTPUT=./reports/deps.json` → files land in `./reports/`.

### Three-key format design note

The `collection / execution / graph_transitive` split means no separate "with/without
transitive" flag is needed — consumers choose which keys to union:
- Conservative (low noise): union `collection` + `execution` only
- Complete (maximum coverage): union all three
- `graph_transitive` key is absent on the first-collected test file (it got everything
  directly in `collection`), present on 2nd+ test files.

### Package __init__.py files in the graph

`transformers.models.bert` → `transformers/models/bert/__init__.py` — a real file, not
noise. If a test reaches it, it imported something from the bert package, so bert file
changes legitimately affect that test. The lazy `transformers/__init__.py` and
`transformers/models/__init__.py` are also real but low-signal (rarely changed; lazy so
they don't pull in model code at import time — a change to them would correctly trigger
all tests anyway).

### BFS path analysis: why vit reaches bart/clip/deepseek/qwen3/nllb_moe

Traced via the module-name graph. Two distinct root causes:

**1. Generation mixin (legitimate)**
```
test_modeling_vit.py
  → tests/test_modeling_common.py         (vit inherits ModelTesterMixin)
  → tests/generation/test_utils.py        (GenerationTesterMixin)
  → transformers/models/bart/modeling_bart.py  (generation tests use bart as reference)
```
Correct — if `modeling_bart.py` changes, `test_generation_utils.py` could break, and
that mixin is exercised by vit's test class. **Survives PR #50000.**

**2. Eager exporter imports (noise, fixed by PR #50000)**
```
test_modeling_vit.py
  → tests/test_modeling_common.py         (ModelTesterMixin → ExportTesterMixin)
  → tests/exporters/test_export.py
  → transformers/exporters/exporter_dynamo.py  (eager @register_patch resolution)
  → transformers/models/deepseek_v2/modeling_deepseek_v2.py
  → transformers/models/nllb_moe/modeling_nllb_moe.py
  → transformers/models/qwen2_vl/modeling_qwen2_vl.py
  → transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py  (etc.)
```
`@register_patch` previously called `_resolve_dotted_path` (which runs
`importlib.import_module`) at decoration time, pulling in ~50 model files the moment any
exporter file was imported. PR #50000 (merged 2026-10-26, not yet in `fetcher-recording`)
stores the dotted-path string and resolves lazily in `apply_patches` instead.

**Effect of PR #50000 on graph (verified by temporary cherry-pick):**

| Metric | Before | After |
|--------|--------|-------|
| auto collection | 248 | 136 (-112) |
| bert graph_transitive | 224 | 111 (-113) |
| vit graph_transitive | 207 | 94 (-113) |
| module graph edges | 1985 | 776 (-1209) |
| deepseek/qwen3/nllb/qwen2_vl in vit | present | **gone** |
| bart in vit | present | **still present** (legitimate) |

Output files saved locally:
- Before: `deps_by_file.json`, `module_graph.json` (repo root)
- After:  `tmp_graph_run2/deps_by_file.json`, `tmp_graph_run2/module_graph.json`

**Conclusion**: once `fetcher-recording` is rebased onto main (picking up PR #50000),
the graph will be ~60% smaller and free of cross-model exporter noise. Bart remains
because it is a real dependency via the generation mixin, not noise.

### Real CI analysis: 8 shards, run 41302880472 (2026-11-01)

Artifacts: `ci_artifacts/run_41302880472/shard{0..7}/deps_by_file.json`

Per-shard summary (xdist, 8 workers per shard, real test execution):
- 65-66 test files per shard, 525 unique test files total across all shards
- 64-65 of 66 have `graph_transitive` (the one without = first collected in its worker)
- Module graph per shard: ~310-350 parent modules, ~3400-3700 direct edges
- All 3 phases populated (`execution` non-empty — real test runs, not just collection)

#### Universal file set (random ideas, not necessary to implement)

Files present in every test file's dep set (all phases unioned) vary by shard (39-112).
Global intersection across all 8 shards = **30 files**.
Excluding `test_modeling_pp_chart2table.py` (outlier — custom arch, no standard modeling
stack) = **38 files**.

These 38 are the true universal framework deps (modeling_utils, configuration_utils,
tokenization_utils_base, generation/utils, cache_utils, modeling_outputs, etc.).

**Random idea**: strip the universal set from each test file's dep list before writing
`fetcher_source_deps.json`. This would:
- Reduce file size significantly
- Remove noise (static fetcher already handles these universally anyway)
- Keep only model-specific edges — the real value the recorder adds

`pp_chart2table` is the only outlier — custom architecture, missing conversion_mapping,
core_model_loading, modeling_layers, modeling_outputs entirely.

**Also noted**: `graph_transitive_collection` vs `graph_transitive_execution` as separate
keys could be useful in future (currently BFS seeds from union of both). Trivial to add.
Not needed now.
