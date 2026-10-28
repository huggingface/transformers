# Migration: unittest.TestCase → Pure pytest

**Status**: Long-term project, not started
**Scope**: Common tests from `ModelTesterMixin` only — integration tests are out of scope for now

Current pattern in concrete model test files:
```python
class BertModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
```
Target: drop `unittest.TestCase` from every such class.

---

## Motivation

- Unlocks pytest fixtures — clean parametrization, no `__init_subclass__` workarounds needed
- Enables `@pytest.mark.parametrize` directly on tests (directly unblocks Raushan's omni-modal tester PR)
- Cleaner test IDs: `test_forward[image]` instead of `test_forward_image`
- pytest assertion rewriting gives richer diffs than `self.assertEqual` — actual upgrade in output quality
- Mixins (`ModelTesterMixin`, `GenerationTesterMixin`, etc.) stay structurally the same

---

## Where the real work is

**`tests/test_modeling_common.py`** is the primary target — this is where `ModelTesterMixin` lives.
It does NOT inherit from `unittest.TestCase` itself, but all its methods rely on `unittest.TestCase`
being somewhere in the MRO of the concrete class. Remove `unittest.TestCase` from the concrete
classes, and all the `self.assert*` / `self.skipTest` / `self.subTest` calls in the mixin break.

Counts in `test_modeling_common.py`:
- `self.assert*` calls: **261**
- `self.skipTest` calls: **113**
- `self.subTest` calls: **24**
- `setUpClass` / `tearDownClass`: **0**
- `unittest.TestCase` directly: **0** (it's a pure mixin)

The concrete model test files (bert, llama, etc.) are secondary — once `test_modeling_common.py`
is converted, those files mostly just need their `unittest.TestCase` removed from the class
declaration. Their own `self.assert*` counts are small (bert: 2, roberta: 8, llama: 15, gpt2: 19).

---

## setup/tearDown compatibility

### Method-level (per test)
**CONFIRMED BY EXPERIMENT**: pytest does NOT call camelCase `setUp`/`tearDown` in plain classes.
They must be renamed:
- `setUp` → `setup_method` (or `setup`)
- `tearDown` → `teardown_method` (or `teardown`)

### Class-level (once per class)
Same — must be renamed:
- `setUpClass` / `tearDownClass` → `setup_class` / `teardown_class` (keep `@classmethod`)

---

## What needs to change (in `test_modeling_common.py`)

### 1. `self.assert*` — 261 instances (mechanical)
Replace with bare `assert` or `pytest.raises`. Tool: `unittest2pytest` handles ~90% automatically.

Common conversions:
- `self.assertEqual(a, b)` → `assert a == b`
- `self.assertTrue(x)` → `assert x`
- `self.assertFalse(x)` → `assert not x`
- `self.assertIsNone(x)` → `assert x is None`
- `self.assertIsNotNone(x)` → `assert x is not None`
- `self.assertIn(a, b)` → `assert a in b`
- `self.assertAlmostEqual(a, b)` → `assert abs(a - b) < 1e-7` (or `pytest.approx`)
- `with self.assertRaises(Exc):` → `with pytest.raises(Exc):`
- `self.assertRaises(Exc, func, arg)` (direct call form) → needs case-by-case

### 2. `self.skipTest` — 113 instances (mechanical)
- `self.skipTest(reason)` → `pytest.skip(reason)`

### 3. `self.subTest` — 24 instances (non-trivial)
`with self.subTest(key):` has no direct pytest equivalent. Options:
- **Simplest**: just remove the `with self.subTest(...):` wrapper — the inner assertions
  still run, you just lose the sub-test label. Acceptable for a first pass.
- **Proper**: convert to `@pytest.mark.parametrize` — cleaner but requires restructuring
  the test method.

The 24 instances in `ModelTesterMixin` are mostly used to loop over `model_class` variants.
Removing the wrapper is safe — assertions inside still execute.

### 4. `@unittest.skip*` decorators (in concrete model files)
- `@unittest.skip(reason)` → `@pytest.mark.skip(reason=reason)`
- `@unittest.skipIf(cond, reason)` → `@pytest.mark.skipif(cond, reason=reason)`
- `@unittest.skipUnless(cond, reason)` → `@pytest.mark.skipif(not cond, reason=reason)`

### 5. Class declarations (in each concrete model test file)
Remove `unittest.TestCase` from the base class list.

---

## Phased plan

### Phase 0 — Audit (before touching anything)
- [x] Count `assertRaises(Exc, func, arg)` direct call form in `test_modeling_common.py` → **0** (all uses are context manager form — clean)
- [ ] Read through the 24 `self.subTest` usages — how many can simply drop the wrapper vs. need parametrize
- [x] Confirm pytest calls `setUp`/`tearDown` in a plain class → **NO** (confirmed by experiment — must rename)

### Phase 1 — Pilot with bert
The mixin changes and the concrete model class changes are done together — you can't test
one without the other.

#### Step 1.1 — First run: observe failures (DONE)
- [x] Added `python_classes = ["Test*", "*Test"]` to `pyproject.toml` (required — otherwise classes
      ending in `Test` are not collected at all without `unittest.TestCase`)
- [x] Removed `unittest.TestCase` from `BertModelTest` declaration
- [x] Ran 6 sample tests (`test_model`, `test_config`, `test_save_load`, `test_training`,
      `test_attention_outputs`, `test_hidden_states_output`) with no other changes

**Findings from first run:**
- All 6 fail with same root cause: `AttributeError: 'NoneType' object has no attribute 'prepare_config_and_inputs'`
- Root cause: `setUp` not called → `self.model_tester` stays `None` (class-level default in `ModelTesterMixin`)
- Single failure mode at this layer: `setUp` renaming
- `self.assert*` and `self.skipTest` not yet reached — next layer after setUp is fixed

#### Step 1.2 — Fix setUp, observe next layer (DONE)
- [x] Renamed `setUp` → `setup_method` in `BertModelTest`
- [x] Re-ran 6 tests

**Findings:**
- `test_training` now **passes** (1/6) — it has no `self.assert*` calls, confirming setUp was the only blocker for it
- 5 still fail with: `AttributeError: 'BertModelTest' object has no attribute 'assertEqual'`
- Next layer confirmed: `self.assert*` in `test_modeling_common.py` — as expected

#### Step 1.3 — Compat mixin approach (IN PROGRESS)
Instead of converting every `self.assert*` call, inject a `_UnittestCompatMixin` into
`ModelTesterMixin` that re-implements only the methods actually needed. Add methods
progressively as failures surface — do not pre-implement the full unittest API.

Mixin location: `tests/test_modeling_common.py`, just above `ModelTesterMixin`.
`ModelTesterMixin` now inherits: `(_UnittestCompatMixin, ExportTesterMixin)`.

Methods added (discovered iteratively — sample run first, then full BertModelTest):
- [x] `assertEqual`
- [x] `assertTrue`
- [x] `assertFalse`
- [x] `assertListEqual`
- [x] `assertIn`
- [x] `assertIsNotNone`
- [x] `assertIsInstance`
- [x] `assertGreater`
- [x] `assertLessEqual`
- [x] `assertRaises`
- [x] `assertRaisesRegex`
- [x] `skipTest`
- [x] `subTest` (no-op context manager)
- [x] `_testMethodName` (property — walks call stack to find active `test_*` frame; used in `test_training_mixin.py` for logging)

**Result: 145 passed, 109 skipped, 0 failed** on full `BertModelTest -m "not slow"`

### Phase 2 — Widen to more models (DONE)
- [x] roberta: removed `unittest.TestCase`, renamed `setUp` → `setup_method`
- [x] llama: removed `unittest.TestCase` (setUp is in `causal_lm_tester.py`, not llama file)
- [x] gpt2: removed `unittest.TestCase` (setUp is in `causal_lm_tester.py`)
- [x] t5: removed `unittest.TestCase` from both `T5ModelTest` and `T5EncoderOnlyModelTest`, renamed both `setUp` → `setup_method`
- [x] `causal_lm_tester.py`: renamed `setUp` → `setup_method` (covers llama + gpt2 + any other CausalLMModelTest subclass)

**Additional compat method discovered: `_testMethodName`**
Used in `test_training_mixin.py` for logging the current test name.
Added as a property that walks the call stack to find the active `test_*` frame.

**Results (all 5 test classes, `-m "not slow"`):**
- 684 passed, 557 skipped, 16 failed
- All 16 failures are environment issues (Windows C++ compiler missing for `torch.compile`, libuv not available for distributed tests)
- **Zero migration-related failures**

### Key insight: zero-risk incremental rollout (CLAIM LATER PROVEN WRONG — see Issue 3 and 4)
The compat mixin is purely additive. Models still carrying `unittest.TestCase` continue to
pass unchanged — pytest's unittest handling still calls their `setUp`, and assert methods are
provided by both `unittest.TestCase` and the compat mixin (compat wins in MRO but both
do the same thing). Migrated and unmigrated models coexist safely at any point in the rollout.

**NOTE**: This claim was based on testing only 5 migrated models (bert, roberta, llama, gpt2, t5).
CI revealed that Issue 3 and Issue 4 below break ALL unmigrated models. The rollout is NOT zero-risk
until both issues are fixed.

### Phase 3 — Tooling & full rollout (TODO)
- [ ] Write a script automating the conversion for any model test file
- [ ] Run across all model test files
- [ ] Run full CI, fix remaining edge cases

### Phase 4 — Integration tests (future, separate discussion)
Out of scope until Phase 3 is complete and stable.

---

## Progress

- Phase 0: **DONE**
- Phase 1, Step 1.1 (first run / observe): **DONE**
- Phase 1, Step 1.2 (fix setUp): **DONE** — `test_training` passes
- Phase 1, Step 1.3 (compat mixin — full BertModelTest): **DONE** (13 methods, 145 passed, 0 failed)
- Phase 2 (roberta, llama, gpt2, t5): **DONE** — 684 passed, 0 migration failures
- Phase 3 (tooling + rollout): not started

---

## Issues found so far

### Issue 1: `python_classes` config missing
**Found**: Step 1.1 (local experiment)
Without `unittest.TestCase`, pytest does not collect `*Test`-named classes (its default is `Test*`
prefix only). Classes like `BertModelTest` become invisible to the test runner.
**Fix**: Add to `pyproject.toml`:
```toml
python_classes = ["Test*", "*Test"]
```
**Status**: Applied in `transformers-2/pyproject.toml`. ✅

### Issue 2: `setUp` / `tearDown` not called
**Found**: Step 1.1 (local experiment)
pytest does NOT call camelCase `setUp`/`tearDown` in plain (non-unittest) classes.
Every concrete model test file that defines `setUp` to initialize `self.model_tester` will
silently leave it as `None`, causing all tests to fail.
**Fix**: Rename `setUp` → `setup_method`, `tearDown` → `teardown_method` in every affected file.
**Status**: Applied to bert, roberta, t5, and `causal_lm_tester.py`. ✅

### Issue 3: `_testMethodName` getter is fragile (pre-push concern, not yet fixed)
**Found**: Before pushing PR — listed as a blocker for making it a proper proposal
**Context**: `_testMethodName` is used in `test_training_mixin.py` to log the current test name.
We implemented it as a `@property` using `inspect.stack()` to walk the call stack and find
the first frame whose function name starts with `test_`.

**Problem**: The `inspect.stack()` approach is fragile:
- Could return the wrong result if a helper method name starts with `test_` is anywhere in the call stack
- Could break in unexpected call patterns (e.g. nested calls, decorators, etc.)
- Not a hard crash — just silently returns wrong value, making test logging unreliable

**Better alternatives**:
- Have `setup_method` explicitly set `self._testMethodName = <current test name>` via pytest's
  `request` fixture, so the value is set at the right time without stack walking
- Or just not provide it in the compat mixin and instead fix `test_training_mixin.py` to not rely on it

**Status**: Not yet fixed. Issue is correctness/reliability of the getter, not a crash.

### Issue 4: `_testMethodName` property missing setter — crashes ALL unmigrated `unittest.TestCase` models
**Found**: CI job 123003870110 (PR #50087, shard 1/8) — 2026-10-28
**Note**: This is a DIFFERENT problem from Issue 3. Issue 3 is about getter reliability. This is about
a hard crash caused by the missing setter.

**Root cause**: Our `_UnittestCompatMixin._testMethodName` is a `@property` with only a getter —
no setter. When `unittest.TestCase.__init__` runs for any class that still inherits `unittest.TestCase`,
it immediately executes `self._testMethodName = methodName`. Since the property has no setter,
Python raises:
```
AttributeError: property '_testMethodName' of '...' object has no setter
```
This happens at **collection time**, before any test runs. Every test in the affected file becomes
an ERROR (hard collection failure, not a test failure).

**Why it wasn't caught locally**: Our local testing only ran 5 already-migrated models (bert, roberta,
llama, gpt2, t5 — all with `unittest.TestCase` removed). The crash only occurs in classes that
STILL have `unittest.TestCase`, which was the entire rest of the test suite (hundreds of files).

**Scope**: Every model test file not yet migrated — all classes with `unittest.TestCase` in their
declaration that also inherit `ModelTesterMixin`. Confirmed from shard 1: CvtModelTest, CwmModelTest,
DFineModelTest, DPRModelTest, DPTModelTest, DabDetrModelTest, DacModelTest, DbrxModelTest,
DebertaModelTest, DecisionTransformerModelTest, DeepseekVLHybridModelTest, DeepseekVLModelTest,
DeformableDetrModelTest, DeiTModelTest, DepthAnythingModelTest, DepthProModelTest, DetrModelTest,
DiaModelTest, DiffLlamaModelTest, DinatModelTest, DistilBertModelTest, DogeModelTest,
DonutSwinModelTest, EdgeTamModelTest, EfficientLoFTRModelTest, EfficientNetModelTest,
ElectraModelTest, EncodecModelTest, EomtForUniversalSegmentationTest, ErnieModelTest,
EsmFoldModelTest, EsmModelTest, EsmcModelTest, EuroBertModelTest, EvollaModelTest,
ExaoneMoeModelTest, FalconModelTest … (and hundreds more across all 8 shards)

**Fix**: Add a setter to the property that captures the value set by `unittest.TestCase.__init__`,
and have the getter return it if available, otherwise fall back to stack walking:
```python
@property
def _testMethodName(self):
    if hasattr(self, '_testMethodName_storage'):
        return self._testMethodName_storage
    import inspect
    for frame_info in inspect.stack():
        if frame_info.function.startswith('test_'):
            return frame_info.function
    return ''

@_testMethodName.setter
def _testMethodName(self, value):
    self._testMethodName_storage = value
```
This resolves the hard crash for `unittest.TestCase` subclasses (setter captures value, getter returns it),
while still providing the stack-walk fallback for pure-pytest classes (where the setter is never called).
Note: this does NOT fix Issue 3 (getter is still fragile for pure-pytest classes) — that remains open.

**Status**: **FIXED** ✅ — setter added to property in `_UnittestCompatMixin`. Verified: 475 passed, 0 errors across 2 migrated (bert, roberta) + 2 unmigrated (deberta, distilbert) models in a single pytest run.

---

## CI analysis — job 123003870110 (PR #50087, shard 1/8) — 2026-10-28

**Run**: https://github.com/huggingface/transformers/actions/runs/40740916200/job/123003870110?pr=50087
**Summary**: **6762 total ERRORs** in shard 1 alone. Two distinct root causes found.

---

### CI Issue 5 (= Issue 4 manifesting in CI): `_testMethodName` no setter — see Issue 4 above for full details
**Confirmed in**: CI job 123003870110, shard 1/8
**Error count**: ~536 errors in shard 1
**Status**: Not yet fixed.

---

### CI Issue 6: `setUp` rename in `causal_lm_tester.py` breaks 97 models + leaks as collected class
**Severity**: CRITICAL — 6226 of 6762 errors (92%) in shard 1 alone are from this issue
**Discovered**: CI job 123003870110, ERROR lines all tagged `CausalLMModelTest`

**Root cause (two parts)**:

**Part 1 — pytest collects `CausalLMModelTest` from every model file that imports it.**
After adding `python_classes = ["Test*", "*Test"]` to `pyproject.toml`, pytest now picks up
any class whose name ends in `Test` that is visible at the top level of a test module.
`CausalLMModelTest` is imported by 97 model test files. Since the name ends in `Test`, pytest
collects it as a test class from each of those files — even though it was never meant to be
run directly. At most 97 × (number of tests in CausalLMModelTest) = thousands of ghost tests.

**Part 2 — `setUp` was renamed to `setup_method` in `causal_lm_tester.py`.**
The 97 model test files that import `CausalLMModelTest` mostly still have `unittest.TestCase`
in their own class declarations (e.g. `class FalconModelTest(CausalLMModelTest, unittest.TestCase):`).
When pytest runs those as unittest-style tests, it calls `setUp` (not `setup_method`). Because
`setUp` no longer exists on `CausalLMModelTest`, `self.model_tester` stays `None`, and every
test errors out immediately.

**Scope**: 97 model test files that reference `CausalLMModelTest` (confirmed by grep):
afmoe, apertus, arcee, axk1, axk2, bailing_hybrid, beam, blt, bloom, cohere2_moe,
cohere_compass, cwm, dbrx, deepseek_v2, deepseek_v32, deepseek_v4, diffusion_gemma,
dots1, ernie4_5, ernie4_5_moe, exaone4, exaone_moe, falcon, flex_olmo, flint, gemma,
gemma2, gemma3, gemma3n, gemma4, gemma4_unified, glm, glm4, glm4_moe, glm4_moe_lite,
glm_moe_dsa, gpt2 ✓, gpt_oss, granitemoe, granitemoe_swa, granitemoeshared, granite_swa,
helium, hrm_text, hunyuan_v1_dense, hunyuan_v1_moe, hy_v3, hy_v4, hyperclovax, inkling,
jais2, jetmoe, kimi_linear, kolibri1, laguna, lfm2, lfm2_moe, lfm2_vl, llama ✓,
longcat_flash, mellum, mimo_v2_flash, minicpm3, minimax, minimax_m2, ministral,
ministral3, mistral, mistral4, mixtral, mllama, modernbert_decoder, nanochat, nemotron,
olmo3, olmo_hybrid, persimmon, phi, phi3, phimoe, qwen2, qwen2_moe, qwen3, qwen3_5,
qwen3_5_moe, qwen3_moe, qwen3_next, qwen4_exp, recurrent_gemma, seed_oss, smollm3,
solar_open, stablelm, starcoder2, vaultgemma, youtu, zaya
(✓ = already migrated to pure pytest; the rest are still unmigratable by this bug)

**Fix strategy**:
1. **Short-term (prevent leaking)**: Rename `CausalLMModelTest` to `_CausalLMModelTestBase`
   (with underscore prefix) so pytest does not collect it. OR: add `__test__ = False` class
   attribute to `CausalLMModelTest` to opt it out of pytest collection.
2. **For setUp rename**: Keep BOTH `setUp` and `setup_method` during transition:
   ```python
   def setup_method(self):           # pure-pytest subclasses call this
       self.model_tester = ...
   def setUp(self):                   # unittest.TestCase subclasses still call this
       self.setup_method()
   ```
   This is safe and backwards-compatible.

**Status**: Fix not yet applied (user asked to analyze only, not code).

---

### Summary of CI impact

| Issue | Root cause | Models affected | Error count (shard 1) | Fix complexity |
|-------|-----------|-----------------|----------------------|----------------|
| 4 / CI-5 | `_testMethodName` property no setter | All non-migrated models (hundreds) | ~536 | Small — add setter to property ✅ FIXED |
| CI-6 | `CausalLMModelTest` pytest collection + setUp rename | 97 models × N tests | 6,226 | Small — `__test__ = False` + keep setUp alias |
| **Total** | | | **6,762** | |

Both fixes are small and safe — they do not require touching any individual model test files.

---

## Open questions

- Of the 24 `subTest` usages in `test_modeling_common.py` — how many loop over `model_class`
  variants (safe to remove wrapper) vs. something more structural?
- Does `GenerationTesterMixin` (`test_generation_utils.py`) have the same scale of issues?
  Should it be included in Phase 1?
