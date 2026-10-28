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

### Key insight: zero-risk incremental rollout
The compat mixin is purely additive. Models still carrying `unittest.TestCase` continue to
pass unchanged — pytest's unittest handling still calls their `setUp`, and assert methods are
provided by both `unittest.TestCase` and the compat mixin (compat wins in MRO but both
do the same thing). Migrated and unmigrated models coexist safely at any point in the rollout.

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
**Found**: Step 1.1
Without `unittest.TestCase`, pytest does not collect `*Test`-named classes (its default is `Test*`
prefix only). Classes like `BertModelTest` become invisible to the test runner.
**Fix**: Add to `pyproject.toml`:
```toml
python_classes = ["Test*", "*Test"]
```
**Status**: Applied in `transformers-2/pyproject.toml`

### Issue 2: `setUp` / `tearDown` not called
**Found**: Step 1.1
pytest does NOT call camelCase `setUp`/`tearDown` in plain (non-unittest) classes.
Every concrete model test file that defines `setUp` to initialize `self.model_tester` will
silently leave it as `None`, causing all tests to fail.
**Fix**: Rename `setUp` → `setup_method`, `tearDown` → `teardown_method` in every affected file.
**Status**: Not yet applied — next step

---

## Open questions

- Of the 24 `subTest` usages in `test_modeling_common.py` — how many loop over `model_class`
  variants (safe to remove wrapper) vs. something more structural?
- Does `GenerationTesterMixin` (`test_generation_utils.py`) have the same scale of issues?
  Should it be included in Phase 1?
