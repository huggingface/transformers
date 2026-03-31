# Doc brief: Rewrite of `docs/source/en/testing.md`

## Overview

Rewrite the testing guide to serve model contributors. The current doc is pytest tips mixed with CI maintainer notes — it says nothing about the mixin-based test architecture, which is the single most important thing a model contributor needs. The new doc leads with a working example, explains the system that makes it work, and pushes generic pytest reference to the end.

This doc also drives adoption of `CausalLMModelTest` and `VLMModelTest` as the recommended paths for new models.

## Audience

- **Primary reader**: A developer adding a new model to Transformers (or modifying an existing model's tests)
- **Assumed knowledge**: Python, pytest basics, PyTorch fundamentals, familiarity with Transformers as a user
- **Goal**: After reading, the contributor can write a complete test file for a new model — knowing which classes to inherit, what methods to implement, what gets tested automatically, and how to run their tests

## Doc type

Mixed: primarily a **how-to guide** (write tests for your model), with **conceptual explanation** of the test architecture and a **reference** appendix for utilities and mixin tests.

## Scope

- **In scope**: Model test architecture (mixins, tester pattern), writing model/config/tokenizer tests, running tests, key decorators and helpers, control flags, tiny model creation
- **Out of scope**: CI infrastructure internals (CircleCI config, GitHub Actions workflows, experimental CI features), contributing workflow (covered in `contributing.md`), detailed model implementation (covered in `add_new_model.md`)

## Prerequisites

- A development install of Transformers (`pip install -e ".[dev]"`)
- PyTorch installed
- Familiarity with `add_new_model.md` (the testing doc picks up where that doc's "Add model tests" section leaves off)

---

## Outline

### 1. Introduction + quick-start (brief)

3-4 sentences on what this page covers. Two test suites exist (`tests/` for API, `examples/` for applications). Link to `add_new_model.md` for the full model addition workflow.

**Quick-start box** right at the top — the 3 commands a contributor needs immediately:

```bash
# Run your model's tests
pytest tests/models/mymodel/test_modeling_mymodel.py -v

# Run a specific test
pytest tests/models/mymodel/test_modeling_mymodel.py::MyModelTest::test_model

# Include slow integration tests
RUN_SLOW=1 pytest tests/models/mymodel/ -v
```

**CI note** (one paragraph): CI runs your model tests without `@slow` on every PR. Slow tests run on a nightly schedule. Link to `pr_checks.md` for details.

This gets the contributor productive in 30 seconds before they read anything else.

---

### 2. Write tests for a causal LM (`CausalLMModelTest`)

**Lead with a complete working example** — the hook. Show the ~10-line Llama-style test file first, then explain what's happening.

Use fictional `MyModel` examples throughout, with callouts like *"See `tests/models/llama/test_modeling_llama.py` for a real example."*

```python
class MyModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MyModel

@require_torch
class MyModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MyModelTester
```

Then explain:

- **What this gives you**: 100+ auto-generated tests across save/load, generation, pipelines, training, tensor parallelism
- **`base_model_class`**: The only required attribute. Config class, CausalLM class, and task head classes are auto-inferred from naming conventions (explain the convention briefly)
- **When to override**: `all_model_classes`, `pipeline_model_mapping`, control flags — show one example of each
- **When this path works**: Standard naming conventions, standard causal LM architecture
- **When it doesn't**: Encoder-decoder, encoder-only, multimodal (→ see Section 4)

---

### 3. Write tests for a vision-language model (`VLMModelTest`)

Same structure as Section 2 but for VLMs. Show the minimal example using `VLMModelTester` and `VLMModelTest`.

- What it provides vs CausalLMModelTest (inherits ModelTesterMixin + GenerationTesterMixin + PipelineTesterMixin — no TrainingTesterMixin or TensorParallelTesterMixin)
- How to set up image/processor inputs in the tester
- Callout to a real example (e.g., `tests/models/gemma3/test_modeling_gemma3.py` or similar)

Keep this shorter than Section 2 — the concepts are the same, just show the differences.

---

### 4. Write tests for other architectures (general path)

For models that don't fit CausalLMModelTest or VLMModelTest: encoder-only, encoder-decoder, multimodal, non-standard architectures.

#### 4a. The two-class pattern

Now explain the architecture that was implicit in Sections 2-3:

- **`ModelTester`** (plain class): Creates tiny configs and dummy inputs. Constructor stores small dimensions (hidden_size=32, num_layers=2, vocab_size=99). Key methods: `get_config()`, `prepare_config_and_inputs()`, `prepare_config_and_inputs_for_common()`.
- **`ModelTest`** (unittest.TestCase + mixins): Inherits from mixins that auto-generate tests. Sets `all_model_classes` and `pipeline_model_mapping`.
- **The contract**: `prepare_config_and_inputs_for_common()` must return `(config, inputs_dict)`. The mixins call this to get test data. This is the glue between your model-specific tester and the generic test infrastructure.

#### 4b. The mixin chain

Briefly explain what each mixin provides (1 sentence each):

| Mixin | What it tests |
|-------|--------------|
| `ModelTesterMixin` | Save/load, gradient checkpointing, forward signature, model attributes |
| `GenerationTesterMixin` | Greedy, sampling, beam search, assisted decoding |
| `PipelineTesterMixin` | One test per task in `pipeline_model_mapping` |
| `TrainingTesterMixin` | Overfitting on a small batch |
| `TensorParallelTesterMixin` | Distributed tensor parallelism |

Point to source files for the curious.

#### 4c. Step-by-step walkthrough

Walk through writing a custom ModelTester + ModelTest:

1. **Write the `ModelTester`**: Show `get_config()`, `prepare_config_and_inputs()`, `prepare_config_and_inputs_for_common()`. Show the helper functions: `ids_tensor()`, `random_attention_mask()`, `floats_tensor()`.
2. **Write the `ModelTest`**: Inherit from appropriate mixins + `unittest.TestCase`. Set `all_model_classes`, `pipeline_model_mapping`. Implement `setUp()` with `self.model_tester` and `self.config_tester`.
3. **Add model-specific tests**: `create_and_check_*` methods on the tester, called from `test_*` methods on the test class.

Show an abbreviated but complete BERT-style example. Callout: *"See `tests/models/bert/test_modeling_bert.py` for the full version."*

#### 4d. File organization

Where test files live: `tests/models/mymodel/`. Show the standard directory layout:

```
tests/models/mymodel/
├── __init__.py
├── test_modeling_mymodel.py          # Model tests (required)
├── test_tokenization_mymodel.py      # Tokenizer tests (if custom tokenizer)
├── test_image_processing_mymodel.py  # Image processor tests (if vision model)
└── test_processing_mymodel.py        # Processor tests (if multimodal)
```

---

### 5. Config tests with `ConfigTester`

How to use `ConfigTester`. What `run_common_tests()` checks: JSON serialization, file I/O, save/load pretrained, num_labels handling, default initialization, common properties (hidden_size, num_attention_heads, etc.).

When to pass `has_text_modality=False` (vision-only models). Pass `**kwargs` to override config defaults for testing.

Note: CausalLMModelTest and VLMModelTest handle this automatically — this section is for the general path.

---

### 6. Integration tests and tiny models

Two parts:

#### 6a. Writing integration tests

The `@slow` test class pattern (e.g., `MyModelIntegrationTest`). Downloads real weights, runs inference, checks outputs against expected values. Mark with `@slow`. Show a concrete example.

When to use `hf-internal-testing/tiny-random-*` models (pipeline tests, fast smoke tests) vs. real checkpoints (accuracy validation).

#### 6b. Creating tiny models

Full walkthrough of `utils/create_dummy_models.py`:

- Your `ModelTester.get_config()` feeds into the script — it extracts your tiny hyperparameters automatically
- Command to generate locally: `python utils/create_dummy_models.py output_dir -m your_model_type`
- Command to upload: `python utils/create_dummy_models.py output_dir -m your_model_type --upload --organization hf-internal-testing`
- The `hf-internal-testing/tiny-random-{ModelName}` naming convention
- `tests/utils/tiny_model_summary.json` tracks all generated models
- CI automation via `.github/workflows/check_tiny_models.yml` runs daily

---

### 7. Control what gets tested

Reference table of boolean flags on `ModelTesterMixin` that toggle auto-generated tests:

| Flag | Default | What it controls |
|------|---------|-----------------|
| `test_resize_embeddings` | `True` | Embedding resizing tests |
| `test_resize_position_embeddings` | `False` | Position embedding resizing |
| `is_encoder_decoder` | `False` | Encoder-decoder specific tests |
| `has_attentions` | `True` | Attention output tests |
| `test_mismatched_shapes` | `True` | Shape mismatch handling |
| `test_missing_keys` | `True` | Missing key warnings |
| `test_torch_exportable` | `True` | torch.export tests |
| `test_all_params_have_gradient` | `True` | Gradient flow (set False for MoE) |

Show how to override: just set the attribute on your test class.

---

### 8. Run tests (full reference)

Expands on the quick-start from Section 1. Practical commands for daily workflow — no obscure plugins.

#### 8a. Filtering and selection

`-k` keyword filtering, `::Class::method` syntax, `--collect-only -q` to list tests, `-m` marker filtering.

#### 8b. Device selection

`CUDA_VISIBLE_DEVICES=""` for CPU-only, `TRANSFORMERS_TEST_DEVICE="cuda"` to force a device, `TRANSFORMERS_TEST_BACKEND` for custom backends. Brief — 5-6 lines.

#### 8c. Parallel execution

`pytest -n auto --dist=loadfile` (what `make test` does). When to use, when not to.

#### 8d. Environment variables

| Variable | Effect |
|----------|--------|
| `RUN_SLOW=1` | Run `@slow` tests |
| `RUN_PIPELINE_TESTS=1` | Run pipeline tests (default: on) |
| `RUN_TRAINING_TESTS=1` | Run training tests |
| `RUN_TENSOR_PARALLEL_TESTS=1` | Run TP tests (default: on) |
| `CUDA_VISIBLE_DEVICES` | Select GPUs |
| `TRANSFORMERS_TEST_DEVICE` | Override device |

---

### 9. Decorators and skip conditions

Reference for the ~15 decorators model contributors actually use. Organized by category:

- **Speed**: `@slow`, `@tooslow`
- **Device**: `@require_torch`, `@require_torch_gpu`, `@require_torch_multi_gpu`, `@require_torch_accelerator`, `@require_torch_large_gpu(memory=20)`
- **Libraries**: `@require_flash_attn`, `@require_bitsandbytes`, `@require_accelerate`, `@require_tokenizers`, `@require_sentencepiece`, `@require_vision`
- **Stacking and ordering**: Decorators must come *after* `@parameterized.expand` — show correct vs. incorrect example
- **Skipping patterns**: `@unittest.skip`, `@pytest.mark.skipif`, `pytest.skip()` inside a test

Don't list all 127 decorators. Mention that more exist in `testing_utils.py`.

---

### 10. Test utilities reference

Brief reference for helpers used inside tests:

- **Tensor generators**: `ids_tensor(shape, vocab_size)`, `floats_tensor(shape, scale)`, `random_attention_mask(shape)` — what they return, when to use each
- **Temp directories**: `self.get_auto_remove_tmp_dir()` — auto-cleaned after test
- **Logger capture**: `CaptureLogger(logger)` — one example for testing warning messages
- **Environment**: `@mockenv(VAR="value")` decorator
- **Reproducibility**: `set_seed(42)` for deterministic tests
- **Parametrization**: `@parameterized.expand(...)` for unittest-style, `@pytest.mark.parametrize` for pytest-style — brief example of each
- **Network resilience**: `@hub_retry()` for Hub downloads (note: auto-applied by ModelTesterMixin)
- **Flaky tests**: `@is_flaky(max_attempts=5)` for non-deterministic tests
- **Path accessors**: `TestCasePlus` provides `self.test_file_dir`, `self.tests_dir`, `self.repo_root_dir`

---

### 11. What the mixins test (appendix with troubleshooting)

Reference table for each mixin: test name, what it checks, and what to fix when it fails. This IS the troubleshooting section — contributors consult it when a mixin test fails.

Format:

#### ModelTesterMixin tests

| Test | What it checks | Common fix |
|------|---------------|------------|
| `test_save_load` | Save/load round-trip produces identical outputs | Check custom serialization in `from_pretrained`/`save_pretrained` |
| `test_model_is_small` | Config dimensions are tiny enough for fast tests | Reduce hidden_size, num_layers, vocab_size in your ModelTester |
| `test_forward_signature` | Forward method accepts expected args | Ensure `forward()` signature matches base class |
| `test_gradient_checkpointing_*` | Gradient checkpointing works | Implement `_supports_gradient_checkpointing` |
| ... | ... | ... |

#### GenerationTesterMixin tests

| Test | What it checks | Common fix |
|------|---------------|------------|
| `test_greedy_generate` | Greedy decoding produces valid output | Check `prepare_inputs_for_generation()` |
| ... | ... | ... |

#### PipelineTesterMixin tests

Brief: one test per task in `pipeline_model_mapping`. If a pipeline test fails, verify the model class works with the corresponding pipeline.

#### TrainingTesterMixin tests

Brief: `test_training_overfit` — verifies model can overfit on a small batch. Configurable via `training_overfit_steps`, `training_overfit_batch_size`, `training_overfit_learning_rate`.

#### ConfigTester tests

| Test | What it checks | Common fix |
|------|---------------|------------|
| `create_and_test_config_to_json_string` | JSON serialization | Ensure all config attributes are JSON-serializable |
| `create_and_test_config_from_and_save_pretrained` | Save/load pretrained config | Check `__init__` accepts all saved attributes |
| `check_config_arguments_init` | Common kwargs are set | Ensure config stores common args (hidden_size, etc.) |
| ... | ... | ... |

---

## Sections from current doc to remove entirely

- "How transformers are tested" CI section (CircleCI/GitHub Actions internals)
- "Working with GitHub Actions workflows" (maintainer-only ci_ branch workflow)
- "Testing Experimental CI Features" (set +euo pipefail, allow-failure workarounds)
- "DeepSpeed integration" (niche, not model-addition)
- pytest-picked, pytest-flakefinder, pytest-repeat, pytest-random-order deep dive
- pytest-sugar, pytest-pspec, pytest-instafail (cosmetic plugins)
- Pastebin, JUnit XML, color control
- "Sending test report to online pastebin service"
- CaptureStdout/CaptureStderr full examples (replaced by single CaptureLogger example)
- Multiple approaches to stdout capture (capsys, contextlib.redirect_stdout)

## Cross-links

- **Points to**: `add_new_model.md` (full model addition guide), `modular_transformers.md` (modular model files), `contributing.md` (general contribution workflow), `pr_checks.md` (CI checks on PRs)
- **Pointed from**: `add_new_model.md` (its "Add model tests" section should link here for details), `contributing.md`

## Open questions

All resolved during planning.

1. ~~Should CausalLMModelTest be the recommended default?~~ **Yes** — lead with it to drive adoption, even though only 15% of existing tests use it. It's the intended path for new causal LMs.
2. ~~Tiny model creation approach?~~ **Resolved** — `utils/create_dummy_models.py` reads `ModelTester.get_config()`. Full subsection with commands.
3. ~~Tokenizer/processor test depth?~~ **Resolved** — brief coverage. Tokenizer tests are much simpler (~34 lines for BERT). `TokenizerTesterMixin` exists. Mention in file organization section, don't give full walkthrough.
4. ~~VLMModelTest coverage?~~ **Yes** — full coverage as a second recommended fast path (Section 3).
5. ~~Example style?~~ Fictional `MyModel` throughout with callouts to real files (Llama, BERT, Gemma3).
6. ~~Section ordering?~~ Example-first: show working code, then explain the architecture.
7. ~~Run tests position?~~ Quick-start at top of doc, full reference later (Section 8).
8. ~~CI content?~~ One paragraph + link to pr_checks.md.
9. ~~Output capture?~~ Single CaptureLogger example only.
10. ~~Troubleshooting?~~ Folded into appendix as "common fix" column in mixin test tables.
